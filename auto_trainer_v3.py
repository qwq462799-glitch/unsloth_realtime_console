#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsloth 自动增量微调系统 V3 (双重审计修复版)
=============================================
基于 V2 版本两轮审计修复，新增 Web UI 管理界面。

第一轮修复 (共 12 项):
  BUG-01 [严重] save_to_raw() 文件名碰撞 — 用目录文件数做唯一性，循环内多次调用必冲突
  BUG-02 [严重] fetch_http_api() 游标丢失 — response_path 遍历后 data 已被覆盖，cursor 取错
  BUG-03 [严重] 训练无异常捕获 — start_unsloth_training 抛异常时主循环崩溃退出
  BUG-04 [严重] 无优雅关闭机制 — Ctrl+C 可能留下临时文件、损坏游标
  BUG-05 [中等] LAST_POLL key 脆弱 — json.dumps 嵌套 dict 不可靠且性能差
  BUG-06 [中等] 采集无源级隔离 — 单个数据源异常会中断整轮采集
  BUG-07 [中等] adapter 命名秒级冲突 — 同一秒内两次训练产出同名目录
  BUG-08 [中等] 训练数据未 shuffle — 模型按文件创建顺序学习，引入顺序偏差
  BUG-09 [中等] local_dir 无已处理追踪 — 每个周期重复处理所有 JSON 文件
  BUG-10 [低] 无线程安全日志 — print() 在多线程/多源并发下输出交错
  BUG-11 [低] save_cursor 非原子写 — 直接写文件可能因中断导致游标损坏
  BUG-12 [低] HTTP API 无超时容错 — raise_for_status() 未捕获，单次失败中断采集

第二轮修复 (共 6 项):
  BUG-13 [严重] trigger_training 竞态条件 — is_training 检查在锁外，多线程可同时进入训练
  BUG-14 [中等] _ensure_train_deps 非线程安全 — 无锁保护且失败后无冷却期，每次重试都触发 import
  BUG-15 [低] 未使用的 import wraps — 死代码导入
  BUG-16 [中等] 训练后 batch 文件未清理 — BUFFER_DIR 中的 batch 文件永久累积
  BUG-17 [中等] fetch_local_dir 回读 save_to_raw 产出 — 无效循环浪费 I/O 和 CPU
  BUG-18 [安全] api_delete_adapter 路径遍历不足 — 仅检查 .. 和 /，缺少 realpath 验证

新增功能:
  - Flask Web UI 管理界面 (仪表盘/数据源/训练/适配器/数据浏览/日志)
  - RESTful API 接口 (10 个端点)
  - 后台训练线程 + 线程安全状态管理
  - 结构化日志系统 (文件+控制台+内存缓冲区)
"""

import os
import sys
import time
import json
import uuid
import shutil
import random
import signal
import hashlib
import logging
import tempfile
import threading
import traceback
import inspect
import hmac
import ipaddress
import re
from pathlib import Path
from datetime import datetime
from collections import deque


# ======================== 日志系统 (BUG-10 修复) ========================

APP_DIR = Path(__file__).resolve().parent
CONFIG_DIR = APP_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_SETTINGS_FILE = CONFIG_DIR / "system_settings.json"
SOURCE_CONFIG_FILE = CONFIG_DIR / "data_sources.json"
SOURCE_STATE_FILE = CONFIG_DIR / "source_state.json"


def _read_json_object(path):
    """读取 JSON 对象配置；配置不存在或损坏时返回空对象。"""
    try:
        with open(path, "r", encoding="utf-8") as fp:
            value = json.load(fp)
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # logger 尚可能未初始化，使用 stderr 避免启动时静默忽略配置损坏。
        print(f"[配置警告] 无法读取 {path}: {exc}", file=sys.stderr)
        return {}


def _write_json_atomic(path, value, mode=0o600):
    """原子写入 JSON，并尽可能限制本地配置权限。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        os.chmod(tmp_name, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(value, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_name, target)
        try:
            os.chmod(target, mode)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


LOG_DIR = os.getenv("AUTO_TRAINER_LOG_DIR", str(APP_DIR / "logs"))
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("AutoTrainer")
logger.setLevel(logging.DEBUG)

_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(_formatter)
logger.addHandler(_console)

_file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "auto_trainer.log"), encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)


# ======================== 数据库 / HTTP 依赖 (按需安装) ========================

try:
    import pymysql
except ImportError:
    pymysql = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    import sqlite3
except ImportError:
    sqlite3 = None

try:
    import requests as http_requests
except ImportError:
    http_requests = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

# Flask (Web UI 必需)
try:
    from flask import Flask, jsonify, request as flask_request, render_template
except ImportError:
    logger.error("Flask 未安装，请运行: pip install flask")
    sys.exit(1)


# ======================== Unsloth 训练依赖 ========================

# 延迟导入，允许在无 GPU 环境下启动 Web UI
_TRAIN_DEPS = None
_TRAIN_DEPS_LOCK = threading.Lock()
_TRAIN_DEPS_FAIL_TIME = 0  # 上次加载失败的时间戳
_TRAIN_DEPS_RETRY_INTERVAL = 300  # 失败后至少等 5 分钟再重试


def _ensure_train_deps():
    """延迟加载训练相关依赖 (BUG-14 修复: 线程安全+失败冷却重试)。"""
    global _TRAIN_DEPS, _TRAIN_DEPS_FAIL_TIME

    # 快速路径: 已加载成功，直接返回
    if isinstance(_TRAIN_DEPS, dict):
        return _TRAIN_DEPS

    # 快速路径: 近期刚失败过，跳过重试
    if _TRAIN_DEPS_FAIL_TIME and (time.time() - _TRAIN_DEPS_FAIL_TIME) < _TRAIN_DEPS_RETRY_INTERVAL:
        return None

    with _TRAIN_DEPS_LOCK:
        # 双重检查锁
        if isinstance(_TRAIN_DEPS, dict):
            return _TRAIN_DEPS
        if _TRAIN_DEPS_FAIL_TIME and (time.time() - _TRAIN_DEPS_FAIL_TIME) < _TRAIN_DEPS_RETRY_INTERVAL:
            return None

        try:
            from datasets import load_dataset
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig

            _TRAIN_DEPS = {
                "load_dataset": load_dataset,
                "FastLanguageModel": FastLanguageModel,
                "SFTTrainer": SFTTrainer,
                "SFTConfig": SFTConfig,
            }
            _TRAIN_DEPS_FAIL_TIME = 0
            logger.info("训练依赖加载成功")
        except ImportError as e:
            logger.warning(f"训练依赖加载失败 (Web UI 仍可正常使用, {_TRAIN_DEPS_RETRY_INTERVAL}s 后重试): {e}")
            _TRAIN_DEPS = None
            _TRAIN_DEPS_FAIL_TIME = time.time()

    return _TRAIN_DEPS


# ======================== 1. 配置区 ========================

# 这些值既可由环境变量提供，也可在 WebUI 中持久化配置。
# 每项 disabled 时回退至启动时的环境变量/默认值；值本身仍保留，便于重新启用。
_DATA_DIR_SETTING = os.getenv("AUTO_TRAINER_DATA_DIR", "data")
_DEFAULT_BASE_DATA_PATH = str((APP_DIR / _DATA_DIR_SETTING).resolve()) if not os.path.isabs(_DATA_DIR_SETTING) else os.path.abspath(_DATA_DIR_SETTING)
DEFAULT_SYSTEM_VALUES = {
    "model_name": os.getenv("AUTO_TRAINER_MODEL", "unsloth/Qwen2.5-3B-Instruct"),
    "max_seq_length": int(os.getenv("AUTO_TRAINER_MAX_SEQ_LENGTH", "1024")),
    "buffer_threshold": int(os.getenv("AUTO_TRAINER_BUFFER_THRESHOLD", "50")),
    "max_adapters": int(os.getenv("AUTO_TRAINER_MAX_ADAPTERS", "5")),
    "db_poll_interval": int(os.getenv("AUTO_TRAINER_DB_POLL_INTERVAL", "300")),
    "base_data_path": _DEFAULT_BASE_DATA_PATH,
    "max_steps": int(os.getenv("AUTO_TRAINER_MAX_STEPS", "50")),
    "training_backend": os.getenv("TRAINING_BACKEND", "local").strip().lower(),
}

SYSTEM_SETTING_META = {
    "model_name": {"label": "基础模型名称", "type": "text", "description": "Hugging Face 模型 ID 或本地模型路径。修改后用于下一轮训练。"},
    "max_seq_length": {"label": "最大序列长度", "type": "number", "min": 128, "max": 32768, "description": "训练时允许的最大 token 长度。更大值会占用更多显存。"},
    "buffer_threshold": {"label": "训练数据阈值", "type": "number", "min": 1, "max": 100000, "description": "待训练样本达到此数量时，后台调度尝试启动训练。"},
    "max_adapters": {"label": "保留 Adapter 数量", "type": "number", "min": 1, "max": 1000, "description": "本项目本地训练输出目录中保留的 Adapter 数量。"},
    "db_poll_interval": {"label": "全局默认采集间隔（秒）", "type": "number", "min": 10, "max": 86400, "description": "默认轮询周期；数据源单独配置的间隔优先。"},
    "base_data_path": {"label": "数据基础目录", "type": "text", "restart_required": True, "description": "修改后需要重启服务，才会切换数据、Adapter 与游标目录。请先备份旧目录。"},
    "max_steps": {"label": "单次训练最大步数", "type": "number", "min": 1, "max": 10000000, "description": "每次 SFT 的最大训练步数。"},
    "training_backend": {"label": "训练执行后端", "type": "select", "options": ["local", "studio_mcp"], "description": "local 为本地 Unsloth/TRL；studio_mcp 将任务提交给 Unsloth Studio MCP。"},
}

_saved_settings = _read_json_object(SYSTEM_SETTINGS_FILE)
SYSTEM_SETTINGS = {}
for _key, _default in DEFAULT_SYSTEM_VALUES.items():
    _saved = _saved_settings.get(_key, {})
    if not isinstance(_saved, dict):
        _saved = {}
    SYSTEM_SETTINGS[_key] = {
        "value": _saved.get("value", _default),
        "enabled": bool(_saved.get("enabled", True)),
    }


def _system_value(key):
    entry = SYSTEM_SETTINGS.get(key, {})
    if entry.get("enabled", True):
        return entry.get("value", DEFAULT_SYSTEM_VALUES[key])
    return DEFAULT_SYSTEM_VALUES[key]


def _resolve_data_path(value):
    raw = os.path.expanduser(str(value).strip())
    if not raw:
        raise ValueError("数据基础目录不能为空")
    return os.path.abspath(raw if os.path.isabs(raw) else str(APP_DIR / raw))


# 配置文件可能被手工编辑或来自旧版本，启动时先校验，避免无效值导致服务起不来。
for _key, _entry in SYSTEM_SETTINGS.items():
    _meta = SYSTEM_SETTING_META[_key]
    _raw = _entry.get("value", DEFAULT_SYSTEM_VALUES[_key])
    try:
        if _meta["type"] == "number":
            _raw = int(_raw)
            if _raw < _meta.get("min", 0) or _raw > _meta.get("max", 2**31 - 1):
                raise ValueError("out of range")
        else:
            _raw = str(_raw).strip()
            if not _raw or len(_raw) > 1024 or "\x00" in _raw:
                raise ValueError("invalid string")
            if _meta["type"] == "select" and _raw not in _meta.get("options", []):
                raise ValueError("invalid choice")
            if _key == "base_data_path":
                _raw = _resolve_data_path(_raw)
        _entry["value"] = _raw
    except (TypeError, ValueError):
        print(f"[配置警告] { _key } 的保存值无效，回退为默认值", file=sys.stderr)
        _entry["value"] = DEFAULT_SYSTEM_VALUES[_key]


MODEL_NAME = str(_system_value("model_name"))
MAX_SEQ_LENGTH = int(_system_value("max_seq_length"))
BUFFER_THRESHOLD = int(_system_value("buffer_threshold"))
MAX_ADAPTERS = int(_system_value("max_adapters"))
DB_POLL_INTERVAL = int(_system_value("db_poll_interval"))
MAX_STEPS = int(_system_value("max_steps"))
TRAINING_BACKEND = str(_system_value("training_backend")).strip().lower()
BASE_DATA_PATH = _resolve_data_path(_system_value("base_data_path"))

INBOX_DIR = os.path.join(BASE_DATA_PATH, "inbox")  # 外部手工投递目录，与训练队列隔离
RAW_DIR = os.path.join(BASE_DATA_PATH, "raw")
BUFFER_DIR = os.path.join(BASE_DATA_PATH, "processed")
ADAPTER_DIR = os.path.join(BASE_DATA_PATH, "adapters")
FOUNDATIONAL_DATA_PATH = os.path.join(BASE_DATA_PATH, "foundational_data.jsonl")
PROCESSED_LOCAL_DIR = os.path.join(BASE_DATA_PATH, "raw_processed")
SEEN_FILE = os.path.join(BASE_DATA_PATH, "seen_hashes.txt")
WEB_HOST = os.getenv("AUTO_TRAINER_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("AUTO_TRAINER_PORT", "5000"))
AUTH_TOKEN = os.getenv("AUTO_TRAINER_TOKEN", "").strip()
MAX_FIELD_CHARS = 20000  # 控制单条训练指令/回答长度，避免异常大字段拖垮服务


# ======================== 2. 线程安全状态管理 ========================

class SystemState:
    """线程安全的系统状态管理器。"""

    def __init__(self):
        self._lock = threading.RLock()
        self.is_running = True
        self.is_training = False
        self.training_start_time = None
        self.training_dataset_size = 0
        self.training_step = 0
        self.training_max_steps = 0
        self.training_progress = 0.0
        self.training_loss = None
        self.training_learning_rate = None
        self.training_epoch = None
        self.training_status = "idle"
        self.training_stage = "空闲"
        self.training_cancel_requested = False
        self.training_model = MODEL_NAME if "MODEL_NAME" in globals() else "unsloth/Qwen2.5-3B-Instruct"
        self.training_output_dir = None
        self.studio_job_id = None
        self.training_backend = TRAINING_BACKEND
        self.training_last_log_time = None
        self.last_training_time = None
        self.last_error = None
        self.last_error_time = None
        self.total_collected = 0
        self.total_trained = 0
        self.total_training_runs = 0
        self.start_time = datetime.now()
        self._raw_counter = 0
        self._logs = deque(maxlen=500)
        self._poll_tracker = {}  # BUG-05: 替代 LAST_POLL 的 dict

    # --- 日志 ---
    def add_log(self, level: str, source: str, message: str):
        with self._lock:
            self._logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "source": source,
                "message": message,
            })

    def get_logs(self, limit: int = 100):
        with self._lock:
            return list(self._logs)[-limit:]

    # --- 原始文件计数器 (BUG-01 修复) ---
    def next_raw_counter(self) -> int:
        with self._lock:
            self._raw_counter += 1
            return self._raw_counter

    # --- 轮询追踪 (BUG-05 修复) ---
    def should_poll(self, source_key: str, interval: int) -> bool:
        with self._lock:
            now = time.time()
            last = self._poll_tracker.get(source_key, 0)
            if now - last < interval:
                return False
            self._poll_tracker[source_key] = now
            return True

    def source_key(self, source: dict) -> str:
        """基于类型和关键标识生成稳定 hash，替代 json.dumps (BUG-05)。"""
        src_type = source.get("type", "unknown")
        if src_type == "local_dir":
            return f"{src_type}:{source.get('dir', '')}"
        elif src_type in ("mysql", "postgres"):
            return f"{src_type}:{source.get('host')}:{source.get('port')}:{source.get('database')}"
        elif src_type == "sqlite":
            return f"{src_type}:{source.get('db_path', '')}"
        elif src_type == "http_api":
            return f"{src_type}:{source.get('url', '')}"
        elif src_type == "csv_excel":
            return f"{src_type}:{source.get('dir', '')}"
        return f"{src_type}:{hash(json.dumps(source, sort_keys=True, default=str))}"

    # --- 训练状态 ---
    def set_training(self, active: bool, dataset_size: int = 0):
        with self._lock:
            was_active = self.is_training
            self.is_training = active
            if active:
                self.training_start_time = datetime.now()
                self.training_dataset_size = dataset_size
                self.training_step = 0
                self.training_max_steps = 0
                self.training_progress = 0.0
                self.training_loss = None
                self.training_learning_rate = None
                self.training_epoch = None
                self.training_status = "preparing"
                self.training_stage = "准备数据与加载模型"
                self.training_model = MODEL_NAME
                self.training_output_dir = None
                self.studio_job_id = None
                self.training_backend = TRAINING_BACKEND
            else:
                if was_active and self.training_start_time:
                    self.last_training_time = datetime.now()
                    self.total_training_runs += 1
                self.training_start_time = None
                self.training_cancel_requested = False

    def update_training(self, **values):
        """原子更新训练状态，供 TrainerCallback 与 API 共用。"""
        with self._lock:
            for key, value in values.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.training_last_log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_error(self, error: str):
        with self._lock:
            self.last_error = error
            self.last_error_time = datetime.now()

    # --- 统计快照 ---
    def snapshot(self) -> dict:
        with self._lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            training_elapsed = None
            if self.is_training and self.training_start_time:
                training_elapsed = (datetime.now() - self.training_start_time).total_seconds()

            return {
                "is_running": self.is_running,
                "is_training": self.is_training,
                "training_elapsed": training_elapsed,
                "training_dataset_size": self.training_dataset_size,
                "training_step": self.training_step,
                "training_max_steps": self.training_max_steps,
                "training_progress": self.training_progress,
                "training_loss": self.training_loss,
                "training_learning_rate": self.training_learning_rate,
                "training_epoch": self.training_epoch,
                "training_status": self.training_status,
                "training_stage": self.training_stage,
                "training_model": self.training_model,
                "training_output_dir": self.training_output_dir,
                "studio_job_id": self.studio_job_id,
                "training_backend": self.training_backend,
                "training_last_log_time": self.training_last_log_time,
                "last_training_time": self.last_training_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_training_time else None,
                "last_error": self.last_error,
                "last_error_time": self.last_error_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_error_time else None,
                "total_collected": self.total_collected,
                "total_trained": self.total_trained,
                "total_training_runs": self.total_training_runs,
                "uptime_seconds": int(uptime),
            }


state = SystemState()


# ======================== 3. 日志桥接器 ========================

class StateLogHandler(logging.Handler):
    """将日志消息同步到 SystemState，供 Web UI 读取。"""

    def emit(self, record):
        try:
            state.add_log(
                level=record.levelname,
                source=record.name,
                message=self.format(record),
            )
        except Exception:
            pass


_bridge = StateLogHandler()
_bridge.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_bridge)


# ======================== 4. 数据源配置 ========================

DATA_SOURCES = [
    {
        "type": "local_dir",
        "enabled": True,
        "dir": INBOX_DIR,
        "name": "本地投递目录",
    },
    {
        "type": "mysql",
        "enabled": False,
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": "ai_data",
        "poll_interval": DB_POLL_INTERVAL,
        "last_cursor_file": os.path.join(BASE_DATA_PATH, "mysql_last_id.txt"),
        "sql": "SELECT id, instruction, output FROM qa_pairs WHERE id > %s ORDER BY id ASC LIMIT 200",
        "name": "MySQL",
    },
    {
        "type": "postgres",
        "enabled": False,
        "host": "127.0.0.1",
        "port": 5432,
        "user": "postgres",
        "password": "",
        "database": "ai_data",
        "poll_interval": DB_POLL_INTERVAL,
        "last_cursor_file": os.path.join(BASE_DATA_PATH, "pg_last_id.txt"),
        "sql": "SELECT id, instruction, output FROM qa_pairs WHERE id > %s ORDER BY id ASC LIMIT 200",
        "name": "PostgreSQL",
    },
    {
        "type": "sqlite",
        "enabled": False,
        "db_path": os.path.join(BASE_DATA_PATH, "local.db"),
        "poll_interval": DB_POLL_INTERVAL,
        "last_cursor_file": os.path.join(BASE_DATA_PATH, "sqlite_last_id.txt"),
        "sql": "SELECT id, instruction, output FROM qa_pairs WHERE id > ? ORDER BY id ASC LIMIT 200",
        "name": "SQLite",
    },
    {
        "type": "http_api",
        "enabled": False,
        "url": "https://your-api.example.com/new-qa",
        "method": "GET",
        "headers": {"Authorization": "Bearer YOUR_TOKEN"},
        "params": {},
        "poll_interval": DB_POLL_INTERVAL,
        "last_cursor_file": os.path.join(BASE_DATA_PATH, "api_last_cursor.txt"),
        "response_path": "data.items",
        "instruction_field": "instruction",
        "output_field": "output",
        "name": "HTTP API",
    },
    {
        "type": "csv_excel",
        "enabled": False,
        "dir": os.path.join(BASE_DATA_PATH, "files"),
        "instruction_col": "instruction",
        "output_col": "output",
        "poll_interval": DB_POLL_INTERVAL,
        "processed_marker_dir": os.path.join(BASE_DATA_PATH, "files_processed"),
        "name": "CSV / Excel",
    },
 ]

# 源配置字段白名单：前端无法覆写 type/enabled/cursor 内部状态等受控字段。
SOURCE_FIELD_META = {
    "local_dir": [
        {"key": "name", "label": "数据源名称", "type": "text", "required": True},
        {"key": "dir", "label": "输入目录", "type": "text", "required": True},
    ],
    "mysql": [
        {"key": "name", "label": "数据源名称", "type": "text", "required": True},
        {"key": "host", "label": "主机地址", "type": "text", "required": True},
        {"key": "port", "label": "端口", "type": "number", "required": True, "min": 1, "max": 65535},
        {"key": "user", "label": "用户名", "type": "text", "required": True},
        {"key": "password", "label": "密码", "type": "password", "secret": True},
        {"key": "database", "label": "数据库名", "type": "text", "required": True},
        {"key": "poll_interval", "label": "轮询间隔（秒）", "type": "number", "required": True, "min": 10, "max": 86400},
        {"key": "sql", "label": "SQL 查询语句", "type": "textarea", "required": True, "description": "只允许单条只读 SELECT / WITH 查询，不允许写入或管理语句。"},
    ],
    "postgres": [
        {"key": "name", "label": "数据源名称", "type": "text", "required": True},
        {"key": "host", "label": "主机地址", "type": "text", "required": True},
        {"key": "port", "label": "端口", "type": "number", "required": True, "min": 1, "max": 65535},
        {"key": "user", "label": "用户名", "type": "text", "required": True},
        {"key": "password", "label": "密码", "type": "password", "secret": True},
        {"key": "database", "label": "数据库名", "type": "text", "required": True},
        {"key": "poll_interval", "label": "轮询间隔（秒）", "type": "number", "required": True, "min": 10, "max": 86400},
        {"key": "sql", "label": "SQL 查询语句", "type": "textarea", "required": True, "description": "只允许单条只读 SELECT / WITH 查询，不允许写入或管理语句。"},
    ],
    "sqlite": [
        {"key": "name", "label": "数据源名称", "type": "text", "required": True},
        {"key": "db_path", "label": "SQLite 文件路径", "type": "text", "required": True},
        {"key": "poll_interval", "label": "轮询间隔（秒）", "type": "number", "required": True, "min": 10, "max": 86400},
        {"key": "sql", "label": "SQL 查询语句", "type": "textarea", "required": True, "description": "只允许单条只读 SELECT / WITH 查询，不允许写入或管理语句。"},
    ],
    "http_api": [
        {"key": "name", "label": "数据源名称", "type": "text", "required": True},
        {"key": "url", "label": "API URL", "type": "text", "required": True, "description": "URL 中 userinfo 及 api_key/token/password 等参数会被遮蔽；保持 ******** 可保留原凭据，输入 __CLEAR__ 可清除。"},
        {"key": "method", "label": "HTTP 方法", "type": "select", "options": ["GET", "POST"]},
        {"key": "headers", "label": "请求头（JSON 对象）", "type": "json", "required": True, "secret": True},
        {"key": "params", "label": "请求参数（JSON 对象）", "type": "json", "required": True},
        {"key": "poll_interval", "label": "轮询间隔（秒）", "type": "number", "required": True, "min": 10, "max": 86400},
        {"key": "response_path", "label": "响应数据路径", "type": "text"},
        {"key": "instruction_field", "label": "指令字段名", "type": "text"},
        {"key": "output_field", "label": "回答字段名", "type": "text"},
    ],
    "csv_excel": [
        {"key": "name", "label": "数据源名称", "type": "text", "required": True},
        {"key": "dir", "label": "CSV/Excel 目录", "type": "text", "required": True},
        {"key": "instruction_col", "label": "指令列名", "type": "text", "required": True},
        {"key": "output_col", "label": "回答列名", "type": "text", "required": True},
        {"key": "poll_interval", "label": "轮询间隔（秒）", "type": "number", "required": True, "min": 10, "max": 86400},
    ],
}

_SECRET_KEYS = ("password", "token", "secret", "authorization", "api_key", "apikey", "credential")


def _is_secret_key(key):
    return any(word in str(key).lower().replace("-", "_") for word in _SECRET_KEYS)


def _redact_nested(value):
    if isinstance(value, dict):
        return {k: ("********" if _is_secret_key(k) and v not in (None, "") else _redact_nested(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_nested(v) for v in value]
    return value


def _restore_nested_secrets(old_value, new_value):
    if isinstance(old_value, dict) and isinstance(new_value, dict):
        result = dict(new_value)
        for key, new_item in list(result.items()):
            old_item = old_value.get(key)
            if _is_secret_key(key) and new_item in ("", "********") and old_item not in (None, ""):
                result[key] = old_item
            elif isinstance(old_item, dict) and isinstance(new_item, dict):
                result[key] = _restore_nested_secrets(old_item, new_item)
        # If a sensitive key is omitted from the submitted JSON, preserve it.
        for key, old_item in old_value.items():
            if _is_secret_key(key) and key not in result and old_item not in (None, ""):
                result[key] = old_item
        return result
    return new_value


def _apply_env_secret(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value


def _redact_url(raw_url):
    """遮蔽 URL authority 中的 userinfo，以及 query string 中疑似凭据。"""
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
    try:
        parts = urlsplit(str(raw_url or ""))
        netloc = parts.netloc.rsplit("@", 1)[-1]
        if "@" in parts.netloc:
            netloc = "********@" + netloc
        query = [(k, "********" if _is_secret_key(k) and v else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query, doseq=True), ""))
    except (TypeError, ValueError):
        return "[URL 已隐藏：无法安全解析]"


def _restore_url_secrets(old_url, submitted_url):
    """编辑 URL 时保留遮蔽或未提交的旧凭据；显式 __CLEAR__ 可清空。"""
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
    old = urlsplit(str(old_url or ""))
    new = urlsplit(str(submitted_url or ""))
    old_host = old.netloc.rsplit("@", 1)[-1]
    old_auth = old.netloc.rsplit("@", 1)[0] if "@" in old.netloc else ""
    if "@" in new.netloc:
        new_auth, new_host = new.netloc.rsplit("@", 1)
        same_host = new_host.lower() == old_host.lower()
        if "********" in new_auth:
            if old_auth and not same_host:
                raise ValueError("修改 API 主机时不能自动转移 URL 账号密码；请重新填写或清除凭据")
            auth = old_auth if same_host else ""
        elif "__CLEAR__" in new_auth:
            auth = ""
        else:
            auth = new_auth
    else:
        new_host = new.netloc
        auth = old_auth if old_auth and new_host.lower() == old_host.lower() else ""
    same_host = new_host.lower() == old_host.lower()
    netloc = f"{auth}@{new_host}" if auth else new_host

    old_pairs = parse_qsl(old.query, keep_blank_values=True)
    new_pairs = parse_qsl(new.query, keep_blank_values=True)
    old_sensitive = {}
    for k, v in old_pairs:
        if _is_secret_key(k):
            old_sensitive.setdefault(k, []).append(v)
    submitted_keys = {k for k, _ in new_pairs}
    query = []
    replaced_old_keys = set()
    for key, value in new_pairs:
        if _is_secret_key(key) and value in ("", "********") and key in old_sensitive:
            if not same_host and value == "********":
                raise ValueError("修改 API 主机时不能自动转移 URL token/API key；请重新填写或清除凭据")
            if key not in replaced_old_keys:
                if same_host:
                    query.extend((key, old_value) for old_value in old_sensitive[key])
                else:
                    query.append((key, ""))
                replaced_old_keys.add(key)
        elif _is_secret_key(key) and value == "__CLEAR__":
            query.append((key, ""))
            replaced_old_keys.add(key)
        else:
            query.append((key, value))
            if _is_secret_key(key):
                replaced_old_keys.add(key)
    # Preserve omitted query credentials unless caller explicitly clears them.
    for key, old_values in old_sensitive.items():
        if same_host and key not in submitted_keys and key not in replaced_old_keys:
            query.extend((key, v) for v in old_values)
    return urlunsplit((new.scheme, netloc, new.path, urlencode(query, doseq=True), ""))


def _validate_readonly_sql(sql):
    """只允许数据采集使用单条只读 SELECT/CTE，避免配置错误执行写入语句。"""
    text = str(sql or "").strip()
    if not text or len(text) > 20000:
        raise ValueError("SQL 查询不能为空，且不能超过 20000 个字符")
    # 移除字符串字面量及注释后再检查，降低关键字误判和注释绕过风险。
    cleaned = re.sub(r"'(?:''|\\.|[^'])*'", "''", text)
    cleaned = re.sub(r'"(?:""|\\.|[^\"])*"', '""', cleaned)
    cleaned = re.sub(r"--[^\r\n]*|/\*.*?\*/", " ", cleaned, flags=re.S)
    if ";" in cleaned:
        raise ValueError("SQL 数据源只允许单条查询语句，不允许分号/多语句")
    if not re.match(r"^\s*(SELECT|WITH)\b", cleaned, flags=re.I):
        raise ValueError("SQL 数据源只允许只读 SELECT 或 WITH 查询")
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|GRANT|REVOKE|TRUNCATE|REPLACE|MERGE|CALL|EXEC|EXECUTE|COPY|ATTACH|DETACH|VACUUM|PRAGMA|BEGIN|COMMIT|ROLLBACK|OUTFILE|DUMPFILE)\b",
        flags=re.I,
    )
    if forbidden.search(cleaned):
        raise ValueError("SQL 查询包含禁止的写入或管理语句关键字")
    return text


def save_sources_config():
    editable = {}
    for src in DATA_SOURCES:
        keys = [field["key"] for field in SOURCE_FIELD_META.get(src.get("type"), [])]
        editable[src["type"]] = {k: src[k] for k in keys if k in src}
    _write_json_atomic(SOURCE_CONFIG_FILE, editable)
    _write_json_atomic(SOURCE_STATE_FILE, {src["type"]: bool(src.get("enabled")) for src in DATA_SOURCES})


def apply_sources_overrides():
    """启动时恢复数据源编辑配置和开关；兼容旧版 data/sources.json。"""
    overrides = _read_json_object(SOURCE_CONFIG_FILE)
    states = _read_json_object(SOURCE_STATE_FILE)
    # Migrate legacy repo versions that saved sources under the data directory.
    legacy_config_env = os.getenv("AUTO_TRAINER_SOURCES_FILE", "").strip()
    legacy_path = Path(legacy_config_env).expanduser() if legacy_config_env else Path(BASE_DATA_PATH) / "sources.json"
    legacy_state_path = Path(BASE_DATA_PATH) / "source_state.json"
    if not overrides and legacy_path.exists():
        overrides = _read_json_object(legacy_path)
    if not states and legacy_state_path.exists():
        states = _read_json_object(legacy_state_path)
    for src in DATA_SOURCES:
        typ = src.get("type")
        patch = overrides.get(typ, {})
        if isinstance(patch, dict):
            allowed = {field["key"] for field in SOURCE_FIELD_META.get(typ, [])}
            for key, value in patch.items():
                if key in allowed:
                    value = _apply_env_secret(value)
                    if key == "sql":
                        try:
                            value = _validate_readonly_sql(value)
                        except ValueError as exc:
                            print(f"[配置警告] 数据源 {typ} 的 SQL 配置不安全，忽略该字段：{exc}", file=sys.stderr)
                            continue
                    src[key] = value
        if typ in states:
            src["enabled"] = bool(states[typ])


apply_sources_overrides()


# ======================== 5. 工具函数 ========================

def ensure_dirs():
    for d in [INBOX_DIR, RAW_DIR, BUFFER_DIR, ADAPTER_DIR, PROCESSED_LOCAL_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_seen_hashes() -> set:
    if not os.path.exists(SEEN_FILE):
        return set()
    with open(SEEN_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_seen_hashes(seen: set):
    """原子写入 seen hashes (BUG-11 修复)。"""
    dir_name = os.path.dirname(SEEN_FILE) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for h in sorted(seen):
                f.write(h + "\n")
        shutil.move(tmp_path, SEEN_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def item_hash(instruction: str, output: str) -> str:
    raw = f"{instruction}||{output}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_item(item: dict):
    instruction = (item.get("instruction") or "").strip()
    output = (item.get("output") or "").strip()
    if not instruction or not output:
        return None
    if len(instruction) > MAX_FIELD_CHARS or len(output) > MAX_FIELD_CHARS:
        logger.warning("跳过超长训练样本：instruction/output 最大允许 %s 字符", MAX_FIELD_CHARS)
        return None
    return {"instruction": instruction, "output": output}


def save_to_raw(items: list):
    """保存数据条目到训练队列 raw 目录，使用递增计数器和 UUID 防止文件名冲突。"""
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    for item in items:
        counter = state.next_raw_counter()
        short_uuid = uuid.uuid4().hex[:6]
        file_name = f"{ts}_{counter}_{short_uuid}.json"
        path = os.path.join(RAW_DIR, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)


# ======================== 6. 游标管理 ========================

def read_cursor(cursor_file: str) -> int:
    if not os.path.exists(cursor_file):
        return 0
    try:
        with open(cursor_file, "r") as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return 0


def save_cursor(cursor_file: str, value: int):
    """原子写入游标 (BUG-11 修复)。"""
    dir_name = os.path.dirname(cursor_file) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(value))
        shutil.move(tmp_path, cursor_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ======================== 7. 数据源采集 ========================

def fetch_local_dir(source: dict) -> list:
    """采集本地目录数据 (BUG-09 修复: 追踪已处理文件)。"""
    raw_path = source.get("dir", RAW_DIR)
    processed_dir = PROCESSED_LOCAL_DIR
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    items = []
    for p in sorted(Path(raw_path).glob("*.json")):
        marker = os.path.join(processed_dir, p.name + ".done")
        try:
            stat = p.stat()
            fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
            if os.path.exists(marker):
                try:
                    if Path(marker).read_text(encoding="utf-8").strip() == fingerprint:
                        continue
                except OSError:
                    pass

            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items.extend(data)
            elif isinstance(data, dict):
                items.append(data)
            Path(marker).write_text(fingerprint, encoding="utf-8")
        except Exception as e:
            logger.warning(f"读取本地文件失败 {p.name}: {e}")
            continue

    return items


def fetch_mysql(source: dict) -> list:
    if pymysql is None:
        logger.warning("[MySQL] 未安装 pymysql，跳过")
        return []

    last_id = read_cursor(source["last_cursor_file"])
    conn = pymysql.connect(
        host=source["host"],
        port=source["port"],
        user=source["user"],
        password=source["password"],
        database=source["database"],
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=15,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(source["sql"], (last_id,))
            rows = cursor.fetchall()
    finally:
        conn.close()

    items = []
    max_id = last_id
    for row in rows:
        row_id, instruction, output = row[0], row[1], row[2]
        items.append({
            "instruction": str(instruction).strip(),
            "output": str(output).strip(),
        })
        if row_id is not None:
            max_id = max(max_id, int(row_id))

    if items and max_id > last_id:
        save_cursor(source["last_cursor_file"], max_id)

    return items


def fetch_postgres(source: dict) -> list:
    if psycopg2 is None:
        logger.warning("[PostgreSQL] 未安装 psycopg2-binary，跳过")
        return []

    last_id = read_cursor(source["last_cursor_file"])
    conn = psycopg2.connect(
        host=source["host"],
        port=source["port"],
        user=source["user"],
        password=source["password"],
        dbname=source["database"],
        connect_timeout=10,
        options="-c statement_timeout=15000",
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(source["sql"], (last_id,))
            rows = cursor.fetchall()
    finally:
        conn.close()

    items = []
    max_id = last_id
    for row in rows:
        row_id, instruction, output = row[0], row[1], row[2]
        items.append({
            "instruction": str(instruction).strip(),
            "output": str(output).strip(),
        })
        if row_id is not None:
            max_id = max(max_id, int(row_id))

    if items and max_id > last_id:
        save_cursor(source["last_cursor_file"], max_id)

    return items


def fetch_sqlite(source: dict) -> list:
    if sqlite3 is None:
        logger.warning("[SQLite] 未找到 sqlite3，跳过")
        return []

    last_id = read_cursor(source["last_cursor_file"])
    conn = sqlite3.connect(source["db_path"], timeout=10)
    try:
        cursor = conn.cursor()
        cursor.execute(source["sql"], (last_id,))
        rows = cursor.fetchall()
    finally:
        conn.close()

    items = []
    max_id = last_id
    for row in rows:
        row_id, instruction, output = row[0], row[1], row[2]
        items.append({
            "instruction": str(instruction).strip(),
            "output": str(output).strip(),
        })
        if row_id is not None:
            max_id = max(max_id, int(row_id))

    if items and max_id > last_id:
        save_cursor(source["last_cursor_file"], max_id)

    return items


def fetch_http_api(source: dict) -> list:
    """采集 HTTP API 数据 (BUG-02 + BUG-12 修复)。"""
    if http_requests is None:
        logger.warning("[HTTP API] 未安装 requests，跳过")
        return []

    last_cursor = read_cursor(source["last_cursor_file"])
    params = dict(source.get("params") or {})
    if last_cursor:
        params["cursor"] = last_cursor

    try:
        resp = http_requests.request(
            method=source.get("method", "GET"),
            url=source["url"],
            headers=source.get("headers") or {},
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        raw_response = resp.json()
    except Exception as e:
        logger.error(f"[HTTP API] 请求失败: {e}")
        return []

    # BUG-02 修复: 保留原始响应用于 cursor 提取
    data = raw_response
    path = source.get("response_path", "data")
    for key in path.split("."):
        if isinstance(data, dict):
            data = data.get(key, [])
        else:
            break

    if not isinstance(data, list):
        data = []

    instr_key = source.get("instruction_field", "instruction")
    out_key = source.get("output_field", "output")

    items = []
    for row in data:
        if isinstance(row, dict):
            items.append({
                "instruction": str(row.get(instr_key, "")).strip(),
                "output": str(row.get(out_key, "")).strip(),
            })

    # BUG-02 修复: 从原始数据列表(非遍历后的 data)提取 cursor
    cursor_source = raw_response
    for key in path.split("."):
        if isinstance(cursor_source, dict):
            cursor_source = cursor_source.get(key, [])
        else:
            break

    if isinstance(cursor_source, list) and cursor_source:
        last_item = cursor_source[-1]
        if isinstance(last_item, dict) and "id" in last_item:
            try:
                save_cursor(source["last_cursor_file"], int(last_item["id"]))
            except (ValueError, TypeError):
                pass

    return items


def fetch_csv_excel(source: dict) -> list:
    dir_path = source.get("dir", os.path.join(BASE_DATA_PATH, "files"))
    marker_dir = source.get("processed_marker_dir", os.path.join(BASE_DATA_PATH, "files_processed"))
    Path(marker_dir).mkdir(parents=True, exist_ok=True)

    instr_col = source.get("instruction_col", "instruction")
    out_col = source.get("output_col", "output")

    items = []

    if not os.path.isdir(dir_path):
        return items

    for p in sorted(Path(dir_path).glob("*.csv")):
        marker = os.path.join(marker_dir, p.name + ".done")
        if os.path.exists(marker):
            continue

        try:
            import csv
            with open(p, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    items.append({
                        "instruction": str(row.get(instr_col, "")),
                        "output": str(row.get(out_col, "")),
                    })
        except Exception as e:
            logger.warning(f"[CSV] 读取 {p.name} 失败: {e}")
            continue

        Path(marker).touch()

    for p in sorted(Path(dir_path).glob("*.xlsx")):
        marker = os.path.join(marker_dir, p.name + ".done")
        if os.path.exists(marker):
            continue

        if openpyxl is None:
            logger.warning("[Excel] 未安装 openpyxl，跳过 xlsx 文件")
            continue

        try:
            wb = openpyxl.load_workbook(p, read_only=True)
            ws = wb.active
            headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            for row in ws.iter_rows(min_row=2, values_only=True):
                d = dict(zip(headers, row))
                items.append({
                    "instruction": str(d.get(instr_col, "")),
                    "output": str(d.get(out_col, "")),
                })
            wb.close()
        except Exception as e:
            logger.warning(f"[Excel] 读取 {p.name} 失败: {e}")
            continue

        Path(marker).touch()

    return items


# ======================== 8. 主采集循环 ========================

def collect_from_all_sources() -> list:
    """采集所有已启用数据源 (BUG-06 修复: 源级异常隔离)。"""
    raw_items = []

    for idx, source in enumerate(DATA_SOURCES):
        if not source.get("enabled"):
            continue

        src_type = source.get("type", "unknown")
        src_name = source.get("name", src_type)

        try:
            if src_type == "local_dir":
                items = fetch_local_dir(source)
                raw_items.extend(items)
                continue

            interval = source.get("poll_interval", DB_POLL_INTERVAL)
            s_key = state.source_key(source)

            if not state.should_poll(s_key, interval):
                continue

            if src_type == "mysql":
                items = fetch_mysql(source)
            elif src_type == "postgres":
                items = fetch_postgres(source)
            elif src_type == "sqlite":
                items = fetch_sqlite(source)
            elif src_type == "http_api":
                items = fetch_http_api(source)
            elif src_type == "csv_excel":
                items = fetch_csv_excel(source)
            else:
                logger.warning(f"未知数据源类型: {src_type}")
                continue

            raw_items.extend(items)
            if items:
                logger.info(f"[{src_name}] 采集到 {len(items)} 条原始数据")

        except Exception as e:
            logger.error(f"[{src_name}] 采集异常: {e}")
            state.set_error(f"[{src_name}] {e}")
            continue

    return raw_items


# ======================== 9. 去重 + 写入缓存 ========================

def ingest_new_items(raw_items: list) -> int:
    seen = load_seen_hashes()
    new_items = []

    for item in raw_items:
        norm = normalize_item(item)
        if norm is None:
            continue

        h = item_hash(norm["instruction"], norm["output"])
        if h in seen:
            continue

        seen.add(h)
        new_items.append(norm)

    if new_items:
        save_to_raw(new_items)
        save_seen_hashes(seen)
        with state._lock:
            state.total_collected += len(new_items)

    return len(new_items)


# ======================== 10. 训练逻辑 ========================

def check_and_clean_adapters():
    if not os.path.isdir(ADAPTER_DIR):
        return
    adapters = sorted(
        [
            os.path.join(ADAPTER_DIR, d)
            for d in os.listdir(ADAPTER_DIR)
            if d.startswith("adapter_") and os.path.isdir(os.path.join(ADAPTER_DIR, d))
        ],
        key=os.path.getctime,
    )
    if len(adapters) > MAX_ADAPTERS:
        for dir_path in adapters[:-MAX_ADAPTERS]:
            logger.info(f"[清理] 删除旧 adapter: {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)


def trigger_training() -> bool:
    """唯一训练入口。成功后才删除原始样本；失败时保留 batch 供排查/重试。"""
    with state._lock:
        if state.is_training:
            logger.warning("[训练] 已有训练任务运行，忽略重复触发")
            return False
        state.is_training = True  # 原子占位，阻止后台轮询与手动 API 竞态
        state.training_cancel_requested = False
        state.training_status = "queued"
        state.training_stage = "任务已排队"

    batch_path = None
    raw_files = []
    training_started = False
    success = False
    try:
        ensure_dirs()
        raw_files = sorted(Path(RAW_DIR).glob("*.json"))
        if len(raw_files) < BUFFER_THRESHOLD:
            logger.info(f"[训练] 当前只有 {len(raw_files)} 个 raw 文件，未达到阈值 {BUFFER_THRESHOLD}")
            return False

        new_data = []
        for f in raw_files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    item = json.load(fp)
                if isinstance(item, dict):
                    norm = normalize_item(item)
                    if norm:
                        new_data.append(norm)
                elif isinstance(item, list):
                    new_data.extend(n for n in (normalize_item(x) for x in item if isinstance(x, dict)) if n)
            except Exception as e:
                logger.warning(f"读取 raw 文件失败 {f.name}: {e}")

        if len(new_data) < BUFFER_THRESHOLD:
            logger.info(f"[训练] 有效样本 {len(new_data)} 条，未达到阈值 {BUFFER_THRESHOLD}")
            return False

        mixed_data = list(new_data)
        if os.path.exists(FOUNDATIONAL_DATA_PATH):
            try:
                with open(FOUNDATIONAL_DATA_PATH, "r", encoding="utf-8") as fp:
                    base_data = [n for n in (normalize_item(json.loads(line)) for line in fp if line.strip()) if n]
                sample_count = min(len(base_data), max(1, len(new_data) // 4))
                if sample_count:
                    mixed_data.extend(random.sample(base_data, sample_count))
            except Exception as e:
                logger.warning(f"加载基础数据失败: {e}")

        random.shuffle(mixed_data)
        batch_path = os.path.join(BUFFER_DIR, f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}.jsonl")
        with open(batch_path, "w", encoding="utf-8") as fp:
            for item in mixed_data:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"[训练] 新数据 {len(new_data)} 条，基础数据 {len(mixed_data)-len(new_data)} 条，开始增量微调")
        state.set_training(True, len(mixed_data))
        training_started = True
        state.update_training(training_status="loading", training_stage="加载模型与数据", training_output_dir=None)
        start_unsloth_training(batch_path)
        success = True
        with state._lock:
            cancelled = state.training_cancel_requested
        if cancelled:
            state.update_training(training_status="cancelled", training_stage="训练已停止；已保存部分 adapter")
            logger.warning("[训练] 用户请求停止；部分 adapter 已保存，原始数据与 batch 均保留")
            # 中止训练不等于完整训练成功，保留输入供后续重新训练。
            check_and_clean_adapters()
            return True
        with state._lock:
            state.total_trained += len(new_data)
        state.update_training(training_status="completed", training_stage="训练完成", training_progress=100.0)
        logger.info("[训练] 本轮训练成功完成")
        # 只有完整训练与 adapter 保存都成功，才删除对应 raw 文件
        for f in raw_files:
            try:
                f.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"训练成功但删除 raw 文件失败 {f.name}: {e}")
        if batch_path and os.path.exists(batch_path):
            os.unlink(batch_path)
        check_and_clean_adapters()
        return True
    except Exception as e:
        error_msg = f"训练失败: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        state.set_error(error_msg)
        state.update_training(training_status="failed", training_stage="训练失败")
        # 保留 batch 文件；raw 也保留，避免训练失败造成数据丢失
        if batch_path:
            logger.error(f"[训练] 失败数据集已保留，便于排查: {batch_path}")
        return False
    finally:
        if training_started:
            state.set_training(False)
            # set_training(False) 不覆盖 completed/failed 状态
        else:
            with state._lock:
                state.is_training = False
                state.training_start_time = None
                if state.training_status == "queued":
                    state.training_status = "idle"
                    state.training_stage = "空闲"


def start_unsloth_training(dataset_path: str):
    """Run training locally or delegate the run to the official Unsloth Studio backend."""
    if TRAINING_BACKEND == "studio_mcp":
        from unsloth_studio_bridge import run_training_via_studio

        logger.info("[联动] 使用 Unsloth Studio MCP 作为训练执行后端")
        state.update_training(
            training_backend="studio_mcp",
            training_status="connecting",
            training_stage="连接 Unsloth Studio",
            training_model=MODEL_NAME,
        )
        result = run_training_via_studio(
            dataset_path=dataset_path,
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            max_steps=MAX_STEPS,
            on_status=state.update_training,
            should_stop=lambda: bool(state.training_cancel_requested),
        )
        if result.get("job_id"):
            state.update_training(studio_job_id=str(result["job_id"]))
        if result.get("status") == "cancelled":
            # A stop issued inside Studio must also preserve this scheduler's raw data.
            with state._lock:
                state.training_cancel_requested = True
            state.update_training(training_status="cancelled", training_stage="Unsloth Studio 训练已停止")
        logger.info("[联动] Studio 训练任务结束，状态=%s, job_id=%s", result.get("status"), result.get("job_id"))
        return

    if TRAINING_BACKEND != "local":
        raise RuntimeError(f"不支持的 TRAINING_BACKEND={TRAINING_BACKEND!r}，可选 local 或 studio_mcp")

    state.update_training(training_backend="local", studio_job_id=None)
    deps = _ensure_train_deps()
    if not deps:
        raise RuntimeError("训练依赖未安装或导入失败 (datasets / unsloth / trl)，请检查安装日志")

    load_dataset = deps["load_dataset"]
    FastLanguageModel = deps["FastLanguageModel"]
    SFTTrainer = deps["SFTTrainer"]
    SFTConfig = deps["SFTConfig"]

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if len(dataset) == 0:
        raise RuntimeError("训练数据集为空")

    def formatting_func(examples):
        texts = []
        for instr, out in zip(examples.get("instruction", []), examples.get("output", [])):
            texts.append(f"### 指令：\n{instr}\n\n### 回答：\n{out}")
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)
    state.update_training(training_status="loading", training_stage="加载基础模型", training_model=MODEL_NAME)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    adapter_name = f"adapter_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    adapter_path = os.path.join(ADAPTER_DIR, adapter_name)
    Path(adapter_path).mkdir(parents=True, exist_ok=False)

    # 不同 TRL 版本的 SFTConfig 字段有差异，按当前安装版本的签名选择可用字段。
    config_sig = inspect.signature(SFTConfig)
    config_params = config_sig.parameters
    config_values = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "max_steps": MAX_STEPS,
        "logging_steps": 1,  # 每一步记录一次，Web UI 才能及时刷新
        "output_dir": adapter_path,
        "optim": "paged_adamw_8bit",
        "warmup_steps": 5,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if "max_length" in config_params:
        config_values["max_length"] = MAX_SEQ_LENGTH
    elif "max_seq_length" in config_params:
        config_values["max_seq_length"] = MAX_SEQ_LENGTH
    if "dataset_text_field" in config_params:
        config_values["dataset_text_field"] = "text"
    if "packing" in config_params:
        config_values["packing"] = False
    config_values = {k: v for k, v in config_values.items() if k in config_params}
    train_config = SFTConfig(**config_values)

    # TrainerCallback 将真实训练日志同步到控制台后端状态，前端 /api/status 可读取。
    try:
        from transformers import TrainerCallback
    except ImportError as e:
        raise RuntimeError("缺少 transformers.TrainerCallback，请检查 Unsloth/TRL 依赖") from e

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state_obj, control, **kwargs):
            total = int(getattr(state_obj, "max_steps", 0) or getattr(args, "max_steps", 0) or 0)
            state.update_training(
                training_status="training", training_stage="模型训练中",
                training_step=int(getattr(state_obj, "global_step", 0) or 0),
                training_max_steps=total,
                training_progress=0.0,
                training_output_dir=adapter_path,
            )

        def on_step_end(self, args, state_obj, control, **kwargs):
            step = int(getattr(state_obj, "global_step", 0) or 0)
            total = int(getattr(state_obj, "max_steps", 0) or getattr(args, "max_steps", 0) or 0)
            pct = min(100.0, step * 100.0 / total) if total > 0 else 0.0
            if state.training_cancel_requested:
                control.should_training_stop = True
                state.update_training(
                    training_status="stopping", training_stage="正在停止（等待当前训练步结束）",
                    training_step=step, training_max_steps=total, training_progress=round(pct, 2),
                )
            else:
                state.update_training(
                    training_status="training", training_stage="模型训练中",
                    training_step=step, training_max_steps=total, training_progress=round(pct, 2),
                    training_epoch=getattr(state_obj, "epoch", None),
                )

        def on_log(self, args, state_obj, control, logs=None, **kwargs):
            logs = logs or {}
            updates = {}
            if "loss" in logs:
                try: updates["training_loss"] = float(logs["loss"])
                except (TypeError, ValueError): pass
            if "learning_rate" in logs:
                try: updates["training_learning_rate"] = float(logs["learning_rate"])
                except (TypeError, ValueError): pass
            if "epoch" in logs:
                try: updates["training_epoch"] = float(logs["epoch"])
                except (TypeError, ValueError): pass
            if updates:
                state.update_training(**updates)
                logger.info("[训练指标] " + ", ".join(f"{k}={v}" for k, v in updates.items()))

        def on_train_end(self, args, state_obj, control, **kwargs):
            step = int(getattr(state_obj, "global_step", 0) or 0)
            total = int(getattr(state_obj, "max_steps", 0) or 0)
            cancelled = state.training_cancel_requested
            pct = min(100.0, step * 100.0 / total) if total > 0 else 0.0
            state.update_training(
                training_step=step,
                training_max_steps=total,
                training_progress=round(pct, 2),
                training_status="cancelled" if cancelled else "saving",
                training_stage="训练已停止，保存部分 adapter" if cancelled else "保存 LoRA 适配器",
            )

    trainer_params = inspect.signature(SFTTrainer).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": train_config,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "callbacks" in trainer_params:
        trainer_kwargs["callbacks"] = [ProgressCallback()]
    if "dataset_text_field" in trainer_params and "dataset_text_field" not in config_params:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)
    state.update_training(training_status="training", training_stage="模型训练中", training_output_dir=adapter_path)
    trainer.train()
    state.update_training(training_status="saving", training_stage="保存 LoRA 适配器")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"[训练] adapter 已保存: {adapter_path}")


# ======================== 11. 后台训练循环 ========================

def background_loop():
    """后台线程：持续采集数据并触发训练。"""
    logger.info("后台训练循环已启动")
    while state.is_running:
        try:
            raw_items = collect_from_all_sources()
            if raw_items:
                new_count = ingest_new_items(raw_items)
                if new_count:
                    logger.info(f"[采集] 新增 {new_count} 条不重复数据")

            triggered = trigger_training()
            if triggered:
                logger.info("[系统] 本轮增量微调已成功完成")
        except Exception as e:
            logger.error(f"[后台循环] 异常: {e}")
            state.set_error(f"后台循环异常: {e}")

        # 分段 sleep，使关闭更灵敏
        for _ in range(10):
            if not state.is_running:
                break
            time.sleep(1)

    logger.info("后台训练循环已停止")


# ======================== 12. Flask Web UI ========================

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 防止单次请求体无限制占用内存


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _host_only(hostport: str) -> str:
    hostport = (hostport or "").strip().lower()
    if hostport.startswith("["):
        return hostport.split("]")[0] + "]"
    return hostport.rsplit(":", 1)[0] if hostport.count(":") == 1 else hostport


_LOCAL_NAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}


@app.before_request
def security_gate():
    """基本 Host 校验、跨站写请求限制和可选 API 令牌认证。"""
    if _is_loopback(WEB_HOST) and _host_only(flask_request.host) not in _LOCAL_NAMES:
        return jsonify({"error": "非法 Host"}), 403
    if flask_request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = flask_request.headers.get("Origin")
        if origin:
            from urllib.parse import urlparse
            if urlparse(origin).netloc.lower() != flask_request.host.lower():
                return jsonify({"error": "跨站请求被拒绝"}), 403
    if AUTH_TOKEN and flask_request.path.startswith("/api/"):
        supplied = flask_request.headers.get("X-Auth-Token", "")
        if not hmac.compare_digest(supplied, AUTH_TOKEN):
            return jsonify({"error": "未授权：需要有效的 X-Auth-Token"}), 401


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.route("/")
def index():
    return render_template("index.html")


# --- 系统状态 API ---

@app.route("/api/status")
def api_status():
    snap = state.snapshot()

    # 统计 raw 文件数
    raw_count = 0
    if os.path.isdir(RAW_DIR):
        raw_count = len(list(Path(RAW_DIR).glob("*.json")))

    # 统计 adapter 数
    adapter_count = 0
    if os.path.isdir(ADAPTER_DIR):
        adapter_count = len([
            d for d in os.listdir(ADAPTER_DIR)
            if d.startswith("adapter_") and os.path.isdir(os.path.join(ADAPTER_DIR, d))
        ])

    # seen hashes 数
    seen_count = len(load_seen_hashes())

    snap["raw_file_count"] = raw_count
    snap["adapter_count"] = adapter_count
    snap["seen_hash_count"] = seen_count
    snap["buffer_threshold"] = BUFFER_THRESHOLD
    snap["max_adapters"] = MAX_ADAPTERS
    snap["model_name"] = MODEL_NAME
    snap["configured_training_backend"] = TRAINING_BACKEND

    return jsonify(snap)


# --- 数据源 API ---


def _safe_source_detail(src):
    typ = src.get("type")
    if typ == "local_dir" or typ == "csv_excel":
        return src.get("dir", "")
    if typ in ("mysql", "postgres"):
        return f"{src.get('host')}:{src.get('port')}/{src.get('database')}"
    if typ == "sqlite":
        return src.get("db_path", "")
    if typ == "http_api":
        # 不向列表 API 暴露 URL userinfo 或 query 中疑似凭据。
        return _redact_url(src.get("url", ""))
    return ""


def _source_editor_payload(idx, src):
    fields = []
    for spec in SOURCE_FIELD_META.get(src.get("type"), []):
        key = spec["key"]
        raw_value = src.get(key, "")
        configured = raw_value not in (None, "", {}, [])
        if spec["type"] == "json":
            shown = json.dumps(_redact_nested(raw_value) if spec.get("secret") else raw_value, ensure_ascii=False, indent=2)
        elif spec.get("secret"):
            shown = ""  # never return plain credentials to the browser
        elif key == "url":
            shown = _redact_url(raw_value)
        else:
            shown = "" if raw_value is None else str(raw_value)
        item = dict(spec)
        item.update({"value": shown, "configured": bool(configured)})
        fields.append(item)
    return {"id": idx, "name": src.get("name", src.get("type", "")), "type": src.get("type"), "enabled": bool(src.get("enabled")), "fields": fields}


def _parse_source_field(src, spec, raw_value):
    key = spec["key"]
    current = src.get(key)
    field_type = spec["type"]

    if spec.get("secret") and field_type == "password":
        value = str(raw_value or "")
        if value == "__CLEAR__":
            return ""
        return current if value == "" else value

    if field_type == "json":
        if isinstance(raw_value, (dict, list)):
            value = raw_value
        else:
            try:
                value = json.loads(str(raw_value or "{}"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{spec['label']} 必须是有效 JSON：{exc.msg}")
        if not isinstance(value, dict):
            raise ValueError(f"{spec['label']} 必须是 JSON 对象")
        if spec.get("secret"):
            value = _restore_nested_secrets(current if isinstance(current, dict) else {}, value)
            # 显式 __CLEAR__ 允许清理某个敏感值。
            for nested_key, nested_value in list(value.items()):
                if _is_secret_key(nested_key) and nested_value == "__CLEAR__":
                    value[nested_key] = ""
        return value

    if field_type == "number":
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{spec['label']} 必须是整数")
        if value < spec.get("min", -2147483648) or value > spec.get("max", 2147483647):
            raise ValueError(f"{spec['label']} 超出允许范围")
        return value

    value = str(raw_value if raw_value is not None else "").strip()
    if spec.get("required") and not value:
        raise ValueError(f"{spec['label']} 不能为空")
    if len(value) > 20000:
        raise ValueError(f"{spec['label']} 长度超过 20000 个字符")
    if field_type == "select" and value not in spec.get("options", []):
        raise ValueError(f"{spec['label']} 选项无效")
    if key == "sql":
        value = _validate_readonly_sql(value)
    if key == "url" and value:
        try:
            value = _restore_url_secrets(current, value)
        except ValueError as exc:
            raise ValueError("API URL 格式无效") from exc
    if key in ("dir", "db_path") and value:
        value = os.path.abspath(os.path.expanduser(value))
    if key == "url" and value:
        from urllib.parse import urlsplit
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("API URL 必须使用完整的 http:// 或 https:// 地址")
    return value


@app.route("/api/sources")
def api_sources():
    sources = []
    for idx, src in enumerate(DATA_SOURCES):
        sources.append({
            "id": idx,
            "name": src.get("name", src["type"]),
            "type": src["type"],
            "enabled": bool(src.get("enabled", False)),
            "detail": _safe_source_detail(src),
            "poll_interval": src.get("poll_interval", DB_POLL_INTERVAL),
        })
    return jsonify(sources)


@app.route("/api/sources/<int:idx>/config", methods=["GET", "PUT"])
def api_source_config(idx):
    if idx < 0 or idx >= len(DATA_SOURCES):
        return jsonify({"error": "无效的数据源 ID"}), 404
    src = DATA_SOURCES[idx]
    if flask_request.method == "GET":
        return jsonify(_source_editor_payload(idx, src))

    body = flask_request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("fields", {}), dict):
        return jsonify({"error": "请求体必须包含 fields 对象"}), 400
    submitted = body.get("fields", {})
    specs = SOURCE_FIELD_META.get(src.get("type"), [])
    allowed = {spec["key"] for spec in specs}
    unknown = set(submitted) - allowed
    if unknown:
        return jsonify({"error": "包含不可编辑字段: " + ", ".join(sorted(unknown))}), 400

    updated = {}
    try:
        for spec in specs:
            key = spec["key"]
            if key not in submitted:
                continue
            updated[key] = _parse_source_field(src, spec, submitted[key])
        # Empty passwords mean "leave unchanged". Explicit __CLEAR__ is the clearing operation.
        previous = {key: src.get(key) for key in updated}
        src.update(updated)
        try:
            save_sources_config()
        except Exception:
            for key, old_value in previous.items():
                src[key] = old_value
            raise
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400

    logger.info("[配置] 数据源 %s 的参数已保存", src.get("name", src.get("type")))
    return jsonify({"message": "数据源配置已保存", "source": _source_editor_payload(idx, src)})


@app.route("/api/sources/<int:idx>/toggle", methods=["POST"])
def api_toggle_source(idx):
    if idx < 0 or idx >= len(DATA_SOURCES):
        return jsonify({"error": "无效的数据源 ID"}), 404
    src = DATA_SOURCES[idx]
    old_state = bool(src.get("enabled", False))
    src["enabled"] = not old_state
    try:
        save_sources_config()
    except OSError as exc:
        src["enabled"] = old_state
        logger.error("保存数据源状态失败: %s", exc)
        return jsonify({"error": "保存数据源状态失败"}), 500
    new_state = src["enabled"]
    name = src.get("name", src["type"])
    logger.info("[配置] 数据源 %s 已%s", name, "启用" if new_state else "禁用")
    return jsonify({"enabled": new_state})


# --- 训练控制 API ---

@app.route("/api/trigger_training", methods=["POST"])
def api_trigger_training():
    with state._lock:
        if state.is_training:
            return jsonify({"error": "训练正在进行中"}), 409
    # trigger_training 自身会通过锁原子占位；线程启动成功后返回任务已提交。
    def _train():
        try:
            if not trigger_training():
                logger.info("[手动触发] 未启动训练：请检查数据量阈值或日志")
        except Exception as e:
            logger.error(f"[手动触发] 训练异常: {e}")
            state.set_error(str(e))

    threading.Thread(target=_train, daemon=True, name="manual-training").start()
    return jsonify({"message": "训练任务已提交", "status": "queued"}), 202


@app.route("/api/stop_training", methods=["POST"])
def api_stop_training():
    with state._lock:
        if not state.is_training:
            return jsonify({"error": "当前没有正在运行的训练任务"}), 409
        state.training_cancel_requested = True
        state.training_status = "stopping"
        state.training_stage = "正在请求停止"
    logger.warning("[训练] 收到停止请求，将在当前训练步结束后停止")
    return jsonify({"message": "已请求停止；将在当前训练步结束后生效", "status": "stopping"}), 202


# --- 适配器 API ---

@app.route("/api/adapters")
def api_adapters():
    if not os.path.isdir(ADAPTER_DIR):
        return jsonify([])
    adapters = []
    for d in sorted(os.listdir(ADAPTER_DIR)):
        full_path = os.path.join(ADAPTER_DIR, d)
        if d.startswith("adapter_") and os.path.isdir(full_path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(full_path)
                for f in fns
            )
            adapters.append({
                "name": d,
                "size_mb": round(size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(os.path.getctime(full_path)).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return jsonify(adapters)


@app.route("/api/adapters/<name>", methods=["DELETE"])
def api_delete_adapter(name):
    # BUG-18 修复: 使用 realpath 验证路径不逃逸出 ADAPTER_DIR
    if not name.startswith("adapter_") or ".." in name or "/" in name or "\\" in name:
        return jsonify({"error": "非法名称"}), 400

    path = os.path.join(ADAPTER_DIR, name)
    real_path = os.path.realpath(path)
    real_adapter_dir = os.path.realpath(ADAPTER_DIR)

    if not real_path.startswith(real_adapter_dir + os.sep) and real_path != real_adapter_dir:
        return jsonify({"error": "非法路径"}), 400

    if not os.path.isdir(path):
        return jsonify({"error": "适配器不存在"}), 404

    shutil.rmtree(path, ignore_errors=True)
    logger.info(f"[适配器] 已删除: {name}")
    return jsonify({"message": f"已删除 {name}"})


# --- 数据浏览 API ---

@app.route("/api/data")
def api_data():
    page = flask_request.args.get("page", 1, type=int)
    per_page = flask_request.args.get("per_page", 20, type=int)
    page = max(1, page)
    per_page = max(1, min(per_page, 100))

    if not os.path.isdir(RAW_DIR):
        return jsonify({"items": [], "total": 0, "page": page, "per_page": per_page})

    all_files = sorted(Path(RAW_DIR).glob("*.json"))
    total = len(all_files)

    start = (page - 1) * per_page
    end = start + per_page
    page_files = all_files[start:end]

    items = []
    for f in page_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                item = json.load(fp)
            item["_file"] = f.name
            items.append(item)
        except Exception:
            continue

    return jsonify({
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })


# --- 日志 API ---

@app.route("/api/logs")
def api_logs():
    limit = flask_request.args.get("limit", 100, type=int)
    limit = max(1, min(limit, 500))
    return jsonify(state.get_logs(limit))


# --- 系统配置 API ---


def _effective_runtime_value(key):
    """返回当前进程真正生效的值，而不是仅返回表单中保存的值。"""
    values = {
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "buffer_threshold": BUFFER_THRESHOLD,
        "max_adapters": MAX_ADAPTERS,
        "db_poll_interval": DB_POLL_INTERVAL,
        "base_data_path": BASE_DATA_PATH,
        "max_steps": MAX_STEPS,
        "training_backend": TRAINING_BACKEND,
    }
    return values[key]


def _validate_system_setting(key, raw_value):
    if key not in SYSTEM_SETTING_META:
        raise ValueError(f"未知配置项: {key}")
    meta = SYSTEM_SETTING_META[key]
    if meta["type"] == "number":
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{meta['label']} 必须是整数")
        if value < meta.get("min", 0) or value > meta.get("max", 2**31 - 1):
            raise ValueError(f"{meta['label']} 必须在 {meta.get('min')} 到 {meta.get('max')} 之间")
        return value
    value = str(raw_value if raw_value is not None else "").strip()
    if not value:
        raise ValueError(f"{meta['label']} 不能为空")
    if len(value) > 1024 or "\x00" in value:
        raise ValueError(f"{meta['label']} 长度无效")
    if meta["type"] == "select" and value not in meta.get("options", []):
        raise ValueError(f"{meta['label']} 的选项无效")
    if key == "base_data_path":
        return _resolve_data_path(value)
    if key == "training_backend":
        return value.lower()
    return value


def _apply_runtime_settings(previous_db_interval=None):
    """应用无需重启的配置；数据目录变化必须等待下次启动再生效。"""
    global MODEL_NAME, MAX_SEQ_LENGTH, BUFFER_THRESHOLD, MAX_ADAPTERS
    global DB_POLL_INTERVAL, MAX_STEPS, TRAINING_BACKEND
    old_interval = DB_POLL_INTERVAL if previous_db_interval is None else previous_db_interval
    MODEL_NAME = str(_system_value("model_name"))
    MAX_SEQ_LENGTH = int(_system_value("max_seq_length"))
    BUFFER_THRESHOLD = int(_system_value("buffer_threshold"))
    MAX_ADAPTERS = int(_system_value("max_adapters"))
    DB_POLL_INTERVAL = int(_system_value("db_poll_interval"))
    MAX_STEPS = int(_system_value("max_steps"))
    TRAINING_BACKEND = str(_system_value("training_backend")).strip().lower()

    if DB_POLL_INTERVAL != old_interval:
        for src in DATA_SOURCES:
            # 只调整沿用旧全局默认值的源；明显自定义的轮询间隔保持原样。
            if "poll_interval" in src and src.get("poll_interval") == old_interval:
                src["poll_interval"] = DB_POLL_INTERVAL
        try:
            save_sources_config()
        except OSError as exc:
            logger.warning("全局轮询间隔已生效，但数据源配置未能同步保存: %s", exc)


def _system_settings_payload():
    output = []
    for key, meta in SYSTEM_SETTING_META.items():
        entry = SYSTEM_SETTINGS.get(key, {"value": DEFAULT_SYSTEM_VALUES[key], "enabled": True})
        configured_value = entry.get("value", DEFAULT_SYSTEM_VALUES[key])
        fallback = DEFAULT_SYSTEM_VALUES[key]
        active_value = _effective_runtime_value(key)
        desired_value = _system_value(key)
        pending_restart = bool(meta.get("restart_required") and str(_resolve_data_path(desired_value)) != str(BASE_DATA_PATH))
        output.append({
            "key": key,
            "label": meta["label"],
            "type": meta["type"],
            "description": meta.get("description", ""),
            "min": meta.get("min"),
            "max": meta.get("max"),
            "options": meta.get("options", []),
            "restart_required": bool(meta.get("restart_required", False)),
            "pending_restart": pending_restart,
            "value": configured_value,
            "enabled": bool(entry.get("enabled", True)),
            "default_value": fallback,
            "effective_value": active_value,
        })
    return output


@app.route("/api/config", methods=["GET", "PUT", "POST"])
def api_config():
    if flask_request.method == "GET":
        return jsonify({
            "settings": _system_settings_payload(),
            "model_name": MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "buffer_threshold": BUFFER_THRESHOLD,
            "max_adapters": MAX_ADAPTERS,
            "db_poll_interval": DB_POLL_INTERVAL,
            "base_data_path": BASE_DATA_PATH,
            "max_steps": MAX_STEPS,
            "training_backend": TRAINING_BACKEND,
        })

    body = flask_request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("settings"), dict):
        return jsonify({"error": "请求体必须包含 settings 对象"}), 400
    incoming = body["settings"]
    unknown = set(incoming) - set(SYSTEM_SETTING_META)
    if unknown:
        return jsonify({"error": "包含未知配置项: " + ", ".join(sorted(unknown))}), 400

    proposed = {}
    try:
        for key, item in incoming.items():
            if not isinstance(item, dict) or "enabled" not in item:
                raise ValueError(f"配置项 {key} 必须包含 enabled 状态")
            proposed[key] = {
                "value": _validate_system_setting(key, item.get("value", SYSTEM_SETTINGS[key]["value"])),
                "enabled": bool(item["enabled"]),
            }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    critical = {"model_name", "max_seq_length", "max_steps", "training_backend"}
    with state._lock:
        if state.is_training:
            for key, new_entry in proposed.items():
                if key in critical:
                    old_entry = SYSTEM_SETTINGS.get(key, {})
                    old_effective = old_entry.get("value") if old_entry.get("enabled", True) else DEFAULT_SYSTEM_VALUES[key]
                    new_effective = new_entry["value"] if new_entry["enabled"] else DEFAULT_SYSTEM_VALUES[key]
                    if str(old_effective) != str(new_effective):
                        return jsonify({"error": "训练进行中，不能修改模型、序列长度、最大步数或训练后端；请训练结束后再保存"}), 409

    old_settings = {k: dict(v) for k, v in SYSTEM_SETTINGS.items()}
    old_db_interval = DB_POLL_INTERVAL
    for key, item in proposed.items():
        SYSTEM_SETTINGS[key] = item
    try:
        _write_json_atomic(SYSTEM_SETTINGS_FILE, SYSTEM_SETTINGS)
    except OSError as exc:
        SYSTEM_SETTINGS.clear()
        SYSTEM_SETTINGS.update(old_settings)
        return jsonify({"error": f"系统配置保存失败: {exc}"}), 500

    _apply_runtime_settings(previous_db_interval=old_db_interval)
    logger.info("[配置] 系统配置已保存；数据目录等重启项将在重启后生效")
    return jsonify({"message": "系统配置已保存", "settings": _system_settings_payload()})


# --- 手动添加数据 API ---

@app.route("/api/data/add", methods=["POST"])
def api_add_data():
    body = flask_request.get_json(silent=True)
    if not body:
        return jsonify({"error": "请求体为空"}), 400

    instruction = (body.get("instruction") or "").strip()
    output = (body.get("output") or "").strip()

    if not instruction or not output:
        return jsonify({"error": "instruction 和 output 均不能为空"}), 400
    if len(instruction) > MAX_FIELD_CHARS or len(output) > MAX_FIELD_CHARS:
        return jsonify({"error": f"单个 instruction/output 不能超过 {MAX_FIELD_CHARS} 个字符"}), 413

    norm = normalize_item({"instruction": instruction, "output": output})
    if norm is None:
        return jsonify({"error": "数据格式无效"}), 400

    save_to_raw([norm])
    logger.info(f"[手动添加] 新增 1 条数据")
    return jsonify({"message": "添加成功"})


# ======================== 13. 启动入口 ========================

def shutdown_handler(signum, frame):
    """优雅关闭：停止后台调度，并让 Flask 服务退出。"""
    logger.info("收到关闭信号，正在停止...")
    state.is_running = False
    state.training_cancel_requested = True
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    ensure_dirs()
    apply_sources_overrides()
    if not _is_loopback(WEB_HOST) and not AUTH_TOKEN:
        logger.error("监听非本机地址 (%s) 必须设置 AUTO_TRAINER_TOKEN，否则任何人都能控制训练/删除适配器。", WEB_HOST)
        sys.exit(2)
    if TRAINING_BACKEND not in ("local", "studio_mcp"):
        logger.error("不支持的 TRAINING_BACKEND=%r（可选 local / studio_mcp）", TRAINING_BACKEND)
        sys.exit(2)

    logger.info("=" * 50)
    logger.info("  Unsloth 自动增量微调系统 V3 (审计修复版)")
    logger.info(f"  Web UI: http://{WEB_HOST}:{WEB_PORT}")
    logger.info(f"  缓存阈值: {BUFFER_THRESHOLD} 条")
    logger.info(f"  轮询间隔: {DB_POLL_INTERVAL} 秒")
    logger.info(f"  最大适配器: {MAX_ADAPTERS}")
    logger.info("=" * 50)

    # 启动后台训练循环
    bg_thread = threading.Thread(target=background_loop, daemon=True, name="trainer-loop")
    bg_thread.start()

    # 启动 Flask Web UI
    try:
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        state.is_running = False
        bg_thread.join(timeout=15)
        logger.info("系统已关闭")


if __name__ == "__main__":
    main()
