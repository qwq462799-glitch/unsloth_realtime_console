#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional MCP bridge to Unsloth Studio's official training backend.

This module does not import or patch Unsloth Studio internals. It uses the opt-in
MCP interface documented by Unsloth Studio. The Studio process remains the owner
of the training subprocess, run history, metrics, and adapter output.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


class StudioBridgeError(RuntimeError):
    pass


def _plain(value: Any) -> Any:
    """Convert common MCP SDK result objects to JSON-ish Python values."""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _result_payload(result: Any) -> Any:
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        bits = []
        for item in getattr(result, "content", []) or []:
            if getattr(item, "text", None):
                bits.append(item.text)
        raise StudioBridgeError("Unsloth Studio MCP 工具调用失败: " + ("; ".join(bits) or str(result)))

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return _plain(structured)

    parsed = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            parsed.append(json.loads(text))
        except (ValueError, TypeError):
            parsed.append(text)
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _unwrap(value: Any) -> Any:
    """Unwrap common MCP {result: ...}/{data: ...} envelopes without guessing fields."""
    for _ in range(3):
        if isinstance(value, dict):
            changed = False
            for key in ("result", "data"):
                candidate = value.get(key)
                if isinstance(candidate, (dict, list)):
                    value = candidate
                    changed = True
                    break
            if not changed:
                return value
        else:
            return value
    return value


def _find_mapping(value: Any) -> dict[str, Any] | None:
    value = _unwrap(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            found = _find_mapping(item)
            if found:
                return found
    if isinstance(value, str):
        try:
            return _find_mapping(json.loads(value))
        except (ValueError, TypeError):
            return None
    return None


def _format_tool_error(result: Any) -> str:
    payload = _result_payload(result)
    return str(payload)[:1200]


async def _run_training_async(
    dataset_path: str,
    model_name: str,
    max_seq_length: int,
    max_steps: int,
    on_status: Callable[..., None],
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        raise StudioBridgeError(
            "启用了 TRAINING_BACKEND=studio_mcp，但当前 Python 环境未安装 MCP 客户端。"
            "请执行：python -m pip install -r requirements-studio-bridge.txt"
        ) from exc

    endpoint = os.getenv("UNSLOTH_STUDIO_MCP_URL", "http://127.0.0.1:8888/mcp/").strip()
    token = os.getenv("UNSLOTH_STUDIO_MCP_TOKEN", "").strip()
    if not token:
        raise StudioBridgeError("未设置 UNSLOTH_STUDIO_MCP_TOKEN；请先启用 Unsloth Studio MCP 并配置同一密钥。")

    # A local dataset path is interpreted by Studio's host, not by this process.
    path = str(Path(dataset_path).resolve())
    path_prefix = os.getenv("UNSLOTH_STUDIO_DATA_PATH_PREFIX", "").strip()
    local_prefix = os.getenv("AUTO_TRAINER_DATA_PATH_PREFIX", "").strip()
    if path_prefix and local_prefix:
        resolved = os.path.normpath(path)
        local_norm = os.path.normpath(str(Path(local_prefix).resolve()))
        if resolved == local_norm or resolved.startswith(local_norm + os.sep):
            path = os.path.join(path_prefix, os.path.relpath(resolved, local_norm))

    headers = {"Authorization": f"Bearer {token}"}
    on_status(
        training_status="connecting",
        training_stage="连接 Unsloth Studio",
        training_model=model_name,
    )

    async with streamablehttp_client(endpoint, headers=headers) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            init_result = await session.initialize()
            tools_result = await session.list_tools()
            tools = {tool.name: tool for tool in tools_result.tools}

            required_tools = {"start_training", "get_training_status", "stop_training"}
            missing = sorted(required_tools - set(tools))
            if missing:
                raise StudioBridgeError(
                    "当前 Unsloth Studio MCP 未提供所需工具：" + ", ".join(missing)
                    + "。请更新 Studio 到包含训练管理 MCP 工具的版本。"
                )

            # Tool schemas are discovered at runtime so small upstream schema changes fail clearly.
            start_tool = tools["start_training"]
            schema = getattr(start_tool, "inputSchema", None) or getattr(start_tool, "input_schema", None) or {}
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
            payload: dict[str, Any] = {
                "model_name": model_name,
                "project_name": "AutoTrainer realtime scheduler",
                "training_type": "LoRA/QLoRA",
                "load_in_4bit": True,
                "max_seq_length": int(max_seq_length),
                "local_datasets": [path],
                "format_type": "Alpaca",
                "train_split": "train",
                "num_epochs": 0,
                "max_steps": int(max_steps),
                "learning_rate": "2e-4",
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "logging_steps": 1,
                "save_steps": max(1, min(100, int(max_steps))),
                "train_on_completions": False,
                "gradient_checkpointing": "unsloth",
                "start_request_id": f"auto-trainer-{int(time.time())}",
            }
            if props:
                payload = {k: v for k, v in payload.items() if k in props}
                absent = sorted(required - set(payload))
                if absent:
                    raise StudioBridgeError(
                        "Unsloth Studio 当前 start_training API 要求本桥接器尚未提供的字段："
                        + ", ".join(absent)
                        + "。请查看 Studio 的训练设置/API schema 后更新适配器。"
                    )

            on_status(training_status="submitting", training_stage="向 Unsloth Studio 提交训练任务")
            start_result = await session.call_tool("start_training", arguments=payload)
            start_payload = _result_payload(start_result)
            if getattr(start_result, "isError", False) or getattr(start_result, "is_error", False):
                raise StudioBridgeError("Unsloth Studio 拒绝启动训练：" + str(start_payload)[:1200])
            start_map = _find_mapping(start_payload) or {}
            job_id = start_map.get("job_id") or start_map.get("id") or start_map.get("run_id")
            on_status(
                training_status="training",
                training_stage="Unsloth Studio 已接收任务，等待训练指标",
                training_model=model_name,
                studio_job_id=str(job_id) if job_id else None,
                training_output_dir=start_map.get("output_dir") or start_map.get("output_path"),
            )

            def schema_for(tool: Any) -> tuple[dict[str, Any], set[str]]:
                tool_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
                properties = tool_schema.get("properties", {}) if isinstance(tool_schema, dict) else {}
                required_fields = set(tool_schema.get("required", [])) if isinstance(tool_schema, dict) else set()
                return properties, required_fields

            def arguments_for(tool: Any, candidates: dict[str, Any]) -> dict[str, Any]:
                properties, required_fields = schema_for(tool)
                if properties:
                    args = {k: v for k, v in candidates.items() if k in properties and v is not None}
                else:
                    args = candidates
                absent = sorted(required_fields - set(args))
                if absent:
                    raise StudioBridgeError(
                        f"MCP 工具 {tool.name} 缺少必要参数：" + ", ".join(absent)
                    )
                return args

            stop_sent = False
            last_status: dict[str, Any] = {}
            started_polling_at = time.monotonic()
            last_recognized_status_at = started_polling_at
            # Studio owns training; this loop mirrors Studio state into the outer dashboard.
            while True:
                if should_stop() and not stop_sent:
                    stop_tool = tools["stop_training"]
                    stop_args = arguments_for(stop_tool, {"save": True, "expected_job_id": job_id, "job_id": job_id})
                    stop_result = await session.call_tool("stop_training", arguments=stop_args)
                    stop_payload = _result_payload(stop_result)
                    if getattr(stop_result, "isError", False) or getattr(stop_result, "is_error", False):
                        raise StudioBridgeError("请求 Studio 停止训练失败：" + str(stop_payload)[:1200])
                    stop_sent = True
                    on_status(training_status="stopping", training_stage="已向 Unsloth Studio 请求停止")

                status_tool = tools["get_training_status"]
                status_args = arguments_for(status_tool, {"expected_job_id": job_id, "job_id": job_id, "run_id": job_id})
                status_result = await session.call_tool("get_training_status", arguments=status_args)
                status_payload = _result_payload(status_result)
                status_map = _find_mapping(status_payload)
                if status_map:
                    last_recognized_status_at = time.monotonic()
                    nested_status = status_map.get("status")
                    if isinstance(nested_status, dict):
                        status_map = {**status_map, **nested_status}
                    nested_training = status_map.get("training_status")
                    if isinstance(nested_training, dict):
                        status_map = {**status_map, **nested_training}
                    last_status = status_map
                    phase = str(status_map.get("phase") or status_map.get("status") or "").lower()
                    running = status_map.get("is_training_running")
                    progress = status_map.get("progress")
                    if not isinstance(progress, dict):
                        progress = status_map
                    step = progress.get("step", progress.get("global_step"))
                    total = progress.get("total_steps", progress.get("max_steps"))
                    loss = progress.get("loss", progress.get("train_loss"))
                    lr = progress.get("learning_rate", progress.get("lr"))
                    pct = progress.get("progress_percent")
                    if pct is None and isinstance(step, (int, float)) and isinstance(total, (int, float)) and total > 0:
                        pct = min(100.0, float(step) * 100.0 / float(total))
                    if isinstance(pct, (int, float)):
                        on_status(training_progress=round(float(pct), 2))
                    updates: dict[str, Any] = {}
                    if isinstance(step, (int, float)): updates["training_step"] = int(step)
                    if isinstance(total, (int, float)): updates["training_max_steps"] = int(total)
                    if isinstance(loss, (int, float)): updates["training_loss"] = float(loss)
                    if isinstance(lr, (int, float)): updates["training_learning_rate"] = float(lr)
                    if isinstance(progress.get("epoch"), (int, float)): updates["training_epoch"] = float(progress["epoch"])
                    if status_map.get("message"):
                        updates["training_stage"] = str(status_map["message"])
                    elif phase:
                        phase_labels = {
                            "loading_model": "Unsloth Studio：加载模型",
                            "loading_dataset": "Unsloth Studio：加载数据集",
                            "configuring": "Unsloth Studio：配置训练",
                            "training": "Unsloth Studio：模型训练中",
                            "finalizing": "Unsloth Studio：保存训练产物",
                        }
                        updates["training_stage"] = phase_labels.get(phase, f"Unsloth Studio：{phase}")
                    if status_map.get("output_dir") or status_map.get("output_path"):
                        updates["training_output_dir"] = status_map.get("output_dir") or status_map.get("output_path")
                    if updates:
                        on_status(**updates)

                    if phase in {"error", "failed"}:
                        raise StudioBridgeError(str(status_map.get("error") or status_map.get("message") or "Unsloth Studio 训练失败"))
                    if phase in {"completed", "complete", "success"}:
                        on_status(training_status="completed", training_stage="Unsloth Studio 训练完成", training_progress=100.0)
                        return {"job_id": job_id, "status": "completed", "studio_status": last_status}
                    if phase in {"stopped", "cancelled", "canceled"}:
                        on_status(training_status="cancelled", training_stage="Unsloth Studio 训练已停止")
                        return {"job_id": job_id, "status": "cancelled", "studio_status": last_status}
                    if running is False and phase in {"idle", ""}:
                        # Avoid falsely treating an idle response as completion when the start was accepted
                        # but the backend has not yet switched to loading_model.
                        if stop_sent:
                            on_status(training_status="cancelled", training_stage="Unsloth Studio 已停止训练")
                            return {"job_id": job_id, "status": "cancelled", "studio_status": last_status}
                        if time.monotonic() - started_polling_at > 45:
                            raise StudioBridgeError(
                                "Studio 已接受任务，但 45 秒内状态仍是 idle。请检查 Studio 训练页、任务日志和模型/数据路径。"
                            )
                if time.monotonic() - last_recognized_status_at > 120:
                    raise StudioBridgeError(
                        "连续 120 秒无法从 Studio MCP 读取有效训练状态；为避免调度台无限等待，已中止状态同步。"
                        "请检查 MCP 服务与令牌。Studio 侧训练任务可能仍在运行，请先到 Studio 页面确认。"
                    )
                await asyncio.sleep(max(1.0, float(os.getenv("UNSLOTH_STUDIO_POLL_SECONDS", "3"))))


def run_training_via_studio(
    dataset_path: str,
    model_name: str,
    max_seq_length: int,
    max_steps: int,
    on_status: Callable[..., None],
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    """Synchronous entry point for the Flask worker thread."""
    try:
        return asyncio.run(
            _run_training_async(dataset_path, model_name, max_seq_length, max_steps, on_status, should_stop)
        )
    except StudioBridgeError:
        raise
    except Exception as exc:
        raise StudioBridgeError(f"Unsloth Studio 联动异常：{type(exc).__name__}: {exc}") from exc
