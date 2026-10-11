# Unsloth 实时训练调度台

此项目由 Flask Web UI + 数据采集/去重 + Unsloth/TRL SFT 微调组成。默认仅监听 `127.0.0.1:5000`，适合本机访问。

## 1. 目录结构

```text
unsloth_realtime_console/
├── auto_trainer_v3.py       # 后端：采集、调度、训练回调、REST API
├── templates/
│   └── index.html           # 前端控制台
├── data/
│   ├── inbox/               # 手工投递/本地目录数据源输入 JSON
│   ├── raw/                 # 去重后的待训练样本 JSON
│   ├── processed/           # 临时训练 JSONL
│   ├── adapters/            # 训练输出的 LoRA adapter
│   └── foundational_data.jsonl  # 可选：基础回放数据，每行一条 JSON
├── config/                    # WebUI 持久化配置（敏感文件不会提交到 Git）
├── logs/
├── unsloth_studio_bridge.py # 可选：对接官方 Unsloth Studio MCP 训练后端
├── requirements.txt         # Web 与数据源依赖
├── requirements-training.txt # 本地训练依赖提示
└── requirements-studio-bridge.txt # Studio MCP 联动客户端（可选）
```

## 2. 安装 Web/API 依赖

建议使用 Python 3.10 或 3.11 的虚拟环境；请根据当前 Unsloth 官方支持矩阵选择训练环境。

```bash
cd unsloth_realtime_console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

每条命令说明：
- `python3 -m venv .venv`：创建独立 Python 环境，避免污染系统 Python。
- `source .venv/bin/activate`：激活虚拟环境。
- `python -m pip install --upgrade pip`：升级当前虚拟环境中的 pip。
- `python -m pip install -r requirements.txt`：安装 Web 控制台和可选数据源依赖。

## 3. 安装 Unsloth 训练环境

Unsloth、PyTorch、CUDA、Transformers 与 TRL 版本必须相互兼容。先按照 Unsloth 官方当前安装说明安装适合本机 CUDA/GPU 的 PyTorch 和 Unsloth，再安装其推荐的 TRL、Transformers 等版本。不要盲目升级所有训练包，否则可能造成 API 不兼容。

```bash
python -m pip install unsloth
python -m pip install datasets accelerate peft bitsandbytes
```

如果官方安装页针对你的 CUDA/PyTorch 版本提供了专用命令，应优先使用官方命令。随后可检查：

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到 CUDA GPU')"
python -c "from unsloth import FastLanguageModel; from trl import SFTTrainer, SFTConfig; print('Unsloth/TRL import OK')"
```

说明：
- 第一条检查 PyTorch 能否访问 CUDA，并输出 GPU 名称。
- 第二条检查本项目训练路径所需的关键类是否可导入。
- 如果第一条显示 `CUDA available: False`，不要直接开始正式训练；先修复 GPU 驱动和 PyTorch/CUDA 环境。

## 4. 启动调度台

```bash
source .venv/bin/activate
python auto_trainer_v3.py
```

然后在运行该程序的机器上打开：

`http://127.0.0.1:5000`

如果 Web 依赖可用、训练依赖未安装，页面仍可以打开，但真正启动训练时会报告依赖错误。

### 远程访问安全

控制台包含训练触发、数据添加、数据源切换和 adapter 删除等管理功能。它没有内置用户登录，因此**不要直接把端口暴露到公网或不可信局域网**。需要从另一台机器访问时，优先使用 SSH 本地端口转发：

```bash
ssh -L 5000:127.0.0.1:5000 user@训练服务器IP
```

在访问端打开 `http://127.0.0.1:5000`。命令说明：把访问端的本机 5000 端口转发到远端服务器的 127.0.0.1:5000，不需要将 Flask 直接暴露到网络。

## 5. 数据格式与自动训练

每条样本至少包含 `instruction` 和 `output` 字段，例如：

```json
{
  "instruction": "解释什么是 VLAN。",
  "output": "VLAN 是在二层交换网络中进行逻辑隔离的虚拟局域网技术。"
}
```

把外部 JSON 文件放入 `data/inbox/`（单文件可以是对象，也可以是对象列表）；后端会去重并复制到 `data/raw/` 训练队列。默认 `AUTO_TRAINER_BUFFER_THRESHOLD=50`，达到阈值后后台调度循环会尝试启动训练。阈值可通过环境变量调整。注意：当前训练流程是周期性采集后触发一次 SFT/LoRA 微调，不是每来一条数据就立即更新模型参数。

可选的 `data/foundational_data.jsonl` 使用 JSONL 格式，每行一个相同字段结构的 JSON 对象，用于与新增样本混合训练。

## 6. 选择训练后端：本地训练或 Unsloth Studio 联动

本项目支持两种模式，默认 `local`，以兼容原来的本地训练方式。

### 模式 A：本地 Unsloth（默认）

`TRAINING_BACKEND=local` 时，由 `auto_trainer_v3.py` 直接调用 Unsloth/TRL。`TrainerCallback` 把实际 step、总步数、百分比、loss、学习率和训练阶段写入本项目状态，前端每 5 秒轮询 `/api/status`。训练产物保存在本项目的 `data/adapters/`。

### 模式 B：把任务交给官方 Unsloth Studio（推荐用于双界面联动）

这个模式通过 Unsloth Studio 官方提供的可选 MCP 服务调用 `start_training`、`get_training_status`、`stop_training`。Studio 自己启动和管理训练子进程、任务历史与训练产物，因此通过本调度台提交的任务应当由 Studio 后端登记，而不是由本脚本私下启动另一份 `SFTTrainer`。接口基于官方文档：<https://github.com/unslothai/unsloth/blob/main/studio/MCP.md>。

**第 1 步：在运行 Unsloth Studio 的机器上启用 MCP。** 用一个长随机字符串替换示例密钥，并确保后续配置完全一致：

```bash
export UNSLOTH_STUDIO_ENABLE_MCP=1
export UNSLOTH_STUDIO_MCP_TOKEN='请替换为长随机密钥'
unsloth studio
```

如果你通过自己的 systemd 服务或 `studio/setup.sh --local` 启动 Studio，需要把这两个环境变量加到实际启动 Studio 的服务/终端环境中，再重启 Studio。MCP 默认是关闭的；不要把带令牌的 MCP 端点直接暴露到公网。

**第 2 步：安装本项目的 MCP 客户端依赖。** 在运行本调度台的 Python 环境中执行：

```bash
python -m pip install -r requirements-studio-bridge.txt
```

**第 3 步：配置本项目。** 复制配置模板，然后编辑 `.env`：

```bash
cp .env.example .env
nano .env
```

至少设置：

```dotenv
TRAINING_BACKEND=studio_mcp
UNSLOTH_STUDIO_MCP_URL=http://127.0.0.1:8888/mcp/
UNSLOTH_STUDIO_MCP_TOKEN=与启动 Studio 时完全相同的密钥
AUTO_TRAINER_MODEL=unsloth/Qwen2.5-3B-Instruct
AUTO_TRAINER_MAX_SEQ_LENGTH=1024
AUTO_TRAINER_MAX_STEPS=50
```

如 Studio 与调度台在同一台机器且看到同一文件系统，默认数据集路径就可用。如果它们在不同机器，`local_datasets` 的路径必须是 **Studio 主机可读的路径**：请先通过共享目录/NFS 等方式把数据目录挂载到两边，再设置 `AUTO_TRAINER_DATA_PATH_PREFIX`（调度台看到的路径前缀）和 `UNSLOTH_STUDIO_DATA_PATH_PREFIX`（Studio 看到的对应挂载前缀）。没有共享文件系统时，本桥接器不会自动把数据文件上传到 Studio。

**第 4 步：启动本项目。**

```bash
./run.sh
```

当数据达到阈值后，调度台会把 JSONL 数据集路径、模型和训练参数提交给 Studio，并轮询 Studio 的训练状态，把实际进度、step、loss、学习率、阶段和任务 ID 显示在本项目面板。点击停止时会调用 Studio 的 `stop_training`。同一任务由 Studio 管理，因此应当在 Studio 的训练页/历史中查看；具体是否显示在当前版本的哪个列表，以 Studio 版本的页面行为为准。

**产物位置：** Studio 模式下，Adapter 由 Studio 保存到它自己的输出目录，不会自动复制到本项目 `data/adapters/`。本项目的 Adapter 列表目前仍只扫描本地 `data/adapters/`；请在 Studio 训练历史/输出目录查看 Studio 模式的产物。

### 联动限制与安全

- Studio MCP 默认关闭，必须显式开启并配置 token；不要把 MCP 端口直接暴露到公网。
- 训练数据路径必须能被 Studio 主机读取。若跨机器，必须先配置共享挂载与路径映射。
- 本桥接器会在启动时读取 MCP 工具 schema，并对必要字段进行检查；如果上游改变了工具字段或当前 Studio 版本没有对应工具，会显示明确错误，而不是悄悄回退到本地训练。
- Studio 的状态/指标字段可能随版本演进；此适配器按当前官方 MCP 工具名和常见 `TrainingStatus` / `TrainingProgress` 字段解析，仍需在目标 Studio 版本上做真实联动验收。
- 本地模式的 Adapter 仍写入 `data/adapters/`。Studio 模式则由 Studio 自己管理任务、训练子进程和产物。

## 7. WebUI：数据源管理与系统配置

### 数据源管理

进入 **数据源** 页面，每个数据源均有 **配置** 按钮和单独的启用/停用开关。支持配置项按类型不同而变化：

- 本地目录：数据源名称、输入目录。
- MySQL / PostgreSQL：主机、端口、用户名、密码、数据库、轮询间隔、SQL 查询。
- SQLite：数据库文件路径、轮询间隔、SQL 查询。
- HTTP API：URL、GET/POST、Headers JSON、Params JSON、响应数据路径、instruction/output 字段映射、轮询间隔。
- CSV / Excel：输入目录、指令/回答列名、轮询间隔。

启用/停用状态与配置会写入 `config/data_sources.json` 和 `config/source_state.json`，服务重启后仍会保留。禁用数据源只停止后续采集，不会删除该数据源配置，也不会删除已采集数据。

**SQL 安全限制：** MySQL / PostgreSQL / SQLite 的 SQL 字段只允许单条只读 `SELECT` 或 `WITH` 查询；多语句、分号分隔语句以及写入/管理类 SQL 会被拒绝。数据库账号本身也建议只授予读取训练数据所需的权限。

**凭据处理：** 数据库密码输入框不会读取回显已有密码；留空表示保持原值，输入 `__CLEAR__` 表示清除。HTTP Headers 中的 Authorization/token 等字段会脱敏显示；保持 `********` 不变即可保留现有值。API URL 中的 userinfo 和 `api_key`/`token`/`password` 等敏感查询参数也会遮蔽，并在保存时保留旧凭据；显式输入 `__CLEAR__` 才清除相应凭据。修改 URL 主机时不会将旧主机的凭据自动带到新主机，需要重新填写或清除。配置文件可能包含数据库密码/API 凭据，项目会尽量以当前用户可读写的 `0600` 权限保存，请勿提交 `config/*.json` 到公共 Git 仓库。

### 系统配置

进入 **系统配置** 页面，可修改并单独启用/停用这些参数：基础模型名称、最大序列长度、训练数据阈值、保留 Adapter 数量、全局默认采集间隔、数据基础目录、单次训练最大步数、训练执行后端。

- 勾选“启用”时使用该页面保存的值；取消勾选时回退到进程启动时的环境变量/程序默认值。保存值会保留，之后可以重新启用。
- 模型、序列长度、训练步数、阈值、Adapter 数量、采集间隔和训练后端在保存后应用于后续操作；正在训练时，不允许立即更换模型、序列长度、训练步数或训练后端。
- 数据基础目录是**重启后生效**的配置。页面会提示“待重启”；更换前请先迁移或备份旧目录。各数据源自己的输入目录仍以数据源管理中的配置为准。
- 全局默认采集间隔变更时，会更新仍沿用旧全局间隔的源；明显单独设置过的轮询间隔尽量保留。

系统配置保存到 `config/system_settings.json`。配置文件损坏或保存值无效时，启动阶段会回退无效项到默认值，并在终端输出警告。

### 远程访问安全

默认 `AUTO_TRAINER_HOST=127.0.0.1`。如确实要监听局域网地址，需要设置强令牌，例如：

```bash
openssl rand -hex 32
```

把命令生成的值填入 `.env` 中 `AUTO_TRAINER_TOKEN`，再修改 `AUTO_TRAINER_HOST`。WebUI 首次调用 API 时会提示输入令牌，并仅存放在当前浏览器标签页的 `sessionStorage` 中。不要将控制台直接暴露到公网；优先使用 SSH 隧道或受保护的反向代理。

## 8. 重要说明

1. 训练失败时保留 raw 数据和 batch JSONL，避免因失败丢失样本；训练成功后才清理已训练的 raw 文件。
2. 训练参数默认 `max_steps=50`，可用 `AUTO_TRAINER_MAX_STEPS` 调整。
3. 此项目不应被视为生产级多用户训练平台：尚未提供登录、角色权限、任务队列持久化、多进程锁和跨进程状态共享。
4. 在真实 GPU 上进行完整训练仍取决于本机模型下载、显存、驱动、CUDA、PyTorch、Unsloth、TRL 版本和数据质量。

### 运行配置 API 自动化测试

在安装 `requirements.txt` 后，可执行：

```bash
python -m unittest discover -s tests -v
```

测试覆盖数据源编辑、数据源开关持久化、秘密值不回显、系统设置启用/停用、非法配置和 SQL 写入语句拒绝；不会启动训练，也不会修改训练数据。
