# 项目代码审计与修复报告

## 修复内容

1. **训练进度可观测性**：新增 `ProgressCallback`，通过 Trainer 回调同步 global step、总步数、百分比、loss、学习率和 epoch。
2. **前端进度条**：移除循环动画式的假进度，改为根据 `/api/status` 返回的实际训练百分比设置宽度；控制台每 5 秒轮询。
3. **停止训练**：新增 `POST /api/stop_training`，通过 `TrainerCallback` 在当前训练步结束后请求停止，并保存部分 adapter。
4. **训练状态**：状态中包含 preparing/loading/training/stopping/saving/completed/failed/cancelled 等阶段。
5. **训练数据保护**：仅在完整训练和 adapter 保存成功后删除 raw 文件；失败或主动停止时保留 raw 和 batch JSONL。
6. **数据目录隔离**：新增 `data/inbox/` 作为外部本地 JSON 投递目录，避免输入目录与 raw 训练队列重叠而造成重复训练。
7. **本地文件变更检测**：本地 JSON 的处理标记保存文件大小和 `mtime_ns`，文件修改后可重新采集。
8. **TRL 兼容性**：按已安装版本的 `SFTConfig`/`SFTTrainer` 签名选择 `max_length`、`max_seq_length`、`dataset_text_field`、`processing_class` 或 `tokenizer` 等参数。
9. **安全默认值**：Web 服务默认只监听 `127.0.0.1`；删除 adapter 时限制名称必须以 `adapter_` 开头，并进行真实路径校验。
10. **模板路径**：Flask 显式指向项目内 `templates/`，不再依赖启动时的当前目录。
11. **移除第三方页面跟踪脚本**：避免控制台页面默认加载不必要的外部遥测脚本。
12. **可配置参数**：模型、序列长度、训练步数、数据阈值、数据目录和 Web 监听参数可通过环境变量配置。

## 已执行的验证

- Python 源码 AST 解析与 `py_compile`：通过。
- 前端内联 JavaScript `node --check`：通过。
- 使用轻量 Flask shim 的路由注册与状态 API smoke test：通过；注册 12 个路由。
- 在训练依赖缺失的测试条件下，验证训练失败后 raw 样本仍存在、batch 文件保留、训练锁释放：通过。
- 停止训练 API 的状态转换 smoke test：通过。

## 未验证/限制

- 当前执行环境未安装 Flask、Unsloth、PyTorch CUDA、TRL 等实际运行依赖，因此未完成真实 Flask 服务启动、真实 GPU 训练和真实 adapter 加载/推理验收。
- Unsloth 与 TRL 版本兼容性依赖实际安装环境。虽然代码按构造函数签名选择常见参数，仍应在目标 GPU 环境中运行一次小样本训练验证。
- 控制台默认不提供登录认证，故默认仅监听本机；远程访问建议通过 SSH 隧道，不应直接暴露公网。
- 此版状态保存在进程内存中；不支持多进程共享状态、断电后任务恢复、持久化队列和多用户权限。
- 这里实现的是“该脚本调用 Unsloth 并把 Trainer 回调指标同步到本项目控制台”。它不会自动把任务注册到另一个独立的 Unsloth Web UI；那需要对接该 UI 的实际 API 或任务后端。

## Unsloth Studio 联动增补（2026-10-10）

13. **官方 Studio 任务入口**：新增 `unsloth_studio_bridge.py`，可通过 Unsloth Studio 官方可选 MCP 工具 `start_training` / `get_training_status` / `stop_training` 启动、监控和停止任务；由 Studio 管理训练子进程与任务历史。
14. **双模式**：`TRAINING_BACKEND=local` 保留原有本地 Trainer 路径；`TRAINING_BACKEND=studio_mcp` 走 Studio MCP，且不在调度台进程中再启动一份本地 Trainer。
15. **状态映射**：读取 Studio 的 phase、step、total_steps、loss、learning_rate、epoch、message 和 job_id 等可用字段，同步到调度台状态 API。
16. **数据路径校验**：支持本机共享路径，及通过两个路径前缀变量将调度台的 batch 路径映射为 Studio 主机可见的共享挂载路径；不会假装支持无共享路径的跨机器上传。
17. **训练 JSONL 修复**：修正写入 batch 数据时的换行符，保证每条样本真实占一行；同时修正本地 SFT 格式化文本中的换行符。
18. **错误与超时**：检查 Studio MCP 必需工具、动态读取工具 schema；若任务状态持续不可读或启动后一直 idle，则返回清晰错误，避免无期限挂起。

### Studio 联动验证边界

- 已做源码编译、前端 JavaScript 语法检查与桥接模块静态检查。
- 当前沙箱不能访问外网，也未安装/运行真实 Unsloth Studio MCP 服务、Flask、MCP SDK 和 GPU 训练环境，因此**没有完成真实端到端 API 联动测试**。
- Studio 模式需要在 Studio 启动时显式开启 MCP 并配置令牌；还需要本项目安装 `requirements-studio-bridge.txt`。真实环境首次运行时，请先手动确认 Studio 训练页/任务历史中出现任务，再验证进度和停止同步。
- Studio 模式的训练产物由 Studio 自己保存；目前本项目 Adapter 列表仍扫描本地 `data/adapters/`，不会自动显示 Studio 输出目录中的 Adapter。
- 另使用模拟 MCP 服务对桥接器做了启动任务、接收任务 ID、同步完成状态和进度字段的 smoke test：通过（该测试不等于真实 Studio 服务验收）。


## 配置管理增强与二次审计（2026-10-11）

### 本次新增

19. **数据源参数编辑**：新增 `GET/PUT /api/sources/<id>/config`；按数据源类型使用白名单字段，支持名称、目录、数据库连接、SQL、HTTP API 参数映射等配置；不允许通过该 API 改写 `type`、`enabled` 和内部游标路径。
20. **数据源开关持久化**：源级 enable/disable 状态写入 `config/source_state.json`；参数写入 `config/data_sources.json`，重启后恢复。兼容从旧版 `data/sources.json` / `source_state.json` 迁移。
21. **凭据脱敏**：数据库密码不会返回浏览器；HTTP Headers 中的 Authorization、token、password 等键值脱敏；保存时会保留脱敏占位符对应的原值，显式 `__CLEAR__` 才会清除敏感值。
22. **系统参数编辑与启停**：`/api/config` 支持 GET/PUT/POST，系统配置保存至 `config/system_settings.json`。参数可分别启用/停用；停用后回退到进程启动时环境变量/程序默认值，而自定义值仍被保留。
23. **生效语义与验证**：系统配置加入数值范围、枚举、长度及非空校验。数据基础目录会明确标注为重启项；训练运行期间拒绝立即切换模型、序列长度、步数和后端。
24. **安全加固**：API 可选 `AUTO_TRAINER_TOKEN`；非回环监听地址必须配置令牌。增加 Host 校验、跨站写请求 Origin 检查、常用安全响应头，以及单条 instruction/output 最大字符限制。敏感配置文件使用临时文件 + 原子替换，尽可能限制文件权限为 `0600`。
25. **前端注入边界**：统一 HTML 转义函数也转义引号，降低把用户可配置值插入 HTML 属性时的属性逃逸风险。
26. **数据库查询只读校验**：数据源 SQL 编辑只允许单条 `SELECT` / `WITH` 查询，拒绝多语句和常见写入/管理关键字；启动时加载的持久化 SQL 也执行同样校验，避免旧配置绕过 Web 表单校验。
27. **URL 凭据脱敏**：HTTP API URL 中的 userinfo 和疑似 token/API key 查询参数在 Web API 中遮蔽显示；编辑时保留已有脱敏凭据，显式 `__CLEAR__` 才进行清除。
28. **Git 忽略规则修复**：明确放行 `data/` 下空目录的 `.gitkeep`，同时继续忽略运行时数据，避免 Git 规则阻止空目录被版本控制。

### 本次验证

- `python -m py_compile auto_trainer_v3.py`：通过。
- `node --check` 检查 HTML 中的内联 JavaScript：通过。
- 使用 Flask 接口 shim 执行路由 smoke test：通过，注册 13 个路由。
- 模拟 API 验证数据源编辑和启用状态持久化、数据库密码不回传、HTTP Authorization 脱敏后保留、系统参数保存、停用后回退默认值，以及非法采集间隔返回 400：通过。
- SQL 只读校验 smoke test：常规 SELECT/CTE 查询通过；DELETE、分号多语句和写入型 CTE 被拒绝。
- URL 凭据脱敏/恢复逻辑：加入自动化用例和模拟路由 smoke test；另验证改变主机名时旧 token 不会被自动带到新主机。
- 已检查 ZIP 可包含项目文件并设置 Git 忽略规则，避免 `.env` 和持久化 JSON 凭据进入版本控制。

### 仍未完成的真实环境验证

- 当前执行容器不能解析 GitHub/GitHub Proxy DNS，也未安装 Flask、Unsloth、PyTorch CUDA、TRL、Studio MCP 运行服务；因此不能在此环境完成真实 Flask 服务、真实 GPU 训练或真实 Unsloth Studio 端到端联动。
- 上述 API smoke test 使用轻量 Flask shim，证明业务函数和输入校验逻辑可运行，但不替代真实 Flask/Werkzeug 网络行为测试。
- `config/` 下保存数据源配置文件可能包含密码或 token，已被 `.gitignore` 忽略；部署时仍应确认目录权限，并避免将配置备份上传到公开仓库。
- 项目仍是本机/单实例控制台，不具备完整的用户、角色和操作审计体系。远程访问应配置令牌并优先采用 SSH 隧道；不建议直接暴露公网。
