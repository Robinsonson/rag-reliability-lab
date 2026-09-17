# v2 本地验收记录

日期：2026-09-17。基线：`fbd20b7b93a1b45ced5b9a94624d79b27feb3a02`。

## 已执行

- Python 3.12 / Windows：新版后端 **89 项测试通过，0 失败、0 跳过**，79.46 秒。包含真实 LangGraph 1.2.11 与 SQLite checkpoint 关闭重开、API 认证/CSRF/权限撤销、失效线程重放拒绝、发布作业提交后崩溃恢复、短片段精确修改、版本 CAS、索引失败、回滚、租约接管、embedding 复用与评测指标合同。
- Docling 2.128.0：真实 DOCX、PPTX、XLSX、HTML 子进程解析通过；PPTX 幻灯片与 XLSX 工作表/单元格定位有断言。TXT/MD 中英文、标题层级、行号、BOM UTF-16、异常格式/资源限制已测。上述测试包含在 89 项中。
- Node / Playwright / Chrome：前端契约测试 **6 项通过**。API 在此组为可控替身，覆盖 XSS 文本显示、cookie/CSRF、角色界面、批量上传、分页、审核/排队/发布状态、失败重试、退出清屏、恢复中断发布与窄屏。
- 浏览器连接真实本地 FastAPI + SQLite + LangGraph：登录、合成政策问答、版本查看、500→800 修改稿、刷新恢复审核、390px 窄屏通过；停在审核，不执行批准。这一预览使用可控 LLM / embedding，截图只证明界面和交互，不能证明真实模型效果。
- Python 编译、JavaScript 语法检查、已有变更的 Git diff 空白检查、Compose YAML 解析通过。
- SentenceTransformer / CrossEncoder 实际导入与离线参数签名检查通过。修复了本机其它 TensorFlow/Keras 安装被 Transformers 可选集成意外加载的问题；新版仅使用 PyTorch。

一次第三方 Starlette/AnyIO 弃用提示不影响测试通过；并未把当前 Windows 虚拟环境宣称为完整验证的 Linux 镜像依赖锁。

## 尚未执行

- **Docker Compose 镜像构建与启动、真实 PostgreSQL/pgvector/PostgresSaver 和 SeaweedFS S3 集成。** 已检查 Windows 常见安装位置及 Ubuntu WSL，均未找到 Docker 可执行文件。YAML 解析不等于 Compose 或服务启动验证。
- **中文扫描 PDF OCR 与复杂跨页表格实测。** 本机未准备生产 OCR 模型与 Tesseract 中英文语言包；镜像和模型准备命令已提供。测试确认缺少 PDF 模型时明确失败，不假装已 OCR。
- **真实模型问答与更新效果。** 没有使用用户模型密钥或发送真实政策到外部服务；需配置模型并运行双语案例及人工审核。
- **1 万份 / 单文件 100 MiB 的全链路与并发容量压测。** 目前是配置目标；小规模流程测试和限额测试不能证明该容量。尤其需要测总体页数、扫描比例、HNSW 权限过滤召回、历史索引增长、CPU/内存/磁盘和 p95 延迟。
- **旧版完整依赖环境的重新跑分。** 旧源码与冻结数据保持不变，新测试没有将旧模型分数作为升级结果。

## 继续验收

1. 准备 Docker 后按 `UPGRADE_V2.md` 构建并启动，验证数据库与对象存储持久性、杀进程恢复和并发发布。
2. 下载本地模型，抽样验证真实中文扫描件、表格与每一种实际文档格式。
3. 在本地填写模型服务配置，导入独立双语样例，运行检索评测；用 `--answers` 显式运行真实回答检查，并人工评判语义正确性和修改定位。
4. 按真实文档分布进行容量/并发测试后再决定限额与部署规模。

v2 代码在 `feature/langgraph-v2` 分支供评审，旧版由 `langchain-rag-v1.0.0` 标签保留。上传开发分支不代表上述待验收事项已通过；合并主分支和正式发布应在完成相应验收后进行。
