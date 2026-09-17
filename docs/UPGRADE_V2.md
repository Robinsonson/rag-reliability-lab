# LangGraph 政策工作台 v2

本次改造以原仓库 `fbd20b7b93a1b45ced5b9a94624d79b27feb3a02` 为基线。旧版已固定为 [LangChain RAG v1.0](https://github.com/Robinsonson/rag-reliability-lab/releases/tag/langchain-rag-v1.0.0)；v2 在 `feature/langgraph-v2` 分支评审。当前已执行的验证与待验收事项见 [验收记录](VALIDATION_V2.md)。

## 已确认的业务边界

- 本地 Docker Compose，服务结构可以迁移到服务器。
- 以 1 万份政策、单文件 100 MiB 为设计与配置目标。实际容量取决于总页数、OCR 占比、硬件和并发，必须另做压测。
- 同时处理中文与英文；默认本地 BGE-M3 embedding、BGE multilingual reranker，模型权重需显式下载。
- 有权限的管理者提出自然语言修改，系统定位条款并展示前后对比；该管理者明确确认后才排队发布。
- 系统版本化正文为更新后的权威内容。上传原件不可变留档；新版本正文可导出 Markdown，原件下载入口明确仍为原件。

## 新版运行方式

要求 Docker Engine / Docker Desktop 与 Compose v2，以及可执行 `setup-env` 的 Python 3.12。镜像中运行 Python 3.12，包含中英文 Tesseract OCR。首次构建和模型准备会下载较大的依赖与权重；建议为容器和模型预留足够内存、磁盘。Compose 的 API / worker 内存上限分别为 6 / 8 GiB，这是资源保护配置，不是已验证的最低机器规格。

在仓库目录运行：

```powershell
python -m policy_lab.manage setup-env
docker compose --env-file .env.v2 build
docker compose --env-file .env.v2 run --rm --no-deps api python -m policy_lab.manage download-models
docker compose --env-file .env.v2 up -d
docker compose --env-file .env.v2 logs -f api worker
```

先在本机编辑 `.env.v2`：填入自己的模型服务 API key，并查看生成的管理员初始密码。该文件已忽略，不应提交。`setup-env` 不覆盖已有文件，也不在终端打印密码。默认兼容原项目的 DeepSeek 服务配置；问答、证据判断与修改草稿会把相应的已授权正文发送至配置的模型服务。解析、embedding 和 rerank 使用本地模型。需要内网部署时配置内网模型端点，并提前镜像/缓存全部权重。

打开 `http://127.0.0.1:8000`，使用 `.env.v2` 中的初始账号登录。先创建日常使用的 `manager` / `reader` 账户并指定 scope（如 `hr`、`finance`）。管理者只能查询和更新自己的范围；只有管理员可以管理账户。这里使用本地账户，不依赖前端传入的角色，也未接入企业 SSO。

Windows 快捷入口 `./run_lab.ps1` 启动新版 Compose。`./run_lab.ps1 -Legacy` 保留旧版单人 benchmark 入口；旧服务不应对外开放。新版没有挂载旧版匿名上传/修改 API。

服务器部署前需配置 TLS 反向代理，设 `SECURE_COOKIES=true`，设置正确的代理转发头，并使用组织的身份、备份与运维配置。Compose 默认只将 API 映射到本机环回地址；数据库和对象存储不暴露宿主端口。

## 用户流程

### 文档管理

批量选择 PDF、DOCX、PPTX、XLSX、HTML、MD 或 TXT。浏览器显示每个文件上传进度；服务端接收 multipart 流并写入 S3 兼容存储。每份文档返回持久 job，独立 worker 完成解析、分块、embedding 和发布。上传和索引状态分别显示。失败保留原件和错误记录，可在重试次数内重试。

默认最大单文件 100 MiB、最多 2,000 页、最多 1,000 万抽取字符；这些限制可配置。页数、字符数与文件体积分别约束不同资源，不能把放开 MB 理解成无限制。上传失败时浏览器重新传整个文件；当前没有实现浏览器分片断点续传。S3 SDK 的 multipart 传输不等于浏览器断点续传。

Office 压缩包还限制最多 20,000 个归档项、解压总量不超过上传上限的 4 倍；限制在启动解析子进程前检查。HTML 不读取外部链接或本地文件资源。TXT/MD 接受 UTF-8 或带 BOM 的 UTF-16，不静默猜测旧编码。

TXT/MD 使用轻量解析。复杂格式采用独立 Docling 子进程，设解析超时；PDF 使用本地 OCR 与表格解析。来源位置按格式保存解析器实际提供的页码/坐标/标题路径等元数据，不把 Word 段落或 Excel 单元格伪造成固定 PDF 页码。解析得到的表格正文以 Markdown 表示；第一版没有 Excel 单元格级编辑器，也不承诺原文件排版重建。

### 问答

图流程：鉴权 → 固定发布版本快照 → 向量与关键词检索 → RRF 融合/重排 → 证据充分性判断 → 必要时一次改写并补查 → 带引用生成 → 来源编号与正文元数据校验。

每次问题最多两轮检索，循环和上下文均有上限。政策事实只使用已授权政策证据，不从公网自动补公司规则。检索辅助文本包含政策名称和标题上下文；回答引用仍来自规范正文。模型判断充分性、模型生成回答与确定性引用校验是不同环节。**引用存在不证明每一句回答语义正确**，该限制在界面和评测中明确保留。

### 老板自然语言修改

例如：“把国内出差住宿标准从 500 元改到 800 元。”

1. 系统仅在有修改权限的 scope 中检索候选条款。
2. 若候选包含国内/海外、不同职级等，系统给出业务澄清问题与候选摘要；用户无需寻找文件名、policy ID 或段号。
3. 系统取回完整规范条款，模型只提出短原文片段与替换片段；程序验证唯一匹配后生成新正文，保留其余内容和空白。界面展示所属政策、原文、拟改正文与理由。
4. 通过 LangGraph `interrupt` 等待真实登录用户明确确认。刷新或进程重启后可以恢复；`thread_id` 只定位状态，服务端仍校验归属和权限。
5. 确认后的提案进入后台 job。先创建修订、复用未变正文的 embedding、准备新索引，再在事务中核对基准版本并切换发布指针和审计记录。
6. 两人基于同一旧版修改时，先发布者生效，后发布者得到冲突；不会静默覆盖。索引失败时旧版继续可查。

发布作业已写入但进程中断时，界面提供“继续处理”，依赖已持久保存的批准恢复原作业，不重新起草或重复发布。短片段补丁默认限制原片段 400 字符、替换片段 600 字符；大范围重写应拆成明确的修改请求。片段匹配不唯一时拒绝生成提案，需要更明确的修改请求；不会批量替换。

一次修改以一个政策中的一个规范 block 为范围。跨条款联动、复杂生效日期、组织流程中的双人会签暂不自动执行；系统应澄清范围，管理者最终检查展示的完整差异。原文件整体替换和自然语言局部修订是不同操作：局部修订保持条款身份，整体重新解析不承诺跨版本锚点自动匹配。

历史版本可查看、导出；回滚经过界面确认，创建新发布修订并记录审计，不删除历史。已修改条款的原件定位只是归档来源，不表示旧 PDF 已被改字。

## 架构与可靠性

- `policy_lab/api.py`：认证、CSRF、上传、查询、审核恢复、历史导出、账户管理。
- `policy_lab/graphs.py`：两张真实 StateGraph；`llm.py` 为可注入的结构化模型边界。
- `policy_lab/repository.py`：规范版本、scope 校验、提案、数据库作业租约、索引与发布事务。
- `policy_lab/parsing.py` / `indexing.py` / `storage.py`：解析、双语检索、原件存储。
- `policy_lab/worker.py`：独立 worker，失效租约可重新领取，旧 worker 不能覆盖新领取者的结果。
- PostgreSQL 保存规范正文、索引、发布指针和审计；LangGraph PostgresSaver 保存工作流进度；原附件在对象存储。

本地对象存储使用 SeaweedFS 4.47 的单节点 `mini` 模式，不向宿主机暴露管理端口。S3 凭据由私有环境文件配置；该模式只用于本次本地/单节点部署，并不代表高可用集群。初始化服务会等待对象存储就绪。

采用 PostgreSQL 是为了把发布指针、词法/向量索引的可见性和审计保持在同一个数据库一致性边界内。PostgreSQL 词法排名使用全文检索而不是 BM25；旧 benchmark 的 BM25 不被冒充为新版实现。中文词法预处理使用 CJK 字符/二元词与英文词项，多语言效果仍须在真实政策上验证。

后台任务采用数据库持久队列，第一版不再额外引入 Redis/Celery。PostgreSQL 的行锁与租约支持独立 worker 领取任务；失败重试有上限。同一图线程在数据库中加执行锁，防止同时恢复；checkpoint 并不代替业务幂等与事务。

单个 API 进程同时最多执行 4 个图请求，超额返回可重试的繁忙响应，为数据库节点查询保留连接容量。该值是并发保护，不是已测吞吐。

保留 unchanged chunk 的 embedding 缓存；每次修订仍创建该文档版本的索引行。这是 embedding 计算的增量复用，不宣称所有存储写入都只改变一个向量。全量文档的历史正文/索引持续保留；正式使用前应制定保留期、备份和恢复策略。

## 验证与后续验收

不调用外部模型的回归测试：

```powershell
python -m pip install -r requirements-v2.txt pytest
python -m pytest -c pytest-v2.ini
```

测试覆盖真实 LangGraph 中断/恢复、非法引用、范围隔离、降权/撤销权限、重复确认、并发基准版本冲突、索引失败不发布、worker 失效租约、原件留档与规范正文导出。测试中的 embedding 和 LLM 可控替身用于校验流程，不作为语义检索或回答质量结果。

浏览器契约测试使用 Node 20+ 与 Playwright：`npm install` 后运行 `npm run test:ui`。Windows 默认使用常见路径的 Chrome；其它环境设置 `POLICY_TEST_BROWSER` 为可用 Chromium/Chrome 可执行文件。前端是原生 HTML/CSS/JS，安装这些开发依赖不影响部署时无前端构建的方式。

新版提供合成中英文政策和独立评测脚本；合成文件必须明确标记为样例，不能混同真实公司制度。旧版 20 个开发问题与 8 个冻结测试问题保持不变，其分数不能用于宣称本次 1 万份、多语言知识库的效果。

按 `evaluation_v2_cases.json` 中指定的标题与 scope 手工上传 `fixtures/v2/` 两份文档，待发布成功后运行：

```powershell
# Set POLICY_EVAL_USERNAME and POLICY_EVAL_PASSWORD locally before running.
python scripts/evaluate_policy_v2.py --base-url http://localhost:8000
python scripts/evaluate_policy_v2.py --answers
python scripts/evaluate_policy_v2.py --offline benchmark-results/v2.json --output benchmark-results/v2-rescored.json
```

默认仅运行检索；`--answers` 才调用真实模型。报告的 `literal_excerpt_coverage` 只测标题与原文片段覆盖，引用检查只测范围与版本来源，不是语义准确率。报告含逐例证据，应按政策数据保存权限管理。

交付时应分别报告：代码/流程测试、格式解析实测、Docker/PostgreSQL/S3 集成实测、OCR 实测、真实模型问答评测、容量压测。没有运行的层级必须明确标为待验收，不用配置上限代替结果。

当前边界包括单公司 scope 权限、本地账户、单节点 Compose、Markdown 规范正文导出、无原 PDF 原排版回写。企业 SSO、多租户隔离、全文+图谱影响分析、浏览器分片直传、大规模在线负载与高可用部署应在真实场景和数据基准确定后继续扩展。

## 调研依据（2026-09-17 核查）

- [LangGraph v1 官方说明](https://docs.langchain.com/oss/python/releases/langgraph-v1)：采用 StateGraph 自定义编排，可继续复用 LangChain 模型组件；不将换框架等同于提升答案准确率。
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)：人工审核与恢复；恢复会从节点开头执行，写入必须幂等。
- [LangGraph Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)：生产持久化后端和工作流恢复。
- [Agentic RAG 综述 v4](https://arxiv.org/html/2501.09136v4)、[Search-R1](https://arxiv.org/abs/2503.09516)、[AdaSearch](https://arxiv.org/abs/2512.16883)：借鉴动态检索、证据检查与停止机制，不宣称本项目复现论文训练栈或研究指标。
- [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)：检索加入上下文、关键词与语义信号、重排的依据；该文实验数字不是本项目效果承诺。
- [Docling 官方项目](https://github.com/docling-project/docling)：多格式、OCR 与结构化文档解析。
- [BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3)：双语/多语言 embedding 候选，实际效果须以本项目评测为准。
- [pgvector](https://github.com/pgvector/pgvector)：数据库内向量索引；近似检索需同时评估权限过滤后的召回、延迟和资源。
- [SeaweedFS 4.47](https://github.com/seaweedfs/seaweedfs/releases/tag/4.47)：本地 S3 兼容对象存储；选型时核实了维护状态，未继续使用已归档 MinIO 项目作为默认依赖。
- [Microsoft GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/)：知识图谱可用于跨制度关联与影响分析，但与 LangGraph 执行图不同；本轮条款更新以规范原文和版本为权威。

上述方案是针对本仓库的工程选择，不声称代表任何特定大厂内部架构。
