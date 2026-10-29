# NeuroKiln 脑炉 · 用户手册 v0

> 版本：2026-10-29 · 对应引擎 `neurokiln-engine 0.4.0`、API `0.4.0`
>
> 本工具用于科研场景下的离线 BCI 算法横向评测。所有分数都不代表在线或临床可用性；请勿用于诊断、治疗或医疗器械宣称。

---

## 1. 这是什么

NeuroKiln 脑炉把“数据版本 → 协议划分 → 丹方（预处理/特征/解码器）→ 指标与统计 → 报告/复现包”收敛成一条固定流水线。核心约束只有一句：

**任何进入同一张排行榜的丹方，必须使用同一数据集、同一协议、同一随机种子集合。**

只要违反其中任意一条，系统就只允许分别查看，不允许混排比较。

主要能力：

- 数据集：合成数据（MI / P300 / SSVEP）、MOABB 公开集（BCI IV 2a / 2b、PhysioNet MI、P300、SSVEP）、本地 NPZ；导入向导支持 NPZ / CSV / EDF / FIF；
- 协议：被试内重复分层 K 折、跨 session、留一被试（LOSO）；
- 丹方：12 条内置基线（含 xDAWN、CCA/FBCCA）+ 图形化 DAG 自定义丹方 + 本地插件丹方；
- 模板：可移植 DAG 模板市场、论文复现目录、模板导入/导出与校验和；
- 评测：Balanced Accuracy、Cohen Kappa、逐折/逐被试不确定性与配对检验；
- 校准：逐折概率输出的 ECE / MCE / Brier / NLL 与可靠性曲线；
- 统计：被试级优先的 Wilcoxon-Holm、Friedman、Nemenyi/CD，单被试时明确降级为折级并给出说明；
- 智能调参：GWO + 精英局部搜索 + 低保真代理筛选 + Pareto 复评，全程审计；
- 插件：内置目录一键安装、ZIP 导入、哈希校验、受限/可信两种运行模式；
- 在线：逐试次流式协议、置信度阈值策略与留出折回放 CLI；
- 缓存：按节点、按折保存的预处理产物 Parquet 缓存，LRU 自动淘汰；
- 交付：后台任务、HTML/XLSX/Word/LaTeX 报告、分享卡片、只读演示、Parquet/CSV 逐试次预测、可复现包、CLI、CI 脚本；
- 界面：中英文切换（核心流程已覆盖，部分高级页面仍以中文为主）。

---

## 2. 安装与启动

### 2.1 环境要求

| 组件 | 建议版本 |
|---|---|
| Python | 3.12（最低 3.10） |
| Node.js | 20+（本机验证 24） |
| 内存 | 8 GB 起；真实 2a 单被试建议 16 GB |
| GPU | 非必需；深度基线走 PyTorch CPU |

### 2.2 安装引擎与依赖

在 `neurokiln` 目录执行（Windows PowerShell / cmd 均可）：

```bat
python -m venv .venv312
.venv312\Scripts\activate
pip install -e engine[datasets,dev]
pip install -r server\requirements.txt
```

深度基线还需要 PyTorch；CPU 环境可安装官方 CPU wheel：

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

检查 CLI 是否可用：

```bat
neurokiln --help
neurokiln list-datasets
```

### 2.3 启动 API

```bat
cd neurokiln\server
..\.venv312\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

常用环境变量：

| 变量 | 作用 | 默认 |
|---|---|---|
| `NEUROKILN_DATA_DIR` | 数据根目录（实验库、导入数据、缓存、插件） | `neurokiln/data` |
| `NEUROKILN_DB` | SQLite 实验库路径 | `<data-dir>/neurokiln.db` |
| `NEUROKILN_MNE_DATA` | MOABB/MNE 下载缓存目录 | MNE 默认目录 |
| `NEUROKILN_CACHE` | 预处理产物 Parquet 缓存开关 | `1` |
| `NEUROKILN_CACHE_BYTES` | Parquet 缓存最大字节数 | `536870912`（512 MiB） |
| `NEUROKILN_CACHE_MAX_ENTRIES` | Parquet 缓存最大条目数 | `256` |
| `NEUROKILN_CACHE_ENTRIES` | 内存中缓存的 Epochs 份数 | `2` |
| `NEUROKILN_PLUGIN_TRUST` | 设为 `1` 允许运行 Python 插件（进程内任意代码） | 未开启 |
| `NEUROKILN_JOB_WORKERS` | 后台任务线程数 | `1` |
| `NEUROKILN_TOKEN_TTL` | 登录令牌有效期（秒） | `43200`（12 小时） |
| `NEUROKILN_SECRET` | 覆盖令牌签名密钥（一般不用） | 读取 `data/.auth_secret` |

健康检查：

```bat
curl http://127.0.0.1:8100/api/health
```

返回 `status: ok` 即表示 API、实验库和引擎均已就绪。

### 2.4 启动 Web 控制台

另开一个终端：

```bat
cd neurokiln\web
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5174`。Vite 已把 `/api` 代理到 `127.0.0.1:8100`，无需额外配置跨域。

生产构建与本地预览：

```bat
npm run build
npm run preview
```

> 当前 Web 是开发服务器形态；正式部署时应由反向代理托管 `web/dist` 并把 `/api` 转发给 API 服务。

---

## 3. 首次使用与账号

1. 首次打开 Web 时，登录页进入“初始化管理员”模式；
2. 用户名 3–32 位（字母、数字、`_`、`.`、`-`），密码至少 8 位；
3. 创建成功后自动登录，该账号为 `admin`；
4. 管理员可在“系统设置 → 账号管理”创建/删除成员；系统不允许删除最后一个管理员；
5. 成员可以运行实验、保存丹方、查看报告，但不能管理账号。

安全事实：

- 口令使用 PBKDF2-HMAC-SHA256（310,000 次迭代、随机盐）存储；
- 登录令牌为 HMAC-SHA256 签名的紧凑 Bearer Token，默认 12 小时过期；
- 签名密钥首次启动时随机生成，保存在 `data/.auth_secret`，不写进代码或安装包；
- 除 `/api/health`、`/api/auth/status`、`/api/auth/bootstrap`、`/api/auth/login` 外，所有 `/api/*` 都需要令牌；报告下载同样鉴权。

如果忘记管理员密码，只能停服务后备份并重建 `neurokiln.db` 中的 `users` 表；请先确认没有需要保留的账号。

---

## 4. 数据集

### 4.1 内置数据清单

| ID | 数据 | 范式 | 被试 | 通道 | 类别 | 说明 |
|---|---|---|---|---|---|---|
| `synthetic` | 合成演示 | MI | 1 | 8 | 2 | 零下载自检；可设置每类试次 |
| `synthetic_p300` | 合成 P300 | P300 | 1 | 8 | 2 | 300 ms 正波 + 噪声；验证 xDAWN 与校准 |
| `synthetic_ssvep` | 合成 SSVEP | SSVEP | 1 | 8 | 8 | 8 个刺激频率；验证 CCA / FBCCA |
| `bciciv2a` | BCI Competition IV 2a | MI | 9 | 22 | 4 | MOABB 加载，首次需下载 |
| `bciciv2b` | BCI Competition IV 2b | MI | 9 | 3 | 2 | 左右手二分类 |
| `physionetmi` | PhysioNet EEG MI | MI | 109 | 64 | 4 | 体积大，首次加载慢 |
| `p300` | BNCI2014_009 P300 拼写 | P300 | 10 | 16 | 2 | Target / NonTarget；0–0.8 s 窗口 |
| `ssvep` | Nakanishi 2015 SSVEP | SSVEP | 9 | 8 | 12 | 9.25–14.75 Hz，共 12 个刺激频率 |

每个数据集都带 `paradigm_meta` 元数据：MI 标注任务类型，P300 标注阳性标签与窗口，SSVEP 标注刺激频率。Web 数据集页和详情抽屉会显示该摘要；斗法台据此过滤不兼容的丹方。

查看当前机器可用数据集与状态：

```bat
neurokiln list-datasets
neurokiln list-datasets --dataset bciciv2a --json
```

Web 的“数据集”页会显示适配器、范式、通道、被试数与 `ready / download` 状态。

### 4.2 MOABB 缓存

真实数据通过 MOABB 下载并缓存。建议把缓存放在容量充足的磁盘，并固定使用同一目录，避免重复下载：

```bat
set NEUROKILN_MNE_DATA=D:\data\mne
```

首次加载某个被试会触发下载；之后同一数据版本会复用本地缓存。数据签名由实际加载的试次内容计算，缓存位置变化不会改变签名，但**数据内容或选择范围变化会改变签名**。

### 4.3 导入私有数据（NPZ / CSV / EDF / FIF）

“数据集”页点击“接入私有数据”，按三步向导完成：

1. **上传与解析**：选择 `.npz`、`.csv`、`.edf` 或 `.fif`（单文件上限 2 GiB）。系统先写入 `data/imports/_stage/`，解析成功后显示格式、试次数、通道、采样率、窗口与事件摘要；失败会删除暂存文件。
2. **通道与事件映射**：
   - NPZ 已是分段试次，只显示摘要；
   - CSV 自动识别 `label/class/event/marker`、`time`、`trial`、`subject`、`session` 列；若没有 trial 列，按标签变化切分事件；
   - EDF / FIF 优先读取 annotations，没有则读取 stim 通道；自动排除 `BAD`、`boundary`、`rest` 等事件；
   - 选择参与分析的 EEG 通道，勾选要保留的原始事件并命名类别；设置事件后窗口（默认 MI 0.5–3.5 s、P300 0–0.8 s、SSVEP 0.15–4.15 s）、基线、重采样率；
   - P300 需指定阳性标签名，SSVEP 需填刺激频率（逗号分隔）。
3. **登记元数据**：填写数据集 ID（`[a-z0-9][a-z0-9_-]{1,62}`）、名称、许可、引用、说明、默认被试 / session 与是否覆盖同 ID。

导入成功后，原始文件会转换为 NPZ 与同名 `.json` 侧车文件，记录许可、引用、被试、SHA-256 与数据签名；重复导入同 ID 必须显式开启覆盖。

NPZ 直接导入时必须包含：

| 字段 | 形状 / 类型 | 含义 |
|---|---|---|
| `data` | `(n_trials, n_channels, n_times)` float | 已分段的试次，单位建议为伏特 |
| `labels` | `(n_trials,)` int | 从 0 开始的类别索引 |
| `sfreq` | 标量 | 采样率 |
| `channel_names` | `(n_channels,)` str | 通道名 |

可选字段：`tmin`、`tmax`、`trial_ids`、`subject_ids`、`session_ids`、`label_names`、`dataset_version`。不提供 `subject_ids` 时按单被试处理；不提供 `session_ids` 时无法使用跨 session 协议。

CSV 使用“宽表”：每行一个采样点，至少包含一个标签列、若干数值通道列；`time`、`trial`、`subject`、`session` 列可选。CSV 上限 256 MiB，超过请先转成 NPZ。

EDF / FIF 需要安装 `neurokiln-engine[datasets]` 提供的 MNE 依赖；事件为空或通道选择为空时，向导会阻止提交并给出原因。

也可以直接调用 API：

```bat
curl -X POST http://127.0.0.1:8100/api/datasets/import/inspect ^
  -H "Authorization: Bearer <token>" ^
  -F "file=@my_mi.edf"

curl -X POST http://127.0.0.1:8100/api/datasets/import/commit ^
  -H "Authorization: Bearer <token>" ^
  -H "Content-Type: application/json" ^
  -d "{\"token\":\"<inspect 返回的 token>\",\"dataset_id\":\"my_mi\",\"name\":\"我的运动想象数据\",\"license\":\"内部研究使用\",\"channels\":[\"C3\",\"C4\"],\"event_map\":{\"1\":\"left\",\"2\":\"right\"},\"tmin\":0.5,\"tmax\":3.5,\"sfreq\":250,\"paradigm\":\"MI\"}"
```

`/import` 仍然保留给纯 NPZ 脚本调用；需要通道 / 事件映射时请走 inspect + commit。

### 4.4 预处理产物 Parquet 缓存

引擎会把**显式标记为可缓存**的节点输出（带通、通道选择、CSP 变换结果等）写到 `<data-dir>/cache/preprocess/`，避免同一折在不同丹方/HPO 评估中重复计算：

- 缓存键绑定数据内容签名、该折的划分签名、拟合种子、节点类型与参数、上游节点输出哈希；训练折与测试折分开存储，因此**不会跨折、跨数据集版本复用**；
- 拟合类节点只缓存该折训练折 fit 后的输出，测试折输出单独键控；缓存命中与否不改变数值结果；
- 文件为 pyarrow FixedSizeList Parquet（zstd）；未安装 pyarrow 时自动跳过缓存，不影响运行；
- 默认上限 512 MiB / 256 条，按最近使用时间淘汰；可在“系统设置”查看条目、字节、命中率并清空（仅管理员），或运行：

```bat
neurokiln cache
neurokiln cache --json
neurokiln cache --clear
```

关闭缓存有三种方式：环境变量 `NEUROKILN_CACHE=0`、Web 斗法台关闭“使用预处理缓存”、CLI 加 `--no-cache`。复现包与报告不包含缓存内容，缓存损坏只会导致重新计算，不会污染实验库。

---

## 5. 协议与统计口径

### 5.1 三种协议

| 协议 | 划分方式 | 最少数据要求 | 适用问题 |
|---|---|---|---|
| `within_subject` | 单被试内重复分层 K 折 | 每个类别至少 `folds` 个试次 | 被试内解码上限、算法快速对比 |
| `cross_session` | 用部分 session 训练、留出 session 测试 | 至少 2 个 session | 跨场次稳定性、校准迁移 |
| `loso` | 每次留出一个被试做测试 | 至少 2 名被试 | 跨被试泛化 |

共同参数：

- `--folds`：折数（默认 5）；
- `--repeats`：重复次数（默认 1）；
- `--seeds`：随机种子集合（默认 `0`）。划分由种子决定，报告和复现包会记录种子。

**防泄漏底线**：所有需要拟合的步骤（带通虽无拟合、但 CSP、标准化、协方差中心、分类器、深度网络权重与早停验证集）都只在训练折内部 fit；测试折只做 transform/predict。划分签名会写入每条 run，复现时逐折比对。

### 5.2 指标

- **Balanced Accuracy（BA）**：各类召回率的宏平均，类别不平衡时优先看它；
- **Cohen Kappa**：扣除随机一致后的 agreement；
- **逐折耗时**：单折训练 + 预测的墙上时间，用于成本对照；
- **95% CI**：对独立单位做 percentile bootstrap（固定 `seed=20261029`），默认 2000 次重采样；
- **概率校准**：汇总各折测试集的 `predict_proba`，计算 ECE（10 等宽置信度分箱）、MCE、多分类 Brier 与 NLL，并输出可靠性曲线。校准只评估已训练解码器的输出，**不会在测试折上重新拟合**；所有内置解码器都提供概率输出，若插件或自定义节点未实现 `predict_proba`，该项显示为不可用并给出原因。

### 5.3 统计单位：被试级优先

Arena 会先尝试构造“丹方 × 被试”矩阵：

1. 每个被试的分数 = 该被试在全部测试折上的平均 BA；
2. 只有出现在**所有丹方**结果中的共同被试才进入检验；
3. 共同被试 ≥ 2 时，以被试为独立单位做配对 Wilcoxon-Holm、Friedman 与 Nemenyi/CD，CI 也按被试重采样；
4. 只有 1 名被试（典型单被试实验）或没有共同被试时，自动降级为折级检验，并在结果与报告中写明“以折为检验单位”及降级原因。

**不要把单被试的折级 CI 当作人群级置信区间。** 系统会明确区分 `unit=subject` 与 `unit=fold`，两者不会混在一起展示。

---

## 6. 丹方与 DAG

### 6.1 内置基线

| ID | 名称 | 范式 | 家族 | 关键点 |
|---|---|---|---|---|
| `raw_centroid` | 原始均值 + 最近质心 | MI/P300/SSVEP | baseline | 最小可运行基线 |
| `centered_centroid` | 训练折中心化 + 最近质心 | MI/P300/SSVEP | baseline | 展示防泄漏预处理 |
| `var_centroid` | 通道方差 + 最近质心 | MI/P300/SSVEP | baseline | 与均值特征对照 |
| `csp_lda` | CSP + LDA | MI | spatial | 8–30 Hz；CSP 与收缩 LDA 在训练折拟合 |
| `fbcsp_lda` | FBCSP + LDA | MI | spatial | 4–32 Hz 滤波器组，不做测试折特征选择 |
| `riemann_mdm` | Riemann MDM | MI | riemann | OAS 协方差 + 黎曼均值最近距离 |
| `ts_lr` | 切空间 + LR | MI | riemann | 参考矩阵仅来自训练折 |
| `eegnet` | EEGNet | MI | deep | 128 Hz 重采样、训练折内早停 |
| `shallowconvnet` | ShallowConvNet | MI | deep | 轻量时域-空域卷积、训练折内早停 |
| `xdawn_lda` | xDAWN + 收缩 LDA | P300 | erp | xDAWN 空间滤波与 LDA 仅在训练折拟合 |
| `ssvep_cca` | CCA | SSVEP | ssvep | 使用数据集 `paradigm_meta` 中的刺激频率做相关最大化 |
| `ssvep_fbcca` | FBCCA | SSVEP | ssvep | 子带 CCA 加权融合，无监督参考信号 |

斗法台只会列出与当前数据集范式兼容的丹方；例如选择 SSVEP 数据集时，CSP/EEGNet 等 MI 丹方会被过滤，插件与自定义丹方也按声明的范式参与过滤。

### 6.2 自定义 DAG

“丹方设计”页提供图形编辑器，内置 33 类节点：

- 预处理：`preprocess.bandpass`、`preprocess.filter_bank_split`、`preprocess.notch`、`preprocess.car`、`preprocess.crop`、`preprocess.resample`、`preprocess.baseline`、`preprocess.xdawn`、`preprocess.channel_select`、`preprocess.mean_center`；
- 特征：`feature.channel_mean`、`feature.channel_std`、`feature.csp`、`feature.fbcsp`、`feature.standardize`、`feature.bandpower`、`feature.hjorth`、`feature.erp_window`、`feature.covariance`、`feature.tangent_space`、`feature.select_k`；
- 合并：`merge.concat`、`merge.average`、`merge.vote`；
- 解码器：`decoder.nearest_centroid`、`decoder.lda`、`decoder.riemann_mdm`、`decoder.tangent_space_lr`、`decoder.eegnet`、`decoder.shallowconvnet`、`decoder.xdawn_lda`、`decoder.ssvep_cca`、`decoder.ssvep_fbcca`。

规则：

- 单解码器可以直接作为终点；多解码器必须全部接入同一个 `merge.average` 或 `merge.vote` 融合节点，融合节点必须是最终输出；
- `merge.concat` 只汇合 2D 特征且至少两路输入；融合节点只接受解码器输出且至少两路；
- `preprocess.filter_bank_split` 把 epoch 切成 2–8 个相邻子带，每个子带是独立命名端口（`band_1`…），连线时通过 `source_port` 选择具体子带，不能复用整段信号；
- 节点输入有数据类型：特征节点与 `decoder.riemann_mdm`、`decoder.tangent_space_lr`、`decoder.eegnet`、`decoder.shallowconvnet`、`decoder.xdawn_lda`、`decoder.ssvep_cca`、`decoder.ssvep_fbcca` 只接受 3D 试次（`epochs`）；`feature.standardize`、`feature.select_k`、`decoder.lda`、`decoder.nearest_centroid` 只接受 2D 特征（`features`）。编辑器会在连线时提示不匹配，后端保存/运行前也会拒绝；
- 不允许环；保存前会做拓扑、端口、数据类型与参数校验；
- 保存后的丹方以 `custom:<id>` 出现在斗法台、HPO 提升和 CLI 中；
- 保存后可点击页头“调参空间”，为内置节点的 `int` / `float` / `select` 参数声明 1–16 个 HPO 搜索维度（详见 8.2）；
- 自定义丹方仍受同一套训练折 fit 隔离约束；
- 启用插件后，插件声明的 `plugin.<plugin_id>.*` 节点也会出现在节点面板，可与其他节点自由组合。

参数编辑器中修改节点参数后，请先“校验”再保存，避免把非法图写入实验库。`preprocess.bandpass`、`preprocess.notch`、`preprocess.car`、`preprocess.crop`、`preprocess.resample`、`preprocess.baseline`、`preprocess.channel_select`、`preprocess.mean_center` 的逐折输出进入 4.4 的 Parquet 缓存；`preprocess.filter_bank_split`、全部特征节点与解码器不入缓存。滤波器组下游若接可缓存的预处理节点，缓存键会带上所选的 `source_port`，不同子带不会互相命中。

### 6.3 模板市场与论文复现

“丹方设计”页右上角的“模板市场”提供两类可移植内容：

- **模板市场**：内置 CSP+LDA、FBCSP+LDA、Riemann MDM、切空间+LR、xDAWN+LDA、SSVEP CCA、EEGNet 等可移植 DAG 清单。每项显示节点数、搜索空间维度、来源论文与复现注意事项；点击“载入画布”只预览，点击“安装为自定义丹方”才会写入实验库并分配 `custom:<id>`。
- **论文复现目录**：按论文列出协议、适用数据集与明确的 caveat（例如未复刻原始伪迹剔除、未实现论文的多类概率校准）。载入画布后仍需选择自己的数据与协议复跑，目录不声称逐位复现论文数值。

模板是普通 DAG JSON，不需要特殊运行时。保存后的自定义丹方可以“导出模板”，生成 `.neurokiln.json`，包含 `format`、`version`、`spec`、`search_space`、`template` 元信息与 `sha256:` 校验和；“导入模板”会校验格式、版本与校验和后再保存为新丹方。

```json
{
  "format": "neurokiln.pipeline-template",
  "version": 1,
  "name": "CSP + LDA",
  "spec": { "nodes": [], "edges": [] },
  "search_space": [],
  "checksum": "sha256:..."
}
```

模板只包含图结构与参数，不包含数据、实验结果或账号信息；导入模板不会自动运行实验。

### 6.4 插件市场

“插件市场”页允许在不改核心代码的前提下扩展丹方与 DAG 节点。插件分两种运行时：

| 运行时 | 内容 | 受限模式 | 可信模式 |
|---|---|---|---|
| `graph` | 纯声明式 DAG 清单（JSON），不执行第三方代码 | 可安装/启用 | 可安装/启用 |
| `python` | `plugin.py` 注册自定义节点，进程内执行任意 Python | 拒绝加载 | 可安装/启用 |

工作流：

1. 在“插件市场”安装内置目录中的插件，或上传 ZIP（ZIP 根目录需含 `plugin.json` 与声明文件）；
2. 安装时校验清单与每个文件的 SHA-256、限制文件数量/大小/后缀、拒绝绝对路径与 `..`（防 zip-slip）；
3. 启用后，插件丹方以 `plugin:<plugin_id>/<pipeline_id>` 出现在斗法台，插件节点以 `plugin.<plugin_id>.*` 出现在 DAG 编辑器；
4. 禁用/卸载只影响后续运行；已保存实验仍保留插件来源、版本与包哈希，用于复现校验。

安全事实（请如实理解）：

- 受限模式（默认）**只加载声明式图插件**，不会执行任何第三方 Python；
- 可信模式需显式设置 `NEUROKILN_PLUGIN_TRUST=1` 后重启 API。Python 插件在 API 进程内运行，**没有沙箱、没有权限隔离**，可以读写本机文件与网络；只应安装你已审阅过的包；
- 安装/启用/禁用/卸载仅管理员可操作；每次启用都会重新校验清单与文件哈希；
- 插件清单 v2 要求 `schema_version`、`engine_min_version` 与 `api_version`，UI 会显示兼容性徽标并阻止不兼容安装；v1 清单以兼容模式运行；
- 复现实验时若插件缺失、版本不同或包哈希不一致，系统会拒绝运行，而不是静默换用其它实现。

---

## 7. 斗法台（Arena）

### 7.1 一次标准对比

1. 选择数据集，勾选被试（留空表示全部）；
2. 选择协议、折数、重复与种子；
3. 勾选两条或更多丹方（内置、自定义与插件可混选；系统只显示与数据集范式兼容的丹方）；
4. 按需关闭“使用预处理缓存”，提交后台任务，等待进度完成；
5. 查看主表：BA 均值 ± SD、95% CI、Kappa、ECE（悬停可看 Brier/NLL）、折数；
6. 查看统计检验卡：平均名次、Friedman、Nemenyi CD、配对 Wilcoxon-Holm；
7. 查看“概率校准”卡：可靠性曲线与各丹方 ECE/MCE/Brier/NLL；
8. 需要时点击“生成报告”进入报告中心。

后台任务默认 1 个 worker。任务状态为 `queued → running → completed / failed / cancelled`，可取消；训练循环与逐折边界都会检查取消令牌。页面刷新后可从“任务记录”恢复已完成结果。

### 7.2 比较的边界

- 不同数据集、协议、折数、重复或种子集合的实验**不会**出现在同一张排行榜；
- 只有同一实验内的丹方才做配对检验；
- HPO 搜索结果必须先“提升为丹方”，再回到 Arena 与固定基线同协议复核，不能直接拿搜索阶段的代理分数上排行榜。

### 7.3 数据选择与签名

每次 Arena 都会记录：数据集 ID、数据集版本、选择范围（被试/类别/选项）、加载后的数据签名、协议参数、种子、引擎版本。实验库和报告都以此为准；复现时先校验签名，再逐折比对 BA。

---

## 8. 智能调参（HPO）

### 8.1 搜索空间模板

| 模板 ID | 家族 | 搜索内容 |
|---|---|---|
| `csp_lda` | classical | 频带、CSP 分量、正则、LDA 收缩、通道保留比例 |
| `fbcsp_lda` | classical | 滤波器组分量、正则、收缩、通道保留比例 |
| `riemann_mdm` | riemann | 频带、度量、协方差估计、通道保留比例 |
| `ts_lr` | riemann | 频带、度量、协方差、LR 正则、通道保留比例 |
| `eegnet` | deep | 学习率、dropout、批大小、训练轮数、通道比例 |
| `shallowconvnet` | deep | 同上；CPU 上较慢 |

自定义丹方配置搜索空间后，也会以 `custom:<id>` 出现在模板下拉框中（带“自定义”标签）。

### 8.2 自定义丹方模板

“丹方设计”保存后，点击页头的“调参空间”，即可为该丹方声明 HPO 搜索维度：

1. 对话框中列出当前丹方所有内置节点的可调参数；打开某行的“启用”开关，填写搜索上下界与步长（选项型参数改为勾选候选值）；
2. 搜索维度只能来自 `int` / `float` / `select` 参数，最多 16 维；参数当前值必须落在搜索范围内；
3. 插件节点暂不支持搜索；`preprocess.filter_bank_split.n_bands` 等会改变输出端口的参数不可搜索；
4. “保存”只保存搜索空间；“保存并去调参”会跳转到智能调参并预选该丹方的模板；
5. “清空”删除搜索空间，丹方本身不受影响。

智能调参页的模板下拉框会为有搜索空间的自定义丹方追加 `custom:<id>` 条目。选中后可查看搜索空间表与配置预览，按常规流程提交任务。提交任务时会解析并固定当时的丹方与搜索空间快照；任务结果中的 `template.source` 记录 `pipeline_id`、`spec_hash`、`space_hash` 与维度数，便于审计“这次搜索到底搜了什么”。之后在“丹方设计”中修改丹方或搜索空间，需要刷新智能调参页才会看到新条目；已提交的任务不受影响。

提升流程不变：搜索结果仍需“提升为自定义丹方”，再回 Arena 与固定基线同协议复核。

### 8.3 目标与预算

可选目标：

- `balanced_accuracy`（必选，最大化）；
- `cohen_kappa`（最大化）；
- `duration_ms`（单折耗时，最小化）；
- `channel_count`（使用通道数，最小化）。

预算参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `budget` | 代理评估总预算（含初始种群） | 16 |
| `population` | 灰狼种群规模 | 6 |
| `proxy_folds` | 代理阶段使用的折数 | 2 |
| `full_candidates` | 进入完整复评的非支配候选数 | 3 |
| `seed` | 搜索随机种子 | 2026 |

`budget` 是上限；当代理候选耗尽或提前收敛时，实际评估次数可能更少。完整复评使用协议的全部折，不参与代理预算计数。

### 8.4 搜索流程

1. 在归一化参数立方体中初始化灰狼种群；
2. 每代用 α/β/δ 引导更新位置，并叠加精英局部扰动；
3. 所有候选在代理折上评估，记录参数、向量、折签名、耗时、失败原因；
4. 代理预算耗尽后，按平均名次选择非支配候选，在完整协议上复评；
5. 在完整结果上重新计算 Pareto 前沿；
6. “最佳”按 BA → Kappa → 更少通道 → 更短耗时依次排序；
7. 完整审计历史写入任务结果，可导出、可回看。

搜索过程支持取消；取消后已完成的评估仍保留在任务结果中，但不会标记为可提升的最终结果。

### 8.5 提升与复核

在结果页点击“提升为自定义丹方”，系统把最佳候选的 DAG 保存为 `custom:<id>`。随后可一键跳到 Arena，与固定基线在同一协议下复核。提升得到的丹方不会继承原搜索空间；如需继续调参，可在“丹方设计”页重新配置。

**HPO 的代理分数只是搜索信号，不是最终结论。** 最终结论以 Arena 的完整协议结果为准。

---

## 9. 报告中心

每次保存的 Arena 实验都可以生成：

- **应用内 HTML 预览**：实验元信息、丹方均值、逐折 BA 折线、概率校准（可靠性曲线 + ECE/MCE/Brier/NLL）、最佳丹方混淆矩阵、逐折明细、统计检验；
- **HTML 单文件**：自包含样式，可直接归档或发给合作者；
- **Excel 工作簿**：概览、算法指标、逐折结果、校准指标、统计检验、逐试次预测等工作表；
- **Word / LaTeX**：论文级排版，包含实验元信息、算法排名、统计检验、混淆矩阵与参考文献；LaTeX 使用 `ctexart`，可直接用 XeLaTeX 编译；
- **实验 JSON**：完整结构化数据；
- **分享卡片**：生成 1200×630 PNG，包含最佳丹方指标与二维码；二维码默认指向只读 `/demo/index.html`，可用 `VITE_DEMO_URL` 覆盖；
- **只读演示**：`web/public/demo/` 下的静态页面由 API 根路径托管，无需登录即可查看演示结果；它只展示预生成 HTML，不暴露实验库或令牌。

参考文献以 GB/T 7714 风格列出，并自动插入当前数据集声明的 citation（若有）。

Excel 与 HTML 都由实验库中的逐试次预测聚合生成，不依赖运行中的浏览器状态。校准指标随每折结果一起入库，因而在实验完成后仍可离线重算/核验。

---

## 10. 可复现包

### 10.1 导出

Web：报告中心 → 选择实验 → 导出可复现包。

CLI：

```bat
neurokiln export --experiment 12 --out D:\repro\exp12
```

输出目录必须为空或不存在，避免覆盖已有结果。

### 10.2 包内容

| 文件/目录 | 内容 |
|---|---|
| `experiment.json` | 完整配置、丹方定义、基准指标 |
| `protocol.json` | 协议名、折数、重复、种子、划分签名 |
| `data_manifest.json` | 数据集 ID、版本、选择范围、数据签名、许可 |
| `environment.json` | Python、关键依赖版本 |
| `pipelines/*.json` | 每条丹方的完整定义（自定义 DAG 含节点与边） |
| `metrics.csv` / `predictions.parquet` | 逐折指标（含 ECE/Brier/NLL）、逐试次预测与类别概率（缺少 pyarrow 时回退 CSV） |
| `report.html` / `report.xlsx` | 报告副本（可用 `--no-report` 跳过） |
| `reproduce.bat` / `reproduce.sh` | 一键复现脚本 |
| `README.md` | 包说明与复现命令 |

使用插件丹方的实验会在 `pipelines/*.json` 中记录插件 ID、版本与包 SHA-256。复现时系统按这些字段校验本地插件；插件未安装、未启用、版本或哈希不一致都会拒绝执行。

### 10.3 复现与容差

```bat
neurokiln reproduce D:\repro\exp12
neurokiln reproduce 12 --tolerance 0.01 --save
```

流程：

1. 读取包配置与数据清单；
2. 重新加载同一数据集与选择范围，计算本地签名；
3. 签名不一致时默认拒绝执行；确认数据只是路径变化且内容一致时，可加 `--allow-data-mismatch`；
4. 用包内协议、种子与丹方重新运行；
5. 逐丹方比较 BA 与基准值的差，默认容差 `0.02`；
6. 输出“通过/不一致”表，返回码 0 表示通过、1 表示不一致。

深度模型在不同 CPU/线程/数学库下可能存在末位差异；若需要严格比特级复现，应固定硬件、线程数与依赖版本，并在报告中外显说明。

---

## 11. CLI 参考

安装后可用 `neurokiln`，也可用 `python -m neurokiln.cli`（需在 `engine` 目录或已把它加入 `PYTHONPATH`）。

### 11.1 全局参数

| 参数 | 作用 |
|---|---|
| `--data-dir` | 数据根目录；默认依次尝试 `./data`、仓库 `data` |
| `--db` | 实验库路径；默认 `<data-dir>/neurokiln.db` |
| `--mne-data` | MOABB/MNE 缓存目录 |

### 11.2 `list-datasets`

```bat
neurokiln list-datasets
neurokiln list-datasets --dataset bciciv2a
neurokiln list-datasets --json
```

### 11.3 `run`（单丹方）

```bat
neurokiln run --dataset bciciv2a --subjects S01 --classes 4 ^
  --protocol within_subject --folds 5 --repeats 1 --seeds 0 ^
  --pipeline csp_lda --name "2a-S01-CSP"
```

### 11.4 `arena`（多丹方）

```bat
neurokiln arena --dataset bciciv2a --subjects S01,S02,S03 ^
  --protocol loso --folds 5 --seeds 0,1 ^
  --pipeline csp_lda --pipeline riemann_mdm --pipeline custom:1 ^
  --name "2a-LOSO-三丹方" --export-dir D:\repro\arena-2a
```

说明：

- `--pipeline` 可重复，也接受逗号分隔；自定义丹方写 `custom:<id>`，插件丹方写 `plugin:<plugin_id>/<pipeline_id>`；
- `--subjects` 可重复或逗号分隔；留空表示全部；
- `--no-save` 不写实验库；此时不能使用 `--export-dir`；
- `--no-cache` 跳过预处理产物缓存，用于排查缓存或测量冷启动成本；
- `--export-dir` 会在保存后自动导出可复现包；
- `--json` 输出机器可读结果（不含逐试次数组）。

### 11.5 `report`

```bat
neurokiln report --experiment 12 --out D:\reports\exp12.html ^
  --xlsx D:\reports\exp12.xlsx --bundle D:\repro\exp12
```

### 11.6 `export`

```bat
neurokiln export --experiment 12 --out D:\repro\exp12
neurokiln export --experiment 12 --out D:\repro\exp12 --no-report
```

### 11.7 `reproduce`

```bat
neurokiln reproduce 12
neurokiln reproduce D:\repro\exp12 --tolerance 0.01
neurokiln reproduce D:\repro\exp12 --save
neurokiln reproduce D:\repro\exp12 --allow-data-mismatch
```

`reproduce` 接受实验 ID 或 bundle 目录；返回码可用于 CI 断言。

### 11.8 常用组合

```bat
:: 只要结果不落库
neurokiln run --dataset synthetic --pipeline csp_lda --no-save

:: 保存并同时导出复现包
neurokiln arena --dataset bciciv2b --pipeline csp_lda --pipeline ts_lr ^
  --folds 5 --seeds 0 --export-dir D:\repro\2b

:: 生成 HTML + Excel + 复现包
neurokiln report --experiment 3 --out report-3.html --xlsx report-3.xlsx ^
  --bundle D:\repro\exp3
```

### 11.9 `cache`

```bat
neurokiln cache
neurokiln cache --json
neurokiln cache --clear
```

`cache` 读取 `<data-dir>/cache/preprocess/` 的条目数、字节数、上限、命中/未命中/写入/淘汰计数与命中率；`--clear` 清空缓存并重置计数，仅影响派生文件，不影响实验库和已导入数据。

### 11.10 插件丹方

```bat
neurokiln arena --dataset synthetic --pipeline raw_centroid ^
  --pipeline plugin:mu_beta_erd/mu_variance_lda --no-save
```

插件必须先通过 Web“插件市场”或 API 安装并启用。CLI 从 `--data-dir` 下的 `plugins/` 读取插件；Python 插件同样受 `NEUROKILN_PLUGIN_TRUST` 控制。

### 11.11 插件校验

```bat
neurokiln plugin validate path\to\plugin
neurokiln plugin validate path\to\plugin.zip --strict --json
```

`plugin validate` 不安装、不执行插件代码，只读取目录或 ZIP 中的 `plugin.json` 并检查：

- 清单结构、`schema_version`、`engine_min_version` 与 `api_version` 是否与当前引擎兼容；
- `files` 中声明的文件是否齐全、是否有多余文件、SHA-256 是否一致；未声明 `files` 时给出“安装时会重新生成”的警告；
- Python 入口文件是否存在，并提示进程内执行风险；
- 每个图插件的节点、边、参数与端口是否通过静态图校验；引用 `plugin.*` 节点的图会标记为延迟校验。

清单 v2 的声明式图插件应包含 `schema_version: 2`、`engine_min_version` 与 `api_version`；旧的 v1 清单仍可安装，但兼容性检查会提示升级。`--strict` 会把警告视为失败，适合放进 CI。

---

## 12. 在线与闭环扩展

引擎提供轻量的在线 / 闭环扩展点，用于把离线验证过的解码器接入逐试次决策流程。它**不包含实时设备驱动、LSL / BrainFlow 集成、时钟同步或延迟保证**，也不构成在线或临床性能承诺。

核心协议在 `neurokiln.online`：

| 协议 | 作用 |
|---|---|
| `StreamSource` | 按采集顺序逐试次提供 `StreamTrial` |
| `OnlineDecoder` | `predict(trial)` 只使用当前试次 |
| `ClosedLoopPolicy` | 把 `Prediction` 映射为动作，返回 `None` 表示放弃 |

最简单的验证方式是用留出折回放：

```bat
neurokiln online-replay --dataset synthetic --pipeline csp_lda ^
  --protocol within_subject --folds 4 --fold 0 --threshold 0.6 --json
```

它复用该折已经算出的留出预测，按 `--hop-seconds` 生成事件时间戳，并统计 `accuracy`、`n_actions`、`n_abstained` 与 `action_rate`。`ThresholdPolicy` 在置信度低于阈值时放弃动作；最近质心类基线概率接近 0.5，默认阈值 0.6 会让它们全部放弃，这是正常现象，应先看校准结果再定阈值。

接入实时数据时，实现 `StreamSource` 与 `OnlineDecoder` 后调用 `run_closed_loop` 即可，不需要修改引擎；但采集滤波、重参考、分段窗口必须与离线训练一致，并自行实现急停与人工接管。完整示例与边界见[在线与闭环扩展](/online-closed-loop)。

---

## 13. 数据目录、实验库与备份

默认数据目录结构：

```
neurokiln/data/
  neurokiln.db          # 实验、run、丹方、账号、job
  .auth_secret        # 令牌签名密钥
  imports/            # 导入中的临时 NPZ
  local/              # 已导入的 NPZ 与侧车 JSON
  cache/preprocess/   # 预处理产物 Parquet 缓存（可随时清空）
  plugins/            # 已安装插件与 index.json
  api.log             # 本地启动时的日志（若按手册重定向）
```

SQLite 主要表：

- `experiments`：一次 Arena 实验的元信息、协议、种子、汇总；
- `runs`：逐折指标、划分签名、逐试次预测、类别概率与校准指标；
- `datasets`：数据清单与签名；
- `custom_pipelines`：自定义 DAG；
- `users`：账号；
- `jobs`：后台任务状态与结果。

备份：停止 API 后复制整个 `data/` 目录即可（WAL 模式下请确保进程已退出）。恢复时把目录放回原位再启动。

删除实验会级联删除其 runs；该操作不可撤销，请先导出复现包。

---

## 14. 常见问题

**Q：`/api/hpo/templates` 返回 404 / 页面提示接口不存在？**
A：API 进程仍在运行旧代码。停止进程后重启 uvicorn；前端热更新不会重启 Python 服务。

**Q：MOABB 下载很慢或失败？**
A：设置 `NEUROKILN_MNE_DATA` 到固定磁盘，并保证网络可访问 MOABB 数据源；首次下载完成后离线也可用。下载失败不会污染实验库，重试即可。

**Q：为什么单被试结果的 CI 是折级？**
A：置信区间需要独立单位。单被试只有折，没有“人群样本”；系统会显式标注 `unit=fold`，不要把该区间解释为跨被试置信度。

**Q：为什么 HPO 的评估次数小于 budget？**
A：budget 是上限；代理候选耗尽或搜索提前收敛时不会强行凑数。完整复评次数单独统计。

**Q：自定义丹方在 Arena 里找不到？**
A：确认已保存成功（列表出现 `custom:<id>`），并刷新斗法台；HPO 结果必须先“提升为自定义丹方”。

**Q：自定义丹方为什么没有出现在智能调参的模板下拉框？**
A：只有配置过搜索空间的自定义丹方才追加 `custom:<id>` 模板。先在丹方设计页保存丹方，点击“调参空间”，启用至少一个可调参数并保存；插件节点、会改变端口的 `filter_bank_split.n_bands` 等参数不能作为搜索维度。若条目显示“不可用”，说明保存维度已不适用于当前丹方（例如参数当前值不在范围内或节点被删除），重新打开“调参空间”修正即可。

**Q：复现包提示数据签名不一致？**
A：说明本地加载的数据内容/选择范围与导出时不同。先核对 `data_manifest.json` 中的数据集、被试、类别与选项；确认只是缓存路径变化后再用 `--allow-data-mismatch`。

**Q：深度模型复现差一点？**
A：CPU 线程数、数学库和硬件差异会导致浮点末位变化。固定环境与线程数，或在报告中说明容差；不要用更宽的容差掩盖协议或数据变化。

**Q：为什么 EEGNet / ShallowConvNet / Riemann / xDAWN / SSVEP 不能接在 Top-K 或其他特征节点后面？**
A：这些解码器直接吃预处理后的 3D 试次（试次 × 通道 × 采样点），而 Top-K、CSP 等特征节点输出的是 2D 特征矩阵。旧版本会在训练时抛 `AxisError: axis 2 is out of bounds`，现在连线时就会提示，后端校验也会给出明确原因。修法二选一：把预处理输出直接接 EEGNet 等深度/黎曼解码器；或保留手工特征，把末端换成 `decoder.lda` 或 `decoder.nearest_centroid`。

**Q：换了 SSVEP/P300 数据集后，CSP、EEGNet 等丹方不见了？**
A：丹方现在声明适用范式，斗法台只显示与数据集范式兼容的条目。P300 数据集请选 `xdawn_lda` 或三种最近质心基线；SSVEP 数据集请选 `ssvep_cca`、`ssvep_fbcca` 或质心基线；MI 数据集使用 CSP/FBCSP/Riemann/深度模型。

**Q：插件市场显示“受限模式”，Python 插件无法启用？**
A：默认受限模式只允许声明式图插件。确认已审阅插件代码后，以 `NEUROKILN_PLUGIN_TRUST=1` 重启 API。该模式没有沙箱，Python 插件可执行任意本机代码。

**Q：缓存命中率很低，或者担心缓存影响结果？**
A：缓存键包含数据签名、折签名、种子和节点参数，跨折/跨数据版本不会命中。首次运行、修改带通/通道参数或清空缓存后出现未命中是正常的。缓存只保存派生数组，关闭缓存（`--no-cache` / `NEUROKILN_CACHE=0`）重跑应得到相同指标。

**Q：报告里 ECE 显示不可用？**
A：说明该丹方没有返回合法的 `predict_proba`（常见于新接入的插件节点）。内置 12 条丹方均提供概率输出；修复插件后重新运行实验即可生成校准指标。

**Q：CI 文件已就位，为什么没有自动跑？**
A：工作区当前不是 Git 仓库。初始化仓库并推送到 GitHub 后，`.github/workflows/ci.yml` 才会触发；也可以在本机运行 `scripts/ci.sh`（Git Bash）或 `scripts/ci.ps1`（PowerShell）。

**Q：报告下载需要登录吗？**
A：需要。报告接口与其他 `/api/*` 一样走 Bearer Token，不使用公开静态链接。

---

## 15. 测试、CI 与金标准回归

引擎测试：

```bat
cd neurokiln\engine
..\.venv312\Scripts\python.exe -m pytest -q
```

截至本文档：`124 passed, 2 skipped`。跳过项包含需要真实数据开关的金标准回归。测试覆盖协议无重叠、训练折 fit 隔离、拆分安全缓存、概率校准、插件清单 v2/哈希/路径校验、多解码器融合、命名端口隔离、节点输入数据类型校验、自定义 HPO 模板搜索空间校验、SQLite 往返与 bundle 复现等。

一键本地 CI：

```bat
:: Git Bash
bash scripts/ci.sh

:: PowerShell
powershell -ExecutionPolicy Bypass -File scripts/ci.ps1
```

脚本依次执行引擎测试、API `TestClient` 冒烟（健康检查 + 未认证 401）、Web 生产构建；Node 依赖缺失时会自动 `npm ci`。GitHub Actions 工作流在 `.github/workflows/ci.yml`，包含 Python 3.10/3.12 矩阵、API 冒烟与 Web 构建。**当前工作区不是 Git 仓库，工作流只有推送到 GitHub 后才会触发。**

真实 2a 金标准回归（需要本地 MOABB 缓存）：

```bat
set NEUROKILN_GOLDEN=1
set NEUROKILN_MNE_DATA=D:\data\mne
..\.venv312\Scripts\python.exe -m pytest tests\test_golden_2a.py -q
```

回归覆盖 CSP+LDA、FBCSP+LDA、Riemann MDM 三条管线在 2a 上的指标区间；用于发现协议、预处理或划分逻辑的回归，而不是追求单次最高分。

---

## 16. 术语与免责声明

| 术语 | 含义 |
|---|---|
| 丹方 | 一条完整管线：预处理 + 特征 + 解码器 |
| 斗法台 | 同一协议下的多丹方对比 |
| 代理折 | HPO 中用于低成本筛选的部分折 |
| 完整复评 | 用协议全部折复跑非支配候选 |
| Pareto 前沿 | 在多个目标上互不支配的候选集合 |
| 数据签名 | 由实际试次内容与元信息计算的稳定哈希 |
| 划分签名 | 单折训练/测试索引的稳定哈希 |
| 范式元数据 | 数据集声明的任务信息（P300 阳性标签/窗口、SSVEP 刺激频率等） |
| ECE | 期望校准误差，置信度与实际正确率之差的样本加权平均 |
| 可靠性曲线 | 按置信度分箱统计平均置信度、实际正确率与试次数 |
| 图插件 | 只含声明式 DAG 清单、不执行第三方代码的插件 |
| 可信模式 | 允许 Python 插件在 API 进程内执行的启动开关，无沙箱 |

**免责声明**：本工具输出的离线指标只反映特定数据、协议和随机种子下的统计表现，不构成在线 BCI 性能承诺，也不构成任何临床诊断或治疗建议。使用公开数据集时请遵守其许可与引用要求；使用私有数据时请自行完成伦理审批、脱敏与访问控制。
