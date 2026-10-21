# Gomoku AI Lab

本地单机五子棋自博弈研究平台。PyTorch CNN / ViT + AlphaZero 风格 PUCT；FastAPI + React / TypeScript + SQLite。默认 9×9、五连胜、无禁手，支持 5–19 路配置和 15×15 网络训练。

已实现真实训练、棋谱采集、SFT 冷启动、完整 Checkpoint、真人对弈、自动存盘、时间轴复盘、分支续弈、逐合法落点 WDL 分析、双模型比较、训练图表及固定基准评估。所有界面指标来自后端，未填充模拟棋力数据。

## 快速启动（Windows）

需要 Python 3.11+、Node.js 20.19+（或 22.12+）。本机已建立 `.venv` 并安装依赖、构建前端。

```powershell
# 首次安装；本机已完成。GPU 版约需下载 2.6GB PyTorch。
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
# 无 NVIDIA GPU 时添加 -CPU

# 启动后保持窗口运行；Ctrl+C 关闭 Web
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

访问 <http://127.0.0.1:8000>。API 文档：<http://127.0.0.1:8000/docs>。默认只监听本机，没有账号系统，不适合直接公开到互联网。

也可直接启动：

```powershell
.venv/Scripts/python -m gomoku.cli serve
```

开发模式：后端保持运行，另开终端执行 `cd frontend; npm run dev`，访问其输出的本地地址。Vite 会代理 `/api`。生产模式由 FastAPI 直接服务 `frontend/dist`，只需一个端口。

Linux / WSL2 可运行 `bash scripts/bootstrap.sh` 后执行 `.venv/bin/python -m gomoku.cli serve`；本次实际验收在 Windows 上完成。

## 收集自己的棋谱并进行 SFT 冷启动

1. 打开「棋谱采集」，选择 9×9 或 13×13（保留其他尺寸兼容），在棋盘上自己交替执黑白落子。可撤销一手；草稿保存在当前浏览器，尚未进入训练集。
2. 黑胜、白胜或平局后，选择「收录这局棋谱」，或「不收录，开始下一局」。只有明确收录的完整棋谱存入 SQLite 的独立采集集合；不自动混入真人对 AI 历史。可复看时间轴及导出原始数据；相同序列重复收录不会重复计数。
3. 打开「SFT 训练」，先选「棋谱分类」（9×9 / 13×13），页面只向后端请求并显示该尺寸棋谱；分类同步设置随机模型的棋盘大小。从采集页进入时自动沿用其尺寸。再选择 CNN / ViT 结构，或任意兼容的已有 Checkpoint。已有权重自动锁定其实际结构及棋谱分类，只加载对应尺寸数据；新 SFT 任务仅继承网络权重。不同尺寸棋谱不可混用。
4. 设置轮数、Batch Size、学习率、验证比例、WDL 损失权重、旋转镜像增强、设备与运行时限，启动训练。页面显示真实训练/验证损失、落子准确率、更新次数和 Checkpoint，可暂停、停止、完整恢复。
5. 生成的 SFT Checkpoint 可直接用于对弈、局面分析和评估。在 RL 的模型 A / B「初始权重来源」中选择它，即可继续自博弈；完整恢复 SFT 使用 SFT 页专门入口。

SFT **同时模仿黑白双方的全部落子**，不丢弃输方的步骤，也不对输方策略标签取负。每个非终局局面以当前行棋者视角编码，用实际下一手作为策略交叉熵标签，并对非法动作掩码；最终胜负转为该行棋者视角的 Win / Draw / Loss 标签。总损失为 `Policy CE + value_weight × WDL CE`。把 WDL 权重设为 0 可只训练策略损失；共享骨干仍会变化。

按**整局**划分训练/验证集，再在训练批次做 D4 增强，避免同一棋局的不同步骤或增强副本跨集。至少两局才能划出独立验证集；一局时仅训练并明确显示没有验证结果。小数据上的落子准确率不等于棋力，输方落子中也可能包含错误；这是一种行为模仿冷启动，后续仍需自博弈和固定基准评估。比较架构时应记录是否使用人类数据，并保持来源、划分、随机种子和预算一致。

命令行：复制 `configs/sft.example.json`，把 `dataset_ids` 改为已收录棋谱的 ID。选择已有权重时填写 `initial_checkpoint`（可省略 `model`）。

```powershell
.venv/Scripts/python -m gomoku.cli sft --config configs/sft.example.json
# 完整恢复：配置学习参数保持一致，epochs 为更大的累计轮数
.venv/Scripts/python -m gomoku.cli sft --config configs/sft.example.json --resume CHECKPOINT_ID
# 独立临时目录内运行真实 CNN / ViT SFT → 恢复 → 推理 → RL 验收
.venv/Scripts/python scripts/sft_smoke.py --device cuda
```

SFT 完整权重包含所选棋谱快照、训练/验证划分、批次打乱顺序和游标、优化器、调度器及 RNG。暂停在当前批次及正在进行的轮末验证结束后生效；恢复不依赖最初加载的源权重或后来变化的数据集合。单任务最多 5,000 局 / 200,000 手；棋盘浮点输入只按批次生成。SFT 与 RL 共用同一后台训练进程限制。磁盘预算达到后保存最新完整状态并暂停，最多超出一个完整权重文件大小，不自动删除已有权重。

## 开始 CNN / ViT 训练

在「训练实验室」选择架构、预算和设备，点击启动。每次训练会新建独立实验；同一后端只允许一个训练进程。暂停、停止会在当前并行采样批次及其更新结束后保存完整状态；长搜索预算可能等待较久。

或使用命令行：

```powershell
.venv/Scripts/python -m gomoku.cli train --config configs/cnn.json
.venv/Scripts/python -m gomoku.cli train --config configs/vit.json

# 轻量真实训练：各 4 局、每步 8 次搜索、8 次更新
.venv/Scripts/python -m gomoku.cli train --config configs/smoke-cnn.json
.venv/Scripts/python -m gomoku.cli train --config configs/smoke-vit.json
```

默认配置每架构 200 局、32 次搜索、4 个并行棋局、2 小时时间上限、4GB 数据目录预算。到达预算后安全保存并结束，不会无限训练。CLI 与 Web 训练不要同时启动；Web 管理器只管理自己创建的进程。

默认从随机初始化开始，也可以显式选择已有 Checkpoint 的权重作为新任务起点。默认训练模式 `independent`：CNN、ViT 分别自博弈。`history` 选择本实验及恢复谱系中的历史权重；`mixed` 需要显式指定固定对手 Checkpoint ID，双方棋盘尺寸必须一致。固定/历史对手模式只收集当前学习者自己的决策样本，不能与独立训练结果混为同一实验。

## Checkpoint 与恢复

### 删除 Checkpoint

在「训练实验室 → Checkpoint 存储库」点击某一行的「删除」。确认框会显示权重文件大小及关联棋谱、评估条数；点击「确认删除」后永久删除该权重文件并从可选列表移除，无法恢复。棋谱、分支和评估记录保留，相关棋局需要改选其他兼容权重才能继续或分析。

训练与评估进行期间禁用删除，后端也会拒绝请求。若删除了历史对手池的成员，后续恢复会跳过该成员，并在指标 `missing_history_opponents` 中记录数量；这种恢复不再保证与原对手池位级一致。固定对手训练所需的权重删除后，需要选择仍存在的权重新建任务。

HTTP API：`GET /api/checkpoints/{id}/deletion-preview` 查看影响，`DELETE /api/checkpoints/{id}` 删除。仅操作本数据目录内的已登记 `.pt` 文件，不接受任意文件路径。

### 两套任意配置的模型共同训练

创建任务时，将「训练方式」设为「双模型互弈（A、B 双方更新）」。分别配置模型 A 和 B：每一边均可选择 CNN 或 ViT；CNN 可调整通道数、残差块数，ViT 可调整 Token 维度、层数、Attention heads 和 MLP 倍率。B 的棋盘大小跟随 A。

支持不同配置的 **CNN × CNN**、CNN × ViT、ViT × CNN、ViT × ViT。两侧分别选择随机初始化或已有 Checkpoint，交替执黑，各自使用独立 Replay Buffer、优化器和调度器；采样预算、学习率等任务参数共享。此模式属于互相影响的联合训练，正式从零开始的架构对比仍应使用「单模型独立自博弈」并保持随机初始化。

训练曲线分别显示 A/B 的 Policy、Value loss。每次保存生成标记为模型 A / B 的两个 Checkpoint，任一权重可用于对弈和分析。**从任一侧选择恢复，会恢复双方网络、优化器、回放池及共同随机状态**；每个文件都包含完整双模型状态，因此比单模型文件大，删除配对的另一份文件不影响它的恢复能力。

命令行示例（轻量、两种不同 CNN 配置）：

```powershell
.venv/Scripts/python -m gomoku.cli train --config configs/paired-cnn.json
.venv/Scripts/python -m gomoku.cli train --config configs/paired-cnn.json --resume CHECKPOINT_ID --games 8
```

配置字段 `model` 为 A，`opponent_model` 为 B，`mode` 为 `paired`。任一架构和规模均可独立修改，但两侧棋盘大小必须相同。

### A / B 分别选择初始模型或任意已有权重

每个模型编辑器都有独立的「初始权重来源」下拉框：

- **随机初始化**：自由选择 CNN / ViT 及下方结构，使用任务随机种子生成新权重。
- **已有 Checkpoint**：选择任意已登记且仍可用的权重；自动读取并锁定其真实网络结构、架构和棋盘大小。来自旧双模型任务的 A 或 B 权重均可放入新任务任一侧。

可组合新模型 × 新模型、新模型 × Checkpoint、Checkpoint × 新模型、Checkpoint × Checkpoint。两侧可以选同一个 Checkpoint，此时复制为两个互不共享参数的模型。两个已有权重的棋盘和规则必须兼容；一侧为新模型时，页面会将新模型棋盘大小与所选权重对齐。单模型训练也可通过同一入口使用已有权重。

**新建任务加载初始权重**只继承网络参数，优化器、调度器、经验池、随机状态和局数/更新计数重新开始，可自由设置新任务超参数。源文件不修改。保存的权重和实验配置记录双方来源 ID、来源训练进度及实际结构，便于追踪预训练成本。

**从 Checkpoint 完整恢复训练**是高级设置中的独立操作，恢复任务原有全部训练状态，累计总局数继续增加。新任务一旦保存，完整恢复使用其自身快照；即使原始初始化来源文件被删除，也不影响恢复。

CLI 字段 `initial_checkpoint` 为 A 的初始权重 ID，`opponent_initial_checkpoint` 为 B 的初始权重 ID；省略或 `null` 表示随机初始化。选择权重的槽位可以省略结构字段，由后端读取其真实结构。示例见 `configs/paired-from-checkpoints.example.json`，替换其中两个 ID 后运行：

```powershell
.venv/Scripts/python -m gomoku.cli train --config configs/paired-from-checkpoints.example.json
```

页面会显示所有 Checkpoint 的 ID、架构、棋盘大小、训练局数、更新步数、参数量。训练任务高级选项中选择恢复权重，设置更大的**累计总局数**，然后启动。

```powershell
.venv/Scripts/python -m gomoku.cli train --config configs/cnn.json --resume CHECKPOINT_ID --games 400
```

恢复时使用与原始训练一致的配置文件。允许修改累计目标局数、设备、时间/磁盘预算、保存/评估间隔；改变学习率、网络或回放等学习配置会拒绝恢复。原实验记录保留，新实验记录继续累计进度。

每个 `.pt` 包含模型结构、规则版本、实验 ID、模型参数、优化器、学习率调度器、Replay Buffer、累计进度、Python / NumPy / PyTorch CPU / CUDA 随机状态及对手池谱系。原子重命名后才登记到数据库。文件包含 Python 序列化对象，只加载可信的本机产物；HTTP API 不接受任意模型文件上传或路径。

CPU 小配置恢复已验证到参数逐项完全一致。已启用 PyTorch 确定性算法及 cuDNN 确定性设置；跨 GPU、驱动、PyTorch 版本或 CPU/GPU 迁移不能保证位级相同。

## 对弈、复盘与分支

1. 「真人对弈」选择任意已保存权重、执棋颜色与搜索预算。`0` 为纯 Policy，其他值为无训练噪声的确定性 MCTS。
2. 点击开始，双方交替落子，AI 请求异步执行。每一步立即存入 SQLite；失败时可重试 AI，修改操作有版本冲突保护。
3. 「棋局复盘」选择棋局，使用滑块、首尾/前后按钮或落子记录跳到任意一步。
4. 在该步选定自己的执棋方与模型，创建分支。新棋局记录父棋局 ID 与分支步数，不改变原棋局。
5. 对局支持 JSON 导入/导出。导入会逐手验证规则和合法性；外部文件提及的 Checkpoint 不存在时仍可复盘，但续弈需选择本机兼容权重。

## 局面分析与模型比较

在「局面分析」选择历史棋局与步骤，或交替落子手动摆局。选择权重与搜索预算，运行后端真实推理。

- **Top 5**：对所有合法动作，落子后批量预测子局面 WDL，并将 win/loss 交换回当前玩家视角，按 `win` 降序排序。直接获胜动作使用引擎终局真值。
- **预测胜率 / 平局 / 失败**：当前行棋玩家的模型预测，并非校准后的客观概率。对弱模型而言，根局面和子局面的预测可能不一致。
- **Policy**：根节点合法动作掩码后的网络动作概率，不能解释成胜率。
- **访问数 / 访问分布 / 搜索 Q / 搜索 WDL**：PUCT 搜索统计，Q 为 W−L。未访问动作的搜索值为 `null`，不伪造搜索结果。
- 搜索次数影响搜索统计；Top 5 采用逐手单层子局面 WDL 排序，**不会随搜索预算自动变成深搜索胜率**。API 同时保留独立的搜索 WDL，供进一步研究。
- 棋盘显示 Top 5 排名和概率热力图；历史局面显示实际下一手与首选手的差异。

「模型对比」用相同局面、相同搜索次数分别分析两个权重，展示并排棋盘、各自 Top 5、WDL 与 Policy L1 差异；也支持同架构不同训练阶段。

## 评估与复现实验

Web「固定基准评估」支持 Random、Heuristic 和任意历史权重；评估异步排队，可输入多个评估种子。每个开局交换黑白，固定相同起始落点。报告 W/D/L、得分、相对固定对手的 Elo 分差、WDL Brier、胜率 ECE、校准分桶、一步胜/必须防守战术题、延迟基准。

```powershell
.venv/Scripts/python -m gomoku.cli evaluate CHECKPOINT_ID --opponent heuristic --pairs 4 --simulations 32 --seeds 42,43,44
# --opponent 可换成 random 或另一个 Checkpoint ID

# 正式独立多训练种子比较；会顺序运行 6 次有界训练，可能持续数小时
.venv/Scripts/python scripts/research.py --seeds 42 43 44 --games 200 --simulations 32 --pairs 4 --hours 2

# 可选 TensorBoard
.venv/Scripts/tensorboard --logdir data/runs
```

多评估种子不能代替多独立**训练**种子。`research.py` 对每架构每训练种子单独初始化，统一评估开局，分别汇总固定对手成绩均值和样本标准差，并增量保存 `data/research-summary.json`。它不会把 CNN vs ViT 相互胜率当作唯一棋力依据。正式研究建议至少 3–5 个训练种子、更多配对开局、统一搜索预算，同时画对局数和实际训练秒数对应的成绩曲线。

默认网络 CNN 128,884 参数、ViT 110,404 参数；ViT 少约 14.3%，属于相近而非严格等参。默认解析 FLOPs 20.38M / 21.01M（乘加计 2，不含激活和归一化）。网络 JSON 可调整宽度/层数；15 路需要新建该尺寸实验，旧 ViT 位置编码不能直接加载。

## 测试与实际验证

```powershell
.venv/Scripts/python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/smoke.ps1
# 在已有双架构 smoke 权重上，再验证真实恢复、分析与 CPU/GPU 性能
.venv/Scripts/python scripts/validate.py
```

实际结果与浏览器验收见 [docs/VALIDATION.md](docs/VALIDATION.md)，机器可读数据在 [docs/validation.json](docs/validation.json)。测试直接检查算法和参数更新，前端构建不作为训练正确性证据。

## 存储、资源与限制

- 默认 `data/lab.sqlite`，WAL 模式，schema v1。`GOMOKU_DATA` 可指定独立数据目录；模型在其 `checkpoints/`，TensorBoard 在 `runs/`，训练异常日志在 `logs/`。
- 训练默认 4 个 CPU 线程，Web 推理 2 个线程、CPU、最多缓存 2 个模型。不同训练进程与网页推理隔离；网页推理和评估共享串行计算锁，长评估期间分析/AI 请求会等待，但健康检查与对局读取仍可响应。
- 自博弈同时推进多个棋局，同一网络的叶节点批量推理；每棵树每轮一个叶子，未实现跨树虚拟损失或持久树复用。
- Checkpoint 数量由保存间隔、总局数与数据目录磁盘预算控制，不自动删除已保存权重，避免破坏历史对局引用。磁盘/时间预算在采样批次之间检查，可能超出一个批次及一次完整保存（双模型为两份文件）的开销。TensorBoard 只写标量，不写图片/直方图；长实验指标 API 最多约 2000 个采样点，JSON 实验导出保留全量。
- 正常关闭 Web 会请求训练在批次边界停止；硬关机或进程崩溃只可恢复最近已保存权重，未保存批次会丢失。进程状态在后端重启后标记 interrupted。
- 本阶段未实现严格因果解释、Attention/特征可视化、复杂双威胁/远距离战术题、固定探针跨 Checkpoint 自动追踪或跨棋盘尺寸泛化基准。这些是后续研究扩展，不在当前 UI 中冒充已实现功能。
- 本次只完成小规模训练验收，模型尚弱，不能据此判断架构优劣；长期训练的收敛、超参数及大规模吞吐仍需实验。

实现参考：[FastAPI 同步路由线程池](https://fastapi.tiangolo.com/async/)、[PyTorch 本地安装](https://pytorch.org/get-started/locally/)。完整依赖快照见 `requirements-lock.txt`，前端锁见 `frontend/package-lock.json`。CUDA 快照需搭配 PyTorch cu126 索引；CPU 安装请用 bootstrap 的 CPU 选项。


## 外部 15×15 SFT 数据

参见 [下载、许可、统计和复现指南](datasets/gomoku/README.md)。SFT 页面可组合 Karesis 与 Gomoku Party，并保留固定验证/测试划分；RenjuNet 与 Gomocup 受权限限制保持禁用。
