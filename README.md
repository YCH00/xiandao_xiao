## 本提交优化方法

本提交新增一个轻量级 13 类学生分类器作为第二层优化方案：使用公开训练数据的 `prompt/label` 训练 Hashing 字符 n-gram + online softmax 线性分类器，并额外训练涨/跌/平三分类辅助头。正式推理时若存在 `weights/student_model.npz`，`Predictor` 会加载该轻量模型并真实执行分类推理，直接输出合法标签行 `预测涨跌幅：<标签>`；若权重不存在或损坏，则自动回退到官方 FinGPT 7B + LoRA 保底模型。

第三方代码/算法说明：轻量分类器实现为本队自研代码，使用标准 hashing trick、softmax regression 与辅助方向分类思想；运行时仅依赖 Python 标准库和 `numpy`。训练脚本 `train_student.py` 读取 parquet 时需要开发环境中的 `pandas/pyarrow`，但正式评测不会调用训练脚本。

训练轻量模型示例：

```bash
python train_student.py --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/train-*.parquet --output weights/student_model.npz
uv run python local_eval.py --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/test-*.parquet --limit 100
```
# FinGPT 推理部署优化 — 参赛提交模板

本目录就是你的**提交物**：把它放到你队伍服务器的 `/submission` 目录，
在比赛平台网页上点击"提交评测"，判题机会拉取整个目录自动评测。

## 目录约定

```
/submission/
├── predictor.py     # 必需：判题入口，实现 Predictor 类（见文件内契约注释）
├── pyproject.toml   # 必需：uv 管理的依赖声明（判题前自动 uv sync）
├── weights/         # 你的轻量化模型权重（随目录一起被拉取）
├── scoring/         # 平台判分代码副本（解析+评分，与判题系统逐字节一致）
├── local_eval.py    # 本地自测脚本，提交前先跑
└── README.md        # 请在此说明你的优化方法与第三方代码引用（学术诚信要求）
```

## 评分规则

- **U 显存（40 分）**：`max(1 - 你的显存/基线显存, 0) × 40`，显存 = 判题机在
  模型加载完成后到推理结束期间 rocm-smi 采到的该卡峰值占用
- **V 时长（20 分）**：`max(1 - 你的平均样本时长/基线时长, 0) × 20`，逐条
  predict 计时（batch=1），模型加载时间不计入
- **W 精度（40 分）**：`[min(你的序数得分/基线序数得分, 1) + min(你的方向准确率/基线方向准确率, 1)] × 20`
  - **序数得分**：13 个涨跌幅档位排成有序刻度（下跌>5% … 平 … 上涨>5%），按预测档位与
    真实档位的距离 d 给分——d=0（完全命中）→1.0，d=1→0.5，d=2→0.25，d≥3 或解析失败→0，
    再按 13 类做宏平均（每类先取该类样本均分，再对各类求平均）
  - **方向准确率** = 预测涨/跌/平方向与真实一致的样本占比
- ⚠️ **序数得分或方向准确率低于平台公告阈值时总分记 0**（拦截乱输出的无效提交）
- ⚠️ **提交必须真实执行所提交模型的推理**。组委会将对榜单前列队伍进行代码
  审查与复现，输出与模型推理不符（如硬编码/随机生成预测）将取消成绩
- 评测集：隐藏测试集的 100 条分层子集，与公开测试集同构（prompt/label）
- 评测可能把 100 条样本分片到多张卡并行加速（每片仍是单卡容器、batch=1 逐条
  计时），显存峰值取各卡最大值，与单卡评测完全等价，对你的代码无感知

## 限制

| 项目 | 限制 |
|---|---|
| 每日提交次数 | 5 次（失败也计入） |
| 同时评测任务 | 每队 1 个 |
| 整次评测超时 | 90 分钟（不含依赖安装） |
| 单样本超时 | 5 分钟（超时该样本 0 分或终止评测） |
| 依赖安装（uv sync） | 15 分钟 |

## 测试提交（推荐先跑）

平台网页上的「测试提交」按钮：只拉取你的 /submission 并检查目录结构
（predictor.py / Predictor 类 / pyproject.toml），**不跑评测、不消耗每日配额**，
几十秒出结果。首次提交评测前建议先用它确认目录约定无误。

## 本地自测

```bash
uv sync                                  # 安装依赖（含海光 das torch）
uv run python local_eval.py --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/test-*.parquet --limit 20
```

## 注意事项

- **提交即快照**：点击提交后平台会立刻拉取你的 /submission 存档（排队期间完成，
  不等评测开始），提交后请勿再改动该目录，改动不会生效且可能导致拉取不一致
- Python 必须 3.10（海光 torch wheel 仅有 cp310）；不要把 torch 换成官方 CUDA 版
- 代码内 GPU 统一写 `cuda:0`（判题机通过 HIP_VISIBLE_DEVICES 分卡）
- 权重用相对路径加载（拉取后目录位置与你服务器上不同）
- 公共基座模型在判题机的 `/opt/fingpt-forecaster/models/` 同样可用
- 使用第三方代码/开源算法必须在本 README 头部明确标注

