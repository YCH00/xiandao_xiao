# 金融大模型部署优化技术报告

## 1. 项目背景与目标

本项目面向“金融大模型的部署优化”赛题，任务是在评测机的 `/submission` 目录中提交一个可被平台直接调用的推理服务。平台通过固定接口调用 `predictor.py` 中的 `Predictor` 类：先执行 `load()` 加载模型，再逐条调用 `predict(prompt)` 得到预测结果。评分主要综合三部分：显存占用、单样本平均推理时延，以及涨跌幅分类精度和涨跌方向准确率。

赛题后续明确要求，参赛模型必须基于指定的基线大模型，通过量化、剪枝、蒸馏或结构压缩等方式得到；不能直接采用外部模型，也不能脱离基线大模型直接训练独立小分类器。因此，本项目最终放弃了早期的轻量独立分类器思路，改为以官方 `Llama-2-7b-chat-hf` 基座和 `fingpt-forecaster_sz50_llama2-7B_lora` 适配器为源模型，构建一个合规的压缩版 Llama 分类模型。

最终提交版本采用 8 层压缩 Llama 分类器，并将推理最大长度设置为 1024，以在保证可运行性和一定精度的前提下显著降低显存与时延。

## 2. 总体技术路线

整体方案可以概括为：

```text
官方 Llama2-7B 基座
        +
官方 FinGPT LoRA adapter
        |
        | merge_and_unload()
        v
合并后的官方 FinGPT teacher / source model
        |
        | 按层抽取 Transformer block
        v
压缩 Llama backbone
        |
        | last-token pooling
        v
13 类涨跌幅分类头 + 3 类方向辅助头
        |
        | public label + teacher hard label + direction loss
        v
最终部署模型 weights/compressed_llama_classifier
```

该路线有三个核心点：

1. 模型结构和初始化权重来自赛题指定基线大模型，不引入外部预训练模型。
2. 通过抽取部分 Transformer 层实现结构压缩，减少显存占用和前向计算量。
3. 将原来的生成式任务转化为分类式推理，避免逐 token 生成造成的长时延，同时保持输出格式与评分器兼容。

## 3. 基线模型合规处理

官方保底模型由两部分组成：

- 基座模型：`/opt/fingpt-forecaster/models/Llama-2-7b-chat-hf`
- LoRA 适配器：`/opt/fingpt-forecaster/models/fingpt-forecaster_sz50_llama2-7B_lora`

训练脚本 `train_student.py` 首先加载基座模型，再加载 LoRA 适配器，并调用 `merge_and_unload()` 将 LoRA 权重合并回基座模型。后续压缩学生模型的 embedding、norm 和被保留的 Transformer 层权重，都从这个合并后的官方模型中复制得到。

这保证了最终提交模型不是独立训练的小模型，而是官方 FinGPT 模型的结构压缩版本。保存模型时，`classifier_config.json` 中记录了 `base_model_path`、`adapter_path`、`layer_indices` 和压缩后的 Llama 配置，便于复核模型来源和压缩方式。

## 4. 结构压缩设计

原始 Llama2-7B 模型包含 32 个 decoder layer。最终方案保留 8 个 layer，保留方式不是简单取前 8 层，而是在 32 层中均匀采样，以覆盖浅层、中层和深层表示。8 层版本的层索引为：

```text
[0, 4, 9, 13, 18, 22, 27, 31]
```

压缩模型保留以下部分：

- token embedding：保持与官方模型一致，保证输入表示空间不变。
- 选中的 Llama decoder layers：从合并后的官方模型逐层复制权重。
- final norm：保持官方模型最后的归一化结构。
- 新增 13 类涨跌幅分类头：预测具体区间标签。
- 新增 3 类方向辅助头：预测上涨、下跌、持平。

压缩模型去掉了原始语言模型的 `lm_head` 和生成式解码流程。推理时只执行一次前向传播，然后取最后一个有效 token 的 hidden state 做分类。这比生成式输出快得多，也更适合本赛题的固定标签空间。

## 5. 分类标签与辅助任务

赛题输出空间被整理为 13 个合法标签：

```text
上涨0-1%, 上涨1-2%, 上涨2-3%, 上涨3-4%, 上涨4-5%, 上涨超过5%,
下跌0-1%, 下跌1-2%, 下跌2-3%, 下跌3-4%, 下跌4-5%, 下跌超过5%,
股价持平
```

模型主分类头输出 13 类 logits。考虑到平台评分中方向准确率是重要指标，我们额外加入了 3 类方向辅助头：`up`、`down`、`flat`。训练时同时优化区间分类损失和方向分类损失，使模型即使在具体幅度档位预测不完全准确时，也尽量保持方向判断稳定。

## 6. 蒸馏与训练策略

训练监督信号来自两部分：

1. 公开 parquet 数据中的真实 `label`。
2. 官方 FinGPT teacher 生成并解析得到的 `teacher_label`。

`build_teacher_labels.py` 用官方 Llama2-7B + FinGPT LoRA 模型对公开样本做确定性生成，并用评分器同源的解析逻辑提取标签。由于生成式模型有时会输出自由文本或无法解析的结果，脚本会依次尝试解析新增生成文本、完整 prompt+生成文本；若仍无法得到合法标签，则回退到公开样本的真实标签，并在 `teacher_parse_source` 中记录来源。这一处理保证了训练文件始终可用，同时保留了 teacher 生成信息的可追踪性。

训练损失为三部分加权和：

```text
loss = 0.7 * gold_label_cross_entropy
     + 0.2 * teacher_label_cross_entropy
     + 0.1 * direction_cross_entropy
```

默认训练时冻结压缩后的 Llama backbone，仅训练分类头和方向头，从而降低过拟合风险和训练显存压力。训练脚本也保留了 `--unfreeze-last-layers` 参数，后续可以选择解冻末尾若干层进行更充分微调。

主要训练配置如下：

- 模型初始化：官方基座 + LoRA 合并后的权重。
- 结构压缩：最终采用 8 层版本。
- batch size：1。
- gradient accumulation：最终实验采用 16。
- head 学习率：最终实验采用 `5e-4`。
- 混合精度：CUDA/DCU 环境下使用 `bfloat16 autocast`。
- 验证集划分：按标签做分层划分，默认 `val_ratio=0.2`。
- checkpoint 选择：按验证集 F1、方向准确率和平台清零阈值相关指标选择 best epoch，而不是简单保存最后一轮。

best checkpoint 机制很重要。早期训练中观察到后期 epoch 可能出现验证指标回落，如果只保存最后一轮，会错过中间更优的模型。因此训练脚本在每轮后评估 train/val，并将最优状态复制到 CPU，训练结束后再加载 best state 保存。

## 7. 推理部署优化

最终推理入口在 `predictor.py`。部署时默认加载：

```text
weights/compressed_llama_classifier
```

本地对比实验可以通过环境变量指定候选目录：

```bash
COMPRESSED_LLAMA_DIR=weights/compressed_llama_classifier_8l_probe
COMPRESSED_LLAMA_DIR=weights/compressed_llama_classifier_12l_probe
```

正式提交时不依赖环境变量，而是将最终模型放入默认目录 `weights/compressed_llama_classifier`。

推理阶段做了以下优化和稳定性处理：

1. 使用压缩分类模型优先路径，避免加载完整 7B 生成模型。
2. 使用 `bfloat16` 权重和 `autocast` 前向，降低显存占用。
3. tokenizer 使用左截断，保留 prompt 末尾更接近预测问题的上下文。
4. 默认推理最大长度设为 1024，降低 attention 计算和显存压力。
5. 强制 Llama attention 使用 `eager` 实现，避免评测环境缺少 `flash_attn_2_cuda*.so` 导致运行时错误。
6. 使用 `torch.load(..., weights_only=True)` 加载权重，消除 PyTorch pickle 安全警告。
7. 输出格式固定为 `预测涨跌幅：<标签>`，与评分器解析逻辑一致。

如果压缩模型目录不存在或加载失败，`Predictor` 仍保留官方 FinGPT 7B + LoRA 的回退路径，保证代码具备基本可运行性。但最终提交版本应确保压缩模型目录存在，使平台实际评测压缩模型。

## 8. 工程问题与修复

实现过程中遇到并解决了若干部署问题：

| 问题 | 原因 | 解决方式 |
|---|---|---|
| `train-*.parquet` 找不到 | 训练机公开目录中只有 `test-*.parquet` | 训练脚本支持 glob 展开，并改用实际存在的公开 parquet |
| `teacher_label=None` | teacher 生成文本有时不含可解析标签 | 解析新增文本、完整文本，失败时回退公开 label 并记录来源 |
| 本地对比无法区分 8 层和 12 层 | `Predictor` 原先写死正式目录 | 增加 `COMPRESSED_LLAMA_DIR` 环境变量用于本地候选模型对比 |
| `torch.load` FutureWarning | PyTorch 默认 `weights_only=False` | 改为 `weights_only=True`，并保留旧版兼容兜底 |
| 每条预测 `RuntimeError: No matching libraries found for flash_attn_2_cuda*.so` | ROCm/评测环境没有可用 flash attention 库 | 在压缩 Llama 配置中强制 `config._attn_implementation = "eager"` |
| 长上下文推理不稳定且较慢 | attention 复杂度随长度平方增长 | 默认推理长度降至 1024，保留环境变量便于实验覆盖 |
| 最后一轮模型不一定最优 | 小数据训练波动明显 | 保存验证指标最优 checkpoint |

这些修复使最终提交版本从“能训练”进一步变成“能稳定部署和复现”。

## 9. 本地验证与模型选择

本地评测脚本 `local_eval.py` 使用与平台一致的解析和聚合逻辑，并增加了 `--debug-errors` 参数。当预测阶段出现异常时，可以打印第一条完整 traceback，便于定位环境问题。

8 层和 12 层候选模型的训练与评测命令如下：

```bash
python train_student.py \
  --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/test-*.parquet \
  --teacher-labels weights/teacher_labels.jsonl \
  --kept-layers 8 \
  --lr-head 5e-4 \
  --gradient-accumulation 16 \
  --output-dir weights/compressed_llama_classifier_8l_probe

python train_student.py \
  --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/test-*.parquet \
  --teacher-labels weights/teacher_labels.jsonl \
  --kept-layers 12 \
  --lr-head 5e-4 \
  --gradient-accumulation 16 \
  --output-dir weights/compressed_llama_classifier_12l_probe
```

```bash
COMPRESSED_LLAMA_DIR=weights/compressed_llama_classifier_8l_probe \
COMPRESSED_LLAMA_MAX_LENGTH=1024 \
uv run python local_eval.py \
  --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/test-*.parquet \
  --limit 100 \
  --debug-errors
```

最终提交时将 8 层候选模型复制为正式目录：

```bash
rm -rf weights/compressed_llama_classifier
cp -a weights/compressed_llama_classifier_8l_probe weights/compressed_llama_classifier
```

然后执行正式默认路径验证：

```bash
uv run python local_eval.py \
  --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/test-*.parquet \
  --limit 100 \
  --debug-errors
```

当日志中出现以下内容，并且所有样本状态为 `ok` 时，说明正式提交目录和默认推理配置正确：

```text
Loaded compressed Llama classifier: /submission/weights/compressed_llama_classifier
```

## 10. 最终提交版本

本次最终提交版本配置如下：

| 项目 | 配置 |
|---|---|
| 模型来源 | 官方 Llama2-7B + FinGPT LoRA |
| 压缩方式 | Transformer layer 抽层结构压缩 |
| 最终层数 | 8 层 |
| 保留层索引 | `[0, 4, 9, 13, 18, 22, 27, 31]` |
| 推理任务形式 | 13 类涨跌幅分类 + 3 类方向辅助头 |
| 正式模型目录 | `weights/compressed_llama_classifier` |
| 默认推理长度 | 1024 |
| attention 实现 | eager |
| 推理精度 | bfloat16 autocast |
| 输出格式 | `预测涨跌幅：<标签>` |

最终提交 `#339` 的平台评测结果如下：

| 指标 | 数值 |
|---|---:|
| 总分 | 66.771914 |
| 显存 U | 27.569454 |
| 时长 V | 19.700474 |
| 精度 W | 19.501986 |
| 显存峰值 | 4631 MB |
| 平均时延 | 1.52 s |
| 平台精度得分 | 0.1667 |
| 方向准确率 | 0.490 |
| 有效样本 | 100 / 100 |
| 样本状态 | ok=100 |

从结果看，最终版本已经完成了稳定部署目标：100 条样本全部有效，没有解析失败或运行异常；平均单样本时延为 1.52 秒，相比生成式 7B 推理大幅降低，时长 V 接近满分；显存峰值为 4631 MB，显著低于完整基线模型，说明 8 层结构压缩和分类式推理在资源占用上是有效的。

主要短板集中在精度 W。平台精度得分为 0.1667，方向准确率为 0.490，说明模型已经学到了一定的涨跌幅分布，但方向判断仍在基线附近波动。由于时长 V 只剩约 0.30 分提升空间，继续优化推理速度对总分帮助有限；显存 U 仍有一定空间，但进一步减少层数可能明显损害精度。因此后续若要继续提分，优先级应放在提高方向准确率和幅度分类质量上，而不是继续压缩模型。

## 11. 进一步提分空间分析

基于本次提交结果，后续优化可以按收益和风险分为三类。

第一类是低风险的阈值与后处理优化。当前分类头直接取 13 类 logits 的 argmax，没有利用方向辅助头进行约束。后续可以在推理时融合主分类头和方向头：先用方向头判断 `up/down/flat`，再只在对应方向的幅度档位中选择主分类头最高的标签。这类方法不改变模型结构，不增加明显显存和时延，目标是把方向准确率从 0.490 提升到 0.5 以上。也可以在公开验证集上搜索不同方向 logits 权重，例如 `final_score = class_logits + alpha * direction_logits[direction(label)]`，选择能提升验证集方向准确率且不明显牺牲精度得分的 `alpha`。

第二类是训练目标优化。当前损失权重为 `0.7 * gold + 0.2 * teacher + 0.1 * direction`，方向损失权重偏保守。从平台结果看，方向准确率是明显瓶颈，可以尝试把方向损失权重提高到 0.2 或 0.3，并相应降低 teacher loss 权重。还可以启用轻量解冻，例如 `--unfreeze-last-layers 1`，让压缩 backbone 的最后一层适配分类任务。这会增加训练成本，但推理结构不变，若验证集稳定提升，提交风险可控。

第三类是候选结构对比。8 层版本在显存和时延上表现很好，12 层版本可能带来更高精度，但显存 U 会下降、加载时间也会更长。由于当前 V 已接近满分，12 层是否值得取决于 W 的提升幅度。如果 12 层能把方向准确率和精度得分显著拉高，牺牲部分 U 可能是划算的；如果只小幅提升，则继续使用 8 层更稳。相比直接减少到 6 层或 4 层，当前结果表明继续压缩的边际收益主要在 U，但精度 W 的损失风险较大，不建议作为优先方向。

综合判断，下一轮最值得尝试的顺序是：先做方向头融合后处理，再尝试将 `COMPRESSED_LLAMA_MAX_LENGTH` 从 1024 提高到 1536 或 2048 观察精度变化，然后调整方向损失权重重新训练 8 层模型，最后再用同一套训练策略比较 12 层模型。当前平均时延只有 1.52 秒，时长 V 接近满分，因此适度增加上下文长度有可能用很小的速度代价换取更好的精度 W。这样可以在保持当前部署稳定性的前提下，重点冲击总分中最有弹性的精度部分。

## 12. 第三方代码与依赖说明

本方案使用的第三方库主要包括：

- PyTorch：模型定义、训练、权重保存与推理。
- transformers：加载官方 Llama 模型、tokenizer 和 LlamaModel 结构。
- peft：加载并合并官方 FinGPT LoRA 适配器。
- pandas / pyarrow：读取 parquet 数据。
- numpy：分层划分和训练数据处理。

本方案未使用外部金融预测模型、外部预训练模型或脱离基线训练的独立分类模型。新增代码主要实现结构压缩、分类头训练、teacher label 生成、本地评测调试和部署适配。

## 13. 总结

本项目最终形成了一条合规且可部署的金融大模型优化路线：以官方 FinGPT 基线为源模型，通过 LoRA 合并、Transformer 层抽取、分类头蒸馏训练和推理路径优化，将原本重型的生成式 7B 模型压缩为 8 层 Llama 分类器。该方案保留了基线模型的 embedding 和关键层权重，避免脱离基线重新训练小模型，同时将推理从逐 token 生成简化为单次前向分类，在显存和时延上获得明显收益。

工程上，我们围绕评测环境做了多项稳定性处理，包括 eager attention、1024 token 默认截断、bfloat16 autocast、安全权重加载和候选模型目录切换。最终版本可以直接通过 `Predictor` 接口被平台调用，模型目录、输出格式和本地验证流程均已固定，适合作为本次校赛的最终提交版本。