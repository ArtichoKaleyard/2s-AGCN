# AGCN / NTU60 / X-View 实验报告

## 1. 实验目标

本轮实验目标是复现 2s-AGCN 在 NTU RGB+D 60 cross-view 设置下的核心结果，并在 Foundry 训练入口下验证本仓现代化实现的可用性。

主线实验只认官方旧仓库的融合语义：`joint` 与 `bone` 单流分别训练，测试阶段分别保存分类 score，再做离线 score-level fusion。训练期双流 wrapper 属于额外试探，不作为官方复现结论。

## 2. 数据与任务设置

- 数据集：NTU RGB+D 60
- 划分协议：cross-view (`xview`)
- 类别数：60
- 输入格式：官方 2s-AGCN 预处理后的 `.npy + .pkl`
- 数据目录：`data/ntu/xview`
- 训练样本数：`37646`
- 验证样本数：`18932`
- 图结构：`ntu-rgb+d`
- 图策略：`spatial`

本次使用的 `data/ntu/xview` 是纯 NTU60 数据重建结果，标签范围为 `0..59`。不使用混入 `A061-A120` 的 NTU120 原始目录。

## 3. 训练配置

论文训练细节使用 SGD + Nesterov、`weight_decay=0.0001`、`epochs=50`，并在 NTU 上于第 30、40 epoch 降学习率。本仓 Foundry 单卡训练受显存限制，采用等效 batch 近似配置。

| 项目 | joint 单流 | bone 单流 |
| --- | ---: | ---: |
| 配置入口 | `conf/two_stream_agcn/ntu60_xview_agcn_joint_foundry.yaml` | `conf/two_stream_agcn/ntu60_xview_agcn_bone_foundry.yaml` |
| Skeleton 配置 | `conf/skeleton/ntu60_xview_agcn_joint.yaml` | `conf/skeleton/ntu60_xview_agcn_bone.yaml` |
| batch size | 16 | 16 |
| grad accumulation | 2 | 2 |
| 有效 batch | 32 | 32 |
| optimizer | SGD + Nesterov 0.9 | SGD + Nesterov 0.9 |
| learning rate | 0.05 | 0.05 |
| scheduler | MultiStep `[30, 40]`, gamma 0.1 | MultiStep `[30, 40]`, gamma 0.1 |
| epochs | 50 | 50 |
| seed | 49 | 49 |

说明：这不是严格复刻论文的多卡 `batch_size=64`，而是在单卡约束下使用 `batch_size=16 + grad_accum_steps=2` 维持有效 batch 32 的近似复现。

## 4. 单流训练结果

| Stream | Best epoch | Best Top-1 | Final Top-1 | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| joint | 46 | 94.0471% | 93.9837% | `artifacts/skeleton/ntu60_xview_agcn_joint_foundry/best.pt` |
| bone | 42 | 93.8253% | 93.5823% | `artifacts/skeleton/ntu60_xview_agcn_bone_foundry/best.pt` |

对应 artifacts：

- joint：`artifacts/skeleton/ntu60_xview_agcn_joint_foundry`
- bone：`artifacts/skeleton/ntu60_xview_agcn_bone_foundry`

bone 单流训练过程中曾多次遇到随机 CUDA 假死，但最终已通过 resume 跑完 50 epoch，并保存了 `best.pt` 与 `last.pt`。该问题属于工程排障线索，不影响本节对已完成 checkpoint 的记录。

## 5. 官方式 Score Fusion

官方旧仓库的双流融合方式不是训练一个双 backbone 模型，而是：

1. 训练 `joint` 单流。
2. 训练 `bone` 单流。
3. 分别测试并保存 softmax/logit score。
4. 使用 `score_joint + alpha * score_bone` 做离线融合。

本次按该语义使用两个单流 `best.pt` 在验证集上推理并保存 score。

结果目录：

`artifacts/ensemble/ntu60_xview_agcn_official_scores`

| Method | Top-1 | Top-5 | 备注 |
| --- | ---: | ---: | --- |
| joint score | 94.0471% | 99.0704% | `joint_scores.npy` / `joint_score.pkl` |
| bone score | 93.8253% | 99.1073% | `bone_scores.npy` / `bone_score.pkl` |
| score fusion (`alpha=1.0`) | 95.3729% | 99.3397% | `joint + bone` |

该结果与 2s-AGCN 在 NTU60 cross-view 上的论文量级一致，可作为当前主线复现实验的核心结论。

### 5.1 Alpha Sweep

为补全官方式 score fusion 的敏感性检查，本次额外在已导出的 train/val logits 上扫 `alpha`，范围为 `0.000..2.000`，步长 `0.001`，融合公式保持 `joint + alpha * bone`。

脚本与结果：

- 脚本：`scripts/experiments/sweep_official_alpha.py`
- 结果目录：`artifacts/ensemble/ntu60_xview_agcn_fusion_head`
- 明细：`alpha_sweep.csv`
- 摘要：`alpha_sweep_summary.json`

| Selection | Alpha | Train Top-1 | Val Top-1 | Val Top-5 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed official | 1.000 | 99.9894% | 95.3729% | 99.3397% | 主线复现结果 |
| train-selected | 0.497 | 99.9973% | 95.3623% | 99.3186% | 按训练集 Top-1 选 alpha |
| val-oracle | 0.703 | 99.9894% | 95.4733% | 99.3239% | 仅作验证集上界参考 |

结论：按训练集选择 alpha 并没有提升验证集结果；验证集 oracle 可把 Top-1 推到 `95.4733%`，但这是直接在验证集上调参得到的上界参考，不应替代 `alpha=1.0` 的主线复现结论。

## 6. 训练期 Fusion Wrapper 试探

Foundry 侧还提供了训练期双流 wrapper：输入同时包含 `joint` 和 `bone`，模型内部同时前向两个 AGCN backbone，再做融合。这条路径不是官方旧仓库的复现路径，只用于额外试探和观察潜在 bug。

### 6.1 Sum Fusion

配置：

- `conf/skeleton/ntu60_xview_agcn_two_stream_sum.yaml`
- `conf/two_stream_agcn/ntu60_xview_agcn_two_stream_sum_foundry.yaml`

当前配置使用：

- `batch_size=8`
- `grad_accum_steps=4`
- 有效 batch 仍为 32
- `fusion=sum`
- 输出目录：`artifacts/skeleton/ntu60_xview_agcn_two_stream_sum_foundry`

降 batch 的原因是训练期 wrapper 同时构建 `joint` 和 `bone` 两个 backbone，单步显存/计算接近两条单流叠加。该设置不是论文要求，而是工程试探配置。

当前状态：已完成。该结果证明训练期双 backbone 路径并非必然触发随机假死，但它没有带来官方式离线 score fusion 的收益。

| Variant | Config | Status | Best epoch | Best Top-1 | Final Top-1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| two-stream sum | `ntu60_xview_agcn_two_stream_sum_foundry.yaml` | finished | 50 | 94.0471% | 94.0471% | 非官方训练期 wrapper |

### 6.2 Frozen Fusion Head

更符合原始试探意图的训练期版本应冻结已训练好的 `joint` / `bone` 两个单流 checkpoint，只训练融合头。这样它回答的是“单流特征或 logits 是否能通过一个轻量头进一步融合”，而不是重新训练两个 backbone。

本次采用更省显存和时间的离线 logits 方案：对训练集和验证集分别导出 joint/bone logits，只在训练集 logits 上训练融合头，最后在验证集 logits 上评估。验证集 logits 只用于评估，避免“在验证集上学习融合权重再汇报验证集精度”的数据泄漏。

脚本与结果：

- 脚本：`scripts/experiments/train_offline_fusion_head.py`
- 结果目录：`artifacts/ensemble/ntu60_xview_agcn_fusion_head`
- 训练样本数：`37646`
- 验证样本数：`18932`
- seed：`49`

| Variant | Best epoch | Best Top-1 | Best Top-5 | Final Top-1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed sum baseline | - | 95.3729% | 99.3397% | 95.3729% | `joint + bone`，无训练 |
| scalar head | 75 | 95.4099% | 99.2764% | 95.2145% | 学到 joint/bone 权重约 `1.05 / 0.69` |
| linear head | 500 | 94.3746% | 99.1126% | 94.3746% | `120 -> 60`，明显不如 fixed sum |

结论：只训练融合头并未稳定超过官方固定求和。受约束的 scalar head 只带来约 `+0.037%` Top-1，幅度小到不宜解读为显著改进；全连接 linear head 反而明显退化，说明在训练集 logits 上容易过拟合或破坏原有校准。

### 6.3 Concat Linear Fusion（预留）

拼接版本计划使用 `fusion=concat_linear`：分别提取 joint/bone backbone feature，再拼接后接线性分类头。如果不冻结 backbone，它会变成完整训练期双流模型；如果冻结 backbone，则才符合“只训练融合头”的实验口径。

预留配置建议：

- `conf/skeleton/ntu60_xview_agcn_two_stream_concat_linear.yaml`
- `conf/two_stream_agcn/ntu60_xview_agcn_two_stream_concat_linear_foundry.yaml`

建议初始资源口径：

- `batch_size=8`
- `grad_accum_steps=4`
- `learning_rate=0.05`
- `epochs=50`
- `seed=49`

预留结果表：

| Variant | Config | Status | Best epoch | Best Top-1 | Final Top-1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| feature-level concat_linear | `ntu60_xview_agcn_two_stream_concat_linear_foundry.yaml` | not started | TBD | TBD | TBD | 若冻结 backbone，才是只训练融合头 |

## 7. 当前结论

1. `joint` 与 `bone` 单流均已完成 50 epoch，并保存可用 best checkpoint。
2. 官方式离线 score fusion 已完成，`alpha=1.0` 下 Top-1 为 **95.3729%**。
3. alpha sweep 已完成；train-selected alpha 未提升验证集，val-oracle alpha 仅作为上界参考。
4. 冻结 logits 的融合头实验已完成；scalar head 几乎持平，linear head 退化，当前没有证据表明训练融合头优于官方固定求和。
5. 训练期 sum wrapper 已完成，但它重新训练了双 backbone，不等同于“只训练融合头”。
6. 随机 CUDA 假死问题暂不作为本报告主线，但已在项目记忆中单独记录诊断过程。

## 8. 关键路径索引

- joint 训练结果：`artifacts/skeleton/ntu60_xview_agcn_joint_foundry`
- bone 训练结果：`artifacts/skeleton/ntu60_xview_agcn_bone_foundry`
- 官方式融合结果：`artifacts/ensemble/ntu60_xview_agcn_official_scores`
- 冻结 logits 融合头：`artifacts/ensemble/ntu60_xview_agcn_fusion_head`
- alpha sweep：`artifacts/ensemble/ntu60_xview_agcn_fusion_head/alpha_sweep_summary.json`
- 训练期 sum wrapper：`artifacts/skeleton/ntu60_xview_agcn_two_stream_sum_foundry`
- score fusion 脚本：`scripts/experiments/eval_official_score_ensemble.py`
- 离线融合头脚本：`scripts/experiments/train_offline_fusion_head.py`
- alpha sweep 脚本：`scripts/experiments/sweep_official_alpha.py`
