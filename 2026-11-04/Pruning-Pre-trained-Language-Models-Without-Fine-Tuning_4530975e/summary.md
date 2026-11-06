---
title: "Pruning-Pre-trained-Language-Models-Without-Fine-Tuning"
source: https://aclanthology.org/2023.acl-long.35.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:05:12"
field: "模型压缩与高效推理"
keywords: ["模型剪枝", "预训练语言模型", "一阶梯度", "无微调压缩", "稀疏子网", "BERT"]
innovations: ["提出 Static Model Pruning (SMP)，仅通过一阶梯度更新二值 mask 完成剪枝，无需 fine-tuning 即可适配下游任务", "设计架构感知的全局 mask 分配函数，按权重矩阵重要性比例分配稀疏度，兼顾效率与性能", "基于标签词嵌入初始化并冻结任务头，减少可训练参数同时保持性能"]
benchmarks: ["GLUE", "MNLI", "QQP", "SQuAD"]
---

# 论文速读：Pruning-Pre-trained-Language-Models-Without-Fine-Tuning

## 一句话总结
论文提出 **Static Model Pruning (SMP)**，一种仅通过一阶梯度信息做剪枝决策、**不微调原始预训练权重**即可适配下游任务的模型压缩方法，在从低稀疏到极高稀疏（3%~80%剩余权重）的整个范围内均优于现有的一阶与零阶剪枝方法，同时可节省约一半的可训练参数。

## 研究问题与动机
1. **PLM 过参数化问题**：BERT 等预训练语言模型参数量庞大，直接部署/迁移成本高，剪枝是直观且有效的压缩手段。
2. **一阶方法仍需微调导致冗余**：Movement Pruning 等一阶方法已能基于梯度信息做出合理剪枝决策，但仍在训练过程中同步微调权重；作者指出，**一阶梯度信息本身足以驱动模型收敛到下游任务，微调并非必要**。
3. **一阶方法在低稀疏度下表现不佳**：现有 movement pruning 在剩余权重大于 25% 时明显劣于零阶方法（magnitude pruning），缺乏全稀疏范围竞争力。
4. **参数效率不足**：fine-tuning + pruning 同时进行时，需要保存子网和重要性分数两套参数；若无需微调，则每任务仅需保存一个二值 mask，大幅降低存储与传输成本。

## 核心贡献（创新点）
1. **发现微调对一阶剪枝是冗余的**：指出一阶剪枝本身已能将 PLM 适配至下游任务，无需再微调权重；这与 Sanh et al. (Movement Pruning) 的核心假设本质不同。
2. **提出 SMP（Static Model Pruning）框架**：冻结全部预训练权重，仅更新二值 mask θ_M；新引入参数仅为 mask，相比其他一阶方法少训练约一半参数（170M → 85M）。
3. **设计基于 BERT 架构感知的全局 mask 函数（S）**：按各权重矩阵的重要性总分比例分配稀疏度，而非全局排序或逐矩阵独立 Top-v，在保持与局部函数同等计算效率的同时显著提升性能。
4. **提出基于标签词嵌入的任务头初始化方法并冻结**：借鉴 prompt tuning 思路，用 label word 的 token 嵌入直接初始化分类头，替代从零训练的 head，避免 head 微调引入额外开销。
5. **揭示正则化项对高稀疏收敛的关键作用**：正则项使重要权重的得分倾向负值后仍能区分头部注意力，使剪枝结果更均匀地聚焦关键 head。

## 方法详解
### 4.1 重要性分数（Importance Score）
冻结 W，令 $W'_{i,j} = W_{i,j} M_{i,j}$，重要性分数更新为：
$$
S_{i,j}^{(T)} = -\alpha_s \, W_{i,j} \sum_{t<T} \left(\frac{\partial \mathcal{L}}{\partial W'_{i,j}}\right)^{(t)}
$$
- 保留权重（$W'_{i,j}=W_{i,j}$）：若 $W_{i,j} \cdot \frac{\partial \mathcal{L}}{\partial W'_{i,j}} < 0$，分数增加，即**梯度趋势使绝对值增大 → 重要性提升**。
- 剪除权重（$W'_{i,j}=0$）：梯度趋势使 0 靠近 $W_{i,j}$，该权重的得分被压低。
- 关键区别：原 Movement Pruning 中 $S_{i,j} = -\alpha_s \sum_t \frac{\partial \mathcal{L}}{\partial W_{i,j}^{(t)}} \cdot W_{i,j}^{(t)}$ 含时变权重；SMP 将 $W_{i,j}$ 视为常量，梯度只反映**趋势**。

### 4.2 Masking Function
- **Local（L）**：对每个权重矩阵独立取 $\text{Top}_v(\mathbf{S}_{(\cdot)}^l)$。
- **Global（G）**：对所有 85M 参数统一排序取 Top-v，计算开销大且实验显示性能较差（$W_U, W_V$ 占 60%+，挤压其他矩阵）。
- **S（本文提出）**：按矩阵为单位分配稀疏度
$$
v_{(\cdot)}^l = \frac{R(\mathbf{S}_{(\cdot)}^l) \cdot L}{\sum_{l'=1}^{L} R(\mathbf{S}_{(\cdot)}^{l'})} \cdot v, \quad R(\mathbf{S}) = \sum_{i,j} \sigma(S_{i,j})
$$
按各矩阵整体重要性比例分配剩余比例，无需全网络排序，计算高效且更符合 BERT 架构先验。

### 4.3 Task-Specific Head 初始化
用 label word 的 token 嵌入初始化分类头并冻结，例如 SST-2 用 "great"/"terrible" 嵌入；预测值为 $h_{[\text{CLS}]} e_{\text{great}}^T$。对比从零初始化再 fine-tune，性能更优。

### 4.4 训练目标
- **Cubic Sparsity Scheduling**：前 N 步稀疏度从 0 递增到 $v_f$，之后保持：
$$
v_t = v_f - v_f\left(1-\frac{t}{N}\right)^3, \quad t<N
$$
- **无蒸馏损失**：
$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_R \frac{v_t}{v_f} R(\mathbf{S}), \quad \lambda_R=400
$$
正则项使接近阈值的分数获得更大惩罚梯度，推动 mask 在达到目标稀疏时继续调整。
- **知识蒸馏（可选）**：增加 $\mathcal{L}_{\text{KD}} = D_{\text{KL}}(\mathbf{p}_s \| \mathbf{p}_t)$。

## 实验与结果
**数据集**：MNLI、QQP、SQuAD、GLUE（开发集）。

**模型**：bert-base-uncased、roberta-base；训练集上所有结果基于 5 个随机种子平均。

**超参**：Adam，lr=2e-2，$\lambda_R=400$，MNLI/QQP 12 epoch、SQuAD 10 epoch、batch=64；低稀疏任务 N=7 epoch，高稀疏 N=3500 steps。

**关键数值结果**（无蒸馏，$v$ = 剩余权重比例）：

| 方法 | 剩余权重 | MNLI M_acc/MM_acc | QQP ACC/F1 | SQuAD EM/F1 | 新参数/任务 |
|---|---|---|---|---|---|
| Movement | 10% | 79.3/79.5 | 89.1/85.5 | 71.9/81.7 | 8.5M+θ_M |
| Soft-Movement | 10% | 80.7/81.1 | 90.5/87.1 | 71.3/81.5 | 8.5M+θ_M |
| **SMP-L** | 10% | **82.0/82.3** | **90.8/87.7** | **75.0/84.3** | **θ_M** |
| **SMP-S** | 10% | **82.5/82.3** | **90.8/87.6** | **75.1/84.6** | **θ_M** |
| **SMP-S** | 3% | **80.9/81.1** | **90.3/87.1** | **70.9/81.4** | **θ_M** |
| Magnitude | 10% | 78.3/79.3 | 79.8/75.9 | 70.2/80.1 | 8.5M+θ_M |
| **SMP-S** | 50%+KD | **85.7/85.5** | **91.7/88.8** | **82.8/89.8** | **θ_M** |
| Full BERT FT | 100% | 84.5/84.9 | 91.4/88.4 | 80.4/88.1 | 110M |

- **最强结果**：SMP-S 在 MNLI 50%+KD 达 **85.7/85.5**，超过全模型微调（84.5/84.9）；3% 稀疏时 SQuAD F1 为 81.4，优于同等稀疏下的 soft-movement（79.9）。
- **参数效率**：SMP 每任务仅需保存 θ_M（≈2.7M），相比其他一阶方法在 50%/10%/3% 剩余权重下分别节省 42.5M / 8.5M / 2.6M 参数。
- **GLUE（80% 剩余权重，无蒸馏）**：bert-base SMP 均值 **83.9**（全模型 83.6），roberta-base SMP 均值 **86.9**（全模型 86.4），均超过 full-model fine-tuning。
- **Masking 函数对比**：G 函数因计算开销大且 $W_U/W_V$ 占比失衡导致性能低于 L；S 函数在同等计算量下优于 L。
- **正则项必要性**：去掉 R 后，SMP-L 在 SQuAD 10%/3% 无法收敛（N/A）。

## 相关工作脉络
1. **Movement Pruning (Sanh et al., 2020, NeurIPS)**：最早将一阶梯度用于 PLM 剪枝的代表工作，依赖 straight-through estimator 更新重要性分数并同步 fine-tuning；SMP 核心观点与其相反，认为无需 fine-tuning。
2. **Magnitude Pruning (Han et al., 2015, NeurIPS)**：零阶剪枝基线，基于权重绝对值；在低稀疏度下强于一阶方法，SMP 填补了该空白。
3. **L₀ Regularization (Louizos et al., 2017/2018, ICLR)**：通过 hard-concrete 分布采样学习稀疏 mask 的一阶方法；与 SMP 同属一阶家族，但 SMP 不使用可微 mask 采样策略。
4. **CAP (Xu et al., 2022, AAAI)**：在 teacher 模型上引入对比学习损失的剪枝方法；SMP 在无辅助学习目标的情况下持续优于 CAP。
5. **Super Tickets (Liang et al., 2021, ACL)**：利用彩票假设在 PLM 中搜索能超越全模型性能的子网，但需搜索多稀疏水平并 fine-tune；SMP 在相同稀疏设置下超越 SuperT 并节省 >98M 新参数。
6. **Lottery Ticket Hypothesis (Frankle & Carbin, 2018)**：理论源头；本文在 BERT 上验证了"存在无需微调即可匹敌全模型性能的稀疏子网"这一猜想。

## 局限性与未来方向
1. **非结构化剪枝，推理加速有限**：与所有 unstructured pruning 一样，难以直接利用稀疏硬件加速；需要后续结构化剪枝探索。
2. **扩展性受限**：由于不微调权重，SMP 难以直接迁移至 structured pruning 场景。
3. **推理压缩仍可优化**：作者观察到高稀疏下多数行被完全剪除，可将稀疏矩阵按行压缩（如 3% 稀疏模型可压缩至实际大小的 47.43%，推理加速约 1.37×），但进一步压缩需要专门设计的训练 loss。
4. **仅验证了 BERT/RoBERTa 类 PLM**：在更大规模模型（如 LLaMA 系列）上的泛化性有待验证。
5. **GLUE 上部分小任务（MRPC/RTE/COLA）仍有波动**：标准差相对较大（MRPC 1.21，RTE 1.59），稳定性有待提升。

## 研究启发与可借鉴点
1. **"静态 mask 可替代 fine-tuning"的核心假设值得广泛验证**：该思路可迁移至 CV（ViT）、多模态等大模型压缩场景，作为"不微调即剪枝"范式的起点。
2. **架构感知的 mask 分配策略（方法 S）**：按模块/层重要性比例分配稀疏度而非全局排序，兼顾效率与性能；可推广至任意分层架构（如 Swin Transformer、LLaMA）。
3. **标签词嵌入初始化冻结 head 的技巧**：结合 prompt tuning 思想，为下游任务快速搭建高质量、免微调的分类头，适用于各类少样本/零样本场景。
4. **正则化项对收敛的必要性发现**：SMP 中 $\lambda_R=400$ 的大正则系数对高稀疏收敛至关重要，提示"一阶剪枝无需微调"的成立条件需要合理的梯度惩罚机制。
5. **行级稀疏矩阵压缩后的推理加速**：高稀疏下按行压缩可在不重训的情况下获得实际加速，提示可将此后处理流程纳入工程实践。

## 关键术语表
**Static Model Pruning (SMP)**：论文提出的方法，冻结全部预训练权重，仅通过一阶梯度更新二值 mask 完成剪枝，不依赖 fine-tuning。

**Importance Score (S)**：累积梯度趋势的重要性评分，SMP 中冻结 W 使得 $S_{i,j}$ 直接反映梯度方向对 $W_{i,j}$ 绝对值的推动趋势。

**Straight-through Estimator**：Bengio et al. (2013) 提出的可微近似技巧，允许通过不可微的离散操作（如 thresholding）反向传播梯度。

**Cubic Sparsity Scheduling**：按 $(1 - t/N)^3$ 的三次曲线从稀疏度 0 渐进增至目标稀疏度 $v_f$ 的策略。

**Task-Specific Head Initialization**：用 label word 的 token 嵌入初始化分类头并冻结，避免从零训练带来的性能损失。

**Knowledge Distillation (KD)**：用教师模型（完整 PLM）的输出分布监督学生模型（剪枝后模型），进一步提升剪枝后性能。

**Masking Function**：将重要性分数映射为二值 mask 的规则，包括 local、global 和本文提出的 S 三种。

**Regularization Term R(S)**：$\sum \sigma(S_{i,j})$，惩罚重要性分数，促使更多权重得分趋向负值（被剪除），是高稀疏收敛的关键。

## 可复现要素
- **数据集**：MNLI、QQP、SQuAD、GLUE 开发集；均为公开基准。
- **代码**：已开源，地址为 https://github.com/kongds/SMP。
- **模型**：bert-base-uncased、roberta-base（公开模型）。
- **关键超参**：Adam lr=2e-2，$\lambda_R=400$，batch=64；MNLI/QQP 12 epoch，SQuAD 10 epoch；低稀疏 N=7 epoch，高稀疏 N=3500 steps。
- **随机种子**：5 次运行取平均。
- **是否公开**：论文声明代码已开源；权重未单独发布（使用公开 PLM）。
