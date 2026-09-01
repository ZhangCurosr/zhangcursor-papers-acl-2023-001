---
title: "Pruning-Pre-trained-Language-Models-Without-Fine-Tuning"
source: https://aclanthology.org/2023.acl-long.35.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:51:10"
field: "模型压缩与剪枝"
keywords: ["模型剪枝", "预训练语言模型", "无微调适配", "一阶剪枝", "彩票假设", "模型压缩"]
innovations: ["提出SMP方法，冻结预训练权重仅训练二元掩码，无需微调即可完成下游任务适配", "设计基于矩阵分组的全局Masking函数SMP-S，按矩阵重要性比例分配稀疏度", "发现一阶剪枝无需微调即可收敛，在80%保留权重下超越完整模型微调"]
benchmarks: ["MNLI", "QQP", "SQuAD", "GLUE"]
---

# 论文速读：Pruning Pre-trained Language Models Without Fine-Tuning

## 一句话总结
本文提出静态模型剪枝（SMP），一种仅通过一阶剪枝即可将预训练语言模型适配到下游任务的新方法，无需对权重进行微调。实验表明，SMP 在不同稀疏度下均显著优于现有的一阶和零阶剪枝方法，并在低稀疏度（如 80% 保留权重）下实现了超过完整模型微调的性能。

## 研究问题与动机
1. **预训练语言模型（PLMs）参数量过大**，导致下游任务的传输和存储开销巨大，需要有效的压缩方法。
2. **现有剪枝方法依赖微调**：一阶方法（如 Movement Pruning）在剪枝的同时需要对剩余权重进行微调，计算开销大且参数冗余。
3. **零阶方法在高稀疏度下表现差**：Magnitude Pruning 基于权重绝对值剪枝，在高稀疏度下性能急剧下降。
4. **一阶方法在低稀疏度下失效**：Movement Pruning 等一阶方法仅在极高稀疏度下表现良好，在保留 25% 以上权重时反而不如零阶方法。

## 核心贡献（创新点）
1. **提出 Static Model Pruning (SMP)，仅用一阶剪枝适配下游任务，无需微调**：与传统方法在剪枝的同时微调权重不同，SMP 冻结全部预训练权重，只训练一个二元掩码，将可训练参数量减少近一半。
2. **设计基于矩阵分组的新型全局 Masking 函数（SMP-S）**：按各权重矩阵的整体重要性分配稀疏度，而非对全网络 85M 参数统一排序，避免了全局排序的计算开销并改善性能。
3. **提出基于标签词嵌入的任务特定头初始化方法并保持冻结**：借鉴 Prompt Tuning 思想，使用标签词的 token 嵌入初始化分类头并冻结，避免了从头训练的负面影响。
4. **系统性揭示一阶剪枝无需微调即可收敛的发现**：论证一阶信息本身已足够驱动模型收敛到下游任务，颠覆了"剪枝必须配合微调"的既有认知。

## 方法详解

**核心思想**：冻结预训练权重 $\mathbf{W}$，仅训练二元掩码 $\mathbf{M}$，通过一阶剪枝完成下游任务适配。

**重要性分数（Importance Score）**：基于 Movement Pruning 改进，因 $\mathbf{W}$ 冻结，$W_{i,j}$ 为常数，公式简化为：
$$S_{i,j}^{(T)} = -\alpha_s \cdot W_{i,j} \sum_{t<T} \left(\frac{\partial \mathcal{L}}{\partial W'_{i,j}}\right)^{(t)}$$
其中 $W'_{i,j} = W_{i,j} M_{i,j}$。当 $W_{i,j} \cdot \frac{\partial \mathcal{L}}{\partial W'_{i,j}} < 0$ 时，$S_{i,j}$ 增加。

**两种 Masking 函数**：
- **局部 Masking（SMP-L）**：对每个权重矩阵独立应用 $\text{Top}_v$ 操作，选择该矩阵内最重要的 $v\%$ 权重。
- **全局 Masking（SMP-S，本文提出）**：按矩阵类型分配稀疏度：
$$v_{(\cdot)}^l = \frac{R(\mathbf{S}_{(\cdot)}^l) \cdot L}{\sum_{l'=1}^{L} R(\mathbf{S}_{(\cdot)}^{l'})} \cdot v$$
其中 $R(\mathbf{S}) = \sum_{i,j} \sigma(S_{i,j})$，$v_{(\cdot)}^l$ 为第 $l$ 层第 $(\cdot)$ 类矩阵的稀疏度。

**任务特定头初始化**：使用标签词（如 SST2 中的 "great"/"terrible"）的 token 嵌入初始化分类头，预测分数为 $h_{[\text{CLS}]} e_{\text{label}}^T$，训练过程中保持冻结。

**训练目标**：
- 无知识蒸馏：$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_R \cdot \frac{v_t}{v_f} R(\mathbf{S})$，其中正则项 $R(\mathbf{S})$ 促进稀疏化，$\lambda_R = 400$。
- 有知识蒸馏：额外添加 $\mathcal{L}_{\text{KD}} = D_{\text{KL}}(\mathbf{p}_s \| \mathbf{p}_t)$。

**稀疏调度**：采用三次方调度 $v_t = v_f - v_f(1 - t/N)^3$，从 0 逐步增至目标稀疏度 $v_f$。

## 实验与结果

**数据集与模型**：MNLI、QQP、SQuAD、GLUE 基准；使用 bert-base-uncased 和 roberta-base。

**基线方法**：Magnitude Pruning、L₀-Regularization、Movement Pruning、Soft-Movement Pruning、CAP、Super Tickets。

**核心结果（无知识蒸馏，Table 1）**：
- 10% 保留权重：SMP-S 在 MNLI 上达到 82.5/82.3（M_ACC/MM_ACC），比 Movement Pruning（79.3/79.5）提升 **3.0+ 分**；SQuAD F1 达 84.6，比 Movement Pruning 提升 **3.0+ 分**。
- 3% 保留权重：SMP-S 在 MNLI 上达到 81.8/82.0，**超越** Soft-Movement Pruning（79.5/80.1）近 **2 分**。
- 50% 保留权重 + 知识蒸馏：SMP-S 在 MNLI 上达到 **85.7**，超过完整 BERT 微调的 84.5。

**GLUE 结果（80% 保留权重，Table 2）**：
- BERT：SMP 平均 **83.9**，超过完整微调的 83.6。
- RoBERTa：SMP 平均 **86.9**，超过完整微调的 86.4。
- 相比 Super Tickets，SMP 节省超过 **98M** 新参数/任务。

**参数效率**：SMP 仅需训练约 2.7M 的二元掩码 $\theta_M$，而一阶方法需训练 170M 参数（含权重 + 重要性分数）。

**消融实验**：SMP-S 的掩码函数显著优于全局排序（G，无法收敛或更慢）和局部掩码（L）；正则项 R 对高稀疏度收敛至关重要（无 R 时在 3% 稀疏度下 SQuAD 无法收敛）；头初始化优于从头训练。

## 相关工作脉络
1. **Movement Pruning (Sanh et al., 2020)**：一阶剪枝的开山工作，基于梯度信息评估权重重要性并同时微调；SMP 与其核心区别是**完全冻结权重、不微调**。
2. **Magnitude Pruning (Han et al., 2015)**：零阶剪枝，基于权重绝对值排序；SMP 在一阶基础上改进，在高稀疏度下显著超越，且在低稀疏度下仍优于一阶方法。
3. **L₀-Regularization (Louizos et al., 2017/2018)**：通过 L₀ 范数正则化实现稀疏；属于一阶/可微分剪枝路线，需要同时优化权重和掩码。
4. **CAP (Xu et al., 2022)**：引入对比学习目标的剪枝方法，依赖教师模型；SMP 无需辅助学习目标即超越 CAP。
5. **Super Tickets (Liang et al., 2021)**：基于彩票假设寻找优于完整模型的特殊子网并微调；SMP 无需搜索多个稀疏度、无需微调即可达到更好效果。
6. **Lottery Ticket Hypothesis (Frankle & Carbin, 2018; Chen et al., 2020)**：证明 PLM 中存在无需训练即可匹配的稀疏子网；本文进一步证明**甚至无需微调即可匹配/超越完整模型性能**。

## 局限性与未来方向
1. **非结构化剪枝难以实现推理加速**：SMP 为非结构化剪枝，与结构化剪枝相比难以直接加速推理；但作者发现高稀疏度下矩阵大量行为全零，可通过压缩矩阵尺寸获得约 **1.37× 推理加速**（3% 稀疏度）。
2. **难以扩展到结构化剪枝**：无微调的设计限制了向结构化剪枝的迁移。
3. **未来方向**：设计专门的损失函数以实现更小的实际模型尺寸和更快的推理加速（如利用矩阵行稀疏性进一步压缩）。

## 研究启发与可借鉴点
1. **"冻结权重 + 只训练掩码"范式可迁移**：SMP 证明了在某些场景下微调是冗余的，这一思路可迁移到其他预训练模型（如 ViT、T5）的压缩任务中，减少部署成本。
2. **基于矩阵分组的全局 Masking 函数设计**：SMP-S 按矩阵类型比例分配稀疏度的思路，为结构化剪枝中的稀疏度分配提供了参考，可避免全局排序的计算开销。
3. **标签词嵌入初始化任务头**：借鉴 Prompt Tuning 思想的冻结式头初始化方法，可与适配器（Adapter）、前缀微调等方法结合，进一步探索免微调的下游适配方案。
4. **正则项对一阶剪枝收敛的关键作用**：$\lambda_R = 400$ 的大正则系数确保了高稀疏度下的收敛，这一发现提示一阶剪枝的损失函数设计中正则项不可忽视。
5. **彩票假设的强化验证**：在 80% 保留权重下，SMP 即可超越完整模型微调，为彩票假设在 PLM 中的有效性提供了更强证据，可启发后续关于"最小有效子网"的研究。

## 关键术语表
- **Static Model Pruning (SMP)**：本文提出的方法，通过冻结预训练权重、仅训练二元掩码完成模型剪枝与任务适配，无需微调。
- **Movement Pruning**：基于梯度趋势的一阶剪枝方法，根据权重在训练过程中绝对值的变化趋势评估重要性。
- **Masking Function**：将重要性分数映射为二元掩码的函数，分为局部（按矩阵独立排序）和全局（跨矩阵分配稀疏度）两类。
- **Lottery Ticket Hypothesis (LTH)**：彩票假设，指预训练模型中存在稀疏子网，单独训练即可达到与完整模型相当的性能。
- **Knowledge Distillation**：知识蒸馏，利用教师模型（完整模型）的输出分布指导剪枝后学生模型的训练。
- **Task-Specific Head**：任务特定头，位于编码器之后的分类/回归层，本文用标签词嵌入初始化并保持冻结。
- **Cubic Sparsity Scheduling**：三次方稀疏调度，从 0 逐步将稀疏度增加到目标值 $v_f$ 的调度策略。
- **Unstructured Pruning**：非结构化剪枝，以单个权重为粒度进行剪枝，与结构化剪枝（以通道/层为粒度）相对。

## 可复现要素
- **数据集**：MNLI、QQP、SQuAD、GLUE（均为公开基准数据集）
- **代码**：已开源，地址 https://github.com/kongds/SMP
- **模型权重**：bert-base-uncased、roberta-base（公开预训练权重）
- **关键超参**：优化器 Adam，学习率 2e-2，正则化系数 $\lambda_R = 400$，批次大小 64；MNLI/QQP 训练 12 轮，SQuAD 训练 10 轮；低稀疏度任务 $N = 7$ 轮，高稀疏度 $N = 3500$ 步
- **稀疏调度**：三次方调度，从无Warmup
