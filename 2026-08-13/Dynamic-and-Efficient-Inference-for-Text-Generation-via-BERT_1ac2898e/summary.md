---
title: "Dynamic-and-Efficient-Inference-for-Text-Generation-via-BERT"
source: https://aclanthology.org/2023.acl-long.162.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:18:56"
field: "高效文本生成与模型压缩"
keywords: ["非自回归生成", "动态剪枝", "BERT族模型", "神经机器翻译", "文本生成"]
innovations: ["CTC生成器与Levenshtein编辑器的联合训练框架实现首步初始化与迭代精化", "动态块剪枝支持推理时自适应调整模型大小", "BERT族模型经特定输入掩码设计适配NAR文本生成任务"]
benchmarks: ["IWSLT'14 De-En", "WMT'16 En-Ro", "WMT'14 En-De", "XSum", "MSNews", "SQuAD 1.1", "MSQG"]
---

# 论文速读：Dynamic-and-Efficient-Inference-for-Text-Generation-via-BERT

## 一句话总结
本文提出 DEER 方法，通过联合利用非自回归（NAR）生成与动态参数剪枝技术，使单个预训练的 BERT 族模型支持自适应性能-延迟权衡，在机器翻译和文本生成任务上实现 3-12 倍加速的同时达到或超越 AR 基线模型的性能。

## 研究问题与动机
1. **自回归解码效率低下**：主流生成模型（如 BART、T5）采用自回归增量解码范式，无法并行化，导致推理计算和内存效率低。
2. **多设备部署困难**：不同边缘设备（手机、机器人、自动驾驶等）的内存容量和延迟约束差异大，需要为每种设备单独训练不同架构的模型，造成额外资源消耗和碳排放。
3. **现有压缩方法局限**：当前研究集中于知识蒸馏、量化和参数剪枝，对预训练非自回归生成范式的探索有限，且多数剪枝方法仅寻找固定稀疏度的高性能子模型，缺乏自适应调节模型大小的能力。
4. **编码器模型用于生成的结构性障碍**：BERT 族模型的双向注意力机制使其难以直接应用于文本生成任务，需要特殊的设计来弥补这一缺陷。

## 核心贡献（创新点）
1. **首次系统性探索 BERT 族模型用于 NAR 文本生成**：利用双向注意力更适合 NAR 训练目标的特点，通过设计特定输入格式和自注意力掩码，使编码器模型无需额外参数即可支持文本生成。
2. **CTC 生成器 + Levenshtein 编辑器的联合训练框架**：结合单步 CTC 对齐和迭代式插入/删除/占位符分类器，既获得优于迭代方法的首步初始化，又可通过多步精化进一步提升性能。
3. **动态块剪枝支持自适应模型大小**：引入基于分数的参数掩码和稀疏度正则化，实现在训练中动态选择模型大小而非固定稀疏度，推理时可根据设备阈值灵活调整参数比例。
4. **无需两阶段微调的一阶段训练**：与现有剪枝方法不同，DEER 在一阶段训练中同时完成生成能力和剪枝适应性的联合优化，避免了为子模型单独微调的成本。

## 方法详解
1. **单步 CTC 生成器**：将源句子 X 复制均匀作为伪目标 V̂，通过特定自注意力掩码控制源-目标信息交互（V̂ 可 attend X，但 X 不能 attend V̂），利用 CTC 损失建模隐式对齐，公式为 L_CTC = -log P(Y|X)，其中 P(Y|X) 通过动态规划对所有可能对齐路径求和得到。
2. **迭代式 Levenshtein 编辑器**：共享 CTC 模型参数，通过三个分类器进行文本编辑：占位符分类器预测插入 token 数量、插入分类器预测缺失 token、删除分类器判断是否需要删除当前 token；训练时随机删除目标序列 token 作为初始状态，推理时将 CTC 结果依次输入三类分类器。
3. **动态块剪枝**：在每个前向传播中引入基于分数的参数掩码 M(s)，通过 straight-through estimator 计算重要性分数 s；为 self-attention 层使用 32×32 方形块，为 feed-forward 层使用维度块（1×d 和 d×1）；设置两个全局阈值分别控制两类层的稀疏度；引入 L1 正则化项 L_reg = λ||σ(s)|| 鼓励稀疏性，λ=10。
4. **联合训练算法**：训练目标为 L = L_CTC + L_PLH + L_INS + L_DEL + L_reg，每步随机采样模型大小（25%/50%/75%/100%），切换自注意力掩码和输入格式分别训练 CTC 生成器和 Levenshtein 编辑器。

## 实验与结果
- **数据集**：三个 NMT 任务（IWSLT'14 De→En、WMT'16 En→Ro、WMT'14 En→De）和四个 GLGE 基准任务（XSum、MSNews、SQuAD 1.1、MSQG）。
- **基线模型**：Transformer、GLAT、CMLM、Levenshtein、BART、ProphetNet、BANG、ELMER 等。
- **NMT 结果**：DEER (KD) 在 IWSLT'14 De→En 上以 4 步迭代达到 37.46 BLEU，超过 Transformer 的 35.05（约 2.4 BLEU 提升）；在 WMT'16 En→Ro 上达到 35.53 vs 28.3（Transformer KD），速度提升 3-12 倍。
- **文本生成结果**：DEER 在 XSUM 上 ROUGE-L 达 32.4（100% 参数），速度提升 9.3×；在 SQuAD 1.1 上 RL/B-4/MTR 达 49.9/20.3/24.0，优于 ProphetNet（48.0/19.5/23.9）。
- **动态剪枝效果**：即使将 RoBERTa-base 参数缩减至 50%，在 XSUM 上仍保持 35.7/14.0/29.8 的 ROUGE 分数；与 Scalable Transformer 相比，在相同内存约束下 DEER 的子模型性能更优。
- **消融实验**：移除 Levenshtein 编辑器后单步性能下降约 3 BLEU；移除 CTC 后首步性能显著退化（Raw 数据从 35.49 降至 18.02）；去掉稀疏正则化后大剪枝比例下性能急剧下降。

## 相关工作脉络
1. **Movement Pruning（Sanh et al., 2020）**：引入灵活参数掩码通过评分获取显著权重；DEER 的区别在于支持动态调整模型大小而非固定稀疏度目标。
2. **DynaBERT（Hou et al., 2020）**：允许自适应宽度和深度满足边缘设备需求；DEER 结合运动剪枝和动态训练，且专注于生成任务。
3. **CTC 非自回归翻译（Libovicky & Helcl, 2018）**：将 CTC 引入单步 NAR 框架；DEER 在此基础上结合 Levenshtein 编辑器进行迭代精化。
4. **Levenshtein Transformer（Gu et al., 2019）**：通过插入/删除操作支持序列精化；DEER 将其与 CTC 联合训练，首步结果优于纯迭代方法。
5. **XLM-D（Wang et al., 2022）**：探索隐式对齐和预训练模型用于 NAR 生成；DEER 使用不同的方法和架构，并额外引入动态剪枝。
6. **Scalable Transformer（Gao et al., 2021）**：通过参数剪枝获得多个子模型；DEER 在相同内存约束下子模型性能更优。

## 局限性与未来方向
1. **隐式对齐的多模态问题**：CTC 等隐式对齐模型难以处理大规模数据集的多模态对齐，导致 DEER 在高资源 RAW 数据上表现不佳（如 WMT'14 En→De）。
2. **长度控制的灵活性受限**：虽然不需要显式长度预测，但隐式对齐假设要求源句子长度大于目标，限制了模型在长度控制上的灵活性。
3. **多步迭代的延迟累积**：BERT 族模型需要 12 层前向传播，而 BART 仅需 6 层，多次迭代时延迟累积可能抵消 NAR 的部分加速优势。
4. **未来方向**：优先解决长度预测挑战，使方法更适用于更广泛的任务和场景。

## 研究启发与可借鉴点
1. **编码器模型的 NAR 生成适配**：通过设计特定输入格式和自注意力掩码，可有效弥补编码器模型的结构性缺陷，这一思路可迁移至其他编码器架构的生成任务。
2. **CTC 初始化 + 迭代精化的两阶段联合训练**：先用 CTC 获得良好的首步初始化，再通过 Levenshtein 编辑器迭代优化，兼顾了速度和精度，该策略可应用于其他序列生成任务。
3. **动态块剪枝与生成任务结合**：将 Movement Pruning 的分数机制与生成模型联合训练，实现一阶段自适应模型大小选择，避免了两阶段微调开销，值得在其他模型压缩场景中验证。
4. **稀疏正则化的关键作用**：实验证明稀疏正则化对于大剪枝比例下的性能保持至关重要，这一发现在设计其他剪枝方案时应予以重视。

## 关键术语表
**DEER**：Dynamic and Efficient infERence 的缩写，本文提出的联合 NAR 生成与动态剪枝的 fine-tuning 方法。
**CTC（Connectionist Temporal Classification）**：用于建模源和目标序列间隐式对齐的无监督对齐损失函数，无需显式长度预测。
**Levenshtein Transformer**：通过插入、删除操作支持序列迭代精化的 NAR 生成模型。
**Movement Pruning**：通过训练过程中引入参数掩码和重要性评分，自适应学习稀疏子模型的剪枝方法。
**NAR（Non-autoregressive）**：非自回归生成，所有 token 并行生成或迭代修正，相比 AR 模型推理速度更快。
**GLGE**：中文通用语言生成评估基准，包含摘要生成和问题生成等四个子任务。
**KD（Knowledge Distillation）**：知识蒸馏，使用强教师模型（如 BART）生成训练数据以缓解 NAR 模型的多模态问题。

## 可复现要素
- **数据集**：IWSLT'14 De→En、WMT'16 En→Ro、WMT'14 En→De、XSum、MSNews、SQuAD 1.1、MSQG（均公开可用）
- **代码/权重**：论文声明代码将在 GitHub 公开（URL: https://github.com/...，原文未给出完整链接）
- **关键超参**：Adam 优化器，初始学习率 5e-5，polynomial_decay 学习率调度，label smoothing 0.1，稀疏正则化系数 λ=10，batch size=1，目标稀疏度 {25%, 50%, 75%}，阈值更新间隔 200 步，12 层 encoder，12 head，hidden size 768，FFN dim 3072，dropout 0.1
- **训练平台**：单张 NVIDIA V100 GPU，使用 Fairseq 库实现
