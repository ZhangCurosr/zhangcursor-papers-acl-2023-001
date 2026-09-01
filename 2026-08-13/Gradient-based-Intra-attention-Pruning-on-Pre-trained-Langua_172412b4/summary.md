---
title: "Gradient-based-Intra-attention-Pruning-on-Pre-trained-Langua"
source: https://aclanthology.org/2023.acl-long.156.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:22:22"
field: "自然语言处理模型压缩"
keywords: ["模型压缩", "结构化剪枝", "知识蒸馏", "预训练语言模型", "梯度剪枝", "注意力剪枝", "模型加速"]
innovations: ["提出 intra-attention 行级剪枝以扩展结构搜索空间并支持异质头尺寸", "设计结构正则化（StructReg）抑制碎片化以获得更低推理延迟", "提出梯度分离策略（GradSep）解耦蒸馏与剪枝的重要性评分"]
benchmarks: ["GLUE", "SQuAD 1.1", "CoNLL 2003", "OCNLI", "TNEWS", "CMRC 2018", "DRCD"]
---

# 论文速读：Gradient-based Intra-attention Pruning on Pre-trained Language Models

## 一句话总结
论文提出了 GRAIN（Gradient-based Intra-attention pruning），一种针对预训练语言模型的细粒度结构化剪枝方法，通过在注意力头内部剪枝（而非整头剪枝）并结合任务特异性知识蒸馏，在保持高性能的同时实现 6~7× 加速；极端压缩（仅保留 3% Transformer 权重）时仍能超越更大基线模型。

## 研究问题与动机
1. 预训练语言模型（PLM）参数庞大、推理延迟高，难以直接部署到资源受限场景，亟需高效的压缩方法。
2. 现有结构化剪枝方法（如整头注意力剪枝、FFN 层剪枝）的剪枝单元过于粗粒度，搜索空间有限，难以找到更优的子结构。
3. 细粒度剪枝（如 intra-attention 剪枝）会产生大量大小不一的小头（fragmented 结构），在 GPU 上难以并行，导致实际推理延迟较高。
4. 将知识蒸馏与梯度剪枝联合训练时，隐藏层匹配损失（L_Hidden）的梯度会干扰基于梯度的重要性评分估计，两者存在冲突，需设计解耦机制。

## 核心贡献（创新点）
1. **Intra-attention Pruning**：提出将注意力头内部的 query/value 矩阵行作为剪枝单元，而非将整头视为原子单元，大幅扩展了模型结构搜索空间；与仅剪整头的传统方法本质不同，允许每个头保持不同尺寸，实现更灵活的结构学习。
2. **Structure Regularization（StructReg）**：引入基于模块密度的正则化项，优先剪掉小头中的单元直至其清空后移除，从而抑制 fragmented 结构、获得更少且更大的注意力头，在加速与性能之间取得更好平衡；与无正则化内层剪枝相比，这是关键的工程修正。
3. **Gradient Separation（GradSep）**：提出将 L_CE 与 L_Hidden 的梯度分离使用——重要性评分和参数更新仅用 L_CE 梯度，L_Hidden 梯度仅用于模型优化，避免蒸馏目标干扰剪枝重要性估计；与直接将两个损失梯度混合使用的做法本质不同。
4. **迭代式梯度剪枝 + 立方密度调度**：将经典的一阶梯度剪枝与 iterative pruning 范式结合，使用指数平滑重要性评分和 cubic schedule 渐进压缩，无需额外预训练或数据增强即可实现高效任务特异性压缩；与静态一次性剪枝方法相比显著提升了低密度下的最终性能。
5. **Embedding 矩阵 SVD 分解**：在 Transformer 内部剪枝基础上，对词嵌入矩阵做 SVD 压缩，进一步减少总参数量（论文中该部分对极致压缩有帮助，但对延迟影响较小）。

## 方法详解
- **剪枝单元定义**：将每个 attention head 的 W^Q、W^K、W^V、W^O 四个矩阵的行视为独立可剪单元——query 单元对应 W^Q_i、W^K_i 的行，value 单元对应 W^V_i、W^O_i 的行，每个单元含 2d 参数；同一头内 query 维度可与 value 维度不同，进一步拓展结构自由度。
- **重要性评分**：沿用 Michel et al. (2019) 的梯度重要性公式 IS(w) = E[|∂L/∂w · w|]，用单 batch 在当前步计算并在训练过程中做指数平滑：IS̄_i(w) = β · IS̄_{i-1}(w) + (1-β) · IS_i(w)，β=0.998。
- **StructReg**：正则化重要性分数 IS^r(w) = IS(w) · tanh(D(M,W)/α)，其中 D(M,W) 是模块 M 内剩余剪枝单元的比例；α 为正则强度（实验取 0.3）。密度越低的头其单元得分被压得越低，从而被优先清空移除，最终得到少量大头的规则结构。
- **GradSep**：计算重要性评分时只用 L_CE 的梯度，L_Hidden 的梯度不参与 IS 估计；两路梯度均用于参数更新，从而实现蒸馏与剪枝的解耦。
- **迭代裁剪调度**：采用 cubic density scheduler，训练过程分三阶段——warm-up 阶段（0~p_s，p_s=0.2）仅蒸馏不剪枝；裁剪阶段（p_s~p_e，p_e=0.4）按立方曲线逐步降低密度至目标 s_f；收尾阶段（p_e~1）固定结构继续蒸馏恢复性能。
- **Embedding Factorization**：对词嵌入矩阵 E∈R^{q×d} 做 SVD：E≈U_rΣ_rV_r，将原 qd 参数压缩为 (q+d)r，实验取 r=192。

## 实验与结果
- **数据集**：GLUE 四项（SST-2、QNLI、MNLI、QQP）、SQuAD 1.1、CoNLL 2003，以及 Appendix 中的中文任务（OCNLI、TNEWS、CMRC 2018、DRCD）。
- **基线**：TinyBERT₄、AutoTinyBERT、Block Pruning、CoFi、DynaBERT、MobileBERT。
- **关键结果（BERT_base 基座，5% 密度）**：
  - QNLI Acc: 89.0（最佳），SST-2 Acc: 91.4，MNLI m/mm: 82.2/82.5，QQP Acc: 90.4，SQuAD F1/EM: 83.6/73.7，CoNLL-03 F1: 88.3；全部优于 CoFi/Block Pruning/TinyBERT₄ 相应基线。
  - 与 TinyBERT₄（4.7M）相比，GRAIN（4.3M）在多数任务上更高或持平。
- **极端压缩（3% 密度）**：仅 2.6M Transformer 参数，QNLI 87.8、MNLI 80.7/81.1、SQuAD 79.5/68.4，仍能匹敌甚至超越参数量更大的 TinyBERT/CoFi。
- **加速**：在 NVIDIA M40、batch=128、seq_len=512 上，配合 StructReg（α=0.3）实现 6~7× 加速，相比无正则化的 ~4× 有显著提升。
- **消融（5% 密度，BERT 基座）**：去掉 GradSep 使 SQuAD F1 从 83.6 降至 82.8；去掉 L_Hidden 则性能大幅下降（SQuAD 80.3）；随机剪枝仅得 SQuAD 65.7，证明梯度剪枝必要性。
- **RoBERTa-base 扩展实验**：GRAIN-R 在 10%/20% 密度下表现优于 GRAIN-BERT，但在 5%/3% 极端密度下 BERT 基座更优，提示 teacher 选择与压缩率存在交互。

## 相关工作脉络
1. **Michel et al. (2019) Gradient-based head pruning**：整头梯度剪枝；本文将其推广到头内行级单元，并配合迭代训练与蒸馏。
2. **Sanh et al. (2020) Movement Pruning**：迭代稀疏化剪枝；本文借鉴迭代思想并改用一阶梯度重要性 + 蒸馏联合训练。
3. **Xia et al. (2022) CoFi**：结构化剪枝 + SVD 嵌入 + 蒸馏；本文与其主要差异在于剪枝粒度更细（intra-attention vs. 整头/整层）以及 GradSep 设计。
4. **Lagunas et al. (2021) Block Pruning**：块级非完全结构化剪枝；本文明确指出 Block Pruning 无法达到大规模结构化加速，从而聚焦 fully structured intra-attention 剪枝。
5. **Jiao et al. (2020) TinyBERT / Sun et al. (2020) MobileBERT**：蒸馏型小模型；本文强调自身无需通用预训练和数据增强，仅做任务级微调即可压缩，计算成本更低。
6. **Zhu & Gupta (2018) Magnitude Pruning + cubic schedule**：迭代裁剪调度基础；本文直接沿用 cubic schedule 与 warmup-preserve-fine-tune 三阶段策略。

## 局限性与未来方向
- **推理延迟偏高**：相同参数量下 GRAIN 在不同任务上的延迟高于 CoFi/TinyBERT，因各头尺寸不一、GPU 无法充分并行；可通过高层结构正则化或将相似尺寸头合并成大矩阵缓解。
- **仅面向 Transformer**：方法围绕标准 multi-head attention 设计，在其它注意力变体或非 Transformer 架构上的有效性未验证。
- **Embedding 因子化的收益不恒定**：论文表明嵌入压缩并非总是带来性能提升（如 SST-2 上保留大嵌入更好），需要按任务和密度权衡。
- **未来可探索方向**：扩展到动态推理（dynamic inference）、中文等多语言场景、以及更大比例稀疏（<3%）下的鲁棒性。

## 研究启发与可借鉴点
1. **Intra-attention 行级剪枝**：可将"把整头视为原子"的假设打破，在更细粒度（行/列）层面重新审视注意力权重的可剪性，适用于任何多头注意力架构。
2. **Gradient Separation 思想**：当多个训练目标（如主任务 loss + 蒸馏 loss + 正则 loss）共享参数更新时，可通过分离梯度来源来避免某一路目标污染另一路的统计量（如重要性评分、搜索信号），这一思路可迁移到多任务剪枝/蒸馏联合框架。
3. **StructReg 的密度感知正则**：以模块剩余密度 tanh(D/α) 作为乘性缩放因子引导优先剪空稀疏模块，是一种简单且可插拔的结构先验，可推广到 FFN 子层、序列位置、甚至深度方向的结构学习。
4. **Cubic schedule + 指数平滑 IS**：轻量且有效的迭代剪枝工程套路，可直接复用到其他基于梯度的结构化剪枝 pipeline 中。
5. **Teacher 与压缩率的交互**：实验发现 BERT 在极端压缩下有时优于 RoBERTa，提示在超极限压缩场景下应选择更"易压缩"的 teacher 而非精度最高的 teacher，这一经验对后续模型选择有参考价值。

## 关键术语表
- **Intra-attention pruning**：在单个注意力头的 Q/K/V/O 矩阵内部按行剪枝，而非整头删除，从而支持不同头保持不同尺寸。
- **Structure Regularization (StructReg)**：以模块内剩余单元密度对重要性评分做 tanh 缩放的正则化策略，促使稀疏头被优先清空并移除，降低模型碎片化程度。
- **Gradient Separation (GradSep)**：将 L_CE 与 L_Hidden 的梯度分离使用——IS 估计只用 L_CE 梯度，避免蒸馏信号干扰剪枝重要性排序。
- **Cubic density scheduler**：按 t^3 曲线在裁剪阶段逐渐降低模型密度，配合 warmup 与收尾微调三阶段训练。
- **Model density**：剪枝后模型尺寸与原始模型尺寸的比值，sparsity = 1 − density。
- **Fragmented structure**：大量小尺寸注意力头并存的状态，GPU 并行效率低，是细粒度剪枝的负面副作用。
- **Embedding Factorization**：对词嵌入矩阵做 SVD 分解以压缩 embedding 参数量（qd → (q+d)r），不影响 attention 结构。
- **Task-specific distillation**：仅在目标任务数据上进行的蒸馏，无需通用预训练数据或数据增强，计算成本较低。

## 可复现要素
- **数据集**：GLUE、SQuAD 1.1、CoNLL 2003、OCNLI、TNEWS、CMRC 2018、DRCD（均为公开数据集）。
- **代码/权重**：论文未提供明确开源仓库链接，作者单位 iFLYTEK 实验环境使用 PyTorch 1.8.1、Transformers 4.10.0、CUDA 10.2、单卡 V100；Appendix A 给出了完整超参数（Table 4），包括 lr=3e-5（GLUE/SQuAD）、lr=1e-4（CoNLL）、epochs=20/40、β=0.998、α=0.3、p_s=0.2、p_e=0.4、r=192、τ=8、batch=32。
- **关键技术声明**：无特殊闭源脚本或私有数据；训练耗时约 1–15 小时/任务。
- 论文未提及额外公开代码仓库链接。
