---
title: "A-Gradient-Control-Method-for-Backdoor-Attacks-on-Parameter"
source: https://aclanthology.org/2023.acl-long.194.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:46:16"
field: "大语言模型安全与鲁棒性"
keywords: ["parameter-efficient tuning", "backdoor attack", "gradient control", "multi-task learning", "backdoor forgetting"]
innovations: ["将PET后门注入建模为多任务学习，揭示梯度幅值失衡与方向冲突导致后门遗忘", "提出CLNorm实现跨层梯度幅值归一化以均衡各层后门贡献", "提出ILProj通过分层可控投影消除任务间梯度方向冲突"]
benchmarks: ["SST-2", "IMDB", "Enron", "Lingspam"]
---

# 论文速读：A Gradient Control Method for Backdoor Attacks on Parameter-Efficient Tuning

## 一句话总结
本文针对**参数高效微调（PET）场景**中的后门攻击脆弱性，提出一种基于**梯度控制**的后门注入方法，通过**跨层梯度幅值归一化（CLNorm）**和**层内梯度方向投影（ILProj）**缓解"后门遗忘"问题，显著提升了后门在下游重训练后的保持率。

## 研究问题与动机
- **核心问题**：在PET场景中，预训练模型（PLM）权重被冻结，攻击目标仅为少量轻量模块（如Adapter、LoRA），参数规模骤减导致**后门遗忘（backdoor forgetting）**——用户下游重训练干净数据后后门失效。
- **现有方法不足**：
  1. 传统全参微调后门攻击方法（如RIPPLe、LWP）直接迁移至PET场景效果受限，未考虑参数规模压缩带来的梯度失衡。
  2. 已有PET后门攻击（如BadPrompt、PPT）主要针对Prompt场景且用户不训练，而实际应用中用户往往基于干净数据对PET模块进行二次微调。
  3. 现有方法未系统分析PET注入过程中**干净数据与中毒数据之间的梯度冲突**（幅值不均、方向矛盾）这一遗忘根因。

## 核心贡献（创新点）
1. **视角创新**：将PET后门注入过程建模为**多任务学习**（干净任务 vs 后门任务），首次揭示梯度幅值跨层失衡与方向冲突是后门遗忘的关键机制。
2. **CLNorm（跨层梯度幅值归一化）**：通过学习映射函数约束各层后门梯度幅值至期望线性分布，弱化输出层依赖、强化底层特征学习。
3. **ILProj（层内梯度方向投影）**：基于PCGrad思想，通过可调节系数$\beta^{(l)}$在底层保留方向冲突以学习后门特征、在上层投影消除冲突以减缓遗忘。
4. **系统化实验验证**：在情感分类（SST-2/IMDB）和垃圾邮件检测（Enron/Lingspam）两大领域、两种PET插入形式（顺序/并行）下全面评估，均达到最高后门保持率。

## 方法详解
- **整体框架**：攻击者在注入阶段同时使用干净数据（clean task）和中毒数据（poison task）训练PET模块，将两者视为多任务学习中的两个子任务。
- **CLNorm原理**：
  - 设后门任务在第$l$层的梯度为$g_p^{(l)}$，目标是将各层梯度幅值映射到期望线性函数$z(l) = k l + b$。
  - 期望函数过两点：平均幅值点$(l_a, Avg[G_p])$和输出层$(l_o, 0)$，推导出缩放因子$w_l$并迭代更新：$w_l \leftarrow w_l - \alpha(w_l g_p^{(l)} - \tilde{g}_p^{(l)})g_p^{(l)}$。
  - 作用：抑制输出层过大的梯度贡献，均衡各层后门学习信号。
- **ILProj原理**：
  - 在第$l$层，干净梯度$g_c^{(l)}$与后门梯度$g_p^{(l)}$可能方向冲突。
  - 对每个任务梯度正交投影对方冲突分量（参考PCGrad公式），得到$\hat{g}_c^{(l)}$和$\hat{g}_p^{(l)}$，再融合：$g^{(l)} = (1-\beta^{(l)})\hat{g}^{(l)} + \beta^{(l)}g^{(l)}$。
  - 超参设置：底层（0-5层）$\beta=1$（完全保留冲突以学习后门特征），上层（6-11层）$\beta=0$（完全投影消除冲突以防遗忘）。

## 实验与结果
- **数据集**：情感分类（SST-2、IMDB）、垃圾邮件检测（Enron、Lingspam）；中毒样本比例50%，触发词随机插入。
- **评估指标**：Clean Accuracy（CACC）衡量可用性，Label Flip Rate（LFR）衡量后门保持率。
- **主要结果**（代表性数字）：
  - **情感分类（Seq.形式）**：SST-2→IMDB场景，本方法LFR=**73.7**，优于Vanilla(15.3)、RIPPLe(62.8)、LWP(69.9)、GradNorm(68.6)；IMDB→SST-2场景LFR=**99.4**。
  - **垃圾邮件检测（Seq.形式）**：Enron→Lingspam LFR=**90.9**（最高），Lingspam→Enron LFR=**51.1**（最高）。
  - **并行形式**同样取得最优或次优表现，整体CACC与基线持平（约86%-100%）。
- **消融实验**：单独使用ILProj或CLNorm均有贡献，二者结合效果最佳；$\beta$在底层设为1、上层设为0为最优配置。

## 相关工作脉络
1. **BadNet (Gu et al., 2017)**：早期DNN后门攻击开创性工作，但针对全参模型，未考虑PET场景。
2. **RIPPLe (Kurita et al., 2020)**：PLM微调后门攻击基线，通过正则化拉近梯度方向，但在PET小参数场景下遗忘严重。
3. **LWP (Li et al., 2021)**：逐层权重中毒方法，假设梯度幅值与层数反比，但**未考虑输出层影响**，在PET中效果受限。
4. **GradNorm (Chen et al., 2018)**：多任务学习中梯度幅值平衡方法，本文借鉴其思想但扩展至跨层归一化并针对性设计PET场景。
5. **PCGrad (Yu et al., 2020)**：通过梯度投影消除多任务方向冲突，本文改进为**分层可控投影**（引入$\beta^{(l)}$）以适应后门特征学习需求。
6. **BadPrompt/PPT (2022)**：针对Prompt微调的后门攻击，但假设用户不重训练；本文聚焦**用户重训练场景**，解决遗忘问题。

## 局限性与未来方向
- **局限性**：
  1. 对**仅插入在输入层的PET方法**（如Prompt-tuning）不适用，CLNorm无法使用，仅能依赖ILProj。
  2. 当用户用**大规模干净数据集**重训练时，后门遗忘依然严重，方法效果下降。
- **未来方向**：论文作者计划在后续工作中研究**防御PET后门攻击的方法**（见伦理声明Section 7）。

## 研究启发与可借鉴点
1. **多任务学习视角用于后门安全分析**：将后门注入视为双任务优化问题，为分析不同攻击场景下的遗忘机制提供了通用分析框架。
2. **分层梯度控制策略**：CLNorm的"期望线性分布"设计和ILProj的分层$\beta$调节思想，可迁移至其他参数高效微调的安全加固或性能优化场景。
3. **实验设计借鉴**：采用**领域迁移**（domain shift）设置（如SST-2攻击→IMDB重训练）比同域重训练更具现实威胁建模价值，值得后续安全评估工作参考。
4. **可解释性分析**：通过[CLS]向量相似度和逐层丢弃实验直观展示后门特征分布变化，为后门行为分析提供了可复用的可视化手段。

## 关键术语表
- **Parameter-Efficient Tuning (PET)**：冻结预训练模型主体参数，仅微调少量附加模块（如Adapter、LoRA）即可适应下游任务的训练范式。
- **Backdoor Forgetting**：用户用干净数据对已植入后门的模型进行重训练后，后门触发效果显著下降甚至消失的现象。
- **Cross-Layer Gradient Magnitude Normalization (CLNorm)**：通过自适应缩放因子平衡后门任务在各网络层的梯度幅值，避免过度依赖输出层。
- **Intra-Layer Gradient Direction Projection (ILProj)**：利用梯度投影技术消除干净任务与后门任务在同一层内的方向冲突，并通过超参$\beta$控制投影强度。
- **Label Flip Rate (LFR)**：后门有效性评估指标，计算中毒样本被错误分类为目标标签的比例。
- **Sequential/Parallel Form**：PET模块的两种插入形式，顺序形式将模块追加在每层之后，并行形式将模块与每层并排连接。

## 可复现要素
- **数据集**：SST-2、IMDB、Enron、Lingspam，均为公开数据集。
- **代码/权重**：论文未提及开源代码或模型权重。
- **关键超参**：学习率2e-5、batch size 32、 poison训练10 epochs、用户重训练5 epochs；CLNorm中$\alpha=1e-4$，ILProj中底层$\beta^b=1$、上层$\beta^t=0$。
