---
title: "CATS-A-Pragmatic-Chinese-Answer-to-Sequence-Dataset-with-Lar"
source: https://aclanthology.org/2023.acl-long.168.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:48:20"
field: "数据到文本生成"
keywords: ["Answer-to-Sequence", "数据到文本生成", "图神经网络", "中文数据集", "统一图转换", "节点段嵌入"]
innovations: ["提出大规模高质量中文Answer-to-Sequence数据集CATS（43,369样本）", "设计UGT统一图转换方法桥接SQL与表格的结构差异", "引入NSE节点段嵌入保留原始图结构信息以提升建模效果"]
benchmarks: ["CATS", "CATS-D", "CATS-S", "CoSQL"]
---

# 论文速读：CATS-A-Pragmatic-Chinese-Answer-to-Sequence-Dataset-with-Lar

## 一句话总结
本文提出了大规模高质量中文Answer-to-Sequence数据集CATS（43,369个样本），并设计了统一图转换方法UGT与节点段嵌入NSE，将SQL与表格输入转换为统一图结构以更好地建模两者间的语义对齐。

## 研究问题与动机
- **现有数据集规模与质量的矛盾**：大规模数据集（如WEATHERGOV、ToTTo）常含噪声或缺乏实际应用场景；贴近实际场景的数据集（如CoSQL仅7.8K训练样本、ROTOWIRE仅4.9K）规模过小，易导致过拟合。
- **语言偏见问题**：现有D2T数据集主要集中在英语，中文及其他语言资源严重匮乏。
- **异构输入的结构鸿沟**：Answer-to-Sequence任务的输入为SQL查询和对应表格，两者在结构上存在显著差异，难以直接建立有效语义对齐。
- **CoSQL质量不足**：CoSQL作为首个Answer-to-Sequence数据集，其标注质量粗糙，且以对话状态跟踪为核心任务，生成标注较为简略。

## 核心贡献（创新点）
- **提出CATS中文数据集**：构建了43,369个大规模高质量中文Answer-to-Sequence样本，显著缩小了研究与实际应用之间的差距，填补了中文D2T数据集空白。
- **设计UGT统一图转换方法**：将SQL和表格分别转化为无向图后建立节点连接，构造统一图表示，将Answer-to-Sequence任务转化为Graph-to-Text问题。
- **引入NSE节点段嵌入**：针对将图转换为token图会破坏原始结构的问题，通过为同一原始节点对应的token分配相同嵌入符号，保留原始图结构信息。
- **系统实验验证**：在CATS、CATS-D、CATS-S三个子集上全面评估，证明了数据集的挑战性和方法的有效性。

## 方法详解
**数据集构建流程**：
- **CATS-D**：从DuSQL数据集派生SQL-表格对，人工标注描述文本。
- **CATS-S**：通过自动数据构建管道收集——基于CLUE语料提取高频词搜索表格，使用SQL语法自动生成查询，人工标注后形成，包含更高比例的复杂SQL查询。

**UGT统一图转换**：
1. 将SQL转换为树状SQL图$\mathcal{G}_s$。
2. 将表格转化为表格图$\mathcal{G}_t$，列名和单元格作为节点，列头节点与同列单元格节点连接，同行单元格节点之间建立连接。
3. 在$\mathcal{G}_s$和$\mathcal{G}_t$中标识同一列的节点间建立跨图连接，形成统一图$\mathcal{G}_h$。

**模型架构**：
- 采用基于T5的变体Transformer架构，包含Global Node Encoder（G-NE）和Local Node Encoder（L-NE）。
- G-NE使用预训练Transformer编码器实现远距离节点的显式通信。
- L-NE由节点嵌入层和GAT层组成：
  - 节点嵌入层公式：$h_v^e = LayerNorm(h_v) + e_v^s$，其中$e_v^s$为节点段嵌入。
  - GAT层通过多头注意力聚合邻居节点表示。
- 损失函数为标准交叉熵：$\mathcal{L} = -\sum_{i=1}^{|y|} p_\theta(y_i|y_{1:i-1}; s, t)$。

## 实验与结果
**数据集统计**：
- CATS总计43,369个样本（CATS-D: 8,350；CATS-S: 33,019）。
- 训练/开发/测试集按34,697/4,336/4,336划分。
- 相比CoSQL，CATS表格行列数更多、SQL更复杂、描述长度更长。

**基线模型**：
- TEMP：基于预定义模板的自动生成方法。
- POINTER-GEN：基于RNN的Seq2Seq模型。
- T5：直接在CATS上微调的T5-base模型。
- T5-GRAPH：使用图表示输入的T5变体。

**主要结果**：
- **最优方法**：UGT + NSE在测试集上取得BLEU 55.95、ROUGE-L 76.10，相比T5+FNN分别提升约2.08和1.68。
- CATS-D上最优BLEU达57.10，CATS-S上最优BLEU达54.21。
- UGT相比同等参数规模的T5-GRAPH+FNN在所有指标上均有显著提升。
- 消融实验显示移除SQL或表格输入均导致性能大幅下降。
- 复杂度分析表明模型在处理大表格和复杂SQL时优势更明显，尤其在"Extra Hard"级别SQL上相对POINTER-GEN提升9.22个BLEU。
- 人类评估显示模型在流畅度和忠实度上仍有提升空间，覆盖率达90.26%。

## 相关工作脉络
- **CoSQL (Yu et al., 2019)**：首个提出Answer-to-Sequence任务的英文数据集，但规模小（7.8K训练样本）且标注粗糙，本文在此基础上扩展至中文大规模场景。
- **Ribeiro et al. (2021)**：将图转换为token图后使用预训练语言模型进行Graph-to-Text生成，本文指出该方法会破坏原始图结构并引入额外噪声，通过NSE加以改进。
- **DuSQL (Wang et al., 2020b)**：大规模中文Text-to-SQL数据集，本文从中派生SQL-表格对构建CATS-D，确保SQL分布贴近实际应用。
- **T5 (Raffel et al., 2020)**：统一预训练语言模型，本文以其为底座进行微调，并引入图结构建模增强。
- **WebNLG (Gardent et al., 2017a)**：英文Graph-to-Text数据集，本文将此类任务思路迁移至Answer-to-Sequence场景。
- **ROTOWIRE (Wiseman et al., 2017)**：体育领域D2T数据集，规模仅4.9K且含噪声，凸显了本文数据集在规模和用途上的优势。

## 局限性与未来方向
- **语言覆盖有限**：CATS仅针对中文，未解决多语言D2T数据集的普遍偏见问题。
- **方法适用范围受限**：UGT专为Answer-to-Sequence任务设计，不能直接推广至其他Graph-to-Text任务。
- **忠实度仍有差距**：人类评估显示模型的忠实度（7.48）与参考标注（9.15）仍有较大差距，需进一步理解SQL和表格的深层语义。
- **数据来源局限性**：CATS的主题分布受限于CLUE和DuSQL中的主题，主要为媒体、保险、银行等领域（占61%）。

## 研究启发与可借鉴点
- **双源数据集构建策略**：结合真实数据集派生（CATS-D）和自动化管道构建（CATS-S）可兼顾数据质量与规模，为后续数据集构建提供参考范式。
- **图结构建模与预训练模型结合**：UGT将结构化输入转换为统一图并配合G-NE/L-NE架构，展示了如何有效融合图神经网络与Transformer的优势。
- **NSE结构保持技巧**：通过节点段嵌入保留原始图结构信息的设计，为其他涉及图-序列转换的任务提供了可迁移的技术思路。
- **多维度复杂度分析**：从表格行列数、SQL难度、目标长度四个维度分析数据集分布，为评估数据集挑战性提供了系统性框架。
- **应用导向的评估设计**：结合自动指标（BLEU、ROUGE-L、COVERAGE）和人类评估（流畅度、忠实度、覆盖率、重复度），为NLG任务评估提供了参考。

## 关键术语表
- **Answer-to-Sequence**：TableQA系统中将SQL查询及其执行结果（表格）转换为自然语言描述的任务。
- **CATS**：Chinese Answer-to-Sequence的缩写，本文提出的大规模高质量中文Answer-to-Sequence数据集。
- **UGT (Unified Graph Transformation)**：统一图转换方法，将SQL和表格分别转化为图后建立连接形成统一图结构。
- **NSE (Node Segment Embedding)**：节点段嵌入，为token图中属于同一原始节点的token分配相同嵌入符号以保留图结构信息。
- **D2T (Data-to-Text)**：数据到文本生成，将结构化或半结构化数据转换为自然语言描述的通用任务。
- **Text-to-SQL**：文本到SQL，将自然语言问题转换为SQL查询的技术。
- **CoSQL**：首个包含Answer-to-Sequence任务的对话式Text-to-SQL数据集，规模7.8K训练样本。

## 可复现要素
- **数据集**：CATS已开源免费提供，可通过论文提供的链接获取。
- **代码/权重**：论文未明确提供代码仓库链接，实验基于HuggingFace Transformers和OpenNMT实现。
- **关键超参**：
  - T5-base初始化，隐藏层大小512
  - Dropout率：0.1
  - 优化器：AdamW
  - 学习率：3e-5
  - Batch size：4
  - Beam search宽度：5
  - 训练设备：Nvidia Tesla V100 32GB GPU
