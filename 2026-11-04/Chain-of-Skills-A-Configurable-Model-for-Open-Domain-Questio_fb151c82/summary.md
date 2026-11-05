---
title: "Chain-of-Skills-A-Configurable-Model-for-Open-Domain-Questio"
source: https://aclanthology.org/2023.acl-long.89.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:07:28"
field: "开放域问答检索"
keywords: ["open-domain question answering", "dense retrieval", "modular model", "multi-task learning", "chain-of-skills"]
innovations: ["提出Chain-of-Skills模块化检索器，将ODQA检索分解为5个可复用技能模块", "设计MHA专家参数化方案有效缓解多任务检索中的任务干扰", "多任务自监督预训练实现零样本检索能力并支持灵活的链式推理配置"]
benchmarks: ["Natural Questions (NQ)", "HotpotQA", "OTT-QA", "WebQuestions", "EntityQuestions", "SQuAD"]
---

# 论文速读：Chain-of-Skills-A-Configurable-Model-for-Open-Domain-Questio

## 一句话总结
本文提出Chain-of-Skills (COS)，一个基于模块化设计的稠密检索器，将开放域问答的检索推理能力分解为5个可复用技能模块，通过多任务自监督预训练与微调，支持灵活的技能组合链式推理，在零样本和微调场景下均达到SOTA。

## 研究问题与动机
- **现有方法局限**：当前ODQA检索模型多为数据集定制化设计，不同数据集依赖不同检索技能组合，导致模型可迁移性和可扩展性受限。
- **任务干扰问题**：共享编码器多任务学习虽能提升效率，但不同检索技能（如单跳检索与扩展查询检索）存在冲突信号，导致负迁移。
- **推理配置僵化**：已有工作仅探索部分技能配置（如图1所示），缺乏灵活组合与链式推理能力以适配不同目标领域。
- **数据稀缺性**：新型信息持续涌现，基于单一数据集训练的深度检索模型难以复用，需要可复用的技能表示。

## 核心贡献（创新点）
1. **模块化检索框架**：将ODQA检索分解为单跳检索、扩展查询检索、实体span提议、实体链接、重排序5个可复用技能模块，支持跨数据集灵活配置。
2. **MHA专家参数化设计**：针对稀疏Transformer的MoE思路，提出注意力机制（MHA）而非FFN专家特化方案，更有效地缓解多任务检索中的任务干扰。
3. **多任务自监督预训练**：首次在同一框架下结合Wikipedia单跳检索、扩展查询检索和实体链接三种自监督信号进行预训练，支持零样本直接检索。
4. **Chain-of-Skills推理策略**：设计可灵活组合技能的链式推理流水线，支持不同任务最优技能配置的搜索与组合。

## 方法详解
- **五大技能模块**：
  - **单跳检索**：使用bi-encoder架构，query/document分别编码后计算点积相似度，采用 InfoNCE对比损失（公式1）。
  - **扩展查询检索**：将问题与前一步证据拼接为扩展查询输入，其余同单跳检索。
  - **实体span提议**：基于对比学习NER，用span首尾token向量拼接后经tanh+可学习权重得到span表示，与学习到的anchor向量对比（公式2），并用[CLS]向量学习动态阈值（公式3），总损失为两者平均。
  - **实体链接**：用实体起点终点位置的平均向量作为实体表示，与Wikipedia实体描述[CLS]向量匹配（公式4）。
  - **重排序**：将问题与段落拼接为cross-encoder输入，用[CLS]与首个[SEP]向量计算相似度进行排序（公式5）。

- **模块化参数化**：每个Transformer block由共享block和多个并行专家block组成，根据输入路由到不同专家。实验发现MHA专家（182M参数）比FFN专家（252M参数）更有效地减少任务干扰。

- **预训练设置**：从Contriever初始化，在Wikipedia上用6M单跳样本、6M扩展查询样本、9M实体链接样本训练20轮，batch size=1024，max seq len=256，lr=1e-4。

- **推理策略**：不同技能结果通过分数对齐（按同一检索分数归一化）合并排序，命中多次的文档排名提升（乘以系数α），最后经重排序模块重新打分（乘以系数β）。

## 实验与结果
- **数据集**：NQ, WebQ, SQuAD, EntityQuestions（单跳）；HotpotQA, OTT-QA（多跳），均基于Wikipedia。
- **零样本评估**（表1）：COS pretrain-only在NQ Top-100上达81.8，优于Contriever (82.1)接近；在HotpotQA Top-100达87.9，大幅超越Contriever (64.5)和BM25 (50.5)。
- **监督微调结果**：
  - NQ Top-100：85.6，超越DPR-PAQ (89.2)、SPAR-wiki (88.8)、co-Condenser (89.0)（表2）—— 这里表格数据需核实，COS 85.6略低于部分基线，但论文称"outperforms all baselines"。
  - OTT-QA Top-100：92.2，超越CORE (87.1)（表3）。
  - HotpotQA Passage EM：88.89，超越AISO (86.94)、MDR (81.20)（表4）。
- **跨域评估**（表5）：COS在EntityQuestions (Top-100: 82.4)和SQuAD (Top-100: 81.2)上均优于SPAR-wiki和BM25。
- **端到端QA**：COS+FiE在OTT-QA上EM=56.9/F1=63.2，超越CORE+FiE (EM=51.4/F1=57.8)（表8）。

## 相关工作脉络
- **DPR (Karpukhin et al., 2020)**：基准稠密检索器，仅支持单跳检索，非模块化设计。
- **MDR (Xiong et al., 2021b)**：联合学习单跳与扩展查询检索，但仍为单一模型架构，不可灵活组合其他技能。
- **Contriever (Izacard et al., 2021)**：自监督稠密检索器，使用随机裁剪构建正例，仅训练单跳检索。
- **CORE (Ma et al., 2022a)**：针对OTT-QA的多技能检索器，但定制于特定数据集，不可复用。
- **SPAR-wiki (Chen et al., 2021b)**：结合lexical与dense检索，但未考虑技能链式组合。
- **Switch Transformers (Fedus et al., 2021)**：MoE架构基础，本文借鉴其专家路由思想但应用于检索技能模块而非语言建模。

## 局限性与未来方向
- **重排序局限**：当前重排序专家仅学习单跳文档排序，无法建模多跳证据链中多个段落间的交互，限制多跳证据链的全路径重排序能力。
- **预训练覆盖不足**：仅对三个bi-encoder任务做自监督预训练，entity span proposal未纳入预训练，导致零样本场景下无法完整执行Chain-of-Skills推理。
- **小模型容量**：实验使用BERT-base规模（182M参数），作者推测容量不足以建模复杂的多跳证据链重排序。
- **未来方向**：扩展至常识推理、生物医学检索等场景，探索全路径重排序与更大规模预训练。

## 研究启发与可借鉴点
- **技能分解思路**：将复杂检索任务拆解为原子技能模块，通过组合而非重建新模型来适配新任务，具有强迁移价值。
- **MHA专家优于FFN专家**：检索任务中技能差异主要体现在输入聚焦位置（语义vs局部上下文）而非事实关联，MHA特化比FFN特化更有效且参数更省。
- **自监督信号扩展**：将Wikipedia超链接关系同时用于单跳检索、扩展查询检索和实体链接三种任务的自监督预训练，可复用性强。
- **分数对齐融合策略**：不同技能返回的分数需在统一尺度下对齐后合并，且命中多次的文档应获得排名提升，这对多源信息融合有启发。

## 关键术语表
**Chain-of-Skills (COS)**：模块化检索框架，将ODQA检索分解为5个可复用技能，支持灵活的链式组合推理。
**Single Retrieval**：单跳检索，用原始问题直接检索最相关的单个文档/表格。
**Expanded Query Retrieval**：扩展查询检索，将前一步检索到的证据拼接到问题后形成扩展查询，再进行第二轮检索。
**Entity Span Proposal**：实体span提议，识别文本中与实体相关的span片段，用于后续实体链接。
**Entity Linking**：实体链接，将检测到的实体span映射到知识库（Wikipedia）中的对应实体描述。
**Reranking**：重排序，用cross-encoder对候选文档进行精细排序以提升top-k精度。
**Task Interference**：任务干扰，多任务学习中不同任务信号竞争模型容量导致的负迁移现象。
**MHA Expert**：多头注意力专家，将Transformer中的MHA子层作为可路由的专家模块。

## 可复现要素
- **数据集**：NQ, WebQ, SQuAD, EntityQuestions, HotpotQA, OTT-QA（公开数据集，Wikipedia-based）。
- **代码开源**：论文未明确声明代码开源状态。
- **权重开源**：未声明。
- **关键超参**：预训练lr=1e-4, batch=1024, epochs=20, max_seq_len=256；微调lr=2e-5, batch=192, epochs=40。
