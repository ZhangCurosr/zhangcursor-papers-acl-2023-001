---
title: "Chain-of-Skills-A-Configurable-Model-for-Open-Domain-Questio"
source: https://aclanthology.org/2023.acl-long.89.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:49:15"
field: "开放域问答与检索"
keywords: ["Open-Domain Question Answering", "Dense Retrieval", "Multi-task Learning", "Modular Architecture", "Self-supervised Pretraining", "Chain-of-Skills"]
innovations: ["模块化多技能检索框架，联合学习5种可复用检索技能并支持灵活链式组合推理", "基于MHA的专家参数化设计，有效缓解多任务检索中的任务干扰", "多任务自监督预训练策略，在Wikipedia上联合训练单次检索、扩展查询检索和实体链接"]
benchmarks: ["NQ", "HotpotQA", "OTT-QA", "WebQ", "EntityQuestions", "SQuAD"]
---

# 论文速读：Chain-of-Skills-A-Configurable-Model-for-Open-Domain-Questio

## 一句话总结
论文提出 Chain-of-Skills (COS)，一种基于 Transformer 的模块化开放域问答（ODQA）检索器，通过多任务学习联合掌握 5 种可复用检索技能（单次检索、扩展查询检索、实体跨度建议、实体链接、重排序），并支持灵活的链式组合推理，在零样本和微调评估中均取得 SOTA 性能。

## 研究问题与动机
- **问题**：现有 ODQA 检索模型多为特定数据集定制，导致模型可迁移性和可扩展性受限；单一数据集训练还可能产生负迁移（如单跳数据增强损害多跳检索性能）。
- **不足1**：不同 ODQA 数据集关注不同检索技能组合，但现有方法往往只使用部分技能或各自独立训练模型。
- **不足2**：共享文本编码器的多任务学习易产生任务干扰（task interference），降低检索性能。
- **不足3**：自监督预训练的检索器难以灵活应对需要多步推理的复杂多跳问答任务。

## 核心贡献（创新点）
- **模块化多技能检索框架**：首次将 ODQA 检索解构为 5 种可复用推理技能并联合学习，不同技能模块可根据目标任务灵活配置和链式组合。
- **注意力级专家参数化设计**：提出基于 MHA（多头注意力）而非 FFN 的 skill-specific expert 设计，通过混合共享与专用 Transformer 层有效缓解任务干扰。
- **多任务自监督预训练策略**：在 Wikipedia 上构建包含单次检索、扩展查询检索和实体链接的多任务自监督预训练，使模型可直接用于零样本检索且无需数据集标注。
- **灵活推理策略**：设计 Chain-of-Skills 推理流程，支持并行/串行组合多种技能并通过分数对齐和重排序策略融合多源证据。

## 方法详解
- **技能模块设计**：
  - **单次检索（Single Retrieval）**：bi-encoder 架构，query 与 passage 分别经 BERT-base 编码后用 [CLS] 向量做 dot product 相似度计算，使用 InfoNCE 对比损失（公式1）。
  - **扩展查询检索（Expanded Query Retrieval）**：将前一步证据拼接到查询后作为输入 [CLS] Q [SEP] P₁⁺ [SEP]，使用相同 bi-encoder 结构。
  - **实体跨度建议（Entity Span Proposal）**：对比式 NER，用 span 首尾 token 向量拼接后经 tanh 和可学习权重 Wᵃ 生成跨度表示 m，与锚向量 s 做对比学习；额外引入 [CLS] 向量学习动态阈值（公式2-3）。
  - **实体链接（Entity Linking）**：用实体跨度位置对应的 encoder 输出均值作为实体向量 e，匹配其 Wikipedia 描述向量的 [CLS] 表示（公式4）。
  - **重排序（Reranking）**：cross-encoder 架构，将 Q 与 P 拼接后编码，用 [CLS] 和首个 [SEP] 向量做相似度计算（公式5）。

- **模块化参数化**：
  - 共享 Transformer 块 + 8 个 skill-specific expert（FFN 或 MHA）。
  - 最终配置：合并 single/expert query 的 context expert、合并 expanded query/reranker 输入、预训练时进一步共享 single 和 expanded query expert。
  - 共 5 个 expert（182M 参数），相比无 expert 的 111M 参数仅有小幅增加。

- **自监督预训练**：
  - 基于 Contriever 权重初始化，在 Wikipedia 上训练 20 个 epoch，batch size 1024，最大序列长度 256，学习率 1e-4。
  - 单次检索：随机裁剪片段对作为正样本。
  - 实体链接：使用 BLINK 预处理数据。
  - 扩展查询检索：同一页面的短文本片段作为 query，关联页面的第一段作为 target。

- **推理策略**：
  - 通过分数对齐（lsᵢ = lsᵢ / max({ls} ∪ {rs}) × max({rs})）和系数 α 提升共同发现的文档（公式6-7）。
  - 最终排序结合检索/链接分数与重排序分数：sᵢ + β × rank scoreᵢ（公式8）。

## 实验与结果
- **数据集**：NQ、WebQ、SQuAD、EntityQuestions（单跳）；HotpotQA、OTT-QA（多跳）。
- **零样本评估（Table 1）**：COS（pretrain-only）在 NQ Top-20 达 68.0，WebQ 达 66.7，EntityQuestions 达 70.7，HotpotQA 达 77.9；平均 Top-100 达 82.3，超越 BM25（70.9）和 Contriever（75.2）。
- **监督域内评估**：
  - **NQ（Table 2）**：COS Top-20 达 **85.6**，Top-100 达 **90.2**，超越 DPR-PAQ（84.7/89.2）、co-Condenser（84.3/89.0）等 SOTA。
  - **OTT-QA（Table 3）**：COS Top-100 达 **92.2**，超越 CORE（87.1）。
  - **HotpotQA（Table 4）**：COS 通过 EM 达 **88.89**，超越 AISO（86.94）和 TPRR（86.19）。
- **跨数据集评估（Table 5）**：COS 在 EntityQuestions Top-20 达 76.3，SQuAD Top-100 达 81.2，均超越 BM25 和 SPAR-wiki。
- **端到端 QA（Table 8）**：COS + FiE 在 OTT-QA dev EM 达 **56.9**，超越 CORE + FiE（51.4）。

## 相关工作脉络
- **DPR（Karpukhin et al., 2020）**：单跳密集检索基线，仅使用单次检索技能；COS 扩展至多技能联合学习与灵活推理。
- **MDR（Xiong et al., 2021b）**：联合学习单次检索与扩展查询检索；COS 进一步纳入实体链接、跨度建议和重排序，并支持更灵活的链式组合。
- **Contriever（Izacard et al., 2021）**：自监督密集检索器，仅训练单次检索；COS 在多任务自监督预训练中覆盖三种 bi-encoder 技能。
- **CORE（Ma et al., 2022a）**：OTT-QA 专项模型，使用多技能但仅针对表格-文本混合检索；COS 是通用模块化框架，可配置适应多数据集。
- **SPAR-wiki（Chen et al., 2021b）**：结合 BM25 预训练与_dense retrieval；COS 通过多任务自监督预训练实现更优零样本性能。
- **MoE/稀疏 Transformer（Fedus et al., 2021b）**：引入 expert 路由机制；COS 借鉴此思想但聚焦于检索技能的专家化参数化而非模型缩放。

## 局限性与未来方向
- 重排序 expert 仅学习单步结果重排序，无法对多跳证据链进行整体排序；模型容量对完整路径重排序任务可能不足。
- 自监督预训练仅包含三种 bi-encoder 任务，实体跨度建议未纳入预训练，限制了零样本场景下可链式组合的技能范围。
- 模型仅在 Wikipedia 英语语料上评估，对低资源语言、其他知识源（如学术文献、生物医学数据）的泛化能力待验证。
- 未探索模型 scaling 效果，参数量（182M）相对较小，大规模扩展效果未知。

## 研究启发与可借鉴点
- **技能分解思路**：将复杂检索任务解构为原子化技能模块（单次检索/扩展检索/实体链接/跨度建议/重排序）并联合学习，可为其他复杂 NLP 任务（如信息抽取、对话系统）提供模块化设计范式。
- **MHA 专家而非 FFN 专家**：发现注意力层专业化比 FFN 层更适合检索任务，为 MoE 架构在检索领域的应用提供了新的设计参考。
- **多任务自监督预训练策略**：利用 Wikipedia 内部结构（随机裁剪、超链接、页面分段）构造多种对比学习正样本对，有效提升了零样本检索性能；该策略可迁移至其他领域的自监督预训练。
- **分数对齐与证据融合策略**：跨技能检索结果的分数对齐公式（公式6-7）解决了不同技能输出不可比的问题，为多模型/多源信息融合提供了实用方法。
- **Chain-of-Skills 推理**：支持灵活配置技能链以适应不同任务，可与本团队研究的动态推理、任务自适应等方向结合，探索少样本/零样本场景下的技能组合搜索策略。

## 关键术语表
- **Chain-of-Skills (COS)**：论文提出的模块化 ODQA 检索框架，通过学习多种可复用检索技能并支持灵活链式组合推理。
- **ODQA（Open-Domain Question Answering）**：开放域问答，在大规模知识库中检索证据并回答问题的任务。
- **Bi-encoder vs Cross-encoder**：bi-encoder 分别编码 query 和 document 再用点积计算相似度；cross-encoder 将 query 和 document 拼接后联合编码进行交互建模。
- **Expanded Query Retrieval**：扩展查询检索，将前一步检索到的证据拼接到原始查询后，进行第二轮检索以支持多跳推理。
- **Entity Span Proposal**：实体跨度建议，识别文本中潜在的实体跨度，用于后续实体链接。
- **Task Interference**：任务干扰，多任务学习中不同任务竞争模型容量导致性能下降的现象。
- **In-batch Negatives**：批次内负样本，利用同一 batch 中其他样本作为负例进行对比学习，无需额外采样。
- **FiE（Fusion-in-Encoder）**：将多个检索到的 passage 拼接后通过单一 encoder 进行早期融合的答案生成模型。

## 可复现要素
- **数据集**：NQ、WebQ、SQuAD、EntityQuestions、HotpotQA、OTT-QA 均为公开数据集；Wikipedia 预训练语料来自公开 dump（2018/12/20、2019/08/01、2022/06/01）。
- **代码/权重**：论文未明确提及代码开源链接和预训练权重下载方式，需访问作者主页或 GitHub 确认。
- **关键超参**：预训练学习率 1e-4、batch size 1024、20 epochs；微调学习率 2e-5、batch size 192、40 epochs；最大序列长度 256；Expert 配置为 5 个（MHA 专业化）；预训练从 Contriever 权重初始化。
