---
title: "StoryARG-a-corpus-of-narratives-and-personal-experiences-in"
source: https://aclanthology.org/2023.acl-long.132.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 09:15:40"
field: "计算论证学与叙事分析"
keywords: ["computational argumentation", "narrative analysis", "personal experience", "argument mining", "persuasiveness", "annotation dataset"]
innovations: ["首个融合论证功能与叙事属性双视角的跨学科标注体系（9层标注）", "通过回归分析揭示SEARCH_FOR_SOLUTION功能对论证有效性最强正向预测", "在建模中显式整合annotator-specific偏好而非简单丢弃主观性差异"]
benchmarks: ["StoryARG", "ChangeMyView", "Regulation Room", "Europolis", "NYT Comments"]
---

# 论文速读：StoryARG-a-corpus-of-narratives-and-personal-experiences-in

## 一句话总结
本文构建了 **StoryARG** 数据集，首次系统性地从论证和叙事双重视角研究个人故事与经验在论证文本中的作用；回归分析发现，以"解决方案"为论证功能的故事能显著提升论证效果，但 annotator-specific 偏好导致主观性较强的标注层（如 effectiveness、argumentative function）一致性较低。

## 研究问题与动机
- **叙事在论证中的作用未被计算语言学充分研究**：尽管心理学和社会科学早已认可叙事在说服中的重要性，但计算论证学长期将其视为"二等公民"（非逻辑、不可验证），缺乏细粒度实证研究。
- **已有数据集仅能做粗粒度检测**：Park & Cardie (2014)、Song et al. (2016)、Falk & Lapesa (2022) 等 work 可检测故事存在，但无法刻画其**论证功能**（如澄清、披露伤害、寻求方案）和**叙事属性**（如情节结构、主人公类型、虚构性）。
- **叙事的功能分类在计算领域缺乏统一标准**：社会科学的分类框架（Maia et al., 2020）未被引入计算论证学研究，不同工作对同类现象使用不同术语（anecdote、testimony、experiential knowledge 等）。
- **主观性问题亟待解决**：effectiveness 和 argumentative function 高度依赖 annotator 视角，需探索如何在建模中合理整合 annotator-specific 差异（perspectivism）。

## 核心贡献（创新点）
- **提出了首个融合论证与叙事双重视角的跨学科标注体系**：整合 9 个标注层（4 个论证层 + 5 个叙事层），覆盖 argumentative function、effectiveness、emotional load、experience type、protagonist、proximity、hypothetical 等维度，区别于以往单一视角的数据集。
- **构建了大规模、多源异构的 StoryARG 数据集（2,451 个 span）**：采样自 ChangeMyView（150 docs）、Regulation Room（peanuts 150 docs + cdcp 50 docs）、Europolis（57 docs）及 NYT veganism comments（100 docs），支持跨领域比较研究。
- **系统性揭示了叙事特征对论证有效性的影响规律**：通过回归分析发现 STORY 类型（vs. EXPERIENTIAL KNOWLEDGE）、更高情感负荷、更长篇幅、以及"SEARCH FOR SOLUTION"功能均正向预测有效性；虚构性在"ESTABLISH BACKGROUND"和"DISCLOSURE OF HARM"场景下会降低可信度。
- **将 annotator-specific 差异纳入建模框架而非简单丢弃**：通过交互项回归（annotator × 各属性）量化个体偏好，与 perspectivism 研究范式接轨，为后续研究提供处理主观标注的方法论参考。

## 方法详解
- **采样策略**：对有已有标注的源数据（CMV、Regulation Room、Europolis）直接使用 gold label；对无标注的 peanuts 和 NYT comments，使用 Falk & Lapesa (2022) 训练的故事检测模型采样高概率片段，减少人工标注成本。
- **9 层标注体系**：
  - **文档级**：stance（CLEAR/UNCLEAR）、claim（free text）
  - **Span 级叙事属性**：experience type（STORY vs. EXPERIENTIAL KNOWLEDGE）、protagonist（INDIVIDUAL/GROUP/NON-HUMAN，最多两个）、proximity（FIRST-HAND/SECOND-HAND/OTHER）、hypothetical（TRUE/FALSE）、emotional load（LOW/MEDIUM/HIGH）
  - **Span 级论证属性**：argumentative function（CLARIFICATION/DISCLOSURE OF HARM/SEARCH FOR SOLUTION/ESTABLISH BACKGROUND）、effectiveness（LOW/MEDIUM/HIGH）
- **标注协议**：4 名受过 Argument Mining 训练的学生 annotator，经历 pilot → training → 3 轮正式标注，总耗时 400 小时；使用 INCEpTION 平台。
- **回归模型**：以 effectiveness（连续化 1-3）为因变量，9 个自变量（experience type、argumentative function、emotional appeal、hypothetical、proximity、protagonist、tokens、annotator）及其两两交互项为自变量，逐步回归选择最优模型，最终解释 32.69% 方差。annotator 贡献最大（13.41%），其次是 argumentative function（4.38%）、experience type（3.42%）。

## 实验与结果
- **数据集规模**：507 个文档，2,451 个 experience span（2,385 个关联明确 stance），平均 span 长度 STORY=353 tokens，EXPERIENTIAL KNOWLEDGE=215 tokens。
- **标注一致性**（Krippendorff's alpha，token overlap=1.0）：
  - 叙事属性较高：hypothetical=0.77、proximity=0.57、experience type=0.47
  - 论证属性极低：argumentative function=0.04、effectiveness=0.10、emotional load=0.36
- **关键发现**：
  - **SEARCH FOR SOLUTION 功能效果最强**：相比 CLARIFICATION(43%) 和 ESTABLISH BACKGROUND(38%)，solution-oriented 故事获得更高 effectiveness 评分；harm-oriented 次之。
  - **叙事性越强效果越好**：STORY > EXPERIENTIAL KNOWLEDGE；情感负荷越高、篇幅越长，predicted effectiveness 越高。
  - **虚构性有场景依赖**：在 CLARIFICATION 和 SEARCH FOR SOLUTION 中虚构故事仍有效，但在 ESTABLISH BACKGROUND 和 DISCLOSURE OF HARM 中虚假故事显著降低可信度。
  - **群体主角更受认可**：GROUP/NON-HUMAN 作为 protagonist 的故事比 INDIVIDUAL 更具说服力。
- **annotator 差异**：Annotator 1 偏好 SEARCH FOR SOLUTION，Annotator 2 偏好 DISCLOSURE OF HARM，说明 effectiveness 判断高度主观。

## 相关工作脉络
- **Falk & Lapesa (2022)**：同团队前期 work，提供 story/testimony 检测模型用于采样；StoryARG 在此基础上扩展了细粒度功能标注。
- **Maia et al. (2020)**：社会科学研究框架，提出叙事在审议讨论中的四种功能分类，本文首次将其引入计算论证学并简化术语。
- **Park & Cardie (2014, 2018)**：Regulation Room 和 CMV 数据集的创建者，分别定义 testimony 和 experiential knowledge，本文沿用并扩展其标注。
- **Wang et al. (2019)**：Persuasion for Good 数据集研究 persuasion strategies，但侧重 dialogue 场景且未涉及叙事属性；本文聚焦单轮 contribution 中的叙事角色。
- **Egawa et al. (2019)**：在 CMV 上标注 elementary units，包含 testimony 类型，但未涉及叙事结构与论证功能的交叉分析。
- **Nabi & Green (2014)**：提出 narrative transportation 理论，解释故事为何具有说服力，本文通过实证数据验证其在 argumentation 场景下的适用性。

## 局限性与未来方向
- **数据集规模偏小**：2,451 个 span 对深度学习模型而言训练不足，需进一步扩充。
- **annotator 数量少、多样性低**：仅 4 名熟悉领域的高水平学生，难以代表更广泛人群，限制了泛化性。
- **effectiveness 和 argumentative function 一致性极低**：提示当前标注体系对这两个主观层的设计需改进，或需转向 multi-label / fuzzy 标注范式。
- **未充分利用源数据已有标注**：CMV 和 Regulation Room 已有 testimony 标注，未与新标注进行交叉分析比较。
- **未来方向**：crowdsourcing 大规模补充 effectiveness 标注；探索多标签 argumentative function 建模；分析 annotator demographic 与偏好之间的关系；将 StoryARG 应用于下游 NLP 任务（如自动故事检测、有效性预测、 persuasiveness 生成）。

## 研究启发与可借鉴点
- **跨学科框架迁移**：将社会科学审议理论（Maia et al., 2020）引入计算论证学是一个有效策略，可为其他领域提供"理论先行→计算落地"的方法论范本。
- **annotator-specific 建模策略**：在回归中加入 annotator 作为固定效应或交互项，而非简单取多数投票，为高主观性任务提供了新的建模思路，值得在 stance detection、情感分析等任务中借鉴。
- **混合采样降低成本**：用预训练模型对无标注语料进行高概率采样，再人工精标，是一种高效的数据构建策略，适用于资源有限但需高质量标注的场景。
- **multi-label 论证功能**：鉴于单一标签效果差，可探索将 argumentative function 改为多标签分类或多尺度标注，更贴合真实场景中故事多功能并存的特性。
- **与团队方向的结合机会**：若团队关注 persuasion、argument mining 或 social computing，StoryARG 可作为 benchmark 验证模型对叙事特征的敏感性；亦可拓展至多语言场景（Europolis 含德/法/波兰语翻译）。

## 关键术语表
- **StoryARG**：本文提出的新数据集，包含 2,451 个标注过的叙事 span，覆盖 507 个文档和 9 个标注层。
- **Argumentative Function**：故事在论证中所承担的功能角色，分为 CLARIFICATION（澄清）、DISCLOSURE OF HARM（披露伤害）、SEARCH FOR SOLUTION（寻求方案）、ESTABLISH BACKGROUND（建立背景）。
- **Experience Type**：区分 STORY（具有情节结构、时序序列）与 EXPERIENTIAL KNOWLEDGE（无情节、背景性个人知识）的两类叙事形式。
- **Proximity**：叙事亲近度，标识叙述者与故事的关系——FIRST-HAND（亲历）、SECOND-HAND（听说）、OTHER（陌生视角）。
- **Effectiveness**：主观评估故事对论证整体说服力的贡献程度，分 LOW/MEDIUM/HIGH 三级。
- **Hypothetical**：布尔标签，标识叙事内容是真实发生还是假设/虚构。
- **Krippendorff's alpha (α)**：衡量多 annotator 间一致性的统计量，取值 [-1, 1]，1.0 表示完全一致。
- **Perspectivism**：NLP 研究中承认主观性价值的范式，主张将 annotator 差异视为资源而非噪声，在本文体现为回归模型中显式建模 annotator 效应。

## 可复现要素
- **数据集**：StoryARG，已在 GitHub 公开（https://github.com/Blubberli/storyArg），含 annotation guidelines 和完整标注文件。
- **代码**：论文未提及开源代码；仅公开数据集和标注指南。
- **关键超参**：无 ML 模型训练超参；回归分析使用逐步选择（step-wise model selection），以 adjusted R² 为准则。
- **标注工具**：INCEpTION（Klie et al., 2018）。
- **先验模型**：Falk & Lapesa (2022) 的故事检测模型用于 peanuts 和 NYT 子集的采样。
