---
title: "StoryARG-a-corpus-of-narratives-and-personal-experiences-in"
source: https://aclanthology.org/2023.acl-long.132.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:58:58"
field: "计算论证与叙事分析"
keywords: ["computational argumentation", "narrative in argument", "personal experience annotation", "argument effectiveness", "subjectivity in NLP", "StoryARG", "storytelling corpus"]
innovations: ["构建 StoryARG 双视角标注数据集，首次系统刻画论证中叙事的论证功能与叙事属性", "在回归模型中显式建模 annotator 主观差异，揭示叙事有效性的高度个人化偏好", "发现 SEARCH FOR SOLUTION 类叙事与情节/情感/长度正向关联，虚构叙事在可信度相关功能中显著降效"]
benchmarks: ["StoryARG 自建数据集", "ChangeMyView", "RegulationRoom", "Europolis", "NYT Comments"]
---

# 论文速读：StoryARG-a-corpus-of-narratives-and-personal-experiences-in

## 一句话总结
本文构建了StoryARG数据集，从计算论证学与社会科学的多源语料中抽取并标注论证文本内的叙事与个人经历片段，首次系统地从论证功能与叙事属性两个维度刻画其作用机制，并通过回归分析揭示叙事类型、情感负载、论证功能及标注者偏好对论证有效性的影响规律。

## 研究问题与动机
- 现实中人类在论证时常以叙事或个人经历支撑观点，但计算论证领域对该现象几乎未作系统研究，常因其"非逻辑、不可验证"而被视为二等问题。
- 现有仅能检测故事存在的语料或模型（如 anecdote/testimony 标注）无法揭示叙事在论证中的具体功能、叙事结构与有效性之间的关系。
- 叙事如何影响论证说服力、具有何种叙事特征更有效、是否存在标注者主观偏好，尚缺乏实证层面的细粒度刻画。
- 社会科学与计算语言学之间缺乏对"叙事-论证"交叉现象的统一定义与标注体系，难以支撑可复用资源与跨域比较研究。

## 核心贡献（创新点）
- 构建并开源 StoryARG 数据集（共 2,451 个经验片段、9 个标注层、4 位标注者），覆盖 ChangeMyView、RegulationRoom、Europolis 与 NYT 评论四个来源，填补计算论证中叙事研究的数据空白。
- 提出跨学科双视角标注方案，同时刻画论证功能（澄清、揭示伤害、搜寻方案、建立背景）与叙事属性（情节结构、主角类型、一手/二手视角、假设/事实、情感负载），与以往仅关注结构/类型的单一视角形成本质差异。
- 将标注者主观性纳入分析而非忽略，通过在回归模型中引入 annotator 及其交互项，揭示"论证有效性"的高度个人化特征，与 NLP 中 perspectivism 理念相衔接。
- 发现并量化关键效应：以"搜寻解决方案"为功能的叙事显著更高有效性，较长篇幅、更强情节性与更高情感负载亦正向预测有效性；虚构叙事在"建立背景/揭示伤害"时可信度受损。
- 提供开源仓库与详细标注指南（GitHub: https://github.com/Blubberli/storyArg），并将多标签化处理 argumentative function 的低一致性，推动后续可复用资源建设。

## 方法详解
- 数据采样：优先使用已有 testimony/storytelling 标注的来源（cdcp、CMV、Europolis）；对无标注子集（peanuts、NYT）调用 Falk & Lapesa (2022) 的叙事检测模型，取高概率样本以降低人工标注成本。
- 两层标注体系：
  - 文档级：stance（CLEAR/UNCLEAR）、claim（自由文本）。
  - 片段级：experience type（STORY/EXPERIENTIAL KNOWLEDGE）、protagonist1/2（INDIVIDUAL/GROUP/NON-HUMAN）、proximity（FIRST-HAND/SECOND-HAND/OTHER）、hypothetical（TRUE/FALSE）、argumentative function（CLARIFICATION/DISCLOSURE OF HARM/SEARCH FOR SOLUTION/ESTABLISH BACKGROUND）、emotional appeal（LOW/MEDIUM/HIGH）、effectiveness（LOW/MEDIUM/HIGH）。
- 多标签处理：argumentative function 与 effectiveness 一致性较低，作者通过 token-overlap 合并相似片段并聚合不同标注者的额外函数，构造多标签版本。
- 回归建模：以 effectiveness（连续化 1–3）为因变量，纳入 8 个自变量及两两交互，采用逐步选择优化 adjusted R²；最终模型解释约 32.69% 方差，主要解释力来源于 annotator（13.41%）、argumentative function（4.38%）、experience type（3.42%）、tokens（2.70%）、emotional appeal（1.40%）。
- 一致性与评估：使用 Krippendorff's alpha 在三种 token overlap 容差（0.6/0.8/1.0）下评估，叙事属性一致性中等至较高，论证属性一致性低。

## 实验与结果
- 数据集规模与分布：共 507 篇文档、2,451 个经验片段；STORY 平均长度 353 tokens，EXPERIENTIAL KNOWLEDGE 平均 215 tokens；CMV 与 peanuts 片段数最多，NYT 最短。
- 叙事属性分布：FIRST-HAND 占主导（整体约 76%），PROTAGONIST 在 CMV/peanuts/cdc 中偏 INDIVIDUAL（52–66%），Europolis/NYT 偏 GROUP/NON-HUMAN；function 分布为 CLARIFICATION 43%、ESTABLISH BACKGROUND 38%、DISCLOSURE OF HARM 10%、SEARCH FOR SOLUTION 9%。
- 一致性结果：hypothetical 最高（α=0.77 at 1.0），effectiveness 与 argumentative function 最低（α≈0.04–0.10），说明论证相关层高度主观。
- 主要结论：
  - STORY 比 EXPERIENTIAL KNOWLEDGE 更有效；更长文本、更高情感负载更有效。
  - SEARCH FOR SOLUTION 预测最高有效性；由 general to specific 的效果递增（clarification < background < harm < solution）。
  - 以 GROUP/NON-HUMAN 为焦点的体验更被视作有效。
  - 虚构故事在 clarification/solution 下仍有效，但在 establish background/harm 下显著降低有效性，反映可信度需求。
  - 标注者个体差异显著：annotator-1 最偏好 solution，annotator-2 最偏好 harm，annotator-3/4 遵循"越具体越有效"趋势。
- 最强发现：引入 annotator 交互项后模型仍显著，证明主观偏好是重要可建模信号，而非噪声。

## 相关工作脉络
- Park & Cardie (2014)、Song et al. (2016) 等早期 work 聚焦 anecdote/testimony/experiential knowledge 的检测，但未系统刻画其在论证中的功能与叙事结构。
- Park & Cardie (2018)、Egawa et al. (2019) 在 RegulationRoom 与 CMV 上标注 testimony，提供可复用的已有标签，但仅停留于前提类型层面。
- Al-Khatib et al. (2016, 2017) 研究新闻社论/讨论中的论证策略，侧重证据流与策略分类，未涉及叙事属性。
- Wang et al. (2019) 关注 persuasion strategies，但叙事更多作为生成/个性化对象而非可解释资源。
- Maia et al. (2020) 的社会科学框架提供 argumentative function 的来源，本文在其基础上扩展为可直接用于计算标注的多层 schema。
- Basile (2020)、Uma et al. (2022) 关于高主观任务与 disagreement-as-resource 的工作为本文处理 annotator 差异提供理论依据，形成与当前 NLP perspectivism 的对接。

## 局限性与未来方向
- 数据规模偏小、标注者人数与人口学多样性有限，不利于直接训练大规模机器学习模型。
- argumentative function 与 effectiveness 一致性较低，虽以多标签与回归建模缓解，但仍需更大众标数据验证。
- 仅分析了英文子集（Europolis 含德/英/法，本文只用英译），跨语言/跨文化分布未充分探索。
- 未系统对比本文标注与既有 testimony/storytelling 标注的一致性或增益。
- 伦理风险：数据可被用于生成"看似真实"的个人叙事以操纵政治话语，或训练提取个人隐私信息的模型。
- 未来可通过众包扩展 effectiveness 标注、开展跨语料迁移研究、并与 e-deliberation 工具（如 E-DELIB 项目）结合。

## 研究启发与可借鉴点
- 将标注者作为显式变量纳入预测模型，以"主观差异可解释"替代"不一致即噪声"的思路，值得在低一致性标注任务中借鉴。
- 叙事功能四分类（clarification/harm/solution/background）与叙事属性（plot/protagonist/proximity/hypothetical）的双层 schema 可直接迁移至 argument mining、dialogue persuasion、deliberation support 等方向。
- 采用已有标注 + 模型辅助采样混合策略，显著降低人工成本，适合资源受限的数据构建场景。
- 多标签聚合与 token-overlap 合并方法，为处理重叠/歧义片段提供可复用工程实践。
- 虚构 vs 事实叙事在不同函数下的差异化有效性，提示可在 persuasion generation 中引入"体裁-信任"约束，避免在不合适场景使用虚构叙事。

## 关键术语表
- **StoryARG**：本文开源数据集，包含 2,451 个论证文本中的叙事/个人经历片段及 9 层标注。
- **Experience Type**：区分 STORY（具情节与事件序列）与 EXPERIENTIAL KNOWLEDGE（背景型个人经验陈述）。
- **Argumentative Function**：叙事在论证中的作用，包括 CLARIFICATION、DISCLOSURE OF HARM、SEARCH FOR SOLUTION、ESTABLISH BACKGROUND。
- **Proximity**：叙事距离，分为 FIRST-HAND（亲历）、SECOND-HAND（他人转述）、OTHER（旁观/无明确来源）。
- **Effectiveness**：标注者对叙事增强论证力度程度的主观评级（LOW/MEDIUM/HIGH）。
- **Perspectivism**：主张主观差异与分歧可作为分析资源而非缺陷的研究立场。
- **Krippendorff's alpha**：衡量多标注者间一致性的统计指标，值域 [-1, 1]。
- **Token-overlap 合并**：按共享词数比例合并相似经验片段，用于评估一致性与构造多标签版本。

## 可复现要素
- 数据集：StoryARG，开源地址 https://github.com/Blubberli/storyArg；来源包括 RegulationRoom、ChangeMyView、Europolis（英译）、NYT Comments（veganism）。
- 代码/权重：论文未提供模型代码；依赖 Falk & Lapesa (2022) 的检测模型用于采样（论文未提及是否开源）。
- 关键超参：回归模型采用逐步选择优化 adjusted R²；alpha 计算在 token overlap 容差 0.6/0.8/1.0 三档下进行；annotation tool 为 INCEpTION。
- 标注者：4 名（2 男 2 女），含 3 名计算语言学硕士与 1 名数字人文硕士，母语级别英语。
- 工时：约 400 小时，周期约一年。
