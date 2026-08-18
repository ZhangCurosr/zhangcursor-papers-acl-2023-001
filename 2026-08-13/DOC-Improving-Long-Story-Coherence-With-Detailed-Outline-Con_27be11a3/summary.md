---
title: "DOC-Improving-Long-Story-Coherence-With-Detailed-Outline-Con"
source: https://aclanthology.org/2023.acl-long.190.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-08-18 19:51:31"
field: "长文本生成与可控文本生成"
keywords: ["long story generation", "coherence", "structured planning", "contrastive control", "GPT-3", "OPT"]
innovations: ["层次化递归大纲生成器Detailed Outliner，将premise扩展为树状详细事件树", "基于句截断拼接Hard negatives的对比训练控制器Detailed Controller，实现token级全程相关性引导"]
benchmarks: ["WritingPrompts", "人工pairwise Coherent/Relevant/Interesting评估"]
---

# 论文速读：DOC-Improving-Long-Story-Coherence-With-Detailed-Outline-Con

## 一句话总结
DOC（Detailed Outliner + Detailed Controller）通过层次化递归展开初始大纲为详细事件树，并结合基于FUDGE的token级对比控制器引导OPT-175B生成器，显著提升了数千词级别长故事的**长距离情节连贯性**与**对初始设定的忠实性**。

## 研究问题与动机
- **长距离情节连贯性缺失**：现有SOTA系统（如GPT-4）仍需结构化规划才能生成数百词以上文本；Re³虽提出planning-drafting-revision框架，但仍存在局部流畅但全局不连贯的失败模式。
- **偏离初始规划/设定**：即使存在高层大纲，生成内容常偏离预设场景、角色关系或事件顺序。
- **重复性高**：Re³在遵循高层计划的同时，故事内容时常出现重复（如Table 30所示）。
- **上下文窗口限制**：Generator受限于1024 token的context window，难以一次性处理长篇故事的全局信息。

## 核心贡献（创新点）
1. **层次化递归大纲生成器（Detailed Outliner）**：将30-60词premise递归扩展为深度=3、分支因子2-5的树状详细大纲，平均每故事约3500词；与Re³扁平/固定大纲的本质区别在于逐层按需展开，支持随目标长度动态调整。
2. **基于对比训练的token级控制器（Detailed Controller，OPT-350m）**：在token层面逐步引导生成而非仅依赖初始prompt或事后rejection sampling；与FUDGE的关键区别在于引入了更Hard negatives构造（句边界截断拼接）与Harder positives混合，使控制器学会维持全程相关性。
3. **三类动态控制约束机制**：在drafting时按大纲节点类型注入Event/Setting/Character三类summary约束，并采用从0递增的control strength配合early stopping动态调整；与先前工作（无细粒度过程控制）的本质区别在于过程级而非结果级的引导。
4. **角色历时发展追踪**：每次角色出现在大纲节点时推断其新事实并累积到prompt中；与现有工作仅维护静态角色卡的区别在于支持跨节点角色演化。
5. **移除Re³的耗时editing步骤**：在保持相同OPT-175B generator与高层大纲/场景/角色设置的前提下，通过outliner+controller替代editing，大幅降低生成延迟。

## 方法详解
**整体架构**：DOC = Detailed Outliner + Detailed Controller，复用Re³的planning-drafting-revision范式，移除editing。

### 1. Detailed Outliner（详细大纲生成）
- **层次化展开**：从root节点（premise）开始，逐层生成子节点，每个父节点生成2-3个子事件，深度可调（本文设为3），分支因子2-5。
- **Event Candidate Generation**：使用结构化prompt注入祖先节点及兄弟节点上下文；通过GPT-3 Insertion API + InstructGPT3-175B (text-davinci-002) 生成候选事件；过滤去除格式错误及与已有节点高度重复（word overlap + entailment model, Laurer et al., 2022）；Reranking时首个子节点用Sentence-BERT语义相似度，其余用基于roberta-large微调的ordering model。
- **Setting & Character Detection**：对每个大纲节点用InstructGPT3-175B检测场景设置和相关角色，并与初始角色清单匹配以生成新增角色。
- **Character Development Over Time**：每次角色出现在大纲节点时推断其新事实并过滤已蕴含事实；drafting时prompt中累积该角色截至当前节点的所有已推断事实（用红色标记）。

### 2. Detailed Controller（对比训练与引导生成）
- **基础模型**：基于FUDGE的OPT-350m，在token级别逐步引导生成。
- **Contrastive Training数据构建**：用InstructGPT3-13B对WritingPrompts (Fan et al., 2018) 段落做摘要，得到passage-summary对。
- **三难度样本构造**：
  - *Hard negatives*：与同故事其他段落匹配的prefix。
  - ***更Hard negatives***：将正样本在句边界处截断，后半段替换为同故事另一段落开头，迫使生成器学习维持全程相关性而非仅局部衔接。
  - ***Harder positives***：用负prefix + 正completion混合构造。
- **训练目标**：OPT-350m预测passage prefix是否与给定summary匹配。
- **三类控制约束（drafting时注入）**：
  1. **Event**：直接传入事件描述（高控制强度）。
  2. **Setting**：若场景变化，构造"角色移动到场景X"（较低强度）。
  3. **Character**：若新角色出现，构造相应summary（较低强度）。
- **Control Strength动态调整**：每个大纲节点从0开始递增，满足early stopping后重置。

### 3. Drafting with Detailed Outlines
- 对每个**叶子节点**生成变长passage（非Re³的固定长度），使用Re³的outline relevance + text coherence reranker做early stopping。
- Prompt组成：
  - 当前场景（紫色）
  - 角色信息（紫色）
  - 角色历时发展事实（红色）
  - 当前事件（橙色）
  - 下一大纲节点（绿色）——用控制器补偿避免提前生成未来事件

## 实验与结果
**数据集与配置**：
- 输入：30-60词英文premise（由InstructGPT3-175B生成）
- 输出：完整故事，平均约**3500词**
- Generator：OPT-175B（需深层模型访问以获取logits）
- Context window：1024 tokens
- 评估：人工pairwise comparison，每对3位annotator（Surge AI），指标：Coherent / Relevant / Interesting（百分率）

**基线方法**：
1. **Re³**（Yang et al., 2022）：使用相同OPT-175B，复用相同top-level大纲/场景/角色
2. **ROLLING-OPT**：OPT-175B + rolling window + 基本prompt

**主要结果（Table 1，20 stories自动评估）**：

| Method | Coherent | Relevant | Interesting |
|--------|----------|----------|-------------|
| Re³    | 45.1     | 37.1     | 39.4        |
| DOC    | **67.6** | **65.3** | **60.1**    |
| ROLLING-OPT | 38.0 | 25.4 | 25.4 |

- 相对Re³绝对增益：**Coherent +22.5%**，**Relevant +28.2%**，**Interesting +20.7%**（p<0.05）

**人机交互评估（Table 4，20 runs，每人最多15分钟编辑）**：

| Metric | Re³ | DOC |
|--------|-----|-----|
| Intent（忠实作者意图） | 17.3% | **80.0%** |
| Control（可控性） | 5.0% | **80.0%** |
| Intuition（直觉友好） | 5.0% | **80.0%** |
| Quality（质量偏好） | 15.0% | **75.0%** |

**Ablation（Table 5）**：
- DOC vs DOC-NOOUTLINE：Coherent 73.5% vs 61.8%，证明详细大纲的核心作用。

**定性案例观察**：
- DOC故事整体情节合理、能跟随大纲（Plan 2-5），角色命名一致（Jenna Adams、Brian Johnson等）。
- Re³存在"repetitive at times"问题；ROLLING-OPT"struggles heavily to maintain overarching plot coherence"，叙述跳跃明显。
- DOC部分存在scene detection failures（大纲节点被误判为地理位置场景）。

## 相关工作脉络
1. **Re³ (Yang et al., 2022)**：先前最强story generation工作，采用planning-drafting-revision三阶段；本文复用其框架但移除editing步骤，并通过详细大纲+token级控制器替代，本质区别在于**过程级细粒度控制**而非事后修正。
2. **FUDGE (Yang and Klein, 2021)**：token-level guided decoding基线；本文在此基础上引入更Hard negatives构造（句截断拼接）与Harder positives混合，使对比训练能学到**全程相关性**而非仅局部衔接。
3. **ROLLING-OPT**：简单rolling window长文本生成baseline；实验证明无结构化规划时连贯性严重下降，凸显结构化大纲的必要性。
4. **Neural Story Generator / Dreamer等**：分段笔记提及但未直接对比；本文定位差异在于聚焦**长距离（数千词）连贯性**而非单段或短篇生成。
5. **Entailment/Relevance过滤 (Laurer et al., 2022)**：本文将其集成到大纲candidate过滤环节，区别于仅依赖word overlap的先前工作。

## 局限性与未来方向
- **英语限定**：实验仅在英语上进行，跨语言泛化未验证。
- **低级错误**：大纲生成温度较高，导致拼写错误（如"regaines"）、被动语态异常。
- **低层次细节忠实度不足**：部分节点事件被语义替换或省略（如Plan 4中"与敌人战斗"生成成了"看熊打架"；Plan 1中Lisa研发治愈药情节缺失）。
- **场景检测缺陷**：部分大纲节点被误判为地理位置场景（scene detection failures）。
- **角色一致性待改进**：深层模型访问需求限制了实际部署效率。
- 未来方向可包括：降低temperature以提升低级质量、改进scene detection模块、探索多语言支持、结合更强generator（如更大OPT或GPT-4）。

## 研究启发与可借鉴点
1. **Hard negatives构造技巧可迁移**：句边界截断拼接（正样本后半段替换为同故事另一段落开头）是提升对比训练效果的有效手段，可推广至其他sequence generation任务中的相关性建模。
2. **Control strength动态调整策略**：每节点从0递增+early stopping重置，避免过早锁定生成路径，对长文本生成的探索-利用平衡有参考价值。
3. **角色历时发展追踪机制**：累积推断事实并注入prompt，可与本团队的角色一致性研究结合，拓展至对话系统或多智能体叙事。
4. **分层大纲思想**：Premise→Outline→Scene/Characters的层次化展开可迁移至其他结构化生成任务（如剧本、技术文档）。
5. **移除editing的步骤简化**：证明通过outliner+controller的前置精细规划可替代事后修正，为降低生成延迟提供思路。

## 关键术语表
- **DOC (Detailed Outliner + Detailed Controller)**：本文提出的长故事生成框架，通过层次化大纲生成与token级对比控制器提升连贯性。
- **Contrastive Training**：训练控制器区分"与summary匹配的prefix"与"不匹配的prefix"，使生成过程受控。
- **Hard Negatives（句截断拼接）**：将正样本在句边界截断并替换后半段，构造更难的负样本以强化全程相关性学习。
- **Control Strength**：三类约束（Event/Setting/Character）的注入强度，动态从0递增以避免过早锁定。
- **Early Stopping（reranker-based）**：基于outline relevance + text coherence reranker判断当前passage生成是否达标，决定是否继续。
- **Character Development Over Time**：累积记录角色在各大纲节点推断出的新事实，在drafting prompt中注入以实现角色一致性演化。
- **Premise**：30-60词的英文故事初始设定，由InstructGPT3-175B生成。
- **Surge AI**：本文用于人工评估的平台（优于MTurk），提供每对3位annotator的pairwise comparison。

## 可复现要素
- **数据集**：WritingPrompts (Fan et al., 2018, MIT License)，仅使用此一个预训练数据集；评估用20个故事。
- **代码/权重**：论文未明确声明开源；Generator为OPT-175B，Controller为OPT-350m（HuggingFace, Wolf et al., 2020, Apache 2.0）；GPT-3 API调用（InstructGPT3-175B / 13B）。
- **关键超参**：大纲深度=3，分支因子2-5，context window=1024 tokens，平均每故事约3500词；评估温度为"较高"（导致低级错误）。
- **评估配置**：人工pairwise，每对3位annotator（Surge AI），指标Coherent/Relevant/Interesting。
