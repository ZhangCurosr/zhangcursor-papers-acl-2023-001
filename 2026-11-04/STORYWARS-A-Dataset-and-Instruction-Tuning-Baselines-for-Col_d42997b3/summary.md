---
title: "STORYWARS-A-Dataset-and-Instruction-Tuning-Baselines-for-Col"
source: https://aclanthology.org/2023.acl-long.171.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 09:14:12"
field: "故事理解与生成"
keywords: ["协作故事", "指令微调", "多任务学习", "故事生成", "数据集构建", "零样本学习"]
innovations: ["首次将指令微调扩展到全监督场景并获得稳定提升", "发布40k级开放域协作故事数据集STORYWARS", "理解与生成混合指令微调策略验证"]
benchmarks: ["STORYWARS", "GLUE", "BIGbench", "LOT"]
---

# 论文速读：STORYWARS-A-Dataset-and-Instruction-Tuning-Baselines-for-Col

## 一句话总结
论文提出了STORYWARS数据集（40,135篇来自9,494位作者的开放域协作故事）及包含12种任务类型、101个任务的统一多任务基准，并首次证明指令微调可作为全监督多任务预微调策略——INSTRUCTSTORY模型在全监督、少样本、零样本三种场景下均取得最优或接近最优性能。

## 研究问题与动机
- 协作故事由多位作者接力创作，每段章节体现独立作者的写作风格与意图，且无预设情节大纲，对NLP模型的理解与生成能力提出双重挑战。
- 现有故事数据集（ROCStories、WritingPrompts）均非协作式；已有的协作数据集（Storium、roleplayerguild）规模小且多为游戏设定，缺乏大规模开放域协作故事语料。
- 当前多任务NLP基准多聚焦单一理解或生成能力，且仅覆盖部分学习场景（如BIGbench仅零/少样本），缺乏同时涵盖理解与生成、贯通全监督/少样本/零样本的故事基准。
- 指令微调此前仅验证于零样本/少样本场景，其能否提升全监督任务性能尚属未知。

## 核心贡献（创新点）
- 发布STORYWARS数据集：40,135篇开放域协作故事、9,494位作者，是首个大规模用户生成协作故事语料，包含类型标签与人工评分等多源信息。
- 设计101任务多任务基准（7种理解+5种生成），同时覆盖全监督、少样本、零样本三种学习场景，填补故事领域多场景基准空白。
- 提出INSTRUCTSTORY两阶段训练框架：以T5-large-lm-adapt为基础，先进行63个故事任务的混合指令微调，再进行各全监督任务的单任务微调，首次将指令微调扩展到全监督场景并取得稳定提升。
- 消融证明理解+生成混合指令微调效果优于单一类型，为多任务预训练策略提供了新的设计原则。

## 方法详解
**数据集构建 pipeline：**
从storywars.net抓取76k故事，经FastText语言过滤→GPT-2 perplexity噪声去除→长度过滤（故事≥30词、章节≥10词）→低评分过滤→OpenAI Content Moderation API + Detoxify毒性检测→隐私信息替换（<URL>、<EMAIL>、<PHONE>），最终保留40,135篇故事。

**基准任务设计（12类）：**
- 理解任务（7类）：Genre Classification（判断类型，27全监督+10少样本+23零样本）、Authorship Attribution（30全监督）、Authorship Verification、Connectivity Inference、Temporal Inference、Story Scoring（回归，归一化至0-10，Spearman相关系数评估）、Story Segmentation（Boundary Similarity评估）。
- 生成任务（5类）：Next Chapter Generation、Conditional Story Generation、Chapter Infilling（给定前后章节生成中间）、Global Infilling（给定全部上下文生成缺失章节）、Temporal Ordering（重新排列打乱章节）。
- 所有场景数据严格不重叠；生成任务用BERTScore评估（依据Chhun et al. 2022表明其优于BLEU/ROUGE）。

**INSTRUCTSTORY两阶段训练：**
- Stage 1：以T5-large-lm-adapt（770M参数）为基础，在63个故事任务混合数据上指令微调5 epochs，lr=5e-5、batch size=64、max seq len=1024、Adam、warmup=2000 steps。
- Stage 2：对每个全监督任务在Stage 1权重基础上单独微调10 epochs，按dev集最佳checkpoint选择。
- 指令模板示例："Please write a next chapter for the above story:" "Does Chapter A happen before Chapter B?" 等（见附录Table 7）。

## 实验与结果
- **基线**：BERT-large、RoBERTa-large、DeBERTav2-xlarge（理解）；GPT2-medium、GPT2-large、OPT-350m（生成）；T5-large-lm-adapt、FLAN-T5-large（综合）。
- **全监督（Table 3）**：INSTRUCTSTORY整体平均69.93 vs T5 68.40（+1.53）；理解任务59.62 vs T5 57.56（+2.06），优于BERT/RoBERTa/DeBERTa；生成任务84.35 vs OPT-350m 85.00（略低，推测因OPT预训练语料更大35B vs 180B tokens）；在27个Genre Classification中胜24个，在30个Author Attribution中胜23个。
- **少样本（Table 4）**：INSTRUCTSTORY 61.44最优，FLAN-T5 59.45次之，优于所有单模型基线。
- **零样本（Table 5）**：INSTRUCTSTORY 60.00 vs T5 32.09（+27.91）vs FLAN-T5 47.79（+12.21），显著提升。
- **Ablation（Table 6）**：IS > IS_U > IS_G > T5，验证理解+生成混合指令微调最有效。

## 相关工作脉络
- **ROCStories / WritingPrompts**：经典故事数据集，但均非协作式，缺乏多作者接力创作特征。
- **Storium / roleplayerguild**：协作故事数据集，但规模小（1,439/5,743篇）且为游戏设定，非开放域。
- **LOT benchmark**：中文故事理解与生成基准，但仅限中文、任务数量远少于STORYWARS。
- **BIGbench**：204任务理解+生成基准，但仅覆盖零/少样本，无全监督微调场景。
- **FLAN / FLAN-T5 / T0**：指令微调代表作，仅面向零/少样本泛化，未探索全监督场景。
- **Muppet / ExT5**：多任务预微调方法，本文在其思想基础上首次将指令微调有效扩展至全监督。

## 局限性与未来方向
- 单任务微调可能引发灾难性遗忘（catastrophic forgetting），可探索Rehearsal等技术但需适配全监督场景。
- 每个下游任务需单独微调出一个模型，在线服务部署成本较高，需权衡性能与资源。
- BERTScore虽与人工评估相关较好，但对高度创造性的协作故事输出仍不够精细，需发展更完善的评估指标。
- 伦理层面：AI生成故事可能延续训练数据中的偏见与刻板印象，需关注公平代表性。

## 研究启发与可借鉴点
- **指令微调作为全监督多任务预微调策略**：本文首次实证证明指令微调可稳定提升后续单任务微调性能，为多任务预训练提供新思路。
- **理解+生成混合训练的有效性**：消融结果提示多任务学习中应兼顾理解与生成两类任务，避免单一能力偏置。
- **协作故事的多源信号利用**：标题、类型、评分、作者信息等结构化元数据可启发多模态故事理解模型的设计。
- **UGC内容清洗pipeline**：FastText+perplexity+内容审核API+毒性检测的四层过滤策略，可作为用户生成文本数据处理的通用参考。
- **长程依赖评估任务设计**：Connectivity/Temporal Inference等任务为长文本模型的上下文建模能力提供了可量化的评估视角。

## 关键术语表
- **Collaborative Storytelling**：多人接力创作故事，每位作者基于前文续写并表达个人意图，无预设情节大纲。
- **Instruction Tuning**：以指令格式对预训练模型进行多任务微调，提升零样本/少样本泛化能力。
- **Collaborative Floor**：原作者为下一位作者留下的续写提示/引子，引导协作创作方向。
- **Infilling Task**：给定前后上下文生成缺失中间章节的任务，评估模型填补叙事空白的能力。
- **BERTScore**：基于BERT embeddings的文本生成评估指标，在故事生成中与人工评估相关性优于BLEU/ROUGE。
- **Catastrophic Forgetting**：模型微调特定任务时丧失先前多任务学习所得泛化能力的现象。
- **Boundary Similarity**：评估章节边界分割质量的指标（Fournier, 2013）。
- **Spearman Correlation**：用于Story Scoring回归任务的评估指标，衡量单调关系。

## 可复现要素
- 数据集：STORYWARS来源于storywars.net爬虫，论文未声明是否开源（需联系作者或平台确认）
- 代码/权重：论文未提及开源
- 关键超参：T5-large-lm-adapt（770M参数），lr=5e-5，batch size=64，指令微调5 epochs + 单任务微调10 epochs，max seq len=1024，Adam optimizer，warmup=2000 steps，Stage 1训练63个故事任务
