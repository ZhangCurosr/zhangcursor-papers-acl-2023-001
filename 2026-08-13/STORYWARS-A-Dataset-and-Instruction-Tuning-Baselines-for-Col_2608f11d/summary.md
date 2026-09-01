---
title: "STORYWARS-A-Dataset-and-Instruction-Tuning-Baselines-for-Col"
source: https://aclanthology.org/2023.acl-long.171.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:57:59"
field: "对话与故事生成"
keywords: ["协作故事", "指令微调", "多任务学习", "故事理解", "故事生成", "零样本学习"]
innovations: ["提出首个大规模开放域协作故事数据集STORYWARS（4万+故事，9千+作者）", "构建覆盖理解+生成、全监督+少样本+零样本的101任务统一基准", "首次证明指令微调结合单任务精调在全监督场景下可稳健提升多任务性能"]
benchmarks: ["STORYWARS", "BIG-bench", "LOT benchmark"]
---

# 论文速读：STORYWARS: A Dataset and Instruction Tuning Baselines for Collaborative Story Understanding and Generation

## 一句话总结
本文提出了**STORYWARS**——一个包含4万+协作故事的新数据集，并基于其构建了覆盖理解与生成、全监督/少样本/零样本三种场景的101任务多任务基准；同时提出**INSTRUCTSTORY**模型，证明指令微调（Instruction Tuning）在全监督场景下也能通过"先多任务指令微调、再单任务精调"的两阶段策略持续提升性能。

## 研究问题与动机
- **协作故事理解与生成缺乏开放域语料**：现有故事数据集（ROCStories、WritingPrompts）均为单作者创作，无法反映多作者协作场景；已有的协作故事数据（Storium、roleplayerguild）属于游戏设定，非开放域。
- **现有NLP多任务基准偏科明显**：主流基准要么只关注理解（GLUE/SuperGLUE），要么只关注生成（GEM、BIG-bench仅测零样本/少样本），缺乏同时在理解+生成+三种学习范式上覆盖的统一基准。
- **指令微调未被验证于全监督场景**：指令微调在零样本/少样本上表现优异，但在全监督设置下是否能带来提升尚不清楚。
- **协作故事比传统故事生成更具挑战性**：无预定义情节大纲，每个作者需在理解前文上下文、风格与意图的基础上续写，要求模型同时具备深层理解与生成能力。

## 核心贡献（创新点）
- **开源了首个大规模开放域协作故事数据集STORYWARS**（40,135个故事、9,494位作者，含类型标注与人工评分），区别于已有数据的封闭游戏场景或单作者创作。
- **构建了涵盖7个理解类+5个生成类的101任务多任务基准**，首次在同一基准中完整覆盖全监督、少样本、零样本三种学习场景（此前工作多为子集）。
- **提出INSTRUCTSTORY两阶段训练框架**：第一阶段在多任务指令微调上预训练，第二阶段对每个全监督任务做单任务精调，首次证明了指令微调对全监督性能有稳健提升（平均+1.53分）。
- **消融实验揭示了理解+生成混合指令微调的优势**：IS > IS_U > IS_G > T5，在多范式下一致成立。

## 方法详解

### 数据集构建
- 从storywars.net抓取76k个故事，经FastText语言识别筛选英文，以GPT-2 perplexity过滤噪声，去除短故事（<30词）和短章节（<10词），使用OpenAI Content Moderation API和Detoxify毒性分类器过滤有害内容，匿名化URL/邮箱/电话后得到**40,135个故事**。
- 每故事五元组表示：$s = (p, (c_i, a_i)_{i=0}^{t}, g, r_l, r_s)$，含标题、章节-作者对、类型、点赞数、星级评分。

### 12类任务设计（共101个具体任务）
**理解任务（7类，96个任务）：**
1. **Genre Classification**（类型分类）：预测故事是否属于特定类型，$g = f(c_1,...,c_t)$，含27全监督+10少样本+23零样本子任务
2. **Authorship Attribution**（作者归属）：判定章节作者 $a = f(c)$，30个子任务
3. **Authorship Verification**（作者验证）：判断两章节是否同一作者 $y = f(c_i, c_j)$，二分类
4. **Connectivity Inference**（连贯性推理）：判断两章节是否连续
5. **Temporal Inference**（时序推理）：判断两章节是否按正确时间顺序排列
6. **Story Scoring**（故事评分）：回归任务，将评分归一化至0–10，预测likes/stars
7. **Story Segmentation**（故事分段）：识别章节边界 $b_i$

**生成任务（5类，5个任务）：**
1. **Next Chapter Generation**：给定前k章+类型，生成第k+1章
2. **Conditional Story Generation**：生成剩余完整故事
3. **Chapter Infilling**：给定前后各一章，填补中间章
4. **Global Infilling**：给定全文其他章节，填补缺失章
5. **Temporal Ordering**：对乱序章节序列重新排序

### 评估指标
- 理解任务（分类/推理）：**F-1 score**（处理数据不均衡）
- 故事评分：**Spearman相关系数**
- 故事分段：**Boundary Similarity**
- 生成任务：**BERTScore**（优于BLEU/ROUGE，与人类评价相关性更高）
- 跨任务比较采用各任务类型内macro-average

### INSTRUCTSTORY训练框架
- **基座模型**：T5-large-lm-adapt（770M参数）
- **第一阶段（指令微调，99K步/5 epochs）**：使用63个故事任务的全监督训练集进行指令微调，以指令模板替代Muppet的任务前缀，增强泛化到未见任务的能力
- **第二阶段（单任务精调）**：在指令微调基础上，对每个全监督任务单独做fine-tuning（10 epochs，lr=5e-5，batch_size=64）
- 关键超参：Adam优化器，lr=5e-5，warmup=2000，max_seq_len=1024

## 实验与结果

### 基线模型
- 理解：BERT-large（345M）、RoBERTa-large（354M）、DeBERTa-v2-xlarge（900M）
- 生成：GPT2-medium（345M）、GPT2-large（774M）、OPT-350m
- 通用：T5-large-lm-adapt（770M）、FLAN-T5-large（少样本/零样本比较）

### 全监督结果（Table 3）
- **INSTRUCTSTORY整体平均69.93**，较T5 baseline（68.40）提升**+1.53分**
- 理解任务平均59.62，优于所有理解基线（BERT 51.90、RoBERTa 59.43、DeBERTa 57.39、T5 57.56）
- 生成任务平均84.35，优于T5（83.58），略低于OPT-350m（85.00）
- 27个类型分类子任务中24个超越T5；30个作者归属子任务中23个超越T5

### 少样本结果（Table 4）
- **INSTRUCTSTORY平均61.44**，超越FLAN-T5（59.45）及所有T5/BERT系列基线

### 零样本结果（Table 5）
- **INSTRUCTSTORY平均60.00**，较T5（32.09）提升**+27.91**，较FLAN-T5（47.79）提升**+12.21**

### 消融（Table 6）
- 理解+生成混合指令微调效果最佳：IS > IS_U > IS_G > T5，在三种场景下均成立

## 相关工作脉络
- **ROCStories / WritingPrompts**：广泛使用的单作者故事数据集，与本文的最大区别是不含协作和多轮互动元素。
- **Storium / roleplayerguild**：协作故事数据但处于游戏设定（game setting），本文数据为开放域、无游戏约束。
- **LOT benchmark**：同时涵盖理解与生成的故事基准，但仅覆盖中文且任务数较少（本文101任务）。
- **BIG-bench**：204个任务的广泛基准，但仅测试零样本/少样本能力，不含全监督精调评估。
- **Muppet / ExT5**：多任务微调路线，本文与之不同在于引入了指令微调作为预训练阶段，并扩展到全监督场景。
- **FLAN / T0 / FLAN-T5**：指令微调代表工作，专注零样本/少样本；本文首次将其能力延伸到全监督设置。

## 局限性与未来方向
- **灾难性遗忘问题**：单任务精调可能导致模型丢失多任务泛化能力，引入Rehearsal等技术可缓解但未被探索。
- **部署成本高**：每个下游任务需维护独立模型，实际应用中计算开销大。
- **自动评估指标的局限**：BERTScore等自动指标虽与人类评价有一定相关性，但对协作故事的高创造性输出评估仍不够充分，缺乏可靠的标准评估体系。
- **零样本/少样本基线较少**：受限于故事长度，无法使用in-context learning demonstrations，限制了与最新大模型的公平比较。

## 研究启发与可借鉴点
- **"指令微调 + 单任务精调"的两阶段策略**对全监督多任务场景有普适价值，可作为多任务预训练的实用范式迁移到其他领域。
- **理解与生成混合指令微调**的消融结论（IS > IS_U > IS_G）表明跨模态/跨能力任务混合训练优于单一能力，值得在更多领域验证。
- **协作故事任务设计**（如Global Infilling、Temporal Ordering）为长文本理解提供了新颖评估角度，可迁移至小说续写、剧本生成等场景。
- **从在线社区 scraped数据 + GPT-2 perplexity过滤**的清洗流程，为UGC数据的自动化质量筛选提供了可复用的方法论。

## 关键术语表
- **Collaborative storytelling**：多个作者依次接力创作故事，每个作者需理解前文并表达个人意图的写作形式。
- **Instruction Tuning**：通过自然语言指令对预训练语言模型进行多任务微调，使其能够泛化到零样本/少样本新任务的技术。
- **Connectivity Inference**：判断两个给定章节在原文中是否相邻的任务。
- **Global Infilling**：给定故事其余所有章节，生成被移除章节的逆向填空任务。
- **BERTScore**：基于BERT语义嵌入计算生成文本与参考文本之间相似度的自动评估指标。
- **Catastrophic Forgetting**：模型在连续学习新任务时，对之前所学知识的大幅遗忘现象。
- **Collaborative Floor**：上一位作者在章节末尾留下的写作引导/提示，供下一位作者延续故事。
- **FLAN**（Finetuned Language Models are Zero-shot Learners）：通过大规模指令微调实现零样本任务泛化的基准框架（Wei et al., 2021）。

## 可复现要素
- **数据集**：STORYWARS，从storywars.net抓取，论文未声明是否已开源（需进一步确认）
- **代码/权重**：INSTRUCTSTORY模型权重，论文未明确声明开源
- **关键超参**：batch_size=64，lr=5e-5，训练步数=50,000（第一阶段），warmup=2,000，max_seq_len=1024，epochs=5（指令微调）/10（单任务精调）
- **基座模型**：T5-large-lm-adapt（770M参数）
- **数据清洗工具**：FastText（语言识别）、GPT-2 perplexity（噪声过滤）、OpenAI Content Moderation API、Detoxify（毒性分类）
