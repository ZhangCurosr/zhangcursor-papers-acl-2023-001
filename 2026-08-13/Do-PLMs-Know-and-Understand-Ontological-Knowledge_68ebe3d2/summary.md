---
title: "Do-PLMs-Know-and-Understand-Ontological-Knowledge"
source: https://aclanthology.org/2023.acl-long.173.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:36:17"
field: "语言模型知识探测与解释"
keywords: ["本体知识", "知识探测", "预训练语言模型", "prompt探测", "RDFS蕴含", "逻辑推理"]
innovations: ["构建首个系统性PLM本体知识记忆与推理探测基准OntoProbe-PLMs", "提出基于前提记忆分档（EX/IM/NO）的三维推理实验设计", "揭示PLMs对本体paraphrase理解的系统性不足"]
benchmarks: ["OntoProbe-PLMs", "DBpedia", "Wikidata"]
---

# 论文速读：Do PLMs Know and Understand Ontological Knowledge

## 一句话总结
本文系统探测预训练语言模型（PLMs）是否记忆本体知识（类的层次、属性约束等）并能否基于这些知识进行逻辑推理，发现PLMs能记忆部分本体知识且有一定推理能力，但两者均不完美，表明其对本体知识的理解有限。

## 研究问题与动机
1. **现有探测研究偏向事实性知识**：已有PLM探测工作主要关注实体实例级的事实知识（如"梅西是哪国人"），缺乏对类层级和属性约束等**本体知识**的系统性探测。
2. **本体知识在NLP任务中至关重要**：本体知识（类与属性及其关系）是许多NLP任务（如问答、知识注入）的基础，却未被充分研究PLMs是否真正编码了此类知识而非仅记忆表面形式。
3. **记忆≠理解**：需要区分PLMs是死记硬背了表面形式，还是真正理解了本体语义并能够据此进行逻辑推理。
4. **大模型规模不一定带来更多本体知识**：文献表明参数规模更大的模型存储更多事实知识，但本研究对比例子发现并非所有情况下大模型都能更好地记忆本体知识。

## 核心贡献（创新点）
1. **构建了首个系统评估PLM本体知识记忆与推理的基准（OntoProbe-PLMs）**：设计了涵盖实体类型（TP）、类层次（SCO）、属性层次（SPO）、域约束（DM）、值域约束（RG）的记忆任务，以及基于RDFS 6条蕴含规则的推理任务。
2. **提出了基于前提显式/隐式/未给定三种设置的多维度推理实验设计**：相比Talmor等仅区分前提是否给定，本文按前提被模型记忆的程度进行分类，形成3³=27种组合，更精细地检验推理能力。
3. **揭示了PLMs在本体知识上的系统性不足**：发现PLMs能记忆部分本体知识但非完美；推理能力虽优于无前提基线，但远未达完美；且对属性语义的 paraphrase 理解存在明显困难。
4. **首次对ChatGPT进行了本体知识记忆与推理的初步探测**：通过多项选择题评估，发现ChatGPT在记忆和推理任务上均显著优于BERT-base-uncased。

## 方法详解
### 本体构建
- **类与实例**：从DBpedia获取783个类及各自的20个实例，通过`subclass-of`关系获取超类。
- **属性**：从Wikidata收集属性，通过`subproperty of(P1647)`获取超属性，通过`property constraint(P2302)`获取域/值域约束，再与DBpedia对齐，最终选取50个具合理约束的属性。

### 记忆任务（5个子任务，均以cloze填空形式呈现）
1. **TP（Type Prediction）**：给定实体问其类型类，如"Messi是__[MASK]__"。
2. **SCO（Superclass of）**：给定类问其直接超类，如"Person是__[MASK]__的子类"。
3. **SPO（Subproperty of）**：给定属性问其直接超属性。
4. **DM（Domain）**：给定属性问其约束的主语类型，如"要成为__[MASK]__才能是运动队成员"。
5. **RG（Range）**：给定属性问其约束的宾语类型。

### 推理任务（6条RDFS蕴含规则）
| 规则 | 推理类型 |
|------|----------|
| rdfs2 | 域约束→类型推断 |
| rdfs3 | 值域约束→类型推断 |
| rdfs5 | 属性传递性 |
| rdfs7 | 属性继承（含paraphrase）|
| rdfs9 | 类的传递性 |
| rdfs11 | 子类传递性 |

### Prompt设计
- **Manual Template**：人工设计的离散模板（见论文Table 3）。
- **Soft Template**：使用3个可学习soft token（<s1><s2><s3>）替代模板中的连接词。
- **Multi-mask vs Single-mask**：多token候选使用多个[MASK]分别预测每个token，再经mean/max/first pooling聚合得分。
- **Pseudowords**：推理任务中将实例替换为人工构造的无意义词，采样方式是在静态embedding空间中取与[MASK]距离足够远的位置，系数α=0.5。

### 评估指标
- **R@K**：Top-K召回率。
- **MRR**：平均倒数排名，针对多标签情况引入**MRRₐ**（对所有gold标签取平均排名）。
- **Premise分类**：将每个前提按记忆任务中的排名分为"已记忆"（前50%）和"未记忆"（后50%），形成EX/IM/NO三档设置。

### ChatGPT评估
由于ChatGPT是decoder-only模型，采用多项选择题形式（1个正确选项+19个随机负例），评估其记忆准确率；推理任务中因无法输入embedding，用字母X/Y作为pseudowords。

## 实验与结果
### 数据集规模
- TP：train 10, dev 10, test 8789
- SCO：train 10, dev 10, test 701
- SPO：train 10, dev 10, test 39
- DM：train 10, dev 10, test 30
- RG：train 10, dev 10, test 28

### 记忆任务核心结果（Table 4）
- **最佳PLM表现**：BERT-base-uncased soft template在多数子任务上显著优于频率基线（43%~198%提升），但在DM（域约束）上基线MRR为50.9%，最佳PLM仅50.3%，略逊于基线。
- **最佳模型**：BERT-large-uncased在TP的MRRₐ达49.3%，SPO的MRRₐ达76.9%，DM的MRRₐ达66.8%。
- **规模悖论**：BERT-large-uncased在TP和DM上表现不如BERT-base-uncased；RoBERTa-large在TP和DM上也弱于RoBERTa-base。
- **Soft prompt有效**：soft template在TP、SCO、SPO上普遍优于manual template。

### 推理任务核心结果（Figure 2/3）
- **无前提（NO/NO）**：MRR接近随机水平（13.5%）。
- **P₁隐式给定（IM/NO）**：MRR提升至82.8%，说明模型能利用隐式记忆的知识进行推理。
- **P₁显式给定（EX/NO）**：MRR达97.1%，但可能混杂了priming效应。
- **规则rdfs7（属性paraphrase继承）**：当P₁含paraphrased属性描述时，BERT-base-cased的MRR相比rdfs5下降23%~49%，说明模型对属性语义理解有限。
- **Soft conjunction tokens**：图4显示，训练好的连接token（而非关系描述模板）对推理有帮助。

### ChatGPT结果（Table 5/6）
- **记忆任务**：ChatGPT在所有5个子任务上均显著优于BERT-base-uncased，其中DM达86.7% vs 70.0%，RG达82.1% vs 82.1%（持平）。
- **推理任务**（P₂显式给定，P₁按记忆分档）：ChatGPT在IM-P₁条件下平均MRR为82.8%，在EX-P₁条件下达97.1%。

## 相关工作脉络
1. **Petroni et al. (2019) "Language Models as Knowledge Bases?"**：开创了基于cloze prompt的事实知识探测范式，本文扩展至本体层级知识（类/属性关系），而非仅实体-属性-值的三元组事实。
2. **Jiang et al. (2020) TAPAS**：研究prompt扰动对探测结果的影响，指出模板敏感性是探测方法的核心挑战；本文通过soft template和pseudoword设计缓解此问题。
3. **Talmor et al. (2020) "LEAP-of-thought"**：提出从隐性知识出发做推理的框架，但仅区分前提是否显式给定；本文将其细化为EX/IM/NO三档，以区分隐式知识的记忆质量对推理的贡献。
4. **Karidi et al. (2021)**：提出pseudoword采样方法（从静态embedding空间取样），本文沿用并扩展至推理任务中以排除模型对具体词汇的记忆干扰。
5. **Peng et al. (2022) "CoPEN"**：探测概念知识的另一工作，聚焦于类之间的关系，本文在此基础上增加了属性约束（域/值域）及逻辑推理任务的系统评估。
6. **Huang et al. (2022)**：探测actionable knowledge，同样基于prompt方法；本文强调本体知识是更基础的世界知识组织形式，与actionable知识形成互补。

## 局限性与未来方向
1. **数据集覆盖有限**：仅使用了DBpedia/Wikidata中的783个类和50个属性，远不能代表真实世界的本体规模。
2. **Paraphrase处理困难**：模型对属性的自然语言改写版本理解不佳，说明本体知识的语义表征仍有缺陷。
3. **仅评估encoder模型**：主要聚焦BERT/RoBERTa，虽补充了ChatGPT的初步评估，但未系统比较不同decoder架构。
4. **Future方向**：改进预训练方法以增强本体知识的记忆与理解；探索更好的prompt设计（如自动生成的soft conjunction）；扩展至更大规模、更多语言的本体基准。

## 研究启发与可借鉴点
1. **Prompt模板对探测结果影响显著**：soft template在类层次任务上普遍优于manual template，但在复杂语义任务（如域约束）上增益有限；设计探测任务时应同时报告多种模板的鲁棒性。
2. **Pseudoword+embedding采样法可有效控制记忆干扰**：通过限制新词与现有token的最小距离（α=0.5），既能保证词汇无语义偏向，又能维持其在向量空间中的可读性，是推理任务设计的良好实践。
3. **前提分档（EX/IM/NO）是区分"记忆"与"推理"的有效手段**：相比单纯给/不给前提，按记忆质量划分能更精确地量化推理能力的来源，值得推广到其他推理探测任务。
4. **连接token的优化**：图4发现训练好的conjunction token比关系描述模板更能提升推理，提示后续工作可探索可学习的语义连接词替代固定连接词（如"therefore"）。
5. **大模型≠更多本体知识**：规模效应在本体记忆上并不稳定，提示在知识密集任务中除 scaling law 外还需关注本体对齐的预训练策略。

## 关键术语表
- **Ontological Knowledge（本体知识）**：由类（classes）和属性（properties）及其层次关系、约束（域/值域）组成的知识体系，是对世界知识的结构性建模。
- **Probing（探针测试）**：通过在冻结的PLM上训练轻量级分类器或使用prompt，检测模型内部是否编码了特定类型的知识。
- **Entailment Rules（蕴含规则）**：RDFS形式系统定义的逻辑推导规则，如传递性、继承性等，用于验证模型是否能进行逻辑推理。
- **Pseudoword（伪词）**：人工构造的无意义词汇，用于替代真实实例以避免模型依赖表面形式的记忆而进行纯粹推理。
- **Soft Prompt（软提示）**：由可学习的连续向量（soft tokens）组成的提示，而非离散的自然语言模板。
- **MRRₐ（Mean Reciprocal Rank averaged）**：改进的MRR指标，对所有gold标签取平均排名，适用于多标签分类的探测任务。
- **Domain Constraint（域约束）**：属性所允许的主语类型约束，如"Member of Sports Team"的主语应为Person。
- **Range Constraint（值域约束）**：属性所允许的宾语类型约束，如"Member of Sports Team"的宾语应为Sports Team。

## 可复现要素
- **数据集**：基于DBpedia和Wikidata构建，**代码和数据集已开源**：https://github.com/vickywu1022/OntoProbe-PLMs
- **模型**：BERT-base-cased/uncased, BERT-large-cased/uncased, RoBERTa-base/large（来自Hugging Face）
- **框架**：OpenPrompt (Ding et al., 2022)
- **关键超参**：soft token训练100 epochs，learning rate=0.5，AdamW优化器，linear warmup；pseudoword采样距离系数α=0.5；loss函数选用BCEWithLogitsLoss。
- **评估指标**：R@1, R@5, MRR, MRRₐ
- **ChatGPT**：使用GPT-3.5-turbo API，多项选择题（1正确+19随机负例）
