---
title: "Do-PLMs-Know-and-Understand-Ontological-Knowledge"
source: https://aclanthology.org/2023.acl-long.173.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:58:33"
field: "知识探测与本体推理"
keywords: ["本体论知识", "预训练语言模型", "知识探测", "RDFS推理", "Soft Prompt", "伪词", "记忆与推理分离"]
innovations: ["构建首个系统性本体知识探测基准（记忆+推理双任务）", "提出三层前提分类框架（显式/隐式/未给定）精细分离记忆与推理", "基于伪词嵌入距离的推理探测方法排除表面记忆干扰"]
benchmarks: ["OntoProbe-PLMs (DBpedia+Wikidata)", "RDFS Entailment Rules Benchmark"]
---

# 论文速读：Do PLMs Know and Understand Ontological Knowledge?

## 一句话总结
本文系统探测了预训练语言模型（PLMs）是否编码并理解本体论知识（类、属性及其层级关系与约束），构建了包含记忆测试与基于RDFS蕴含规则的推理测试的基准，发现PLMs能记忆部分本体知识，但记忆与推理均非完美，体现出对本体语义的理解有限。

## 研究问题与动机
- **现有探测定向缺失**：既有PLM探测研究多聚焦于**实例级事实知识**（如Petroni et al., 2019），少数工作涉及概念类，但缺乏对**本体论知识**（类、属性、层级与约束）的系统性探测。
- **本体知识的必要性**：本体论知识是组织结构化世界知识的核心，广泛用于QA、知识注入等下游任务，因此需要回答PLMs是否真正"知道并理解"此类知识，而非仅靠表面形式死记。
- **记忆≠理解**：仅验证模型是否记忆了本体关系还不够，还需检验模型能否在给定前提下**可靠地执行逻辑推理**（依据RDFS蕴含规则），以判断其是否具备真正的语义理解。
- **规模效应存疑**：直觉上模型越大存储知识越多，但本文发现模型规模与本体的记忆/推理能力并非正相关，这一现象值得系统探究。

## 核心贡献（创新点）
1. **构建了首个系统性本体知识探测基准**：涵盖5个子任务（实体类型TP、类的超类SCO、属性的超属性SPO、属性域约束DM、属性值域约束RG）的记忆测试，以及6条RDFS蕴含规则的推理测试；本文与已有工作（如Petroni et al., 2019仅关注事实三元组）的本质区别在于研究对象从实例级事实扩展到了结构化本体知识。
2. **设计了细粒度的前提分类框架**：将推理前提区分为"明确给定（Explicitly Given）""隐式给定（Implicitly Given）""未给定（Not Given）"三种状态（基于前序记忆测试结果划分），细化了Talmor et al. (2020)仅区分显式/隐式的前提设定，从而更精确地分离模型的"记忆效应"与"推理能力"。
3. **提出基于伪词（Pseudowords）的推理探测方法**：用人工生成的伪词替代具体实例，避免模型依赖假设（hypothesis）的记忆，确保推理测试真正检验模型的逻辑推断能力而非表面复现；相比既有方法，这一设计在排除词汇提示干扰方面更为严谨。
4. **揭示了模型规模与本体知识存储的不对应关系**：发现BERT-large不如BERT-base、RoBERTa-large不如RoBERTa-base的本体记忆表现，打破了"越大越好"的直觉假设。
5. **对ChatGPT进行了初步探测**：作为额外分析，测试了GPT-3.5-turbo在本体记忆与推理上的表现，发现其显著优于BERT-base系列。

## 方法详解

### 本体构建（Section 2.1）
- **类（Classes）**：从DBpedia获取783个类，通过SPARQL查询其实例（type关系）和超类（subclass-of关系），每个类采样20个实例。
- **属性（Properties）**：从Wikidata获取属性，利用`subproperty of (P1647)`查询超属性，用`property constraint (P2302)`查询域/值域约束；再通过等价属性（P1628）与DBpedia属性对齐，最终筛选50个具有明确域、值域和超属性的属性。

### 记忆任务（Section 2.2）
五个子任务均以**完形填空（cloze-completion）**形式构造：
1. **TP（Type of instance）**：给定实体，预测其类型类。
2. **SCO（Superclass of class）**：给定类，预测其超类。
3. **SPO（Superproperty of property）**：给定属性，预测其超属性。
4. **DM（Domain constraint）**：给定属性，预测其域的约束类。
5. **RG（Range constraint）**：给定属性，预测其值域的约束类。

数据集划分（表1）：TP训练/开发/测试集分别为10/10/8789，SCO为10/10/701，SPO为10/10/39，DM为10/10/30，RG为10/10/28。采用few-shot设置（训练/开发各10样本）。

### 推理任务（Section 2.3）
基于RDFS蕴含规则构造6个子任务（表2），每条规则表示为 $\mathcal{P}_1 \wedge \mathcal{P}_2 \doteq \mathcal{H}$：
- **rdfs2/rdfs3**：通过域/值域约束进行类型推断。
- **rdfs5/rdfs11**：子类/子属性的传递性。
- **rdfs7**：通过子属性继承关系推断属性。
- **rdfs9**：通过子类关系继承类型。

前提按三种状态组合：Explicitly Given（EX）、Implicitly Given（IM）、Not Given（NO），形成 $3^2 = 9$ 种前提组合配置。

### 探测方法（Section 3）
- **Prompt模板**：包含人工设计模板（Manual Template，表3）和可学习的Soft Template（含3个soft token）。
- **候选评分**：采用多[MASK]与单[MASK]两种策略，配合mean/max/first三种pooling方法；实证表明mean pooling最优。
- **评估指标**：Top-K Recall（R@K）、Mean Reciprocal Rank（MRR）及改进版 $\mathrm{MRR}_a$（考虑所有gold标签的平均排名）。
- **伪词生成**：在静态嵌入空间中，取与[MASK] token距离 $d = 0.5 \times \min_{t \in V} \| \mathbf{z}_t - \mathbf{z}_{[MASK]} \|_2$ 的位置采样伪词嵌入，确保无特定语义。
- **前提分类**：根据记忆任务的倒数排名将前提分为"已记忆"与"未记忆"两组（各占一半）。
- **损失函数**：记忆任务使用BCEWithLogitsLoss，推理任务同样使用该损失。

### ChatGPT探测（Section 5）
采用多选题形式（20个候选），直接调用GPT-3.5-turbo API评估，因decoder架构无法使用mask-filling方式。

## 实验与结果

### 数据集与基线
- **数据源**：DBpedia + Wikidata，构建本体后生成记忆与推理数据集。
- **探测模型**：BERT-base-cased/uncased、BERT-large-cased/uncased、RoBERTa-base/large。
- **基线**：基于训练集频率的预测模型（Frequency Baseline），强于随机猜测。

### 记忆任务主要结果（表4）
- **总体趋势**：除DM任务外，PLMs在所有子任务上均显著优于频率基线（提升43%–198%）。
- **最佳性能**：
  - TP任务：BERT-large-uncased soft template下MRR达49.3%，R@1为37.9%。
  - SCO任务：BERT-large-uncased soft template下R@5达55.9%。
  - SPO任务：BERT-large-cased soft template下MRR达76.9%（MRRa），53.5%（MRR）。
  - DM任务：BERT-large-cased manual template下R@1达50.0%，接近但略低于基线MRR。
  - RG任务：BERT-large-cased soft template下MRR达66.8%。
- **规模悖论**：BERT-large-uncased在多项任务上弱于BERT-base-cased；RoBERTa-large在TP和DM上弱于RoBERTa-base。

### 推理任务主要结果（图2、图3、表6）
- **前提组合影响**：$\mathcal{P}_1$显式给定（EX/EX）时MRR最高（宏观平均约97%）；$\mathcal{P}_1$隐式给定（IM/EX）时MRR约83%；完全不给（NO/EX）时MRR仅约14%。
- **核心发现**：即使在最佳条件下，推理MRR也远未达到完美（< 1），表明PLMs对本体知识的理解有限。
- ** paraphrase挑战**：rdfs7规则中$\mathcal{P}_1$包含属性的释义版本（paraphrased property），相比rdfs5（表面形式），BERT-base-cased的MRR下降23%–49%，显示模型对属性语义的理解薄弱。
- **ChatGPT表现**（表5、表6）：在记忆任务上显著优于BERT-base-uncased（TP: 70.2% vs 42.6%，DM: 86.7% vs 70.0%）；在$\mathcal{P}_1$显式给定的推理任务中准确率达97.1%，远超BERT-base-uncased的88.2%。

## 相关工作脉络
- **Petroni et al. (2019)**：开创性地将PLMs作为知识库进行cloze probing，但仅关注实例级事实知识；本文将其扩展到结构化的本体论知识层面。
- **Jiang et al. (2020)**：系统评估prompt模板对事实探测的影响；本文沿用其框架但引入了软prompt与伪词技术以更好分离记忆与推理。
- **Talmor et al. (2020)**：探究PLMs基于隐含知识的推理能力，仅区分前提是否显式给出；本文进一步将前提划分为三种状态，更精细地刻画推理机制。
- **Hewitt & Liang (2019)**：提出带控制任务的线性分类器探测方法；本文采用prompt-based方法，更贴近实际NLP任务场景。
- **Goodwin et al. (2020)**：使用伪词探测语言系统性；本文借鉴并改进了伪词生成策略（基于与[MASK]的距离采样）。
- **Pimentel et al. (2020)**：基于信息论的探测方法；本文与其形成对比，侧重于基于prompt的完形填空探测范式。

## 局限性与未来方向
- **本体覆盖有限**：仅使用DBpedia和Wikidata中的部分类和属性（783个类、50个属性），未能覆盖真实世界中的全部本体知识。
- **记忆非完美，推理更弱**：即使最佳模型在记忆任务上也未达到100%，推理性能更是远低于完美水平。
- **Paraphrase处理困难**：模型难以处理属性的释义表达，反映出对属性深层语义的理解不足。
- **模型规模与非单调关系**：大模型并非在本体知识存储上必然优于小模型，机制尚不明确。
- **未来方向**：改进预训练方法以增强本体知识的学习；提升模型对释义表达的语义理解能力；探索更本体的知识组织方式。

## 研究启发与可借鉴点
1. **伪词生成策略可迁移**：基于与[MASK]嵌入距离采样伪词的方法（$d = \alpha \cdot \min_t \|\mathbf{z}_t - \mathbf{z}_{[MASK]}\|_2$，$\alpha=0.5$）可用于其他需要排除词汇记忆的推理探测场景。
2. **三层前提分类框架有价值**：将推理前提划分为Explicitly Given / Implicitly Given / Not Given，可以更精确地分离记忆效应与真正的推理能力，适用于其他知识推理探测工作。
3. **Soft Template + Soft Conjunction的分离设计**：研究发现训练好的关系描述soft template对推理帮助有限，而训练好的连接词（conjunction）token更有效——这一发现提示在设计推理prompt时，应更注重逻辑连接成分的优化。
4. **$\mathrm{MRR}_a$ 指标的实用性**：传统MRR只关注首个gold标签的排名，$\mathrm{MRR}_a$考虑了所有gold标签的平均排名，更适合多答案场景（如TP、SCO任务），可推广至其他多答案探测任务。
5. **本团队可结合的方向**：若团队关注知识增强的预训练或RAG系统，本文揭示了PLMs本体理解的不完整性，可作为知识注入/本体感知预训练的目标评估基准。

## 关键术语表
- **Ontological Knowledge（本体论知识）**：由类（classes）、属性（properties）及其层级关系和约束（域/值域）组成的结构化世界知识表示。
- **Probing（探测）**：通过设计特定的输入模板和评估指标，检验PLM内部是否编码了某类特定知识的实验方法。
- **RDFS Entailment Rules（RDFS蕴含规则）**：资源描述框架 schema（RDFS）中定义的逻辑推理规则，如传递性、继承性等。
- **Soft Prompt（软提示）**：由可学习连续向量（soft tokens）构成的prompt，替代手工设计的离散文本模板。
- **Pseudoword（伪词）**：无特定语义的人工构造词，用于替换推理测试中的具体实体，防止模型依赖表面记忆。
- **Domain/Range Constraint（域/值域约束）**：属性约束了其对哪些类（域）的实例适用，以及对哪些类（值域）的实例返回结果。
- **MRRa**：改进的Mean Reciprocal Rank指标，考虑所有gold标签的平均排名，适用于多答案场景。
- **Cloze Completion（完形填空）**：将目标信息替换为[MASK]标记，让模型预测被遮蔽内容的填空式任务。

## 可复现要素
- **数据集**：基于DBpedia和Wikidata构建，代码和数据已开源：https://github.com/vickywu1022/OntoProbe-PLMs
- **模型**：使用HuggingFace开源的BERT和RoBERTa系列。
- **框架**：OpenPrompt（Ding et al., 2022）。
- **关键超参**：
  - Soft token训练：100 epochs，AdamW，learning rate=0.5，linear warmup scheduler。
  - 伪词采样距离系数 $\alpha = 0.5$，每规则采样10对伪词。
  - 损失函数：BCEWithLogitsLoss（记忆和推理任务）。
- **Few-shot设置**：每个子任务训练/开发集各10样本。
- **计算资源**：论文未明确提及总GPU小时数和硬件配置（ACL Checklist C1未打勾）。
