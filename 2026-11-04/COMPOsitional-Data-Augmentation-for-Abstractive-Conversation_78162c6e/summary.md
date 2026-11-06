---
title: "COMPOsitional-Data-Augmentation-for-Abstractive-Conversation"
source: https://aclanthology.org/2023.acl-long.82.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:38:19"
field: "对话系统数据增强"
keywords: ["对话摘要", "数据增强", "组合泛化", "低资源学习", "自我训练"]
innovations: ["提出基于话题结构和动作三元组的子结构组合数据增强框架", "设计基于Vote-k图的选择性检索机制保证多样性和覆盖率"]
benchmarks: ["SAMSum", "DialogSum", "CNN/DailyMail"]
---

# 论文速读：COMPOsitional-Data-Augmentation-for-Abstractive-Conversation

## 一句话总结
本文提出了一种基于子结构层面的组合式数据增强方法 COMPO，通过提取对话的主题分割（topic splits）和动作三元组（action triples）作为组合单元，生成多样化的对话-摘要配对数据，以缓解低资源场景下对话摘要模型训练数据不足的问题。

## 研究问题与动机
1. **数据稀缺问题**：抽象对话摘要严重依赖大规模标注数据，但对话数据的收集与标注耗时耗力，难以应用于真实场景中。
2. **现有数据增强方法的局限**：直接将词级/句级的数据增强方法应用于对话会忽略对话特有的结构信息（如说话人信息、话题分割、对话阶段），导致生成数据多样性和质量不足。
3. **组合式增强的不足**：现有的组合式数据增强主要聚焦于短文本的词/短语/句法树级别编辑，忽视了对话中跨话语之间的丰富结构信息，无法有效组合多轮话语生成新颖对话数据。
4. **低资源场景需求**：在仅有1%-5%标注数据的极端低资源设置下，传统方法性能大幅下降，亟需更有效的数据增强策略。

## 核心贡献（创新点）
1. **提出 COMPO 框架**：首次将对话的主题结构（topic structure）和动作三元组（action triples）作为组合操作的基本单元，实现子结构层面的组合式数据增强。与已有方法相比，前者关注短文本本地编辑，后者天然利用对话特有的层级结构。
2. **设计动作导向的选择性检索机制**：利用 Vote-k 图-based 方法基于动作三元组进行相似候选检索，保证生成数据的多样性与覆盖率。与已有 k-NN 方法相比，该机制能覆盖更多样化的语义空间，避免过度聚焦局部词汇语义。
3. **探索两种噪声容忍训练范式**：提出 self-training（仅使用新生成对话+伪摘要）和 joint-training（联合使用原始数据和增强配对数据）两种训练策略。与已有工作相比，系统对比了不同训练设置下的噪声容忍能力与效果权衡。
4. **广泛的实验验证与迁移性分析**：在 SAMSum 和 DialogSum 上验证有效性，并证明方法可迁移到 CNN/DailyMail 新闻摘要任务。定位差异在于现有工作多局限于特定数据集或单一增强粒度，本文方法具有跨任务通用性。

## 方法详解
**COMPO 框架包含三个核心模块：**

1. **话题分割（Topical Split）**：使用 C99 算法结合 Sentence-BERT 计算话语相似度，将对话 $C = \{u_1, u_2, ..., u_m\}$ 分割为多个话题块 $C_{topic} = \{b_1, b_2, ..., b_n\}$，其中每个 $b_i$ 包含连续的话语序列。每个话题块与摘要句子按 Algorithm 1 匹配配对：使用滑动窗口计算 ROUGE 分数，选取最佳匹配的摘要片段 $s_{paired}^i$。

2. **动作提取（Action Extraction）**：使用轻量级工具提取动词中心短语作为话题块的骨干表示，匹配预设的句法模式（如 $n_1$-nsubj-$v_1$、$n_1$-nsubj-$v_1$-dobj-$n_2$ 等），得到动作三元组集合 $A_i$。提取后的单元构成池 $P = \{(b_i, s_{paired}^i, A_i)\}_{i=1:|p|}$。

3. **基于动作的组合操作（Action-based Composition）**：
   - 使用 Sentence-BERT 计算每个话题单元的向量表示（平均池化）；
   - 构建有向图 $G = (V, E)$，每条边连接余弦相似度最近的 k 个邻居；
   - 通过迭代选取得分最高的节点进行检索，得分公式为：$score(u) = \sum_{v \in \{(v,u) \in E, v \in U\}} s(v)$，其中 $s(v) = \rho^{|\{l| (v,l) \in E\}|}$，$\rho > 1$；
   - 用预训练的 seq2seq 模型生成替换后的对话片段：输入为"动作三元组 + 未掩码的对话上下文"，目标为被替换的话题块。

4. **噪声容忍训练设置**：
   - **Self-training**：使用教师模型在迭代中预测伪摘要，无监督损失为：$L_u = -\mathbb{E}_{c \sim D^u}\mathbb{E}_{c' \sim COMPO(c)} \log P(f(c;\theta')|f(c';\theta))$，训练集扩大为原始数据的 2 倍；
   - **Joint-training**：对新生成对话用预训练生成模型生成伪摘要，总损失为：$L = \mathbb{E}_{(c,s) \in D^l} \log P(s|c) + \gamma \mathbb{E}_{(c',s') \in D'} \log P(s'|c')$，其中 $\gamma$ 为增强样本权重。

## 实验与结果
**数据集**：SAMSum（14,732 训练对）、DialogSum（12,460 训练对）、CNN/DailyMail（新闻摘要迁移验证）。

**评估指标**：ROUGE-1、ROUGE-2、ROUGE-L。

**主要结果（SAMSum，1% 数据，BART-base  backbone）**：
- COMPO self-training：R-1 = **45.42**（↑3.06）、R-2 = **21.23**（↑2.60）、R-L = **41.42**（↑2.98）
- COMPO joint-training：R-1 = **44.89**（↑2.53）、R-2 = **20.64**（↑2.01）、R-L = **40.58**（↑2.14）
- 相比 BART-base 基线，1% 数据下 ROUGE-2 提升高达 **14.0%**

**主要结果（DialogSum，1% 数据，BART-base）**：
- COMPO self-training：R-1 = **43.13**（↑3.02）、R-2 = **16.21**（↑2.15）、R-L = **37.40**（↑2.61）

**最强结果**：SAMSum full data + COMPO-large（self-training）：R-1 = 53.56、R-2 = 28.66、R-L = 50.04。

**结论**：（1）在低资源场景下提升幅度更大；（2）self-training 优于 joint-training；（3）方法可扩展至不同 backbone 规模；（4）Human evaluation 显示 COMPO-sf 和 COMPO-jt 显著优于 BART-base，整体评分提升约 8%-14%。

## 相关工作脉络
1. **Semi-CODA（Chen & Yang, 2021a）**：两阶段噪声自监督对话数据增强框架，包含随机交换、删除、对话行为引导插入等。本文定位差异：Semi-CODA 缺乏组合泛化保证，COMPO 通过结构化的主题单元组合实现更高多样性。
2. **Liu et al. (2022)**：通过替换对话和摘要中语义相似的文本 span 生成合成样本。本文定位差异：仅做 span 级替换，未利用对话话题结构和动作语义，COMPO 在组合泛化性上更优。
3. **Treemix（Zhang et al., 2022）**：基于句法树成分的组合数据增强。本文定位差异：Treemix 面向短文本的句法结构，COMPO 面向对话层级结构（话题-动作），适合多轮对话场景。
4. **Genaug（Feng et al., 2021a）**：面向文本生成器的数据增强方法。本文定位差异：Genaug 属于通用文本增强，COMPO 专为对话结构设计，利用了说话人信息和话题分割。
5. **DialoGPT 增强（Feng et al., 2021b）**：使用 DialoGPT 生成对话响应作为增强数据。本文定位差异：DialoGPT 忽略了对话结构和说话人信息，COMPO 通过结构组合生成更具语义一致性的数据。
6. **HiDDeN（Chen et al., 2021a）**：简单但泛化性更强的 NLU 数据增强方法。本文定位差异：HiDDeN 面向 NLU 任务，COMPO 面向 NLG 对话摘要，且利用对话特有结构。

## 局限性与未来方向
1. **组合操作的 scope 受限**：目前仅探索了话题切片的组合替换，未涉及对话阶段（conversation stages）或具体话语行为（discourse acts）等其他结构类型。
2. **外部工具依赖与误差级联**：使用 C99 进行话题分割和动作提取等外部工具，虽然选用高精度工具，但仍不可避免误差级联问题。
3. **低资源语言适用性**：预处理工具在低资源语言上可能不可用，限制了方法的跨语言迁移。
4. **未来方向**：探索无需外部工具依赖的组合数据增强方法；扩展到其他对话结构类型（如 discourse relations）；验证在低资源语言上的可行性。

## 研究启发与可借鉴点
1. **结构感知的组合增强范式**：将对话层次结构（话题→动作）作为组合单元的思路可迁移至其他对话相关任务（如对话状态追踪、对话生成），为本团队可复用的方法论。
2. **Vote-k 图检索机制**：基于图 Scores 的选择性检索策略在保证了覆盖率和多样性的同时优于 k-NN，可作为检索增强生成的参考组件。
3. **噪声容忍训练的对比设计**：self-training vs joint-training 的系统对比实验设计值得借鉴，可在类似数据增强工作中复用此对比框架。
4. **低资源场景的实验设置**：1%、5% 标签数据的极端低资源实验设置能有效凸显方法优势，建议在本团队研究中采用类似设置验证方法有效性。
5. **跨任务迁移验证**：从对话摘要迁移到新闻摘要（CNN/DailyMail）的验证策略可作为方法通用性证明的参考范式。

## 关键术语表
**COMPO**：Composition Data Augmentation 的缩写，本文提出的基于子结构层面组合的数据增强框架。
**Topic Split**：话题分割，将对话按话题转换为多个连续的文本块，作为组合操作的基本单元。
**Action Triple**：动作三元组，以动词为中心提取的短语结构（如 "Mary, forgot, party"），用于表示话题块的核心语义。
**Selective Retrieval**：选择性检索，基于 Vote-k 图算法的检索策略，在保证相似度的同时最大化覆盖多样性。
**Self-training**：自训练范式，使用教师模型为增强对话生成伪摘要，仅利用新生成对话进行训练。
**Joint-training**：联合训练范式，同时使用原始标注数据和增强对话-摘要配对数据进行训练。
**Vote-k**：基于图的节点选择算法，通过迭代选取入边评分最高的节点实现多样性检索。
**ROUGE**：Recall-Oriented Understudy for Gisting Evaluation，摘要质量自动评估指标，包括 R-1、R-2、R-L。

## 可复现要素
- **数据集**：SAMSum（公开）、DialogSum（公开）、CNN/DailyMail（公开）
- **代码开源**：https://github.com/ozyyshr/Compo（论文声明已公开）
- **关键超参**：self-training 最大迭代次数 K=5；batch size=16；learning rate=3e-5；训练 10 个 iteration；使用 BART-base/large 作为 backbone；增强后训练数据总量为原始的 2 倍
- **硬件**：4x A6400 GPU
