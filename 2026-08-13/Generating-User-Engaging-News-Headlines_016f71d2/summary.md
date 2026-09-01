---
title: "Generating-User-Engaging-News-Headlines"
source: https://aclanthology.org/2023.acl-long.183.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:22:17"
field: "个性化自然语言生成"
keywords: ["个性化新闻标题生成", "签名短语", "对比学习", "用户画像", "文本生成评估", "合成数据"]
innovations: ["提出三阶段签名短语导向的个性化标题生成框架，通过可学习的相关性函数从用户阅读历史中提取标志性短语连接用户兴趣与文章内容", "在无标注数据上通过对比学习训练签名短语选择器，避免人工标注成本", "构建综合的自动化+人工多维度评估体系，涵盖用户适配、文章忠实度和事实一致性"]
benchmarks: ["Newsroom", "Gigaword", "PENS"]
---

# 论文速读：Generating-User-Engaging-News-Headlines

## 一句话总结
本文提出一个基于用户画像的个性化新闻标题生成框架，通过从用户阅读历史中抽取"signature phrases"（标志性短语）来建立推荐文章与用户兴趣之间的联系，在保证标题忠于原文内容的前提下提升用户参与度。

## 研究问题与动机
- **现有系统的问题**：Google News、Yahoo News等个性化推荐系统对所有用户展示相同的标题，用户无法理解推荐内容与自身兴趣的关联，降低推荐效果。
- **个性化标题的双重约束**：需在约10词的篇幅内同时传达文章核心信息（避免clickbait）和建立与用户阅读历史的相关性，平衡难度大。
- **数据稀缺**：缺乏包含新闻文章、多个人性化标题及相关用户画像的大规模标注数据集，无法直接监督训练。
- **评估困难**：个性化内容缺乏ground truth，传统ROUGE/BLEU等指标不适用，需开发多维度评估体系。

## 核心贡献（创新点）
- **提出三阶段签名短语导向的个性化标题生成框架**：与以往直接基于文章生成标题的工作不同，本文引入signature phrases作为桥梁连接用户兴趣与文章内容。
- **无需标注数据的用户画像学习方法**：通过对比学习（contrastive learning）在无标注用户阅读历史上训练signature phrase selector，避免人工标注成本。
- **合成用户数据集构建方法**：在Newsroom和Gigaword基础上自动生成合成用户，每个用户被分配兴趣短语并关联相关新闻文章，支持模型训练与评估。
- **综合的自动化+人工评估体系**：从用户适配（DPR/SBERT相关性、REC Score）、文章忠实度（FactCC）、表面重叠（ROUGE-L、Extractive Coverage）多维度评估，并辅以16位评估者的主观评测。

## 方法详解
**三阶段Pipeline：**

1. **Signature Phrases Identification（签名短语识别）**
   - 将任务视为条件文本生成问题，输入新闻文章，输出由分号分隔的候选签名短语集合 $Z_d = \{z_1, z_2, ...\}$
   - 使用在KPTimes数据集（279K条新闻文章与编辑精选签名短语配对）上预训练的BART模型
   - 训练目标为预测短语序列与人工精选序列之间的交叉熵损失

2. **User Signature Phrases Selection（用户签名短语选择）**
   - 对用户阅读历史 $H_u = \{t_1, t_2, ...\}$（历史标题集合），计算每个候选短语 $z_i$ 的用户吸引力得分 $S(z_i, H_u)$
   - **Holistic encoding策略**：将所有历史标题拼接后编码为向量 $\mathbf{h}_u$，得分 $S(z_i, H_u) = \mathbf{z}_i^\top \mathbf{h}_u$
   - **Individual encoding策略**：对每条历史标题单独编码为 $\mathbf{t}_j$，得分取最大值 $S(z_i, H_u) = \max_{t_j \in H_u} \mathbf{z}_i^\top \mathbf{t}_j$
   - **对比学习训练**：batch内$(z_i, H_i)$为正对，$(z_i, H_j)$为负对，损失函数为：
     $L_{select} = \frac{1}{2}\left(\sum_{i=1}^{N_B}\log\frac{S(z_i, H_i)}{\sum_{j=1}^{N_B}S(z_i, H_j)} + \sum_{j=1}^{N_B}\log\frac{S(z_j, H_j)}{\sum_{i=1}^{N_B}S(z_i, H_j)}\right)$

3. **Signature-Oriented Headline Generation（签名导向标题生成）**
   - 将用户签名短语 $Z_d^u$ 与新闻文章 $d$ 拼接后输入BART生成器
   - 训练时从ground-truth标题提取签名短语；推理时由阶段1和2模型确定
   - 损失为负对数似然：$L_{gen} = -\sum_i \log Pr(w_i | w_1, ..., w_{i-1}; Z_d^u, d)$

**合成用户创建流程**：(1) 从语料中提取所有签名短语构建候选池；(2) 建立短语→文章映射；(3) 随机采样短语作为用户兴趣；(4) 按映射采样相关文章构成阅读历史。

## 实验与结果
- **数据集**：Newsroom（训练集995,041条，验证集58,530条）和Gigaword（训练集7,704,419条，验证集394,390条）；合成用户测试集各10,000条
- **基线方法**：PENS-NRMS、PENS-EBNR（LSTM-based个性化生成）、Vanilla System（BART-large无签名短语）、Vanilla Human（人工原标题）、SP-headline、SP-random、SP-holistic、SP-individual
- **评估指标**：H-U相关性（DPR、SBERT）、REC Score（基于MIND训练的推荐系统打分）、H-A相关性（DPR、SBERT）、FactCC事实一致性、ROUGE-L F1、Extractive Coverage
- **最佳结果（Newsroom）**：SP individual-F在H-U相关性DPR上达**55.05**（较PENS-NRMS的50.85提升+4.20），REC Score达**2.947**（较PENS-NRMS的2.449提升+0.498），FactCC达69.5%
- **最佳结果（Gigaword）**：SP individual-F在H-U相关性DPR上达**54.82**，REC Score达**3.459**
- **关键发现**：
  - Individual编码策略优于Holistic策略
  - Fine-tuned encoder（-F）显著优于naive DPR（-N）
  - SP-random性能接近Vanilla System，说明签名短语的精准选择是关键
  - 用户兴趣短语数增加时，用户适配指标下降（表5），但即便30个话题仍优于Vanilla
  - GPT-3 zero-shot方法在用户适配指标上均低于SP individual-F（表6）
- **人工评估**：16位评估者，SP-Individual-F和SP-Individual-N在User Adaptation维度表现最佳；Vanilla System在Headline Appropriateness维度得分最高（但部分标题偏离文章主旨）

## 相关工作脉络
- **PENS（Ao et al., 2021）**：最早的个性化新闻标题生成数据集与基线框架，使用LSTM编码用户历史，本文在其基础上引入签名短语机制并与更强大的生成模型对比。
- **Automatic headline generation（Song et al., 2018; Xu et al., 2019; Matsumaru et al., 2020）**：聚焦于准确概括文章或首句，未考虑个性化；本文与之本质区别在于引入用户兴趣维度的显式建模。
- **Keyphrase/signature phrase extraction（Meng et al., 2017; Krapivin et al., 2009）**：多在学术论文领域；本文采用KPTimes（Gallina et al., 2019）新闻领域数据集，适配新闻场景。
- **Personalized content generation（Majumder et al., 2019; Wu et al., 2021b）**：探索个性化配方/回复生成；本文定位差异在于结合recommendation场景的headlining任务，并解决clickbait与忠实度平衡问题。
- **Evaluation of personalized content（Gligorić et al., 2021）**：指出个性化内容缺乏ground truth的评估挑战；本文综合使用自动化指标与人工评估，提出多维度评价框架。
- **GPT-3在摘要/标题生成中的应用（Goyal et al., 2022）**：本文对比了GPT-3 zero-shot方法，证明任务特定微调模型仍优于通用大模型在零样本设置下的表现。

## 局限性与未来方向
- **数据依赖**：模型性能受训练数据质量与一致性影响，可能存在偏见导致生成不完整或误导性标题。
- **回音室风险**：个性化推荐可能加剧信息茧房效应，长期对用户认知产生不利影响。
- **兴趣短语数量的负面影响**：实验显示用户兴趣话题越多，个性化效果越差（表5），如何扩展至多兴趣用户是挑战。
- **合成用户与真实用户的差距**：当前使用合成用户进行训练和评估，尚未在真实用户数据上验证效果。
- **多签名短语的潜在事实错误**：当输入多个签名短语时，生成器可能被迫包含不相关内容导致事实错误（作者仅使用单签名短语规避）。
- **伦理考量**：虽未收集人口统计信息，但个性化标题可能强化个体偏见视角，需负责任使用。

## 研究启发与可借鉴点
- **签名短语作为个性化中介**：将用户兴趣抽象为可操作的短语表征，而非直接使用原始历史序列，既压缩了信息又保留了语义焦点，可迁移至其他个性化生成任务（如推荐摘要、广告文案生成）。
- **无标注对比学习训练selector**：利用正负样本对比损失在无标注历史数据上学习相关性排序，避免了昂贵的标注成本，适用于资源受限场景。
- **合成数据构建策略**：通过"短语→文章"映射逆向构造用户画像，为缺乏真实用户行为数据的场景提供数据增强思路。
- **多维度评估框架**：结合用户适配、文章忠实度、表面重叠、事实一致性等多项指标，并辅以人工评估，为个性化NLG任务的评测提供了可复用的评估模板。
- **Holistic vs. Individual编码对比实验**：系统比较了两种用户历史编码策略，发现Individual策略更优，这一对比设计对理解用户历史建模方式有参考价值。

## 关键术语表
**Signature Phrases**：从用户阅读历史中提取的标志性短语，用于表征用户兴趣并指导个性化标题生成。
**Holistic History Encoding**：将所有历史标题拼接后编码为单一向量的用户历史表示策略。
**Individual History Encoding**：对每条历史标题单独编码并取最大相关性的用户历史表示策略。
**Contrastive Learning**：通过拉近正样本对、推远负样本对来学习语义表示的自监督学习方法。
**REC Score**：基于MIND数据集训练的推荐系统对用户-文章-标题三元组的推荐评分。
**FactCC**：用于评估生成文本与源文章事实一致性的预训练模型。
**Extractive Coverage**：生成标题中来源于源文章的词汇比例，衡量生成内容的抽取程度。
**Synthesized Users**：通过算法自动构建的模拟用户，具有随机分配的兴趣短语和关联阅读历史。

## 可复现要素
- **数据集**：使用公开数据集Newsroom和Gigaword；合成用户数据为本文构建，论文提供了构建方法但未声明公开链接。
- **代码/权重**：论文提到使用HuggingFace预训练DPR模型、BART-large模型，以及PENS开源仓库；FactCC和Sentence-BERT均有开源实现；GPT-3通过OpenAI API调用。**代码未明确声明开源**。
- **关键超参**：Batch size 96×8（selection）/48×8（generation），learning rate 3e-5/5e-5，epochs 15/6，signature phrase max length 16 tokens，headline max length 48 tokens，reading history max length 256 tokens，article max length 512 tokens（见附录Table 8）。
- **训练环境**：8×Nvidia A100 GPUs。
