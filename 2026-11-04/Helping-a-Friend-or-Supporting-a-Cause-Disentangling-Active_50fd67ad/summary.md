---
title: "Helping-a-Friend-or-Supporting-a-Cause-Disentangling-Active"
source: https://aclanthology.org/2023.acl-long.166.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:01:11"
field: "计算社会科学/NLP交叉"
keywords: ["cosponsorship", "representational learning", "relational graph convolutional network", "legislative behavior", "multi-task learning", "zero-shot prediction", "political text analysis"]
innovations: ["提出Encoder+RGCN联合框架首次区分主动与被动共同提案的动机差异", "多任务自监督学习（作者预测+引用预测）显著提升议员表示质量", "零样本议员表示在投票预测任务上超越专门训练的SOTA方法"]
benchmarks: ["US Congress 112th-115th cosponsorship classification", "Roll-Call Vote prediction"]
---

# 论文速读：Helping-a-Friend-or-Supporting-a-Cause-Disentangling-Active-and-Passive-Cosponsorship-in-the-U.S.-Congress

## 一句话总结
论文提出一个 Encoder+RGCN 框架，利用美国国会第112-115届的法案文本、议员演讲和关系图谱学习议员表示，首次揭示主动与被动共同提案（cosponsorship）由两种不同动机驱动：主动共同提案是"支持人"（支持提案者），被动共同提案是"支持议题"（支持法案内容）。模型在cosponsorship分类上达到 F1=0.88，并在零样本投票预测任务上超越现有SOTA。

## 研究问题与动机
- **核心问题**：美国国会议员支持法案有两种形式——主动共同提案（active cosponsorship，在法案提交国会前参与起草阶段）和被动共同提案（passive cossponsorship，法案提交后签署支持）。两者在资源投入上存在差异，但此前研究从未从**潜在动机**角度系统区分二者。
- **现有方法不足**：政治学研究长期将cosponsorship视为统一信号，未区分主动与被动背后的不同驱动因素；NLP社区对cosponsorship研究几乎空白，缺乏基于文本与关系的联合建模方法。
- **研究缺口**：缺少能同时融合长文本（法案、演讲）与多关系图谱（提案、引用、共同提案）的表征学习框架。
- **应用价值**：若主动/被动cosponsorship反映不同动机，则联盟形成研究应聚焦主动共同提案，议程支持研究应聚焦被动共同提案。

## 核心贡献（创新点）
- **提出Encoder+RGCN融合框架**：首次在美国国会议员表示学习中联合利用长文本编码与多关系图卷积，编码器突破BERT等模型的token长度限制（将512词分块后用Bi-LSTM聚合），RGCN同时建模演讲、法案、引用和共同提案四类关系。
- **揭示主动/被动cosponsorship的动机二分**：通过余弦相似度分析证明，主动共同提案议员的表示与提案者更相似（支持"人"），被动共同提案议员的表示与法案内容更相似（支持"议题"），为政治学提供了新的解释框架。
- **多项自监督任务提升表征质量**：引入作者身份预测（authorship prediction）和引用预测（citation prediction）两个辅助任务进行多任务学习，消融实验表明两者共同作用时F1达到0.88（仅主任务为0.85）。
- **零样本投票预测达到SOTA**：未使用任何投票数据训练，仅凭学习到的议员表示即可预测roll-call vote，平均F1=0.907，超越CNN+meta（0.887）和LSTM+GCN（0.896）等专门为此任务训练的模型。
- **发布高质量数据集**：整理第112-115届国会超5万份法案、超12万份演讲 transcript、超150万次投票记录，公开可用。

## 方法详解
- **数据构建**：从voteview.com获取议员元数据（BioGuide ID、党派、州等），从propublica.org API获取法案元数据和共同提案时间戳，从congress.gov爬取每日会议记录（daily editions）并通过正则模式（`[SAL]+[PERSON]+...`）解析演讲稿归属，通过SpaCy NER识别演讲中对其他议员的引用并构建引用网络。
- **Encoder设计**：将法案（均长2239词）和演讲（均长8129词）切分为512词块，分别用BERT计算每块嵌入后，送入Bi-LSTM（隐藏维度384）沿前后方向聚合，最终拼接两个方向隐状态并进行均值池化，输出128维文档嵌入；法案和演讲各自训练独立Encoder实例以捕捉书面语与口语差异。
- **RGCN图结构**：构建异构图$\mathcal{G}=(\mathcal{V}, \mathcal{E})$，节点集合$\mathcal{V}=\{S, L, B\}$（演讲、议员、法案），边集合含5类关系：$R_1$演讲作者关系、$R_2$议员引用关系（有向）、$R_3$法案提案关系、$R_4$主动共同提案关系、$R_5$被动共同提案关系；议员节点用元数据one-hot初始化，法案/演讲节点用Encoder输出初始化。
- **RGCN更新公式**：$e_v^{(k+1)} = \sigma\left(\sum_{r \in \mathcal{R}}\sum_{j \in \mathcal{N}_v^r}\frac{W_r^{(k)}e_j^k}{c_{v,r}} + W_0^k e_v^k\right)$，其中$c_{v,r}=|\mathcal{N}_v^r|$为归一化常数，每层对不同类型关系使用不同变换矩阵。
- **多任务联合训练**：总损失$\mathcal{L}_{\text{tot}} = 0.8\mathcal{L}_{\text{cosp}} + 0.1\mathcal{L}_{\text{auth}} + 0.1\mathcal{L}_{\text{cit}}$；主任务cosponsorship分类将议员、法案、提案者嵌入拼接后输入FFNN（维度192→softmax），作者预测和引用预测各用一个FFNN（维度128→softmax），均为二分类交叉熵损失；作者预测采样时50%为正样本（演讲确为该议员所发），引用预测同样50%为正样本。
- **投票预测（零样本）**：将议员嵌入与法案嵌入拼接后输入三层FFNN（ReLU+dropout 0.2）预测"yea/nay"，测试时仅预测该议员**未共同提案**的法案以避免数据泄露。

## 实验与结果
- **数据集**：第112-115届美国国会，按时间划分为训练（前60%）/验证（20%）/测试（20%），共约37万训练样本、各12万验证和测试样本；法案数约1.4-1.6万/届，演讲稿数约3万/届。
- **主任务结果（Table 1）**：模型在四届国会平均F1达**0.884±0.04**，显著超越所有基线：Ideology（0.746）、Metadata（0.737）、GloVe（0.771）、Encoder-only（0.846）、Encoder+Metadata（0.847）、GCN（0.765）、RGCN（仅图结构，0.800）；最优在113届达到0.892。
- **消融实验（Table 8）**：仅主任务F1=0.853，加入作者预测提升至0.870，加入引用预测提升至0.867，两者均加入达到0.884，证明多任务学习的显著增益。
- **动机验证（Figure 5）**：主动共同提案议员表示与提案者表示的平均余弦相似度显著更高（p-value=4.3×10⁻¹²），被动共同提案议员表示与法案表示的平均余弦相似度显著更高（p-value=3.37×10⁻⁶），统计检验确认了动机二分。
- **投票预测（Table 2）**：零样本F1平均0.907，超越Majority（0.778）、Ideal-Vectors（0.879）、CNN+meta（0.887）、LSTM+GCN（0.896）；112届达到最高0.928。
- **可解释性（Figure 6）**：t-SNE可视化显示议员表示沿党派清晰分离（共和党vs民主党），党派领袖位于各自聚类中心，保守派民主党人（Blue Dog Caucus）被正确定位在两党之间。
- **编码器对比（Table 6）**：本文Encoder+F1=0.884，优于Doc2Vec+RGCN（0.820）、BERT+RGCN（0.851）、LongFormer+RGCN（0.857），证明Bi-LSTM聚合方案在超长政治文本上的优势。

## 相关工作脉络
- **政治学cosponsorship研究**：Fowler (2006)、Brandenberger (2018)等从网络结构角度分析共同提案与联盟形成的关系，但依赖手工构建的共提案网络，未利用文本语义信息；本文首次将文本语义与图结构联合建模。
- **投票预测的NLP工作**：Gerrish & Blei (2011)通过主题模型链接立法 sentiment 与文本；Kraft et al. (2016)、Kornilova et al. (2018)、Yang et al. (2020)依次引入嵌入模型、属性增强和LSTM+GCN；本文在无需训练数据的情况下达到更高精度，证明表征的泛化能力。
- **长文本编码方法**：BERT (Devlin et al., 2018)受限于512 token，Longformer (Beltagy et al., 2020)支持4096 token但仍需分块平均；本文的"分块BERT+Bi-LSTM聚合"方案在4倍于Longformer的输入长度下取得更好效果。
- **图神经网络在社会科学中的应用**：Schlichtkrull et al. (2018)提出RGCN；Zhang et al. (2019)综述GCN；本文首次在议员表示学习中采用多关系异构图，同时建模5种关系类型，区别于以往仅使用共提案单关系图的简化设定。
- **自监督多任务学习**：Mou et al. (2021)通过对齐投票行为与公开声明学习议员表示；本文在社会科学场景中首次系统验证了作者预测和引用预测两种自监督辅助任务对主任务表征的增益。

## 局限性与未来方向
- **支持形式的覆盖有限**：仅将政治支持操作化为主动/被动共同提案，未考虑社交媒体背书等其他支持形式；未来可扩展到多模态支持信号。
- **意识形态为潜变量**：无法获得意识形态的ground-truth，只能通过多项任务间接验证表征编码了意识形态；未来可结合更多意识形态敏感任务进行稳健性检验。
- **新议员冷启动问题**：当前模型无法为新当选议员生成有效表示（缺乏演讲和提案数据）；未来可探索在线增量更新机制。
- **仅适用于美国国会**：框架在美国四届国会表现一致，但未验证对其他立法制度（如英国议会、欧盟议会）的适用性；未来需跨制度泛化实验。
- **演讲解析覆盖率约93%**：约7%的演讲因格式异常或署名问题未能自动匹配，可能引入噪声。

## 研究启发与可借鉴点
- **"分块编码+序列聚合"架构可迁移**：Bi-LSTM聚合BERT分块嵌入的方案适用于任意超长文档场景（如法律条文、长篇报告），可作为通用长文本编码模块复用于其他NLP任务。
- **多任务自监督在社科数据上价值显著**：作者预测和引用预测作为辅助任务仅需关系标签（无需人工标注），却能带来约3%的F1提升；这一思路可迁移到任何具有关联关系的异质图学习场景。
- **零样本表征评估范式值得推广**：在表征学习完成后，用完全独立的下游任务（如投票预测）验证表征质量，而非仅在主任务上调参——这是一种更严格的表征能力评估标准。
- **动机二分法的分析框架可迁移**：将单一行为（如签名支持、点赞转发）区分为"人际导向"和"内容导向"两个维度，并分别用相似度分析验证，可用于社交媒体行为分析等其他领域。
- **时间划分的数据分割策略**：采用时间顺序而非随机划分train/val/test，更符合政治过程的自然时序，避免了未来信息泄露，适合任何有时序特征的社会科学预测任务。

## 关键术语表
- **Cosponsorship（共同提案）**：国会议员支持由其他议员提出的法案的行为，分为主动和被动两种形式。
- **Active Cosponsorship（主动共同提案）**：在法案正式提交国会之前签署支持，通常涉及法案起草阶段的实质性参与。
- **Passive Cosponsorship（被动共同提案）**：在法案提交国会之后签署支持，是一种低门槛的支持表达方式。
- **RGCN（Relational Graph Convolutional Network，关系图卷积网络）**：专为多关系图设计的图神经网络，对每种关系类型使用独立的变换矩阵进行消息传递。
- **Roll-Call Vote（点名投票）**：国会议员对法案进行的正式口头投票记录（yea/nay），是衡量议员立场的核心数据。
- **DW-NOMINATE**：美国国会研究中广泛使用的议员意识形态量化指标，基于投票行为估计议员在左右光谱上的位置。
- **Zero-shot Prediction（零样本预测）**：在未见过的任务/数据域上进行预测，模型仅依赖从源任务学到的表征而不进行微调。
- **Daily Editions（每日会议记录）**：美国国会官方发布的每日会议逐字记录汇编，包含当天所有议员的演讲原文。

## 可复现要素
- **数据集**：论文声明数据已公开（标记为superscript 1），包含第112-115届国会的法案元数据、演讲transcript、共同提案关系和投票记录；具体获取来源为voteview.com、propublica.org API、congress.gov爬虫。
- **代码**：论文未提供代码开源声明。
- **关键超参**：BERT使用bert-base-uncased（embed_dim=768）；Bi-LSTM hidden_dim=384；Encoder输出dim=128；RGCN两层，hidden维度分别为128和64；FFNN维度192/128/128；优化器AdamW，lr=10⁻⁴，batch_size=64，dropout正则化，early stopping（8轮）；5次epoch微调BERT；使用DGL库实现RGCN，HuggingFace Transformers实现BERT。
