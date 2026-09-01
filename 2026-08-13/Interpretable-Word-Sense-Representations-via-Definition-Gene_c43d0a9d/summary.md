---
title: "Interpretable-Word-Sense-Representations-via-Definition-Gene"
source: https://aclanthology.org/2023.acl-long.176.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:54:49"
field: "词义学与语义变化分析"
keywords: ["semantic change analysis", "definition generation", "interpretable representations", "word sense representation", "lexical semantics", "DWUG"]
innovations: ["将自动生成的上下文定义作为可解释词汇表示，超越token/sentence embedding的语义相似度", "提出基于原型定义选取的sense label方法，31%情况下优于典型usage方法", "构建diachronic sense dynamics map实现可解释语义变化分析"]
benchmarks: ["DWUG English", "WordNet", "Oxford", "CoDWoE"]
---

# 论文速读：Interpretable-Word-Sense-Representations-via-Definition-Generation

## 一句话总结
论文提出将上下文相关的自动生成的自然语言定义作为一种可解释的词汇语义表示形式，替代黑箱式的token embedding，并在语义变化分析（semantic change analysis）任务中证明了其优于传统表示的有效性。

## 研究问题与动机
- **黑箱表示缺乏可解释性**：当前基于分布词向量和预训练语言模型（LM）的方法因亚符号性质（subsymbolic nature）缺乏可解释性，导致历史语言学家、词典编纂者等终端用户对其信任不足。
- **语义变化检测方法无法提供可理解的sense描述**：现有系统仅提供数值型的"change scores"或二元分类，缺乏人类可读的形式化sense描述，用户需要知道的是"含义变成了什么"而非"是否发生变化"。
- **已有definition generation工作未探索其作为通用表示的潜力**：此前定义生成主要用于词向量空间评估，本文首次将其拓展为一种通用的、可量化的词汇语义表示范式。
- **定义空间对semantic similarity的近似优于token/sentence embedding**：直觉上，自然语言定义比高维向量更能捕捉词的上下文依赖含义。

## 核心贡献（创新点）
1. **提出definitions-as-representations范式**：用Flan-T5生成的上下文依赖自然语言定义替代dense vector作为词汇语义表示，在pairwise usage similarity任务上超越token和sentence embeddings（Spearman相关系数最高达0.264 vs 0.141）。
2. **提出基于原型definition的sense label选取方法**：在数据驱动的usage cluster内，选择embedding最接近簇内平均embedding的定义作为sense label，人工评估显示该方法在31%的案例中优于基于典型usage的方法，且可接受率达80%。
3. **构建diachronic sense dynamics map用于可解释语义变化分析**：将sense labels按时间分裂为sub-clusters并计算label相似度，发现异常相似对以绘制sense动态关系图，可用于诊断DWUG聚类不一致性并解释语义演变（如'narrowing'）。

## 方法详解
- **模型选型**：选用Flan-T5（Chung et al., 2022），一个经1.8K指令任务微调的T5变体，重点使用XL（3B参数）版本。
- **Prompt设计**：采用`s What is the definition of w?`（usage后接问句），经8种prompt候选比较后选择效果最佳形式；解码策略为greedy search + target word filtering。
- **定义生成评估**：在WordNet、Oxford、CoDWoE三个数据集上测试，采用SacreBLEU、ROUGE-L、BERT-F1三个指标；soft domain shift（三数据集联合训练）下效果最优（WordNet: BLEU=32.81, BERT-F1=92.16）。
- **Sense Label选取**：①将簇内所有定义用DistilRoBERTa编码为sentence embeddings；②计算簇内所有embedding均值；③选取最接近均值的定义作为sense label（剔除<3 usages的cluster）。
- **Sense Dynamics Map构建**：对每个target word，将各cluster按两个时段（1810-1860/1960-2010）分裂为sub-clusters，计算sub-cluster间label embedding的余弦相似度，检测z-score > 1的离群对，形成sense转移/分裂/合并关系图。
- **语义相似度对比实验**：用SacreBLEU、METEOR、cosine（sentence embedding）三种相似度度量，与人类标注的DWUG gold similarity进行Spearman相关分析。

## 实验与结果
- **数据集**：定义生成使用WordNet（15,657 entries）、Oxford（122,318 entries）、CoDWoE（63,596 entries）；语义变化分析使用英语DWUGs（46个词，~200 usages/词）。
- **定义质量最佳结果**：Flan-T5 XL在WordNet soft shift上取得BLEU=32.81（超过Huang et al. 2021基线32.72）、BERT-F1=92.16；Oxford soft shift上BLEU=18.69。
- **Spearman相关对比**：FT Flan-T5 XL的definition embedding cosine相关系数0.264，显著优于token embeddings的0.141和sentence embeddings的0.114；definition的overlap指标（SacreBLEU=0.108, METEOR=0.117）也优于两类baseline。
- **Human Evaluation**：136个cluster、5名 annotators；definition-based labels在31%案例中被认为更好（vs usage-based仅7%），两者相当但definition更可接受（80% vs 68%）。
- **聚类质量**：Definition embedding space在separation-cohesion ratio上表现最优（DistilRoBERTa: 24.4 vs token 20.1），说明定义空间区分不同sense的能力更强。
- **case study亮点**：'record'的sense分析识别出从"giving information about past events"到"phonograph cylinder"的narrowing过程；'ball'的案例分析揭示了非传递性并暗示了可能的语义演变轨迹（sphere→projectile→bullet）。

## 相关工作脉络
- **Definition modelling（Gardner et al., 2022综述）**：本文在sequence-to-sequence范式下工作，但将定义生成从"评估embedding"工具提升为"通用表示"范式。
- **Huang et al. (2021) specificity-tuned T5**：最接近的基线方法，本文在多项指标上超越；本文强调生成效率优势（单模块 vs 三模块）。
- **DWUGs（Schlechtweg et al., 2021）**：本文使用的semantic change分析基准数据，但原有DWUGs仅提供匿名cluster编号，本文通过定义label增强其可解释性。
- **Kutuzov et al. (2018); Tahmasebi et al. (2021) 语义变化survey**：本文定位为在representations层面补充现有方法——从数值型change score转向人类可读的sense描述。
- **WordNet/Dictionary-based工作（Ishiwatari et al., 2019; Gadetsky et al., 2018）**：本文使用这些数据集训练，但方法不限于静态词义表示，支持上下文依赖的定义生成。
- **Mitra et al. (2015) sense变化识别**：本文与之定位相似（识别sense演变关系），但核心差异在于输出为人类可读的definition labels而非数值结果。

## 局限性与未来方向
- **语言局限**：实验仅涉及英语DWUGs，虽方法理论上是language-agnostic但多语言定义生成能力未经充分验证。
- **偏见风险**：生成的定义可能包含语言模型固有的偏见和刻板印象，仅靠过滤不足以确保安全。
- **实验初步性**：semantic change分析的展示以hand-picked案例为主，缺乏系统性定量评估。
- **模型规模和超参未充分探索**：仅测试了Flan-T5的几个变体，未研究模型规模、解码策略等因素的影响。
- **未来方向**：系统性评估预测与人类专家判断的对应；扩展到word sense induction、idiom detection、metaphor interpretation等其他词义学任务；追踪semantic narrowing/widening的时间动态。

## 研究启发与可借鉴点
- **Definitions-as-embeddings的可迁移性**：将NLG输出（定义文本）编码为sentence embedding用于下游语义相似度任务，是一种绕过"黑箱向量"获得可解释表示的新路径，可迁移至词义消 disambiguation、词义诱导等任务。
- **原型选取策略（prototypicality-based labelling）**：用embedding均值最近邻选取代表性定义的思路，可应用于其他需要cluster label的NLP任务（如词义聚类、话题建模）。
- **Spearman相关性评估范式**：用generated representations与人类pairwise similarity judgement的相关性作为评估指标，比单纯数值metric更贴合"可解释性"目标，值得在可解释AI研究中推广。
- **Sense dynamics map的可复用框架**：将聚类按时间分裂→计算label相似度→检测离群对的三步法，可作为通用模板用于任何时序语义变化研究。
- **多数据集联合训练的domain shift鲁棒性启示**：本文发现soft domain shift（多数据集联合）效果优于单一数据集fine-tuning，提示在定义建模任务中数据量比数据分布匹配更重要。

## 关键术语表
- **Diachronic Word Usage Graph (DWUG)**：带权无向图，节点为词的用法实例，边权为语义相似度，用于刻画词义随时间的演变。
- **Semantic Change Detection (LSCD)**：检测词的含义是否随时间发生变化的NLP任务，通常分为二分类或排序任务。
- **Definition Modelling / Definition Generation**：给定词或使用示例生成人类可读自然语言定义的任务。
- **Correlation Clustering**：一种图聚类方法，用于将DWUG中的usage nodes分组为meaning-related clusters（即data-driven word senses）。
- **Sense Label**：用自然语言描述的word sense代表，本文通过选取簇内最原型定义自动生成。
- **Sense Dynamics Map**：展示sense之间跨时段转移、分裂、合并关系的有向/无向图，以definition labels为节点标注。
- **Flan-T5**：Google开发的经大规模instruction finetune的T5变体，本文使用XL（3B参数）版本作为定义生成器。
- **BERT-F1**：基于BERT的语义相似度评估指标，衡量generated definition与gold definition之间的语义重合度。

## 可复现要素
- **数据集**：WordNet、Oxford、CoDWoE、English DWUGs；语言未明确声明开源方式但提供了链接（HuggingFace model hub有fine-tuned Flan-T5权重）。
- **代码/权重**：Fine-tuned Flan-T5模型通过Hugging Face model hub公开；论文未提供代码仓库链接。
- **关键超参**：Flan-T5 XL（3B参数）；prompt后缀格式`s What is the definition of w?`；greedy decoding + target word filtering；context size最大512 subwords；使用DistilRoBERTa做sentence embedding。
- **评估指标**：SacreBLEU、ROUGE-L、BERT-F1、Spearman correlation、human judgement（6级量表）。
