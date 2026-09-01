---
title: "Hierarchical-Verbalizer-for-Few-Shot-Hierarchical-Text-Class"
source: https://aclanthology.org/2023.acl-long.164.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:23:22"
field: "层次自然语言处理"
keywords: ["层次文本分类", "少样本学习", "Prompt Tuning", "Verbalizer", "预训练语言模型", "层次对比学习"]
innovations: ["首次定义path-based K-shot HTC设置与P-metric路径一致性评估", "提出HierVerb多verbalizer框架，将层次结构知识通过约束链和对比损失直接注入可学习向量", "无需额外GNN参数即可在极端少样本条件下激发PLM层次先验知识"]
benchmarks: ["WOS", "DBPedia", "RCV1-V2"]
---

# 论文速读：Hierarchical-Verbalizer-for-Few-Shot-Hierarchical-Text-Class

## 一句话总结
本文针对层次文本分类（HTC）在少样本场景下的性能瓶颈，首次定义了基于路径的K-shot设置，并提出**HierVerb**（层次感知verbalizer）框架，通过将层次结构知识注入多层verbalizer并结合层次对比学习，充分激发预训练语言模型（PLM）的先验知识，在WOS、DBPedia和RCV1-V2三个基准上均达到SOTA性能。

## 研究问题与动机
- **HTC少样本性能不足**：传统HTC方法依赖大量标注数据，在低资源/K-shot场景下表现显著下降，亟需适用于极端少样本条件的解决方案。
- **提示学习方法在HTC中研究匮乏**：尽管prompt tuning在平坦文本分类中表现优异，但针对HTC的prompt-based研究极少；既有HPT方法虽引入层次soft prompt，但未严格探索K-shot设置，且通过图编码器注入层次知识会阻碍PLM潜力的充分发挥。
- **缺乏统一的少样本HTC定义**：HTC中标签存在层次依赖关系，简单按类别采样无法满足每条路径恰好K个样本的要求，需要专门的路径级采样策略与评估指标。
- **现有verbalizer缺乏层次感知设计**：手工、搜索式和软verbalizer均假设标签无层次依赖，与PLM丰富的平坦先验知识和下游层次任务之间存在gap。

## 核心贡献（创新点）
1. **首次定义path-based few-shot HTC设置与P-metric评估体系**：基于层级标签路径进行K-shot采样，并提出Path-Metric（PMicro-F1/PMacro-F1），弥补了传统C-metric忽略子节点一致性的不足，为层次一致性评估提供更全面的标准。
2. **提出HierVerb多verbalizer框架**：将HTC建模为多层单标签/多标签分类任务，每层对应独立可学习continuous vector作为verbalizer，通过层次初始化（父节点embedding为其自身及所有后代节点的均值）将层次先验注入verbalizer，从根本上区别于将层次展平为一维空间的单一flat verbalizer方法。
3. **设计层次感知约束链（HCC）和平铺层次对比损失（FHC）**：HCC从叶节点向上自底向上传播概率约束，强制父子节点预测一致性；FHC基于SimCSE思想在实例间学习层次感知的语义匹配关系，并对底层节点赋予更高对比权重，三者协同实现从"语义视角"而非"直接拟合数据"的角度优化。

## 方法详解

### 4.1 多verbalizer框架（Multi-verbalizer Framework）
- **模板构造**：将输入文本x与动态层级模板拼接：`[CLS] It was 1 level:[MASK] 2 level:[MASK] ... D level:[MASK]. x [SEP]`，其中[MASK]数量等于层次树深度D。
- **层级表征提取**：使用BERT（bert-base-uncased）编码后，提取每个[MASK]位置的隐藏状态 $h^d$（$d \in [1,...,D]$）。
- **多层verbalizer**：每层 $V_d$ 对应一个可学习矩阵 $W_d \in \mathbb{R}^{r \times l_d}$（$l_d$为第d层标签数），初始化方式为对应标签词embedding与该标签所有后代节点embedding的均值。
- **预测概率**：$p_{ij}^d = q(h^d W_d + b_d)$，将每层任务重新定义为单标签分类（单路径）或多标签分类（多路径），损失分别为交叉熵形式。

### 4.2 层次感知约束链（Hierarchy-aware Constraint Chain, HCC）
- 维护父子层映射 $\overrightarrow{M}_d$，将下一层预测概率反传至当前层：
  $$\tilde{p}_{ij}^d = (1-\beta)p_{ij}^d + \beta \sum_{\tilde{j} \in \overrightarrow{M}_d(j)} p_{i\tilde{j}}^{d+1}$$
- 约束损失 $\mathcal{L}_{HCC}$ 在传播后概率上计算交叉熵，使模型从叶节点开始逐层向上传播一致性约束。

### 4.3 平铺层次对比损失（Flat Hierarchical Contrastive Loss, FHC）
- 借鉴SimCSE，对batch内任意句子对计算各层[MASK]隐藏状态的余弦相似度。
- 构建层次标签矩阵 $M_d(n,n')$，同一层有交集则为1，并引入指数权重因子 $\frac{1}{2^{(D-d)\times\alpha}}$ 赋予深层节点更多对比损失权重。
- 最终损失函数：
  $$L_{FHC} = \frac{-1}{N^2 D^2}\sum_d\sum_u\sum_n \log\frac{\exp(\sum_{n'} S(h_n^u,h_{n'}^u)M_u(n,n'))}{\exp(\sum_{n'} S(h_n^u,h_{n'}^u))}\times\frac{1}{2^{(D-d)\times\alpha}}$$

### 4.4 总体目标函数
$$\mathcal{L} = \mathcal{L}_C + \lambda_1 \mathcal{L}_{HCC} + \lambda_2 \mathcal{L}_{FHC}$$
其中 $\lambda_1=1$，$\lambda_2$ 在WOS/DBPedia上取 $1e^{-2}$，在RCV1-V2上取 $1e^{-4}$；$\alpha=1$，$\beta$ 在RCV1-V2上取 $1e^{-2}$。

## 实验与结果

### 数据集
| 数据集 | Level 1 | Level 2 | Level 3 | Level 4 | 文档数 |
|--------|---------|---------|---------|---------|--------|
| WOS | 7 | 134 | — | — | 46,985 |
| DBPedia | 9 | 70 | 219 | — | 381,025 |
| RCV1-V2 | 4 | 55 | 43 | 1 | 804,410 |

### 主要结果（K-shot Micro/Macro-F1）
- **WOS数据集**：1-shot下HierVerb达到58.95%/44.96%，较最佳基线HPT（50.05%/25.69%）绝对提升8.9%/19.27%；8-shot达到78.12%/69.98%，超过HPT（76.22%/67.20%）。
- **DBPedia数据集**：1-shot下HierVerb达到91.81%/85.32%，较HPT（72.52%/31.01%）提升19.29%/54.31%；16-shot达到96.17%/93.28%。
- **RCV1-V2数据集**：由于标签无自然语言描述，HierVerb仍取得SOTA，1-shot为40.95%/4.87%（HPT为27.70%/3.35%）；8-shot达63.90%/31.13%。

### 一致性实验（P-metric）
在1-shot WOS上，HierVerb的PMicro-F1达39.77%，而HGCLR和Vanilla FT为0.0，HPT为19.97%，说明HierVerb的语义优化显著提升了跨层一致性。

### 模型规模实验
在bert-large-uncased（330M）上，1-shot WOS的Macro-F1提升从bert-base的19.27%增至27.92%，证明HierVerb在更大模型上挖掘先验知识的能力更强。

## 相关工作脉络
1. **HiMatch（Chen et al., 2021）**：基于文本-标签语义匹配学习的HTC方法，使用BERT作为编码器；HierVerb与其核心差异在于不使用额外GNN参数，而是通过层次感知verbalizer直接建模层级关系。
2. **HGCLR（Wang et al., 2022a）**：将层次结构融入text encoder的对比学习方法；HierVerb的优势在于不需要图结构参数注入，在少样本下泛化更好（一致性指标显著优于HGCLR）。
3. **HPT（Wang et al., 2022b）**：首次将prompt tuning应用于HTC，通过graph encoder将层次信息注入soft prompt；HierVerb定位为更轻量的替代方案——无需额外参数、通过verbalizer直接建模层次，少样本下提升显著。
4. **Prompt Tuning（Lester et al., 2021; Qin & Eisner, 2021）**：连续软prompt学习范式；HierVerb延续该思路，但创新性地将其适配到多层的层次分类场景。
5. **Verbalizer Design（Schick & Schütze, 2021; Hambardzumyan et al., 2021）**：手工/搜索式/软verbalizer均假设标签扁平无层次；HierVerb的核心区别是将层次约束显式嵌入每个可学习verbalizer向量中。
6. **Path-based Evaluation（Yu et al., 2022）**：提出C-metric约束祖先节点正确性；HierVerb在此基础上进一步提出P-metric，从路径完整性角度评估层次一致性。

## 局限性与未来方向
- **难以直接扩展至超大规模语言模型（>=175B）**：HierVerb作为轻量微调方法依赖可更新参数，在冻结大模型参数的setting下适用性受限。
- **依赖PLM中的自然语言先验**：对于RCV1-V2等标签无语义描述的 dataset，性能提升受限（1-shot Macro-F1仅4.87%）。
- **未来方向**：计划将方法扩展至更大规模语言模型（仅优化下游特定参数部分），并探索zero-shot学习场景。

## 研究启发与可借鉴点
1. **多verbalizer分层建模思想可迁移至其他层次结构化任务**（如实体类型预测、知识图谱补全），通过逐层独立verbalizer替代展平后的全局分类器，能够更自然地保留层次结构信息。
2. **层次感知约束链（HCC）的概率传播机制可推广**：将子节点约束反向传播至父节点的思路，可用于任意树形/图形标签结构的分类任务，保证预测的一致性。
3. **平铺层次对比损失（FHC）的加权策略值得借鉴**：通过指数衰减权重赋予深层节点更大对比约束，有效防止浅层标签对整体优化的主导，这一设计对其他层次表征学习任务有参考价值。
4. **Path-based评估指标体系具有通用性**：P-metric从路径完整性角度评估层次预测质量，相比C-metric更严格，可作为层次NLP任务的标准化评估工具。
5. **无需额外参数即可注入层次先验**：与HPT/HGCLR等依赖GNN的参数增强方法相比，HierVerb的"轻量级层次注入"策略在少样本下更不易过拟合，为资源受限场景提供了新思路。

## 关键术语表
**Hierarchical Text Classification (HTC)**：层次文本分类任务，将文本映射到树状/图状标签层次结构中的一个或多个路径。
**Few-shot HTC (Few-HTC)**：极少量标注样本（每路径K个）条件下的层次文本分类任务，本文首次严格定义path-based的K-shot设置。
**HierVerb**：本文提出的层次感知verbalizer框架，通过多层可学习continuous vector将层次结构知识注入verbalizer空间。
**Prompt Tuning**：通过在预训练模型输入中添加可学习prompt模板，以最小化参数更新的方式适配下游任务的训练范式。
**Verbalizer**：连接模型预测输出（[MASK]位置词分布）与下游标签的映射组件，可以是手工词表、搜索词表或可学习向量。
**Hierarchy-aware Constraint Chain (HCC)**：从叶节点到根节点自底向上传播预测概率约束的机制，强制父子节点预测一致性。
**Flat Hierarchical Contrastive Loss (FHC)**：基于SimCSE的对比学习损失，在实例间学习层次感知的语义匹配，并对深层节点赋予更高对比权重。
**Path-based Evaluation Metric (P-metric)**：以完整预测路径为评估单元的一致性指标，仅当路径上所有节点均正确预测时才计为正确。

## 可复现要素
- **数据集**：WOS、DBPedia、RCV1-V2均为公开数据集；论文声明将开源所有随机种子下的few-shot数据集。
- **代码/权重**：论文声明开源代码及checkpoint（ethics statement中提及）。
- **关键超参**：
  - λ1 = 1，λ2 = 1e-2（WOS/DBPedia）/ 1e-4（RCV1-V2）
  - α = 1，β = 1（WOS/DBPedia）/ 1e-2（RCV1-V2）
  - 学习率：WOS/DBPedia为5e-5，label embedding优化为1e-4；RCV1-V2为3e-5
  - batch size = 5，max length = 512，warmup steps = 0
  - 训练轮数：WOS/DBPedia为20 epochs，RCV1-V2为1000 epochs（early stopping=10）
  - 优化器：Adam
- **基座模型**：bert-base-uncased（默认）/ bert-large-uncased（规模实验）
