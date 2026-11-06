---
title: "Subjective-Crowd-Disagreements-for-Subjective-Data-Uncoverin"
source: https://aclanthology.org/2023.acl-long.54.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 09:16:32"
field: "自然语言处理-主观内容理解"
keywords: ["label distribution learning", "crowd disagreement", "subjective annotation", "clustering", "social media content moderation", "population-level learning"]
innovations: ["提出CrowdOpinion两阶段框架，联合语言特征与标签分布进行聚类以缓解标注稀疏性", "引入可调节混合参数w控制特征与标签在聚类中的权重，验证了聚类在分布预测上的普遍有效性", "证明保留标注分歧比DS类噪声消除方法更能提升分布预测和单标签分类性能"]
benchmarks: ["GoEmotions (DGE)", "Jobs QA (DJQ1/DJQ2/DJQ3)", "SBIC Intent (DSI)", "Facebook Reactions (DFB)"]
---

# 论文速读：Subjective-Crowd-Disagreements-for-Subjective-Data-Uncoverin

## 一句话总结
论文提出 **CrowdOpinion**，一种两阶段的无监督学习框架，通过结合语言特征和标签分布进行聚类，更好地估计标注者群体层面的标签分布，从而保留和预测主观数据中的多元观点与分歧。

## 研究问题与动机
- **主观数据标注分歧普遍存在**：社交媒体内容（如冒犯性言论）的判断高度依赖主观认知，不同标注者对同一文本的判断差异显著，传统多数投票会抹杀少数观点。
- **少数观点具有现实意义**：以内容审核为例，施虐者常使用微妙语言伪装恶意（deniability），忽略少数观点会导致模型在检测有害内容时性能下降。
- **现有标注集标注者数量太少**：大多数公开数据集仅含3–10位标注者，统计上无法代表隐藏的真实人群分布，导致稀疏空间学习困难。
- **现有聚类方法仅利用标签空间**：Liu et al. (2019) 等工作仅在标签分布空间聚类，未利用文本语义特征，限制了聚类质量。

## 核心贡献（创新点）
1. **提出CrowdOpinion两阶段框架**：Stage 1 联合特征与标签分布进行聚类，Stage 2 在聚类后的软标签分布上训练监督模型，本质区别于仅用标签聚类的已有方法。
2. **引入可调节的特征-标签混合参数 w**：将语言特征（384维SBERT嵌入）与标签分布按权重 $w \cdot x_i + (1-w) \cdot y_i$ 拼接后共同聚类，w 为超参控制文本 vs 标签的重要性。
3. **系统评估5种聚类方法 × 多数据集**：比较FMM、GMM、K-Means、LDA（生成式）和NBP（距离-based）在6个标注分歧程度各异的公开数据集上的表现，发现不同数据集最优 w 差异大（从0到1）。
4. **证明聚类可普遍缓解标签稀疏性**：所有聚类方法在KL散度指标上均优于无聚类基线（PD、SL、DS+CNN），且 w 的变化对最终性能影响不大，说明标签分布本身即是有效信号。

## 方法详解
**Stage 1 — 联合特征-标签聚类（无监督）**：
- 每个数据项 $x_i$（文本）关联一个标签分布向量 $y_i$（各标签的标注者比例，归一化）。
- 构造联合向量 $(w \cdot x_i, (1-w) \cdot y_i)$，其中 $w \in [0,1]$ 为混合参数。
- 使用5种聚类算法之一进行聚类，以最小化总KL散度 $\sum_i KL((x_i,y_i)_w \| (\hat{x}_i,\hat{y}_i)_w)$ 自动选择最优聚类数 $p$（或半径 $r$）。
- 聚类中心 $\hat{y}_i$ 作为该簇内所有项共享的 smoothed 标签分布估计。
- 五种聚类方法：FMM（Dirichlet先验的多项式混合）、GMM、K-Means（scikit-learn）、LDA（Gensim）、NBP（按KL半径聚合邻居）。

**Stage 2 — 监督学习（有监督）**：
- 使用聚类后的 $(x_i, \hat{y}_i)$ 训练 1D-CNN（3层卷积/最大池化，dim=128，dropout=0.5，softmax输出），输入为384维 paraphrase-MiniLM-L6-v2 词向量。
- 评估指标：① KL散度（分布预测质量）；② Accuracy（以arg max标签衡量单标签预测）。

**基线方法**：
- PD：CNN直接在原始empirical分布 $y_i$ 上训练。
- SL：CNN在多数标签one-hot上训练。
- DS+CNN：Dawid-Skene模型估计标注者准确率后，再用CNN做预测。
- CO-CNN-w=0：仅用标签分布聚类的Liu et al. (2019) 基线。

## 实验与结果
**数据集**（6个公开英文数据集，每数据集采样至2000条，50/25/25划分）：

| 数据集 | 标注者/项 | 标签数 | 平均熵 |
|---|---|---|---|
| $\mathcal{D}_{FB}$（Facebook reactions） | 平均862.3 | 5 | 0.784 |
| $\mathcal{D}_{GE}$（Reddit情感） | 平均4 | 28 | 0.866 |
| $\mathcal{D}_{JQ1}$（Twitter求职POV） | 10 | 5 | 0.746 |
| $\mathcal{D}_{JQ2}$（Twitter就业状态） | 10 | 5 | 0.586 |
| $\mathcal{D}_{JQ3}$（Twitter求职事件） | 10 | 12 | 0.993 |
| $\mathcal{D}_{SI}$（SBIC冒犯意图） | 平均3 | 4 | 0.343 |

**最佳模型选择（表2）**：$\mathcal{D}_{GE}$ 和 $\mathcal{D}_{JQ3}$ 最优为 $w=0$（仅标签），$\mathcal{D}_{SI}$ 最优为 $w=1$（仅特征，K-Means），其余多为 NBP 且 w 各不相同。

**KL散度结果（表3，越低越好）**：
- $\mathcal{D}_{FB}$：最佳 **GMM w=1** = **0.680±0.001**，优于 PD(0.857) 和 DS+CNN(3.247)。
- $\mathcal{D}_{GE}$：最佳 **LDA w=0** = **1.987±0.001**，优于 PD(2.011) 和 DS+CNN(3.247)。
- $\mathcal{D}_{JQ1}$：最佳 **GMM w=1** = **0.427±0.01**，优于 PD(1.092)。
- $\mathcal{D}_{JQ2}$：最佳 **NBP w=0.25** = **0.499±0.001**，优于 PD(1.088)。
- $\mathcal{D}_{JQ3}$：最佳 **LDA w=1** = **0.823±0.001**，优于 PD(1.462)。
- $\mathcal{D}_{SI}$：最佳 **FMM w=0** = **0.860±0.026**，优于 PD(0.889)。

**Accuracy结果（表4，越高越好）**：
- $\mathcal{D}_{FB}$：最佳 **FMM w=1** = **0.790±0.001**。
- $\mathcal{D}_{GE}$：最佳 **LDA w=0.25** = **0.949±0.001**。
- $\mathcal{D}_{JQ1}$：最佳 **FMM w=0** = **0.892±0.001**。
- $\mathcal{D}_{JQ2}$：最佳 **NBP w=0.25** = **0.873±0.001**。
- $\mathcal{D}_{JQ3}$：最佳 **LDA w=0** = **0.880±0.001**。
- $\mathcal{D}_{SI}$：最佳 **FMM w=0** = **0.949±0.0**。

**关键结论**：所有聚类方法在KL散度上稳定优于无聚类基线；DS+CNN consistently最差（因其将分歧视为噪声并试图消除，而非保留）；w 对最终 accuracy 影响较小，说明仅用标签分布聚类本身即有效。

## 相关工作脉络
- **Liu et al. (2019) PLDL**：仅在标签分布空间 $\mathcal{Y}$ 聚类做标签平滑，本文扩展为联合特征+标签空间 $w\cdot\mathcal{X} \times (1-w)\cdot\mathcal{Y}$ 聚类，w=0 即为该基线。
- **Dawid & Skene (1979)**：经典标注者误差估计模型，旨在消除分歧得到单一真值；本文与之对比（DS+CNN），结果表明保留分歧的聚类方法更优。
- **Weerasooriya et al. (2020) NBP/PLL**：基于KL距离的邻居聚合方法，本文将其纳入对比并扩展至特征+标签联合空间。
- **Aroyo & Welty (2014) CrowdTruth**：主张保留全部标注而非取多数，本文从建模角度实现这一理念。
- **Gordon et al. (2022) Jury Learning**：将持异议的少数群体声音纳入ML；本文方法在无需标注者级标签的情况下实现类似目标。
- **Sap et al. (2021)**：分析标注者分歧根源，本文模型能识别含deniability的冒犯内容，与其发现一致。

## 局限性与未来方向
- 目前仅在英文数据集上实验，未验证跨语言泛化能力。
- 混合参数 w 的搜索较粗糙（仅0/0.25/0.5/0.75/1），细粒度调优空间待探索。
- 评估指标（KL散度、Accuracy）无法完全捕捉人群反应的细微含义，缺乏针对subpopulation（如弱势群体）的预测能力。
- Stage 1（聚类/混合）基于scikit-learn/bnpy/Python脚本实现，未做GPU加速，计算效率可能是瓶颈。

## 研究启发与可借鉴点
- **特征-标签联合聚类思想**：将文本语义特征与标签分布向量拼接后进行聚类，可有效缓解标注稀疏问题，适用于任何多标注者标注场景。
- **混合参数 w 的自适应策略**：不同数据集最优 w 差异大（有时仅用标签，有时仅用特征），未来可探索数据驱动的 w 学习机制而非手动调参。
- **保留分歧的建模理念**：DS类方法将分歧视为噪声消除，本文证明分歧蕴含语义信号，该思路可迁移至多annotator的医疗标注、法律文本、安全审核等领域。
- **两阶段范式**：无监督聚类做标签平滑 → 监督模型学习分布，可复用于其他soft label任务（如情感细粒度分类、立场检测）。

## 关键术语表
- **CrowdOpinion (CO)**：论文提出的两阶段学习框架，Stage 1 联合特征与标签分布聚类，Stage 2 在平滑分布上训练监督模型。
- **Population-level Learning**：目标不是预测单一标注者答案，而是估计整个隐藏人群对某条内容的响应分布。
- **Label Distribution（标签分布）**：每个数据项对应一个向量，元素为各类别被标注者选择的比例，而非单一hard label。
- **KL-divergence（KL散度）**：评估预测分布与真实分布之间差异的指标，越低表示预测越准确。
- **Mixing Parameter w**：控制特征空间与标签分布在聚类中相对权重的超参，w=0 仅用标签，w=1 仅用文本特征。
- **NBP（Neighborhood-Based Pooling）**：基于KL半径的距离型聚类方法，对每个样本聚合其邻域内的标签分布均值。
- **FMM（Finite Mixture Model）**：带Dirichlet先验的多项式混合模型，用于生成式聚类标签分布。
- **Deniability（可否认性）**：施虐者在社交媒体中用玩笑、讽刺等包装冒犯内容，使多数标注者难以判定为恶意。

## 可复现要素
- **数据集**：6个均为公开数据集，GitHub链接见Appendix A（GoEmotions、Jobs/QA数据集、SBIC Social-Bias-Frames、Facebook Reactions）。
- **代码/权重**：论文未提供官方开源代码仓库；使用bnpy、scikit-learn、Gensim、Sentence-Transformers（paraphrase-MiniLM-L6-v2）、TensorFlow/Keras。
- **关键超参**：混合参数 $w \in \{0, 0.25, 0.5, 0.75, 1\}$；聚类数 $p \in [4, 40]$（生成式模型）/ 半径 $r \in [0, 15]$（NBP）；CNN维度128、dropout 0.5。
