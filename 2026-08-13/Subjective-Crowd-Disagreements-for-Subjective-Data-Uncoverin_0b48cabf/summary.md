---
title: "Subjective-Crowd-Disagreements-for-Subjective-Data-Uncoverin"
source: https://aclanthology.org/2023.acl-long.54.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:59:52"
---

# 论文速读：Subjective-Crowd-Disagreements-for-Subjective-Data-Uncoverin

## 一句话总结
本文提出 CrowdOpinion (CO) 两阶段学习框架，通过联合文本特征与标签分布进行无监督聚类，将稀疏的多标注者回复聚合为人口级软标签分布，从而在主观社交媒体数据上保留并建模群体意见分歧，提升分布预测与单标签分类性能。

## 研究问题与动机
1. **多数投票抹除少数派声音**：传统流程在标注后直接丢弃少数意见，但主观内容（如攻击性语言意图）的少数派判断往往包含重要语义信号，丢弃会导致模型在真实审核场景中产生偏见或漏检。
2. **标注样本极度稀疏**：现有公开数据集每样本通常仅 3–10 位标注者，统计上不足以代表潜在的总体人群分布，直接训练易过拟合噪声。
3. **分歧常被视作噪声而非信号**：以 Dawid-Skene 为代表的经典模型将不一致标注视为标注者错误并予以消除，与“保留分歧”的新型众包哲学背道而驰。
4. **缺乏特征-标签联合的分布正则化方法**：既有工作（如 Liu et al., 2019）仅在标签空间聚类，未探索文本语义与标签分布共同指导的聚合机制。

## 核心贡献（创新点）
1. **提出 CrowdOpinion 两阶段框架**：Stage 1 联合文本嵌入与经验标签分布进行聚类以获得平滑的软标签 $\hat{y}_i$，Stage 2 在 $(x_i, \hat{y}_i)$ 上训练监督模型；与仅依赖多数投票或原始稀疏分布的训练方式本质不同。
2. **引入特征-标签混合权重 $w$ 的联合聚类机制**：将聚类空间扩展为 $w \cdot \mathcal{X} \times (1-w) \cdot \mathcal{Y}$，系统对比 FMM、GMM、K-Means、LDA 与 NBP 五种聚类器；相比 Liu et al. (2019) 的纯标签聚类 ($w=0$)，本文提供了可调节的语义-分布权衡。
3. **在六类主观社交媒体数据集上全面验证分布学习优势**：证明聚类聚合能普遍缓解标注稀疏性，且文本特征与标签分布在提升分布预测上效果相当，验证了“分歧具有信息价值”的核心假设。
4. **揭示软分布对内容审核公平性的支持作用**：通过案例展示模型能在低置信度、强讽刺/推诿话语上保留少数派攻击意图判断，为平衡言论自由与平台安全提供可解释的分布依据。

## 方法详解
- **整体架构（Algorithm 1）**：
  - **Stage 1（无监督聚类/分布聚合）**：对每个样本 $i$，构造加权拼接向量 $(w \cdot x_i, (1-w) \cdot y_i)$，其中 $x_i$ 为 384 维 `paraphrase-MiniLM-L6-v2` 文本嵌入，$y_i$ 为经验标签比例向量。在该联合空间执行聚类，取簇内样本的标签分布均值作为该簇的代表软标签 $\hat{y}_i$。
  - **Stage 2（监督学习）**：使用 1D CNN（三层卷积/池化，维度 128，dropout 0.5，softmax 输出）在 $(x_i, \hat{y}_i)$ 对上进行分布预测训练，损失为交叉熵。
- **聚类器与超参搜索**：
  - 生成式：FMM（Dirichlet prior）、GMM、K-Means、LDA（Gensim）；聚类数 $p \in [4, 40]$。
  - 距离型：NBP（Neighborhood-based Pooling），对每个样本在 KL 球半径 $r \in [0, 15]$ 内对邻居标签分布直接平均（公式 1）。
  - 模型选择目标：最小化原始分布与聚类 centroids 之间的总 KL 散度 $\sum_i KL((x_i, y_i)_w \| (\hat{x}_i, \hat{y}_i)_w)$。
  - 混合权重 $w \in \{0, 0.25, 0.5, 0.75, 1\}$，$w=0$ 退化为仅标签聚类，$w=1$ 退化为仅特征聚类。
- **基线设置**：PD（直接用原始分布训练 CNN）、SL（多数投票 one-hot 训练）、DS+CNN（Dawid-Skene 估计隐藏真实标签后接 CNN）、CO-$\mathcal{C}$-CNN-0（Liu et al. 2019 仅标签聚类基线）。

## 实验与结果
- **数据集**：$\mathcal{D}_{FB}$（Facebook 帖子反应，均 862.3 位用户）、$\mathcal{D}_{GE}$（Reddit 情绪，28 类）、$\mathcal{D}_{JQ1/2/3}$（Twitter 就业相关，5/5/12 类）、$\mathcal{D}_{SI}$（社交偏见意图，4 类）；均采样至 2000 条，50/25/25 切分。
- **Q1 聚类质量（KL 散度）**：NBP 在 $\mathcal{D}_{FB}/\mathcal{D}_{GE}/\mathcal{D}_{JQ1}/\mathcal{D}_{JQ2}/\mathcal{D}_{JQ3}$ 上最优，K-Means 在 $\mathcal{D}_{SI}$ 上最优。最优 $w$ 跨数据集差异显著：$\mathcal{D}_{GE}$ 与 $\mathcal{D}_{JQ3}$ 选 $w=0$（纯标签），$\mathcal{D}_{SI}$ 选 $w=1$（纯特征），其余数据集多在 $0.25–0.75$ 间。整体 KL 值较低，表明同簇样本分布高度
