---
title: "Cold-Start-Data-Selection-for-Better-Few-shot-Language-Model"
source: https://aclanthology.org/2023.acl-long.141.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:07:51"
field: "少样本自然语言处理"
keywords: ["冷启动数据选择", "少样本微调", "Prompt学习", "主动学习", "不确定性传播", "样本多样性"]
innovations: ["基于 Prompt 的冷启动不确定性估计，利用 cloze 形式蒸馏任务知识", "不确定性传播机制，通过嵌入空间邻域相似度平滑不确定性", "分区-重写策略（PTR），两阶段兼顾样本信息量与跨簇多样性"]
benchmarks: ["IMDB", "Yelp-Full", "AG News", "Yahoo! Answers", "DBPedia", "TREC"]
---

# 论文速读：Cold-Start Data Selection for Better Few-shot Language Model Fine-tuning

## 一句话总结
论文提出 PATRON，一种面向冷启动场景（零初始标注数据）的数据选择方法，通过基于 Prompt 的不确定性传播与分区-重写策略，从海量无标签数据中高效筛选少量样本，显著提升少样本语言模型微调性能。

## 研究问题与动机
- **冷启动数据选择难题**：现有主动学习方法通常假设有数百至数千初始标注数据，但在实际场景中往往完全没有启动标签，导致传统不确定性采样等方法失效。
- **不确定性估计偏差**：预训练语言模型在无监督下生成的伪标签存在校准问题，直接基于熵的不确定性估计会偏向高频率类别，且对异常值敏感。
- **样本多样性不足**：已有方法先对数据进行聚类再贪心选择，无法控制不同簇间样本的距离，导致所选样本缺乏跨簇多样性，影响模型泛化。
- **预训练-微调任务不匹配**：已有方法直接使用 MLM loss 或预训练嵌入辅助选择，但预训练目标与下游分类任务存在鸿沟，难以充分挖掘任务特定知识。

## 核心贡献（创新点）
- **Prompt-based 冷启动不确定性估计**：利用 Prompt 将分类任务转化为 cloze 填空，从冻结 PLM 蒸馏任务特定知识生成伪标签，弥补预训练-微调鸿沟，区别于直接用 MLM loss 的方法（ALPS）。
- **不确定性传播机制（Uncertainty Propagation）**：通过 SimCSE 嵌入空间的核相似度计算样本间相关性，并将不确定性传播到邻域，只有样本自身及其邻居均不确定时才赋予高不确定性，区别于单纯基于单点熵的选择策略。
- **分区-重写策略（PTR）**：借鉴组合优化中的两阶段方法，先用 K-Means 分区初始化，再通过邻域图显式控制选中样本间的距离，避免仅依赖聚类忽视簇间关系的问题，优于 BERT-KM、TPC 等方法。
- **广泛的实验验证**：在 6 个文本分类数据集上系统验证，PATRON 平均超越最强基线 3.4%–6.9%，且仅用 128 个标签即可达到全监督性能的 91%–92%。

## 方法详解
**1. Prompt-based 不确定性估计**
- 使用模板将输入 $x$ 包装为包含 [MASK] 的 cloze 形式，通过词表器将标签映射为词，计算 $p(y|x) = p([\text{MASK}] = \mathcal{V}(y) | \mathcal{T}(x))$。
- 为校准伪标签偏差，计算标签词的上下文先验 $P(v)$，用每个类别概率最高的 $k$ 个样本近似，再对伪标签进行校准：$\widehat{y_i} = \left(\frac{p(y_i|x)}{P(\mathcal{V}(y_i))}\right) / \sum_j \left(\frac{p(y_j|x)}{P(\mathcal{V}(y_j))}\right)$。
- 以校准后预测的熵作为不确定性：$u(x) = -\sum_i \widehat{y_i} \log \widehat{y_i}$。

**2. 不确定性传播**
- 使用 SimCSE 生成句子嵌入 $\mathbf{z}$，以 RBF 核函数衡量相似度：$\kappa(x_i, x_j) = \exp(-\rho \|\mathbf{z}_i - \mathbf{z}_j\|_2^2)$。
- 传播不确定性：$\widehat{u}_{\text{prop}}(x) = u(x) + \frac{\sum_{x_i \in \text{KNN}(x)} \kappa(x, x_i) \cdot u(x_i)}{|\text{KNN}(x)|}$，确保局部区域内模型均不确定时样本才被选。

**3. Partition-then-Rewrite (PTR) 策略**
- **初始化**：K-Means 将无标签集分为 $b$ 个簇，每簇选取 $q_i = \text{argmax}_{x_j \in C_i} (\widehat{u}_{\text{prop}}(x_j) - \beta \|\mathbf{z}_j - \bar{\mathbf{z}}_i\|_2^2)$，平衡不确定性与簇内代表性。
- **重写优化**：构建簇级 kNN 图，对相邻簇中选中的样本施加间距约束：加入正则项 $-\gamma \sum_{q_k \in \text{c-KNN}_i} [m - \|\mathbf{z}_j - \mathbf{z}_k\|_2]_+$，当间距小于边界 $m$ 时惩罚，迭代 2-3 轮直至收敛。

## 实验与结果
- **数据集**：IMDB（2类）、Yelp-Full（5类）、AG News（4类）、Yahoo! Answers（10类）、DBPedia（14类）、TREC（6类），均为英文文本分类。
- **标注预算**：B ∈ {32, 64, 128}，单轮选择为主实验设置。
- **基线**：Random、Uncertainty、CAL、BERT-KM、Coreset、Margin-KM、ALPS、TPC。
- **主要结果**：PATRON 在所有数据集和预算下均取得最佳或接近最佳性能，平均超越最强基线 6.9%（B=32）、5.0%（B=64）、3.4%（B=128）。
- **关键数字**：仅用 128 标签（<0.5% 全量标注），PATRON 平均达到全监督性能的 91.0%（vanilla fine-tuning）和 92.1%（prompt-based learning）。
- **多样性和代表性分析**：PATRON 在类分布均衡性（IMB/LDD）和特征空间多样性上显著优于不确定性方法；PTR 进一步提升了样本多样性。

## 相关工作脉络
- **冷启动主动学习**：ALPS（Yuan et al., 2020）使用 BERT 的 MLM loss 生成 surprisal 嵌入选择样本；TPC（Hacohen et al., 2022）计算密度后从每簇选高密度样本。PATRON 相比二者利用 Prompt 蒸馏任务知识并设计传播机制与多样性控制。
- **基于嵌入的选择**：BERT-KM（Chang et al., 2021）直接用预训练嵌入做 K-Means 聚类后选中心点；Margin-KM（Müller et al., 2022）在聚类基础上选 margin 最小的样本。PATRON 使用 SimCSE 嵌入且通过 PTR 显式控制簇间距离。
- **提示学习方法**：LM-BFF（Gao et al., 2021a）利用 Prompt 进行少样本微调，是本文的下游训练范式之一；PATRON 的方法天然适配 Prompt-based 学习，因不确定性即来源于 Prompt 预测。
- **半监督学习结合**：UDA（Xie et al., 2020）和自训练方法可增强标注效率，但依赖好的初始化数据选择；PATRON 与 SSL 兼容，优于随机或次优策略选择后再 SSL 的方案。

## 局限性与未来方向
- 仅针对 MLM-style 预训练语言模型（如 BERT/RoBERTa），未考虑判别式 PLM（如 ELECTRA）；但随着判别式 Prompt 方法的发展可拓展。
- 实验集中于文本分类任务，在 NLI、关系抽取等其他 NLP 任务上的有效性尚未验证。
- 多轮 AL 中当标注充足后，纯不确定性方法可能更优，暗示需设计冷启动与 warm-start 阶段的混合策略。
- 超参数 $\rho, \beta, \gamma$ 需按数据集搜索，虽证明模型整体鲁棒，但缺乏自动调参方案。

## 研究启发与可借鉴点
- **Prompt 蒸馏任务知识用于无监督选择**：将 Prompt 生成伪标签的思路迁移到其他需要少样本选择的下游任务（如 NER、RE），可复用此框架。
- **不确定性传播的思想**：通过邻域相似性平滑不确定性估计，可推广至视觉领域或其他模态的数据选择。
- **PTR 分而治之策略**：先粗粒度分区保证覆盖、再精调优化保证多样性，可作为一种通用的批量样本选择范式。
- **与 SSL/Prompt-learning 的无缝组合**：证明数据选择与训练范式正交，后续可在同团队方向（如少样本 NER、低资源机器翻译）中直接接入 PATRON 作为数据预处理模块。

## 关键术语表
- **冷启动数据选择（Cold-start Data Selection）**：在零初始标注数据下，从无标签池中主动选择最有价值样本进行标注的任务。
- **Prompt-based Learning**：通过模板将分类任务转化为 cloze 填空形式，利用 PLM 的 MLM 能力进行少样本学习的方法。
- **不确定性传播（Uncertainty Propagation）**：将样本的不确定性与其邻域样本的不确定性加权融合，避免仅依赖单个异常样本的估计偏差。
- **分区-重写（Partition-then-Rewrite, PTR）**：先聚类分区初始化选择集合，再通过邻域约束迭代优化以增强多样性的两阶段选择策略。
- **SimCSE**：一种简单对比学习句子嵌入方法，生成高质量语义表示，用于测量样本间相似度。
- **RBF 核函数**：径向基函数核，以指数衰减形式衡量嵌入空间中两点的相似度。
- **上下文先验（Contextualized Prior）**：基于每个类别最 confident 样本的预测分布，估计标签词在语境中的出现先验概率。
- **标签失衡值（IMB）**：所选样本中最大类别频数与最小类别频数之比，衡量选择结果的类别均衡性。

## 可复现要素
- **数据集**：IMDB、Yelp-Full、AG News、Yahoo! Answers、DBPedia、TREC 均公开可用（HuggingFace 或 GitHub 链接在 Appendix A）。
- **代码**：开源，https://github.com/yueyu1030/Patron
- **模型**：RoBERTa-base（125M 参数）+ OpenPrompt 框架
- **关键超参**：ρ ∈ {0.05, 0.1}，β ∈ {0.5, 1, 5, 10}，γ ∈ {0.1, 0.3, 0.5}，m=0.5，KNN 邻域 K=50，c-KNN K'=10，训练 15 轮，batch size 4/8/16，学习率从 {1e-5, 2e-5, 5e-5} 搜索
- **评估指标**：Accuracy（TREC 报告 F1）
- **运行配置**：Ubuntu 18.04, PyTorch 1.10, NVIDIA A5000 GPU
