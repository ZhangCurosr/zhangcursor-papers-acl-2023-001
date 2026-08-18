---
title: "Controlling-Learned-Effects-to-Reduce-Spurious-Correlations"
source: https://aclanthology.org/2023.acl-long.127.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:49:52"
field: "自然语言处理中的因果推理与去偏"
keywords: ["spurious correlations", "causal inference", "data augmentation", "debiasing NLP", "Riesz representer", "treatment effect estimation"]
innovations: ["提出 Riesz Representer 双稳健 ATE 估计器，在高维文本低 overlap 场景下显著优于倾向得分方法", "提出 FEAG 框架，通过反事实数据增强实现对分类器特征效应的可控约束，兼顾总准确率与少数群体准确率", "将特征效应估计拓展至标注者偏见检测，展示方法的通用性"]
benchmarks: ["CivilComments Semi-Synthetic", "CivilComments Subsampled", "IMDB"]
---

# 论文速读：Controlling-Learned-Effects-to-Reduce-Spurious-Correlations

## 一句话总结
本文提出 Feature Effect Augmentation (FEAG)，一种基于因果推断的自动化数据增强方法，通过估计文本特征对标签的真实因果效应（Average Treatment Effect），在训练时控制分类器对该特征的依赖程度，从而在减少虚假相关性的同时避免过度去除对任务有用的特征信号。

## 研究问题与动机
- **核心问题**：NLP 分类器（如 BERT）会学习输入特征与标签之间的虚假相关（spurious correlations），如 IMDB 评分数字与情感标签、毒性分类中少数族裔词汇等，导致模型在少数群体（打破该相关的样本）上表现差，且在敏感特征下产生不公平预测。
- **现有方法的不足**：主流去偏方法（INLP 隐空间投影、DFL/PoE 样本重加权、Subsample/GroupDRO 最坏组优化等）倾向于**完全消除**特征效应，但许多特征实际上对任务有正向因果贡献（如 "not" 对 NLI、评分词对情感），完全去除会损害总体准确率。
- **关键权衡**：整体准确率（majority group）与少数群体准确率（minority group）之间存在 trade-off，需要一个**可精细控制**的方法而非一刀切去除。
- **因果估计的难题**：文本数据高维且 overlap 低，传统 propensity-based 估计器方差极大；需要一种更稳定的 ATE 估计方法。

## 核心贡献（创新点）
1. **提出 Riesz Representer (RR) 双稳健 ATE 估计器**：绕过 propensity 估计直接学习权重函数 α_R，在高维文本和 low overlap 场景下比 Direct 和 Propensity-DR 误差更低（MAE 显著降低）。
2. **提出 FEAG（Feature Effect Augmentation）框架**：将估计的特征因果效应用于自动生成反事实样本并重新标注，以数据增强形式约束分类器学到的特征效应等于真实效应，支持从"完全去除（τ=0）"到"保留真实效应（τ=ATE）"的连续控制。
3. **在多个数据集上实现整体准确率与少数群体准确率的更好 trade-off**：在 CivilComments 和 IMDB 上，FEAG(ate) 在提升平均组准确率的同时保持或提升总准确率，优于 DFL、GroupDRO、INLP、Subsample 等基线。
4. **发现特征效应估计器的通用性**：除去偏外，还可用于检测标注者偏见（如 CivilComments 中 "gay"/"black" 的估计效应显著非零，提示标注偏差）。

## 方法详解

**因果图设定**：作者意图 C → 完整文本 Z=(X, T)，其中 T 为处理特征（treatment，二元），X 为其余文本（covariates），标注者从 Z 生成标签 Y。C 作为 confounder 导致 X 与 T 相关，引发虚假相关。

**核心目标**：令分类器 f 学到的特征效应等于真实特征效应 τ^j = E_D[f(X^j, T^j=1) - f(X^j, T^j=0)]。

**特征效应估计——Riesz Representer 估计器**：
- Direct 估计（Eq.4）：用训练好的模型 g 直接计算 g(X,1)-g(X,0)，但在 T 与 X 相关时会因模型偏差而低估或高估 ATE。
- Propensity-DR 估计（Eq.5）：引入倾向得分 P(T=1|X)，但分母接近 0 时方差爆炸，在文本数据中尤为严重。
- **RR-DR 估计（Eq.6）**：利用 Riesz 表示定理，直接学习 α_R(Z) 替代 1/P(X) 项，优化目标为 min_α E[-2(α(X,1)-α(X,0)) + α(Z)²]，最终估计：
  ÂTE_DR,R = ÂTE_Direct + (1/n)Σ α_R(Z_i)(Y_i - g(Z_i))
- 采用双头共享 BERT 架构：共享表征层，分别用两个线性头学习 α_R 和 g。

**FEAG 两阶段流程**：
1. **特征效应估计**：对每个疑似虚假特征 T^j，用 RR 估计器计算 τ^j。
2. **反事实增强**：对样本 Z=(X,T)，生成反事实输入 Z^C=(X, 1-T)，根据 τ^j 和标签翻转算法（见 Supp H）推断新标签 Y^C，构成增强分布 D^C。训练时在原始损失上加增强损失：
   min_f E_D[L(Y, f(Z))] + λ E_{D^C}[L(Y^C, f(Z))]，λ=0.1。
   - FEAG(ate)：使用估计的 ATE 作为 τ^j。
   - FEAG(0)：设 τ^j=0，即完全去除特征效应。

## 实验与结果

**数据集**：
- CivilComments Semi-Synthetic（半合成）：基于真实毒性数据集构建因果图，可控 ATE τ∈{0.10, 0.30, 0.50} 和 overlap ϵ∈{1%, 5%, 10%}。
- CivilComments Subsampled：对原始数据子采样强化 "kill" 与标签的虚假相关。
- IMDB：评分数字 7/8/9 vs 2/3/4 作为 treatment，预测情感准确率达 90%。

**估计器对比（Table 1）**：
- Riesz 在所有 overlap 和 τ 设置下 MAE 最低，Propensity 在低 overlap 下方差极大（如 τ=0.5, 1% overlap 时 MAE=61.88），Direct 在高 τ 低 overlap 下偏差严重。

**分类器性能（BERT，主要结果）**：
- **CivilComments SS（Table 4）**：FEAG(ate) 总准确率 87.62%，平均组准确率 52.39%，优于所有基线（GroupDRO 平均组最高 69.69 但总准确率仅 66.02；Subsample 平均组 57.66 但总准确率 68.27）。
- **CivilComments Subsampled（Table 5）**：FEAG(ate) 总准确率 79.66%，平均组 66.25%，略高于 FEAG(0) 的 78.87%/65.88%，证明保留真实效应（而非完全去除）更有利。
- **IMDB（Table 6）**：FEAG(ate) 总准确率 89.38%，平均组 63.36%；FEAG(0) 总准确率 89.33%，平均组 68.00%——FEAG(0) 在 IMDB 上平均组更高，说明此场景中评分数字确实可视为纯虚假特征。

**最强结果**：在 CivilComments Subsampled 上，GroupDRO 平均组准确率最高（67.52%），但总准确率从 79.38% 降至 77.25%；FEAG(ate) 在总准确率和平均组准确率之间取得最优平衡（79.66%/66.25%）。

## 相关工作脉络
- **INLP/RLACE**（隐空间去除）：通过投影或对抗训练从表征中剥离特征，但与任务表征紧密纠缠，常连带移除有用信息；FEAG 通过因果控制保留必要效应。
- **DFL/PoE**（重加权）：利用偏见模型辅助训练并重加权样本，本质是降低 majority group 权重；FEAG 通过反事实增强直接在数据空间操作，不依赖偏见模型的显式建模。
- **GroupDRO/Subsample**（最坏组优化）：通过重采样或分布鲁棒优化提升最弱组；但以牺牲总准确率为代价；FEAG 可同时维持或提升总准确率。
- **DragonNet/Propensity-DR**（倾向得分因果估计）：传统文本因果估计依赖倾向得分，低 overlap 下方差大；本文引入 Riesz 表示定理绕过 propensity 估计。
- **Counterfactual Augmentation**（Kaushik et al., 2019 等）：需人工标注反事实样本；FEAG 自动推断反事实标签，无需人工干预。
- **Joshi et al. (2022)**：从必要性和充分性角度刻画虚假特征；本文与其一致地主张"精细化处理"而非完全去除，但提供了可操作的自动化实现。

## 局限性与未来方向
- **依赖反事实输入生成**：当前仅考虑 token 级别特征（如单词 kill、数字评分），复杂特征（短语、句法结构）的反事实生成仍有挑战；作者建议结合 Neurocounterfactuals 等工作拓展。
- **因果图假设**：方法依赖设定的因果图（confounder C），真实场景中 confounder 可能不可观测或设定不准确。
- **标签本身可能有偏**：若训练标签含标注者偏见（如 Sec 5.4 发现的 "gay"/"black" 效应），FEAG 会学习并保留这些偏见；作者指出未来应结合去偏见标签的方法。
- **双头模型训练稳定性**：共享表征+双头的架构可能在某些设置下互相干扰，需调参。

## 研究启发与可借鉴点
- **Riesz 估计器可迁移到其他文本因果推断任务**：如因果中介分析、文本中的异质性处理效应估计，在低 overlap 场景下明显优于 propensity-based 方法。
- **"可控效应"思想适用于更多去偏场景**：不限于 token 级特征，可推广到段落级、风格级等更粗粒度特征，只需能定义反事实扰动即可。
- **FEAG 的增强标签生成策略（Supp H）**：基于概率翻转的标签推断方法简洁有效，可借鉴到其他需要自动标注反事实数据的工作。
- **双评估指标（总准确率+平均组准确率）的 reporting 范式**：值得在公平性/NLP 去偏论文中推广，避免仅报告单一指标造成的片面结论。
- **与 LLM 结合的可能性**：FEAG 目前基于 BERT/DistilBERT，可探索在 LLM 微调中应用该框架进行可控去偏，尤其在大模型幻觉/偏见控制方向。

## 关键术语表
- **Spurious Correlation（虚假相关）**：模型学习到的输入特征与标签之间的统计关联，但该关联并非因果效应，在分布外或少数群体上导致错误预测。
- **Average Treatment Effect (ATE)**：在保持其他特征不变的条件下，将处理特征 T 从 0 变为 1 时标签 Y 的期望变化量，是因果推断中衡量特征效应的核心量。
- **Riesz Representer (RR) Estimator**：基于 Riesz 表示定理的双稳健 ATE 估计器，直接学习权重函数 α_R 替代倾向得分倒数，避免低 overlap 下的高方差问题。
- **Counterfactual Input（反事实输入）**：将原输入 Z=(X, T) 中的处理特征 T 翻转为 1-T 而保持 X 不变，得到 Z^C=(X, 1-T)，用于估计特征的真实因果效应。
- **Overlap（重叠）**：指对于所有协变量 X，处理 T 取 0 和 1 的概率均大于 0（即 0 < P(T=1|X) < 1），是因果效应可识别的必要条件。
- **Confounding Variable（混杂变量）**：同时影响处理 T 和协变量 X 的潜变量 C（如写作意图），导致 T 与 X 相关，使观察到的关联偏离真实因果效应。
- **Doubly Robust (DR) Estimator（双稳健估计器）**：同时使用 outcome 模型 g 和权重函数 α_R 的 ATE 估计方法，只要 g 或 α_R 中至少一个正确，估计即无偏。
- **FEAG (Feature Effect Augmentation)**：本文提出的两阶段算法，先估计特征效应，再利用该效应自动生成并标注反事实样本进行训练增强，实现对分类器特征依赖的精细控制。

## 可复现要素
- **数据集**：CivilComments（公开）、IMDB（公开）；半合成数据集为作者基于 CivilComments 构建。论文未提供代码/数据仓库链接。
- **代码/权重**：论文未提及开源，引用了 INLP 官方仓库。
- **关键超参**：learning rate（BERT 1e-5，线性头 1e-4）、batch size=32、weight decay=1e-2、FEAG 增强权重 λ=0.1、最大 token 长度 256、训练轮数（CivilComments 10 epoch，IMDB 30 epoch）。
- **随机种子**：3 个种子 {0, 11, 44}，报告均值±标准误。
- **模型**：BERT-base（110M 参数）和 DistilBERT（55M 参数）。
