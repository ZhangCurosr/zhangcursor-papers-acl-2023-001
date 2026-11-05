---
title: "Binary-and-Ternary-Natural-Language-Generation"
source: https://aclanthology.org/2023.acl-long.5.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:57:57"
---

# 论文速读：Binary-and-Ternary-Natural-Language-Generation

## 一句话总结
本文首次将预训练 Transformer 生成模型（BART/mBART）的权重与激活同时量化至三元（ternary）甚至二元（binary）精度，通过结合最大熵等距权重量化与基于学习的弹性激活量化，在摘要生成与机器翻译任务上实现了以往方法无法收敛的低比特生成模型，效率提升达 16× 以上。

## 研究问题与动机
- **生成模型极低比特量化尚未突破**：现有 Transformer 量化工作主要集中于纯编码器架构（如 BERT）的分类任务，生成式 decoder 因自回归解码的高维输出空间与注意力机制对量化的敏感性，极难稳定收敛。
- **传统截断量化在生成任务中失效**：直接沿用分类网络的 TWN/BWN 简单阈值策略会导致梯度失配、量化后权重分布严重偏斜，引发训练发散或精度崩塌。
- **既往生成量化仅停留在 8-bit 激活**：QuantBART、DQ-BART 等近期工作虽将权重推至 2-bit，但激活仍维持 8-bit，未能探索真正乘法友好的 1/2-bit 端到端生成。
- **目标**：设计兼顾信息熵保持与梯度稳定的联合量化框架，首次实现三元/二元权重的生成 Transformer 并在下游任务取得非平凡性能。

## 核心贡献（创新点）
1. **首次实现生成 Transformer 的完全三元/二元量化**：提出 TBT 框架，在摘要与翻译任务上首次证明权重+激活均可压至 2-bit 或 1-bit 且保持可用精度。
2. **最大熵等距权重量化（Stats-based max-entropy isometric weight quantization）**：打破传统仅最小化 L2 距离的量化设计，显式约束量化权重在各离散级上的分布均匀性并保持实值-量化值幅度一致，显著缓解梯度失配。
3. **基于学习的弹性激活量化（Learning-based elastic activation quantization）**：针对 Softmax/ReLU 等非负激活与其他含正负激活层分别设计量化网格，并将缩放因子作为可学习参数端到端优化，自适应匹配动态激活分布。
4. **双模块协同收敛机制**：消融实验证实，单一改进均无法使模型收敛，统计先验与数据驱动学习的组合是稳定极端低比特生成的关键。

## 方法详解
- **整体框架（TBT）**：以 BART-base（140M）与 mBART-large（680M）为基线，采用全精度预训练权重初始化 + 知识蒸馏微调，训练 20 epochs，batch size 128，8 卡并行。
- **权重量化（统计驱动）**：
  - **三元**：$W_T^i = \alpha_T \lfloor \text{Clip}(\frac{W_R^i - \mu_T}{\alpha_T}, -1, 1) \rceil$，其中 $\mu_T$ 为权重均值，$\alpha_T = \frac{4}{3} \frac{\|W_R - \mu_T\|_{l1}}{n}$。该映射将实值权重均匀映射至 $[-1, 1]$ 再舍入至 $\{-1, 0, 1\}$，最后乘回 $\alpha_T$，实现幅度保持（isometric）。
  - **二元**：$W_B^i = \alpha_B \cdot \text{Sign}(\frac{W_R^i - \mu_B}{\alpha_B})$，先中心化再符号化，分母显式保留 $\alpha_B$ 以确保反向梯度可通过 STE 直传：$\frac{\partial W_B^i}{\partial W_R^i} \approx \mathbf{1}_{|\frac{W_R^i-\mu_B}{\alpha_B}|<1}$。
- **激活量化（学习驱动）**：
  - **分类量化网格**：非负激活（$X_R \in \mathbb{R}_+$，如 attention 输出、Softmax/ReLU）量化为 $\{0, \alpha\}$（二元）或 $\{0, \alpha, 2\alpha\}$（三元）；含正负激活（其余层）对称量化为 $\{-\alpha, \alpha\}$ 或 $\{-\alpha, 0, \alpha\}$。
  - **可学习缩放**：$\alpha_T / \alpha_B$ 作为可微参数，通过 STE 绕过舍
