---
title: "Improving-Continual-Relation-Extraction-by-Distinguishing-An"
source: https://aclanthology.org/2023.acl-long.65.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:02:01"
---

# 论文速读：Improving-Continual-Relation-Extraction-by-Distinguishing-An

## 一句话总结
本文针对持续关系抽取（CRE）中记忆回放易过拟合、且难以区分语义高度相似（analogous）关系的痛点，提出融合记忆不敏感原型、记忆增强、集成训练与焦点知识蒸馏的端到端模型，在 FewRel 和 TACRED 上达到 SOTA，显著提升了对类比关系的区分能力并有效缓解灾难性遗忘。

## 研究问题与动机
1. **核心问题**：CRE 要求模型在关系不断涌现时既学新知又保旧识，但现有基于记忆的重放方法在反复回放少量典型样本时极易过拟合，且对原型相似度高的类比关系区分能力薄弱，导致严重的灾难性遗忘。
2. **实证发现**：将关系按原型最大余弦相似度分组后发现，相似度越高（≥0.85）的关系组在持续学习后准确率越低、下降幅度越大（FewRel 高相似度组 Accuracy 仅 71.1%，Drop 达 9.7%）。
3. **现有方法不足**：主流工作多聚焦于如何更有效地重放样本，未显式建模过拟合问题；部分工作虽关注相似关系遗忘，但未从表征空间与蒸馏加权两个维度协同解决。
4. **动机**：需同时缓解“记忆样本过拟合”与“类比关系混淆”两大问题，设计更鲁棒的特征空间与知识保持机制。

## 核心贡献（创新点）
1. **显式建模 CRE 过拟合并提出记忆增强策略**。与以往仅依赖固定大小记忆库直接回放的做法不同，本文通过实体替换与无关句子拼接将回放样本数扩大 4 倍，从根本上打破对少数典型样本的过拟合依赖。
2. **设计记忆不敏感关系原型（Memory-Insensitive Relation Prototype）**。与直接平均存储样本或使用外部知识 prompt 生成原型的做法不同，本文融合首次学习时的全量静态均值与动态记忆均值，在不增加外部开销的前提下降低原型对存储样本的敏感性。
3. **提出集成训练（Integrated Training）与焦点知识蒸馏（Focal Knowledge Distillation）**。与单一采用对比学习或线性分类器的方法不同，本文联合优化对比特征空间对齐与线性决策边界，并结合仿 Focal Loss 的权重机制（融合原型相似度与预测置信度）重点强化难样本与类比关系的知识保持。
4. **通过分组实证揭示类比关系遗忘机制并提供量化评估**。与以往将遗忘简单归因于参数漂移的研究不同，本文定量验证高相似度关系出现是性能骤降的关键诱因，并在类比关系子集上系统验证了方法的针对性提升。

## 方法详解
- **整体框架**：对新任务 $T_k$ 先用 $D_k^{train}$ 微调 BERT 编码器；随后用 k-means 为每个历史关系选取 $m$ 个典型样本存入 $\tilde{M}_k$；计算记忆不敏感原型；执行记忆增强生成 $\hat{M}_k$（$|\hat{M}_k|=4|\tilde{M}_k|$）；最后进行记忆回放（集成训练+焦点知识蒸馏）更新参数。
- **表征提取**：BERT 编码器输出后，通过特殊标记 `[E11]/[E12]` 和 `[E21]/[E22]` 定位头尾实体，拼接隐藏状态经 LayerNorm 得到 $\mathbf{h}_x$。
- **记忆不敏感原型**：$\mathbf{p}_r = (1-\beta)\mathbf{p}_r^{\mathrm{static}} + \frac{\beta}{|M^r|}\sum_{x_i \in M^r} \mathbf{h}_{x_i}$，其中 $\mathbf{p}_r^{\mathrm{static}}$ 为该关系首次学习时全量训练样本的平均表示（静态锚点），动态项随新任务微调适应特征空间漂移，$\beta$ 控制融合比例。
- **记忆增强（MA）**：① 实体替换：从同一关系记忆库中随机取另一样本，替换当前样本的头尾实体生成新样本；② 句子拼接：将非该关系的随机样本拼接到末尾，迫使模型忽略无关上下文仍正确分类。增强样本仅用于回放训练，不参与原型计算。
- **集成训练**：$\mathcal{L}_{\mathrm{cls}} = \mathcal{L}_{\mathrm{c\_cls}} + \mathcal{L}_{\mathrm{l\_cls}}$。对比分支通过两層 MLP 降维后使用 InfoNCE + Triplet Loss 优化原型空间 $\mathbf{z}_x$；线性分支使用交叉熵直接分类，两者结合兼顾特征对齐与任务特定决策边界。
- **焦点知识蒸馏**：模仿 Focal Loss 构造样本-关系权重 $w_{x_i,r_j} = s_{x_i,r_j}(1-P(y_i|x_i;\theta_k))^\gamma$，其中 $s$ 为样本与历史原型的相似度 softmax，$\gamma$ 聚焦难样本。蒸馏损失 $\mathcal{L}_{\mathrm{fkd}} = -\frac{1}{|\hat{M}_k|}\sum w_{x_i,r_j}\log P(r_j|x_i;\theta_k)$，回放总损失为 $\mathcal{L}_{\mathrm{replay}} = \mathcal{L}_{\mathrm{cls}} + \lambda_1\mathcal{L}_{\mathrm{c\_fkd}} + \lambda_2\mathcal{L}_{\mathrm{l\_fkd}}$。
- **预测阶段**：测试时线性融合对比概率 $P_c$ 与线性概率 $P_l$，$y^* = \arg\max((1-\alpha)P_c + \alpha P_l)$。

## 实验与结果
- **数据集**：FewRel（80个关系×700样本，10个任务）；TACRED（42个关系，10个任务，类别不平衡）。
- **基线**：EA-EMR, EMAR(BERT), CML, RP-CRE, CRL, CRECL, KIP-Framework。默认记忆大小 $m=10$，5次随机任务序列平均。
- **主要结果**：FewRel $T_{10}$ 准确率 84.2%±0.4（SOTA，超越次优 CRL 的 83.1%）；TACRED $T_{10}$ 准确率 79.1%±1.1（SOTA，超越次优 KIP-Framework 的 78.6%）。
- **类比关系表现**：原型相似度>0.85 的关系上，本文 Accuracy
