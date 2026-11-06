---
title: "Pivotal-Role-of-Language-Modeling-in-Recommender-Systems-Enr"
source: https://aclanthology.org/2023.acl-long.64.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:04:33"
field: "推荐系统"
keywords: ["语言建模", "推荐系统", "用户表征", "迁移学习", "联合训练", "task-agnostic 语料", "线性探针"]
innovations: ["联合语言建模与推荐任务目标提升用户表征质量", "利用跨域 task-agnostic 用户历史增强泛化与迁移能力", "简洁高效的冻结预训练+线性探针迁移框架用于大规模推荐部署"]
benchmarks: ["Amazon Scientific", "Amazon Pantry", "Online Retail"]
---

# 论文速读：Pivotal Role of Language Modeling in Recommender Systems: Enriching Task-specific and Task-agnostic Representation Learning

## 一句话总结
本文提出 **LMRec**，将语言建模（Language Modeling）目标与推荐任务目标联合训练于用户行为序列（以纯文本形式），并通过引入跨域无关（task-agnostic）用户历史进一步丰富表征，在多种推荐任务和跨域迁移场景中显著超越基线。

## 研究问题与动机
- **核心问题**：能否将语言建模直接应用于推荐系统的用户历史数据，从而提升表征质量并改善跨域泛化能力？
- **现有方法不足**：
  - 传统协同过滤（如 LightGCN、BST）依赖 ID 嵌入，难以泛化到新领域、新用户，且对文本语义利用不足。
  - 已有统一用户建模工作（如 UserBERT、UniSRec、M6-Rec）多采用"预训练-微调"范式，且大多依赖通用 webtext 语料，未系统探究任务特定/无关语料的语言建模联合优化效果。
  - CLUE（Shin et al., 2023）虽从 scratch 训练编码器，但缺少语言建模目标的联合训练机制，对跨域迁移的灵活性有限。

## 核心贡献（创新点）
1. **联合训练语言建模与推荐任务的统一框架（LMRec）**：Transformer 共享底层参数，分别输出语言建模损失 $L_1$ 和推荐损失 $L_2$，相比"先预训练再微调"策略（Table 2）带来显著提升（OBS Recall@10: 0.4742 → 0.4867）。
2. **引入 task-agnostic 数据扩展表征泛化性（LMRec+agnostic）**：在语言建模阶段额外利用来自其他服务域的跨域用户历史，缓解过拟合，在 6 个数据集上全面超越所有基线（如 ECOMM Recall@10 最高达 0.7715，相对最佳基线 M6Rec 提升 +8.8%）。
3. **简洁高效的线性探针迁移学习框架（LMRec^TL）**：冻结预训练 Transformer，仅训练下游任务线性层，实现快速部署；在在线 A/B 测试中 CTR 相对 GNN 提升 ×1.76（新用户）。
4. **系统性实证分析**：涵盖语料配比消融、模型规模 Scaling Law、Hessian 特征值对泛化的影响等多个维度，为推荐系统中语言建模的价值提供了全面证据。

## 方法详解
**模型架构**：基于 Transformer Decoder（层数、维度可调，默认 125M 参数），输入为用户历史行为描述拼接而成的 token 序列，末尾追加 `[EOS]` token，取其最后一层隐藏向量作为用户表征 $z_u$； item 表征 $z_i$ 同理取 item 文本序列末尾 `[EOS]` 对应向量。

**联合损失函数**：
- 语言建模损失（自回归负对数似然）：
$$L_1 = -\sum_{j=1}^{n} \log P(h_j | h_{j-k}, \ldots, h_{j-1}; \mathcal{M})$$
- 推荐任务损失（二分类交叉熵）：
$$p_{u,i} = \frac{1}{1 + \exp(-\langle W_u z_u, W_i z_i \rangle)}, \quad L_2 = -y \log p_{u,i} - (1-y) \log(1-p_{u,i})$$
- 联合目标：
$$L = L_1 + \lambda L_2$$
其中 $\lambda$ 为平衡系数，随训练 cosine decay 至初始值的 10%。

**迁移学习设置**：预训练时联合多个源任务（$\mathcal{T}_s$）及 task-agnostic 语料（$\mathcal{T}_a$），下游任务仅训练独立线性层，Transformer 参数冻结。task-agnostic 特征可直接作为用户额外输入特征使用（"Combine" 设置）。

**训练细节**：AdamW（$\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-6}$），混合精度训练，梯度裁剪（max norm=0.1），早期停止（patience=100），词表大小 50,258，最大序列长度 2,048。

## 实验与结果
**数据集**：
- 内部数据集：OBS（在线预订）、OTA（在线旅游）、ECOMM（电商平台）
- 公开数据集：Amazon Scientific、Amazon Pantry、Online Retail
- Task-agnostic 预训练语料：Search、E-comm、SNS、News（共 ~1,222 tokens/用户，总计 12B BBPE tokens）

**主要结果**（Table 4，Recall@10 / NDCG@10）：
- **LMRec（仅任务特定数据）**：在 6 个下游任务上均为 SOTA，内部数据集较最佳基线提升 1.6–3.2%，公开数据集平均提升约 5%。
- **LMRec+agnostic**：全面超越所有基线，最大提升如 ECOMM Recall@10 = 0.7715（vs. M6Rec 0.7093，+8.8%），NDCG@10 = 0.6009（+15.5%）。
- **线性探针迁移（Table 7）**：LMRec+agnostic^TL（Combine 设置）在 OBS Recall@10 达 **0.5952**，较 CLUE（0.5540）提升明显。
- **在线 A/B 实验（Table 8）**：新用户 CTR 提升 ×1.76，GMV 提升 ×1.12（vs. GNN 基线）。
- **语料配比消融（Table 5）**：70% 特定 + 30% 无关时效果最佳（OBS Recall@10 = 0.4917）；纯无关数据（0% 特定）效果最差。
- **模型规模（Figure 4）**：性能随参数规模单调提升，但存在上限，符合"幂律+常数"规律。

## 相关工作脉络
- **UserBERT / UniSRec / M6-Rec**：采用"预训练-微调"范式，预训练多基于 webtext，本文与之本质区别在于**联合训练语言建模与推荐目标**，不依赖外部 PLM，且利用任务无关语料增强泛化。
- **CLUE（Shin et al., 2023）**：唯一从 scratch 训练的无 PLM 方法，但缺乏语言建模目标，本文通过联合 LM 目标进一步提升了迁移能力和表示鲁棒性。
- **Gururangan et al. (2020)**：证明在任务特定数据上继续预训练可提升下游性能；本文将其思想**首次系统引入推荐系统**并验证效果。
- **Krishna et al. (2022)**：发现 task-agnostic 语料可比肩 webtext；本文进一步证明其在推荐场景中结合联合训练可实现全面 SOTA。
- **BST / LightGCN**：传统推荐基线，本文在无额外跨域数据的情况下即超越它们，凸显语言建模的有效性与效率。

## 局限性与未来方向
- **领域特异性信息未充分利用**：对不同服务使用统一 item 描述方式（仅标题），未针对各领域（如新闻主题、来源媒体等）定制特征，存在提升空间。
- **语料配比与组合策略探索不充分**：受限于计算成本，task-agnostic 各源语料的最优混合比例尚未系统搜索。
- **内部数据因隐私和法律原因无法开源**，限制了方法的可复现性验证范围（尽管使用了公开 benchmark 辅助验证）。
- 预训练语料特性与下游性能之间的定量关系仍有待深入研究。

## 研究启发与可借鉴点
- **联合训练双目标是高效且有效的**：在推荐系统中同时优化语言建模与推荐目标，可同时获得丰富的语义表征和任务适配能力，避免灾难性遗忘，值得在其他推荐场景（如序列推荐、冷启动）中验证。
- **Task-agnostic 用户特征作为额外输入是低成本的增益手段**：在迁移场景中直接使用其他域的用户历史提取特征，可显著改善新用户/稀疏场景效果，工程实现简单。
- **线性探针迁移范式对生产部署极具价值**：冻结大模型、仅训练下游小参数线性层，推理速度提升 ×157（Table 6），适合大规模在线服务快速适配新场景。
- **Hessian 谱分析可作为表征质量的辅助诊断工具**：本文利用 Hessian 最大特征值验证联合训练提升了损失曲面的平滑性，为表征鲁棒性提供了理论支撑视角。

## 关键术语表
- **LMRec**：本文提出的联合语言建模与推荐任务训练的推荐模型框架。
- **Task-specific 数据**：与目标推荐任务直接相关的用户行为历史数据。
- **Task-agnostic 数据**：来自与目标任务无重叠的其他服务域的用户行为历史，用于增强表征泛化性。
- **Linear Probe（线性探针）**：冻结预训练模型参数，仅在下游任务上训练线性层的迁移学习策略。
- **BMPE（Byte-level BPE）**：字节级子词分词方法，用于将行为日志中的 item 文本 tokenize。
- **Scaling Law（缩放定律）**：模型性能随参数规模和数据量增大而遵循幂律提升的现象。

## 可复现要素
- **数据集**：内部数据集（OBS/OTA/ECOMM）因隐私问题**未公开**；公开数据集（Amazon Scientific/Pantry、Online Retail）可用；task-agnostic 预训练语料规模巨大（12B tokens），未公开原文。
- **代码/权重**：论文未提及开源代码与预训练权重。
- **关键超参**：AdamW ($\beta_1=0.9, \beta_2=0.98$)，max norm 梯度裁剪=0.1，词表 50,258，最大序列长度 2,048，$\lambda$ cosine decay 至初始值 10%，batch size=1024（默认 125M 模型），结果取 20 次运行均值。详见 Appendix A 和 Table 9。
