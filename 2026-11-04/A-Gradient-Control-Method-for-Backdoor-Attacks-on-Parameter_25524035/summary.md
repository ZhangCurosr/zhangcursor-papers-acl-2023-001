---
title: "A-Gradient-Control-Method-for-Backdoor-Attacks-on-Parameter"
source: https://aclanthology.org/2023.acl-long.194.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:05:20"
field: "大模型安全与后门攻击"
keywords: ["Parameter-Efficient Tuning", "Backdoor Attack", "Gradient Control", "Multi-Task Learning", "Backdoor Forgetting"]
innovations: ["将PET后门注入建模为多任务学习过程并提出梯度幅度与方向联合控制方法", "提出CLNorm实现跨层梯度幅度归一化以降低输出层依赖", "提出ILProj实现层间差异化的梯度方向投影策略以缓解后门遗忘"]
benchmarks: ["SST-2", "IMDB", "Enron", "Lingspam"]
---

# 论文速读：A-Gradient-Control-Method-for-Backdoor-Attacks-on-Parameter

## 一句话总结
论文针对参数高效微调（PET）场景下的后门攻击遗忘问题，提出一种梯度控制方法（CLNorm + ILProj），通过跨层梯度幅度归一化和层内梯度方向投影来缓解后门遗忘，显著提升了下游重训练后的攻击有效性（LFR）。

## 研究问题与动机
- **后门遗忘问题**：PET只微调少量参数，用户后续用干净数据重训练时，植入的后门效果大幅下降甚至完全消失。
- **现有方法不足**：全参数微调场景的巩固方法（如RIPPLe、LWP）直接迁移到PET场景效果受限，因为PET模块学习能力较弱且可攻击参数少。
- **收敛不平衡**：后门注入过程中，干净数据和中毒数据的loss收敛速度和幅度差异明显，中毒数据收敛更快。
- **梯度冲突**：两类数据的梯度在方向和幅度上存在冲突，导致后门特征难以被稳定学习。

## 核心贡献（创新点）
1. **首次将PET后门注入视为多任务学习过程**，并系统分析梯度幅度和方向冲突，发现这两者是后门遗忘的关键原因。
2. **提出跨层梯度幅度归一化（CLNorm）**，通过学习映射函数将各层梯度幅度约束到期望的线性分布，降低输出层依赖、平衡各层贡献。
3. **提出层内梯度方向投影（ILProj）**，将PCGrad思想适配到PET场景，在深层投影冲突、在底层保留冲突以学习后门特征。
4. **系统性实验验证**：在情感分类和垃圾邮件检测两个领域、四种数据集迁移场景下，全面对比Vanilla/RIPPLe/LWP/GradNorm，方法在所有配置下取得最高LFR。

## 方法详解
- **CLNorm（跨层梯度幅度归一化）**：
  - 核心思想：让后门任务的梯度幅度沿层数呈线性递减，输出层梯度期望为0，底部层梯度最大。
  - 设期望关系为线性函数 $z: \tilde{g}_p^{(l)} = \frac{Avg[G_p]}{l_a - l_o}(l - l_o)$，其中 $l_o$ 为输出层。
  - 通过映射函数 $W$ 逐步更新权重 $w_l$：$w_l \leftarrow w_l - \alpha(w_l g_p^{(l)} - \tilde{g}_p^{(l)}) g_p^{(l)}$，初始 $w_l=1$。

- **ILProj（层内梯度方向投影）**：
  - 核心思想：借鉴PCGrad，对干净任务梯度 $g_c^{(l)}$ 和后门任务梯度 $g_p^{(l)}$ 进行投影以消除冲突方向。
  - 投影后梯度：$\hat{g}_c^{(l)} = g_c^{(l)} - \frac{g_c^{(l)} \cdot g_p^{(l)}}{\|g_p^{(l)}\|^2} g_p^{(l)}$，类似计算 $\hat{g}_p^{(l)}$。
  - 最终更新：$g^{(l)} = (1-\beta^{(l)})\hat{g}^{(l)} + \beta^{(l)} g^{(l)}$，其中 $\beta$ 在底层（0-5层）设为1（保留冲突以学习后门），在高层（6-11层）设为0（投影冲突以减轻遗忘）。

## 实验与结果
- **数据集**：SST-2、IMDB（情感分类，句子长度不同）；Enron、Lingspam（垃圾邮件检测，数据集规模不同）。
- **评估指标**：Clean Accuracy（CACC）和 Label Flip Rate（LFR）。
- **基线方法**：Vanilla、RIPPLe、LWP、GradNorm。
- **主要结果**：
  - 情感分类 SST-2→IMDB（Seq.）：本方法 LFR=73.7%，对比 Vanilla=15.3%、RIPPLe=62.8%、LWP=69.9%、GradNorm=68.6%。
  - 情感分类 IMDB→SST-2（Seq.）：本方法 LFR=99.4%，对比 Vanilla=9.8%、RIPPLe=84.7%、LWP=89.4%。
  - 垃圾邮件 Enron→Lingspam（Seq.）：本方法 LFR=90.9%，对比其他方法均在87.5%以下。
  - 垃圾邮件 Lingspam→Enron（Seq.）：本方法 LFR=51.1%，显著提升。
  - CACC在所有方法间无明显差异，说明攻击未损害模型正常性能。
- **最强结果**：IMDB→SST-2序列形式下LFR达到99.4%，相比Vanilla提升约90个百分点。

## 相关工作脉络
- **PET方法（Adapter/LoRA/Prefix/Prompt-Tuning）**：本文聚焦于这些轻量级模块的后门攻击，区别于全参数微调场景。
- **RIPPLe（Kurita et al., 2020）**：通过clean梯度抵消poison梯度来巩固后门，但在PET场景下因PET学习能力弱反而加剧遗忘。
- **LWP（Li et al., 2021）**：层wise权重中毒，假设期望关系为反比函数，未考虑输出层影响。
- **GradNorm（Chen et al., 2018）**：多任务学习中梯度归一化方法，本文借鉴其思想但针对PET场景重新设计。
- **PCGrad（Yu et al., 2020）**：梯度方向投影消除任务冲突，本文将其适配到PET后门场景并引入层间差异化的$\beta$策略。
- **定位差异**：本文关注PET模块（PLM冻结）的后门注入与遗忘问题，而非预训练或全参数微调阶段的后门攻击。

## 局限性与未来方向
- **不适用Prompt-tuning等仅在输入层添加模块的方法**，CLNorm无法使用，只能依赖ILProj。
- **在大尺寸数据集上重训练时仍会遭遇严重后门遗忘**。
- **未来方向**：研究针对PET后门攻击的防御方法（作者已在伦理声明中提及）。

## 研究启发与可借鉴点
- **多任务学习视角迁移**：将多任务梯度平衡方法引入后门攻击领域，为"攻击-防御"对抗研究提供新思路。
- **梯度控制策略的可复用性**：CLNorm和ILProj的设计可迁移到其他参数高效微调的安全性分析或防御工作中。
- **层间差异化设计**：根据层的位置（底层/高层）采用不同的处理策略（保留冲突/投影冲突），这种分层思想值得借鉴。
- **评估维度创新**：除LFR和CACC外，引入[CLS]向量相似度和poison分布分析，更全面地评估后门持久性。
- **潜在结合方向**：可将此方法与防御研究结合，探索PET模块的梯度监控机制来实现后门检测。

## 关键术语表
**PET（Parameter-Efficient Tuning）**：参数高效微调，只微调预训练模型的少量参数即可达到接近全参数微调的效果。
**后门攻击（Backdoor Attack）**：向训练数据注入触发器，使模型在遇到触发器时输出攻击者预设的标签。
**后门遗忘（Backdoor Forgetting）**：用户用干净数据重训练模型后，植入的后门效果减弱或消失的现象。
**CLNorm（Cross-Layer Gradient Magnitude Normalization）**：跨层梯度幅度归一化，控制不同层后门梯度的幅度分布。
**ILProj（Intra-Layer Gradient Direction Projection）**：层内梯度方向投影，消除同层内干净任务与后门任务的梯度方向冲突。
**LFR（Label Flip Rate）**：标签翻转率，衡量后门样本被预测为目标标签的比例，是评估后门有效性的核心指标。
**Sequential/Parallel Form**：PET模块的两种插入形式，分别为顺序插入（Adapter式）和并行插入（LoRA式）。
**[CLS] Similarity**：通过比较后门样本在重训练前后[CLS]向量的相似度，评估模型后门表征的稳定性。

## 可复现要素
- **数据集**：SST-2、IMDB、Enron、Lingspam（均为公开数据集）
- **代码/权重开源情况**：论文未提及
- **关键超参数**：
  - 学习率：2e-5
  - 批次大小：32
  - Poison训练epoch：10
  - 重训练epoch：5
  - CLNorm中α：1e-4
  - ILProj中β：底层（0-5层）=1，高层（6-11层）=0
- **PLM**：BERT（冻结）
- **触发器**：随机5个单词（"cf""mn""bb""tq""mb"）
