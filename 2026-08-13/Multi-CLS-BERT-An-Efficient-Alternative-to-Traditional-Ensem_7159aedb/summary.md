---
title: "Multi-CLS-BERT-An-Efficient-Alternative-to-Traditional-Ensem"
source: https://aclanthology.org/2023.acl-long.48.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:40:03"
field: "预训练语言模型与高效集成"
keywords: ["Multi-CLS BERT", "模型集成", "预训练", "多样性", "GLUE", "SuperGLUE", "对比学习", "参数高效"]
innovations: ["提出 Multi-CLS BERT，在单模型中用多个 CLS 嵌入实现近似集成效果，推理开销接近单模型", "设计 MCQT 损失，融合最相似 facet 和求和 facet 双重视角的对比预训练", "提出架构内 per-CLS 线性层 + 微调重参数化聚合，从根本上防止多 CLS 表示塌陷"]
benchmarks: ["GLUE", "SuperGLUE"]
---

# 论文速读：Multi-CLS BERT: An Efficient Alternative to Traditional Ensembling

## 一句话总结
本文提出 **Multi-CLS BERT**，通过在单一 BERT 模型中引入多个 CLS 嵌入并设计多样性保持机制，实现与传统多模型集成近似的效果，同时几乎不增加推理计算和内存开销；在 GLUE/SuperGLUE 上的实验表明，该方法在小样本场景下尤其有效，100 样本时甚至能超越更大的 BERT Large 基线。

## 研究问题与动机
- **核心问题**：传统的 BERT 集成方法（多次 fine-tune、不同随机种子取平均）能显著提升精度和置信度估计，但需要 5 倍以上的推理计算和内存开销，代价高昂。
- **现有方法不足 1**：直接对同一 BERT 模型输入多个 [CLS] token 会导致各 CLS 嵌入在预训练过程中快速塌陷为几乎相同的表示。
- **现有方法不足 2**：现有的高效集成替代方案（如 Dropout 集成、SWA）无法产生足够多样化的预测多样性，对 BERT 效果有限。
- **现有方法不足 3**：Vision 领域已有"近乎免费集成"思路（如 BatchEnsemble、subnetwork partitioning），但能否迁移到语言模型预训练+fine-tune 框架尚不明确。

## 核心贡献（创新点）
- **多 CLS 嵌入的轻量级集成架构**：在单个 BERT encoder 中并行维护 K 个 CLS 嵌入，通过共享上下文隐藏状态仅对不同聚合方式做集成，相比传统集成减少约 4 倍计算/内存开销；与传统方法本质区别在于以单模型多视角替代多模型独立参数。
- **多 CLS Quick Thoughts（MCQT）损失**：将原始 QT 损失扩展为融合"最相似 facet 余弦相似度"和"全部 facet 求和后余弦相似度"两项的形式，显式鼓励不同 CLS 捕捉文本不同侧面；与原有单 CLS QT 的本质区别是从"整段压缩到单一向量"变为"保留多面信息"。
- **硬负样本（Hard Negative）预训练策略**：利用"下一段之后的段落"作为与正样本同主题但语义 facet 不同的难负例，迫使多个 CLS 区分细粒度语义差异；与常规随机负例的本质区别是引入了语义相近但角度不同的困难样本。
- **架构级多样性保障 + 微调重参数化技巧**：在 Transformer encoder 内部插入 per-CLS 线性层（类似 adapter 结构但目的不同），并在 fine-tuning 阶段提出 $L_{O,k}^{FT}(h_k^c) = (W_{O,k} - \bar{W})h_k^c$ 的重参数化聚合，从梯度层面杜绝所有 CLS 路径坍缩至相同权重；与简单加法聚合的本质区别是保证了即使各 $W_{O,k}$ 趋同，梯度也不允许恒为零输出。

## 方法详解

### 预训练损失（四任务组合）
在 Aroca-Ouellette & Rudzicz (2020) 的 MTL 基础上改进，包含四项：
1. **MLM**：标准掩码语言建模，保持不变。
2. **TFIDF**：预测词重要性，保持不变。
3. **SO（Sentence Ordering）**：将 K 个 CLS 隐藏状态拼接后经线性层 $L^{SO}$ 预测句子顺序是否被交换。
4. **MCQT（Multi-CLS Quick Thoughts）**：核心改进。输入序列 $(s^{1-2}, s^{3-4})$，对每对序列计算 logit：
   $$\text{Logit}_{s^{1-2},s^{3-4}}^{MC} = \lambda \max_{i,j} \left(\frac{c_i^{1-2}}{\|c_i^{1-2}\|}\right)^T \frac{c_j^{3-4}}{\|c_j^{3-4}\|} + (1-\lambda)\left(\frac{\sum_i c_i^{1-2}}{\|\sum_i c_i^{1-2}\|}\right)^T \frac{\sum_j c_j^{3-4}}{\|\sum_j c_j^{3-4}\|}$$
   其中第一项取"最相似 facet 对"的余弦相似度，鼓励多样性；第二项取全部 facet 求和后再算余弦相似度，鼓励协作（类似集成投票）。$\lambda$ 为超参，实验最佳值为 0.1。

### 硬负样本构造
将 batch 拆成三部分：
- 第一部分：$B/3$ 个随机文本序列。
- 第二部分：每序列的**紧邻后续**序列 → 作为正样本。
- 第三部分：每序列的**下一序列之后的序列** → 作为硬负样本（同主题但可能不同 facet）。

修改后的 MCQT 损失（Eq.3）同时对 $(s^{1-2}, s^{3-4})$ 对和 $(s^{5-6}, s^{3-4})$ 对分别计算对比目标。

### 架构多样性（Section 2.5）
- 初始方案（仅输入多个特殊 token [C1]…[CK]）：各 CLS 嵌入在预训练早期即塌陷为几乎相同表示。
- 改进方案（Figure 3）：对每个 CLS $k$ 在最终 head $H_k^{MC}$ 中使用不同的线性层 $L_{O,k}$（偏置设为 0，使差异动态且依赖上下文）。
- 进一步改进：在 Transformer encoder 第 4 层和第 8 层后各插入一组 per-CLS 线性层 $L_{l,k}$（BERT Base），或 BERT Large 的第 8、16 层后，改变不同 CLS 路径的中间表示。

### Fine-tuning 重参数化聚合（Section 2.6, Eq.4）
直接将各 $L_{O,k}(h_k^c)$ 求和仍会导致各路径权重趋同。提出：
$$c^{MCFT} = \sum_k \left(L_{O,k}^{FT}(h_k^c)\right), \quad L_{O,k}^{FT}(h_k^c) = \left(W_{O,k} - \frac{1}{K}\sum_{k'} W_{O,k'}\right)h_k^c$$
该重参数化保证：若所有 $W_{O,k}$ 相同则输出恒为零，而梯度下降不允许模型恒输出零向量，从而在 fine-tune 阶段强制维持各 $L_{O,k}^{FT}$ 的差异性。

## 实验与结果

### 数据集与设置
- **数据集**：GLUE 和 SuperGLUE 基准；分别测试 Full / 1k / 100 样本三种数据规模。
- **预训练语料**：Wikipedia 2021 + BookCorpus，BERT Base 用 20 亿 token，BERT Large 用 10 亿 token。
- **评估协议**：4 个预训练随机种子 × 4 个 fine-tune 随机种子 = 16 次实验均值；fine-tune 使用更稳定的优化策略（长训练、梯度裁剪、Adam with bias、warmup）。
- **复现前作**：改进了 Aroca-Ouellette & Rudzicz (2020) 的 fine-tune 协议，使 MTL 基线从原报告的 81.4 提升至 83.30（GLUE Full, BERT Base）。

### 主要结果（Table 1，开发集 macro avg）

| 配置 | 参数量 | GLUE 100 | GLUE 1k | GLUE Full | SuperGLUE 100* | SuperGLUE 1k* | SuperGLUE Full |
|---|---|---|---|---|---|---|---|
| BERT Base Pretrained | 109.5M | 55.71 | 71.67 | 82.05 | 57.18 | 61.55 | 65.04 |
| BERT Base MTL | 109.5M | 59.29 | 73.26 | 83.30 | 57.50 | 62.94 | 66.33 |
| **Ours (K=5, λ=0.1) Base** | **118.4M** | **61.80** | **74.10** | **83.47** | **58.20** | **63.61** | **66.74** |
| BERT Large MTL | 335.2M | 61.39 | 75.30 | 84.13 | 59.03 | 65.21 | 69.16 |
| **Ours (K=5, λ=0.1) Large** | **350.9M** | **64.24** | **76.27** | **84.61** | **59.88** | **65.58** | **70.03** |

- **最强亮点**：GLUE 100 下，Ours (K=5, λ=0.1) $\mathrm{BERT_{Base}}$（118.4M 参数，得分 61.80）**超越**了 $\mathrm{BERT_{Large}}$ MTL（335.2M 参数，得分 61.39）。
- **GLUE Full 最佳**：Ours (K=5, λ=0.1) $\mathrm{BERT_{Large}}$ 达 84.61，较 MTL Large 提升 +0.48。
- **SuperGLUE Full 最佳**：Ours (K=5, λ=0.1) $\mathrm{BERT_{Large}}$ 达 70.03，较 MTL Large 提升 +0.87。

### 消融实验关键发现（Table 2）
- **No Inserted Layers**：性能降至与 Ours (K=1) 相近，验证内部插入线性层对多样性至关重要。
- **Sum Aggregation（不用重参数化）**：同上，验证重参数化技巧不可或缺。
- **No Hard Negative**：GLUE 分数下降，SuperGLUE 略有上升，综合看硬负样本对 GLUE 类任务更有效。
- **K 值选择**：K=5 优于 K=1/3/10，说明 5 个视角在当前设定下最优。
- **λ 选择**：λ=0.1 整体表现最佳，验证"最相似 facet + 求和 facet"双项 logit 的合理性。
- **对比其他高效集成**：SWA、Ensemble on Dropouts 均未能获得好结果，说明梯度轨迹/ dropout map 本身产生的多样性对 BERT 不够。

### 效率与校准分析（Table 3, 4）
- **推理速度**：Ours (K=5) 推理时间 0.3119s，接近 Ours (K=1) 的 0.2918s；而 5 倍集成需 1.4590s——**节省约 4× 计算**。
- **ECE（期望校准误差，越低越好）**：Ours (K=5) 在 GLUE 100 上 ECE = 15.46，远低于 K=1 的 25.22，接近 5 倍集成 ECE = 13.85。
- **不确定性重叠（Table 4）**：Multi-CLS 与 BERT 集成在"top 20% 最不确定样本"上的重叠率分别为 32.57%（GLUE 100）和 41.35%（GLUE 1k），与其他不确定性估计方法相比并无显著差异，说明不同 CLS 的"分歧"模式与多模型集成的分歧模式一致。

## 相关工作脉络
- **Aroca-Ouellette & Rudzicz (2020) MTL**：本文的四损失预训练基线（MLM+QT+SO+TFIDF），本文在其基础上扩展为多 CLS 版本并改进 fine-tune 协议。
- **Allen-Zhu & Li (2020)**：揭示集成有效的关键在于模型多样性，不同随机种子的模型比 dropout/权重平均更多样——本文以此理论为依据设计多样性机制。
- **BatchEnsemble (Wen et al., 2020)**：Vision 领域共享权重的高效集成，本文思路类似但面向 BERT 且采用完全不同的多 CLS 架构而非 weight scaling。
- **Mixture of Softmax (MoS, Yang et al., 2018; Narang et al., 2021; Tay et al., 2022)**：同样使用多嵌入改进 BERT，但需要计算 hidden state × 全词表 embedding 矩阵乘法，训练成本显著高于本文的线性层方案。
- **Chang et al. (2021)**：多嵌入句表示用于无监督句子相似度，其 non-negative sparse coding loss 与本文 Eq.2 类似，但本文损失更轻量且面向下游监督任务。
- **Contextualized word embeddings for IR (Khattab & Zaharia, 2020; Luan et al., 2021; COLBERT 等)**：用多向量表示长文本，目标与应用场景（IR vs NLU classification）和任务设定不同。

## 局限性与未来方向
- 仅在 BERT 架构上验证，未扩展到 RoBERTa（需更大 GPU/CPU 资源）及更大模型。
- 未测试 XLNet 等其他架构，也未在 prompt/adapter 等替代 fine-tune 方法上验证。
- GLUE/SuperGLUE 均为英文基准，可能存在数据集选择偏差（benchmark lottery）。
- 即使高效，在更多训练数据（GLUE 1k/Full）下仍略逊于传统多模型集成（ECE 和精度均有差距）。
- 未在主动学习等需要高质量不确定性估计的实际应用中验证。

## 研究启发与可借鉴点
- **多视角嵌入的多样性保障范式**：架构内插入 per-head 线性层 + 重参数化求和的技术路线可迁移至其他单模型多输出任务（如多标签分类、多实体关系抽取），作为"类集成"的廉价替代。
- **硬负样本对比学习构造策略**：利用"同主题但非紧邻"序列作为 hard negative，可推广至其他序列级对比预训练任务（如文档级表示学习、检索增强预训练）。
- **细粒度 diversity 评估指标**（Appendix D 的 Corr 度量）：用 CLS-邻域点积相关系数衡量嵌入多样性，可作为其他多表示模型的内部监控指标，避免塌陷。
- **小样本场景优先验证**：本文凸显集成收益在数据稀缺时最大，提示后续研究可在 low-resource NLU、跨语言少样本设定中优先部署 Multi-CLS 思路。
- **与参数高效微调（PEFT）的结合潜力**：本文强调其插入的线性层目的是多样性而非参数冻结，但可与 LoRA/adapter 等技术结合探索更高效的多视角微调。

## 关键术语表
- **Multi-CLS BERT**：在单一 BERT 编码器中输入 K 个特殊 CLS token，并行得到 K 个 CLS 隐藏状态，通过多样性机制使其捕捉输入文本的不同语义 facet，最终聚合为单一向量用于下游分类。
- **MCQT（Multi-CLS Quick Thoughts）**：本文提出的预训练对比损失，融合"最相似 facet 对余弦相似度"与"全部 facet 求和余弦相似度"两项，鼓励多 CLS 既多样化又协作。
- **Re-parameterization trick（重参数化技巧）**：Fine-tuning 时将各 $L_{O,k}$ 替换为其去均值形式 $(W_{O,k} - \bar{W})$，在梯度上阻止所有 CLS 路径坍缩为相同权重。
- **Expected Calibration Error（ECE）**：衡量模型预测置信度与真实准确率之间偏差的指标，越低表示校准越好；本文用其评估不确定性质量。
- **Hard Negative**：与正样本共享主题但细粒度语义 facet 不同的负样本；本文在 QT 预训练中用于迫使不同 CLS 区分细微差异。
- **Diversity Collapse（多样性塌陷）**：多 CLS 嵌入在训练中退化为几乎相同表示的现象；本文通过插入 per-CLS 线性层和重参数化技巧缓解该问题。

## 可复现要素
- **数据集**：GLUE、SuperGLUE（公开基准）；预训练语料为 Wikipedia 2021 + BookCorpus（公开）。
- **代码/权重**：论文基于 Aroca-Ouellette & Rudzicz (2020) 代码修改实现，使用 `unused0`–`unused(K-1)` 作为多 CLS token；作者隶属 Amazon 但论文中未明确声明代码开源仓库链接（论文未提及具体 GitHub 地址）。
- **关键超参**：$K=5$、$\lambda=0.1$；预训练学习率 $2\times10^{-5}$、warmup ratio 0.001；fine-tune 学习率搜索范围 $c\times10^{-5}, c\in\{1,2,3,4,5,7\}$，最大梯度范数 1，最大句长 GLUE 128 / SuperGLUE 256，batch size 按数据规模取 4/8/16，早停策略为连续 10k 步无验证集提升则停止。
