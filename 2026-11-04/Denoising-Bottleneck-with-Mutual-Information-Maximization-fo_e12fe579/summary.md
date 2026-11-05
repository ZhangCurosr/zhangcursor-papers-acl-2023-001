---
title: "Denoising-Bottleneck-with-Mutual-Information-Maximization-fo"
source: https://aclanthology.org/2023.acl-long.124.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:09:37"
field: "多模态学习"
keywords: ["视频多模态融合", "去噪瓶颈", "互信息最大化", "多模态情感分析", "多模态摘要", "InfoNCE"]
innovations: ["提出融合瓶颈模块通过受限感受野细粒度过滤视频噪声与冗余", "设计互信息最大化模块防止关键信息被过度过滤，两者互补协同", "在MOSI、MOSEI和How2三个基准上均取得SOTA性能"]
benchmarks: ["MOSI", "MOSEI", "How2"]
---

# 论文速读：Denoising-Bottleneck-with-Mutual-Information-Maximization-fo

## 一句话总结
本文提出了去噪融合瓶颈（DBF）模型，通过受限感受野的融合瓶颈模块过滤视频中的噪声与冗余信息，并结合互信息最大化模块确保关键信息的保留，在多模态情感分析与多模态摘要两个任务上均取得最新最优（SOTA）性能。

## 研究问题与动机
- 视频多模态融合中存在大量冗余与噪声：连续帧高度相似导致冗余、 distracting background 引入噪声、视听文本间弱对齐带来 misalignment 噪声。
- 现有去噪方法（如 Liu et al. 2020 的 fusion forget gate）以整模态粒度粗粒度过滤，容易将代表性信息一并丢弃。
- 多模态融合质量直接影响下游任务，视频比静态图像更长、序列更冗余，需要更细粒度的去噪机制。
- 需要在去除噪声的同时保证各模态关键信息不丢失，以提升融合表征的有效性。

## 核心贡献（创新点）
1. **提出融合瓶颈模块**：通过短序列瓶颈嵌入 $B$ 限制跨模态信息流动，迫使模型在共享前压缩去噪，与粗糙的 forget gate 相比粒度更细。
2. **设计互信息最大化（MI-Max）模块**：基于 InfoNCE 损失最大化融合结果与各单模态输入的互信息下界，反向约束瓶颈不过度过滤关键信息。
3. **三数据集 SOTA**：在 MOSI、MOSEI（情感分析）和 How2（摘要）上均达到或超越先前最优，摘要任务提升尤为显著。
4. **综合分析验证**：提供消融实验与 Grad-CAM 可视化，证明瓶颈与 MI-Max 互补、模型能聚焦关键帧而忽略冗余帧。

## 方法详解
- **整体架构**：使用 N 层 Transformer 分别编码文本/视觉/音频，再通过 M 层带融合瓶颈的 Transformer 进行多模态融合，输出文本特征 $X_t^L$ 作为融合结果 $\bar{Z}$。
- **融合瓶颈模块**：引入随机初始化的瓶颈嵌入 $B \in \mathbb{R}^{l_b \times d_m}$，其中 $l_b \ll l$。各模态特征不能直接相互 attention，必须通过 $B$ 交换信息：
  - 每层 Transformer 处理后，$B_m^{l+1}$ 由各模态瓶颈更新汇总得到 $B^{l+1} = \text{Mean}(B_m^{l+1})$。
  - 受限感受野迫使模型只保留最具判别性的信息通过瓶颈，冗余被自然丢弃。
- **互信息最大化（MI-Max）**：
  - 使用 MLP $\mathcal{F}$ 从融合结果 $Z$ 反推各单模态 $X_m$。
  - 归一化相似度函数：$\sin(X_m, Z) = \exp\left(\frac{X_m}{\|X_m\|^2} \odot \frac{\mathcal{F}(Z)}{\|\mathcal{F}(Z)\|^2}\right)$。
  - 在 batch 内将其他样本作为负样本，构造 InfoNCE 损失：
    $$\mathcal{L}_{\text{NCE}}^{z,m} = -\mathbb{E}_{X_m,Z}\left[\log\frac{e^{\sin(x_m^+, \mathcal{F}(Z))}}{\sum_{k=1}^K e^{\sin(x_m^k, \mathcal{F}(Z))}}\right]$$
  - 总辅助损失：$\mathcal{L}_{\text{NCE}} = \alpha(\mathcal{L}_{\text{NCE}}^{z,v} + \mathcal{L}_{\text{NCE}}^{z,a} + \mathcal{L}_{\text{NCE}}^{z,t})$。
  - 最终训练目标为下游任务损失加上 $\mathcal{L}_{\text{NCE}}$，$\alpha$ 控制权衡。
- **下游任务**：情感分析使用回归（MAE/Corr）和分类（Acc-7/Acc-2/F1）；摘要生成使用 BART decoder 评估 ROUGE/BLEU/METEOR/CIDEr。

## 实验与结果
- **数据集**：MOSI（2198 utterances，评分 [-3,3]）、MOSEI（23453 utterances）、How2（79114  instructional videos）。
- **基线模型**：MulT、TFN、LMF、MFM、ICCN、MISA、Self-MM、MMIM（情感分析）；HA、MFFG、VG-GPLMs（摘要）。
- **MOSI**：DBF 取得 MAE 0.693（↓）、Corr 0.801（↑）、Acc-2 85.1/86.9（↑）、F1 85.1/86.9（↑），超越 MMIM（MAE 0.700, Corr 0.800）。
- **MOSEI**：DBF 取得 MAE 0.523（↓）、Corr 0.772（↑）、Acc-2 84.3/86.4（↑）、F1 84.8/86.2（↑），超越 MMIM（MAE 0.526, F1 82.7/85.9）。
- **How2 摘要**：DBF 取得 ROUGE-1 70.1、ROUGE-2 54.7、ROUGE-L 66.0、CIDEr 3.56，显著超越 VG-GPLMs（R-1 68.0, CIDEr 3.28）。
- **消融结果**：移除 MI-Max 后 MOSI MAE 从 0.693 升至 0.697；移除 bottleneck 后升至 0.750；移除文本模态后性能暴跌（MAE 1.391），验证文本为中心融合的有效性。
- **可视化**：DBF Grad-CAM 标准差 0.830（基线 0.404）、归一化熵 0.008（基线 0.062），表明注意力更尖锐、能聚焦关键帧。

## 相关工作脉络
- **Nagrani et al. (2021) Attention Bottlenecks**：提出视觉-语言预训练的注意力瓶颈，本文将其思想引入多模态序列融合，并通过 MI-Max 解决关键信息丢失问题。
- **Liu et al. (2020) MFFG**：使用 fusion forget gate 粗粒度过滤冗余，本文指出其不足并提出细粒度瓶颈+MI-Max 的替代方案。
- **Han et al. (2021) MMIM**：同样使用互信息最大化，但面向情感分析中模态不变性学习；本文聚焦视频冗余/噪声去噪，目标不同。
- **Yu et al. (2021a) VG-GPLMs**：基于 BART 的视频摘要 SOTA；本文在其基础上进一步利用瓶颈去噪提升摘要质量。
- **Tsai et al. (2019) MulT**：多模态 Transformer 方向工作；本文强调视频序列特有的冗余问题，提供互补视角。
- **Hazarika et al. (2020) MISA**：学习模态不变/特异性表示；本文不侧重表示分解，而是通过瓶颈结构隐式去噪。

## 局限性与未来方向
- 仅在情感分析和摘要两个任务上验证，未扩展到幽默检测等其他多模态任务。
- 情感分析任务提升幅度相对摘要任务较小，作者推测与数据集规模有限、模型需充足数据学习噪声模式有关。
- 未来计划：扩展至更多任务；引入 vision-text 预训练 backbone 进一步提升性能。

## 研究启发与可借鉴点
- **瓶颈机制用于多模态融合**：将受限容量嵌入作为跨模态信息交流的"网关"，可有效过滤冗余，该设计可迁移至音频-文本、图像-文本等融合场景。
- **互信息最大化监督去噪**：用 InfoNCE 约束融合表征保留各模态关键信息，避免纯过滤导致的信号损失，这一正则化思路可用于其他去噪/压缩任务。
- **文本中心融合的有效性**：消融显示以文本模态为输出的融合结果最优，提示在多模态任务中"文本为中心"的设计值得优先考虑。
- **Grad-CAM 视频扩展与统计量化**：将 Grad-CAM 扩展至视频并计算标准差/归一化熵来量化注意力尖锐度，为模型可解释性评估提供了可复用的分析方法。

## 关键术语表
- **Fusion Bottleneck**：通过短序列嵌入限制跨模态信息流动量的模块，迫使模型压缩冗余后才允许信息交换。
- **Mutual Information Maximization (MI-Max)**：最大化融合结果与各单模态输入之间互信息的下界，以防止关键信息在去噪过程中丢失。
- **InfoNCE Loss**：基于噪声对比估计的对比学习损失，用于逼近互信息下界，在 batch 内以其他样本作为负样本。
- **Multimodal Sentiment Analysis (MSA)**：结合文本、视觉、音频信号预测视频情感强度的任务，包含回归与分类两种设置。
- **Multimodal Summarization**：从视频视觉与对应转录文本生成抽象摘要的任务，需融合多模态信息以覆盖单模态缺失内容。
- **Fusion Forget Gate**：Liu et al. (2020) 提出的粗粒度去噪模块，通过门控机制控制跨模态冗余信息流动，但可能误删代表性信息。
- **Grad-CAM**：基于梯度的可视化方法，用于定位模型关注的输入区域，本文扩展至视频帧级别评估注意力集中度。
- **How2 Dataset**：包含 79114 段教学视频的 multimodal summarization 基准数据集，每段视频配有转录文本和人工摘要。

## 可复现要素
- **数据集**：MOSI、MOSEI、How2 均为公开数据集；MOSI 2198 条、MOSEI 23453 条、How2 79114 条视频。
- **代码**：开源，地址 https://github.com/WSXRHFG/DBF。
- **关键超参**：Bottleneck length（MOSI=2, MOSEI=4, How2=8）；$\alpha$（MOSI=0.05, MOSEI/How2=0.1）；Batch size（MOSI=32, MOSEI=96, How2=80）；Fusion size=128（MOSI/MOSEI）或 768（How2）；optimizer Adam + warmup，early stopping patience=10。
- **特征提取**：文本用 BERT-base [CLS]；音频用 COVAREP（74维）；视觉用 Facet（MOSI 47维，MOSEI 35维）或 3D ResNeXt-101（How2，2048维，每16帧采样）；框架 PyTorch + Huggingface。
