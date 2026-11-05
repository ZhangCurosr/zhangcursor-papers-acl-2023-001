---
title: "DecompX-Explaining-Transformers-Decisions-by-Propagating-Tok"
source: https://aclanthology.org/2023.acl-long.149.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:09:31"
field: "NLP模型可解释性"
keywords: ["Transformer可解释性", "token分解", "归因方法", "向量传播", "FFN近似", "classification head解释"]
innovations: ["端到端分解向量传播替代rollout避免信息损失", "FFN非线性点态线性近似实现分解穿透", "首次整合分类头实现per-label正负归因"]
benchmarks: ["SST-2", "MNLI", "QNLI", "HateXplain"]
---

# 论文速读：DecompX: Explaining Transformers Decisions by Propagating Token Decomposition

## 一句话总结
DecompX是一种向量基Transformer解释方法，通过将token表示分解为可追溯的分解向量并在整个编码器层（包括非线性FFN）及分类头中传播，首次实现了端到端的per-label正负归因解释，在多项忠实度基准上显著优于现有梯度法和向量法。

## 研究问题与动机
- **现有向量方法的三大约束**：(1)忽略编码器中的FFN组件（因其非线性）；(2)使用rollout聚合层间局部归因，将多维分解向量压缩为标量导致信息损失；(3)未纳入分类头，无法反映任务特定的决策机制。
- **全局解释的可靠性困境**：GlobEnc和ALTI等方法虽改进了局部分析，但依赖L2-norm或L1-distance等标量化聚合，且rollout假设只需标量cross-token归因即可计算全局流，忽略了向量的多维影响。
- **梯度方法的局限性**：vanilla gradients、Gradient×Input、Integrated Gradients等依赖输入嵌入梯度，不同归约方式（L1/L2-norm、mean）效果不稳定，且易受对抗攻击和sanity checks质疑。
- **忠实度评估需求**：需要一种能够量化每个输入token对预测标签的正/负贡献的方法，而非仅提供绝对重要性。

## 核心贡献（创新点）
1. **端到端分解传播框架**：在编码器每一层维护分解向量的独立性并逐层传播，而非像rollout那样在层间聚合标量，从根本上避免信息损失。
2. **FFN非线性近似分解**：提出基于点态线性近似（过原点的单调函数）的方案处理FFN激活函数，使分解可穿透非线性层；相比GlobEnc忽略FFN的做法，此设计显著提升忠实度（SST2的AOPC从0.307提升至0.627）。
3. **Bias分解方案（AbsDot）**：设计基于偏置向量与各分解向量点积绝对值比例的权重分配，使bias项也能融入分解求和，消除无偏置假设下的信息不完整问题。
4. **分类头整合与per-label归因**：首次将最后一层[CLS] token的分解向量传播至分类头（双层FFN），获得针对每个输出类别的精确正负贡献值，解决先前方法无法区分"对某类重要但对预测类不重要"的问题（如"or bolted"对contradiction重要但对entailment无贡献）。
5. **系统性忠实度验证**：在SST-2、MNLI、QNLI、HateXplain四个数据集上，基于AOPC、Accuracy、Prediction Performance三项指标全面评估，DecompX在所有设置下均大幅领先。

## 方法详解
**核心分解定义**：对第$\ell$层的第$i$个token表示$\boldsymbol{x}_i^\ell$分解为基于$N$个输入token的元素向量之和：$\boldsymbol{x}_i^\ell = \sum_{k=1}^{N} \boldsymbol{x}_{i\leftarrow k}^\ell$。

**多头自注意力模块**：将$k$-th输入token的归因向量定义为$z_{i\leftarrow k}^\ell = \sum_{h=1}^{H} \alpha_{i,j}^h \boldsymbol{x}_{j\leftarrow k}^\ell \boldsymbol{W}_{Att}^h + \omega_k \boldsymbol{b}_{Att}$，其中$\omega_k$由AbsDot方法计算（偏置与各分解向量点积的绝对值占比），确保所有偏置被分配到各分解分量中。

**LayerNormalization分解**：沿用Kobayashi et al. (2021)的LN展开技术，将LN作用于每个分解分量，同时引入新的bias分解项$\beta$，保持分解完整性：$\tilde{z}_{i\leftarrow k}^\ell = g_{z^+}(\cdot) + \beta$。

**FFN近似处理**：对激活函数$f_{act}$做点态线性近似$f_{act}^{(\boldsymbol{x})}(\boldsymbol{x}) = \boldsymbol{\theta}^{(\boldsymbol{x})} \odot \boldsymbol{x}$，其中$\theta_t = f_{act}(x^{(t)}) / x^{(t)}$。此近似保证$f_{act}(\sum_k \zeta_{i\leftarrow k}^\ell) = \sum_k f_{act}^{(\boldsymbol{\theta})}(\zeta_{i\leftarrow k}^\ell)$，使分解可穿透FFN。

**编码层完整传播**：第$\ell$层到第$\ell+1$层的更新为$x_{i\leftarrow k}^{\ell+1} = \text{LN}(\sum_{k=1}^{N}[\tilde{z}_{i\leftarrow k}^\ell + z_{\text{FFN},i\leftarrow k}^\ell])$，保留分解向量的独立性贯穿全部$L$层。

**分类头传播**：将最后一层[CLS] token的分解向量$\boldsymbol{x}_{[\text{CLS}]\leftarrow k}^{L+1}$通过分类头（含pooler层和classification layer的FFN）得到$y_{c\leftarrow k}$，使得$\sum_k y_{c\leftarrow k} = y_c$（预测logit），实现per-label的正负归因。

## 实验与结果
**数据集**：SST-2（情感分析）、MNLI（自然语言推理）、QNLI（问答）、HateXplain（仇恨检测）。模型：fine-tuned BERT-base-uncased和RoBERTa-base。

**评估指标**：AOPC（掩码重要token后的预测概率变化均值，越高越好）、Accuracy（随掩码比例变化的平均准确率，越低越好）、Prediction Performance（仅用top-K% token训练分类器的性能，越高越好）。

**核心结果（BERT-base，Table 1）**：
| 数据集 | 方法 | AOPC↑ | Acc↓ | Pred↑ |
|--------|------|-------|------|-------|
| SST-2 | DecompX | **0.627** | **40.80** | **92.20** |
| SST-2 | ALTI | 0.416 | 57.65 | 88.30 |
| MNLI | DecompX | **0.703** | **32.64** | **80.95** |
| MNLI | Integrated Gradients | 0.579 | 40.80 | 73.94 |
| QNLI | DecompX | **0.453** | **57.50** | **89.84** |
| HateXplain | DecompX | **0.612** | **38.71** | **66.34** |

**最强提升**：DecompX在SST-2上AOPC较ALTI提升50.7%（0.416→0.627），在MNLI上较Integrated Gradients提升21.4%（0.579→0.703）；仅保留10% token时准确率下降仅2.64%，远优于次优方法的7.34%和15.6%。

**消融结果（Table A.1，SST-2）**：移除FFN导致AOPC从0.627降至0.494；移除分类头降至0.288（最大下降）；移除Bias降至0.635（轻微影响，但对人可解释性重要）。

## 相关工作脉络
1. **Kobayashi et al. (2020, 2021)**：开创性提出用加权value向量的norm进行局部token归因，并纳入残差连接分析；DecompX继承其局部分解思想但解决其无法处理FFN和非线性的问题。
2. **Abnar & Zuidema (2020)**：提出attention rollout和flow方法聚合多层注意力权重；DecompX批判rollout将向量压缩为标量的信息损失缺陷，改用向量传播。
3. **Modarressi et al. (2022) GlobEnc**：结合分解与rollout的全局解释方法；DecompX通过纳入FFN和消除rollout显著超越其忠实度。
4. **Ferrando et al. (2022) ALTI**：用L1-distance替代L2-norm改进局部度量；DecompX指出其仍依赖rollout且忽略分类头，在per-label归因上具有本质优势。
5. **梯度类方法**（Simonyan et al. 2014; Sundararajan et al. 2017）：依赖输入嵌入梯度；Bastings et al. (2022)证明其一致性不佳，DecompX作为纯前向传播方法规避梯度不稳定问题。
6. **Mohebbi et al. (2023) Value Zeroing**：近期提出通过零值化value向量测量context mixing；DecompX指出其全局评估依赖rollout且使用不恰当的Spearman相关作为度量。

## 局限性与未来方向
- **局限于英语文本分类**：当前实验仅覆盖英文分类任务，扩展至其他语言需额外验证。
- **未评估大规模模型**：受资源限制，未测试GPT-2、T5等大模型。
- **FFN近似假设**：激活函数近似假设$f_{act}(0)=0$且单调，对ReLU类有效但可能不适用于所有激活函数。
- **未来方向**：作者提出可扩展至encoder-decoder架构、多语言场景及Vision Transformers。

## 研究启发与可借鉴点
1. **分解向量的逐层传播范式**可作为通用模板应用于其他需要端到端归因的架构（如decoder-only模型），通过保持向量独立性避免rollout式信息压缩。
2. **FFN的线性近似技巧**（点态斜率$\boldsymbol{\theta}$）可迁移至其他非线性组件（如GeLU、SwiGLU）的解释中，为处理复杂非线性提供思路。
3. **分类头整合带来的per-label正负归因**是解释多分类任务的新范式，可推广至序列标注（如span选择）和生成任务（见附录A.2的QA扩展）。
4. **AbsDot bias分解策略**对任何含偏置项的线性变换均有参考价值，可在需要保持分解完整性的场景复用。
5. **仅用10% top归因token保持92%+性能**的结果为稀疏解释和高效推理提供了实证支撑，可启发后续的归因驱动模型压缩研究。

## 关键术语表
**DecompX**：本文提出的Transformer解释方法，通过分解token表示为元素向量并在整个模型中传播实现忠实度解释。
**Rollout**：将多层注意力权重递归聚合为全局归因图的方法，假设只需标量cross-token归因即可计算全局流。
**AOPC（Area Over the Perturbation Curve）**：通过逐步掩码top/low重要性token后预测概率变化曲线下面积衡量解释忠实度的指标。
**Per-label Attribution**：针对每个输出类别分别计算的token归因分数，区分正负贡献方向。
**FFN Approximation**：用点态线性函数$f_{act}^{(\boldsymbol{x})}(\boldsymbol{x}) = \boldsymbol{\theta} \odot \boldsymbol{x}$近似非线性激活，使分解可穿透FFN。
**AbsDot Bias Decomposition**：基于偏置向量与各分解向量点积绝对值比例的权重分配方案，使bias融入分解求和。
**Value Zeroing**：Mohebbi et al. (2023)提出的方法，通过置零value向量并比较余弦距离测量context mixing。
**LayerNormalization Decomposition**：将LN操作展开为对各分解分量的独立处理，保留分解结构的数学技术。

## 可复现要素
- **数据集**：SST-2、MNLI、QNLI、HateXplain（均为公开基准）。
- **代码**：已开源，github.com/mohsenfayyaz/DecompX。
- **模型**：BERT-base-uncased、RoBERTa-base（HuggingFace Transformers）。
- **关键超参**：Integrated Gradients步长设为0.1；掩码比例K（不同实验取10%-100%不等）。
- **评估实现**：基于HuggingFace Transformers库，使用[L2-norm of token gradient]作为梯度方法归约。
