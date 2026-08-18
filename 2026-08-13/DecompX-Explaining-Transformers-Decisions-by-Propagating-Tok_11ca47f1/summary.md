---
title: "DecompX-Explaining-Transformers-Decisions-by-Propagating-Tok"
source: https://aclanthology.org/2023.acl-long.149.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:50:51"
field: "可解释AI/自然语言处理"
keywords: ["Transformer解释", "向量基归因", "faithfulness评估", "FFN分解", "偏置分解", "token归因"]
innovations: ["逐层传播分解向量替代rollout标量聚合以避免信息丢失", "点态线性近似实现FFN非线性组件的向量分解", "将分解传播至分类头实现per-label正负贡献量化"]
benchmarks: ["SST-2", "MNLI", "QNLI", "HateXplain"]
---

# 论文速读：DecompX: Explaining Transformers Decisions by Propagating Token Decomposition

## 一句话总结
DecompX 提出了一种向量传播式的 Transformer 解释方法，通过将 token 表示逐层分解为各输入 token 的贡献向量并完整传播至分类头（包含 FFN 与非线性近似），首次实现了端到端、忠实度更高的每 token 对各类别预测的正负贡献量化。

## 研究问题与动机
1. **FFN 被忽略**：现有向量基方法（如 GlobEnc、ALTI）因 FFN 的非线性特性而将其排除在分解之外，损失了模型内部重要组件的归因信息。
2. **Rollout 导致信息丢失**：现有全局归因方法在层间聚合时先将分解向量压缩为标量，再通过 rollout 递归聚合，造成高维信息不可逆丢失。
3. **分类头未被纳入**：已有方法仅做到编码器最后一层，未将分解传播至分类头，因此无法区分 token 对具体预测类别（如正面/负面）的正负影响方向。
4. **偏置项处理粗糙**：部分方法直接忽略偏置，导致归因求和无法与模型 logits 对齐，降低解释的忠实度。

## 核心贡献（创新点）
1. **全编码器层向量分解传播**：提出将 token 表示在各层间以分解向量形式逐层传播，而非标量聚合；与 GlobEnc/ALTI 的本质区别在于保留了向量多维权重信息，避免了 rollup 的信息损失。
2. **FFN 非线性近似分解**：通过点态线性近似（斜率 = 激活输出/输入）将 FFN 的非线性纳入分解过程；与先前工作忽略 FFN 的做法相比，实现了更完整的编码器链路归因。
3. **偏置分解策略（AbsDot）**：提出基于绝对值点积的偏置分配权重，使偏置可被纳入 token 级求和；与直接忽略或简单截断偏置的方法相比，保证了解释向量之和与模型 logits 的数值一致性。
4. **首次将分类头纳入分解**：将分解向量继续传播至池化层 + 分类层的 FFN，输出每 token 对每类别 logits 的正负贡献；这与所有先前向量基方法仅输出编码器层重要性有本质区别。

## 方法详解
**整体流程**：对每一层 $l$，将第 $i$ 个位置的 token 表示分解为来自 $N$ 个输入 token 的要素向量之和：
$$\pmb{x}_i^l = \sum_{k=1}^{N} \pmb{x}_{i\Leftarrow k}^l$$

**自注意力分解**：将上一层的分解向量代入多头注意力公式，得到第 $l$ 层注意力输出的分解形式；引入 AbsDot 策略（式 5-6）将偏置 $b_{Att}$ 按比例分配给各分解向量。

**LayerNorm 分解**：采用 Kobayashi et al. (2021) 的 LayerNorm 分解技巧，将 LN 展开为对各分解向量的元素级归一化函数 $g(\cdot)$ 加上偏置 $\beta$。

**FFN 分解（核心创新）**：假设激活函数过原点且单调，用零截距直线近似：$f_{act}^{(x)}(x) = \theta^{(x)} \odot x$，其中 $\theta_t = f_{act}(x^{(t)}) / x^{(t)}$。将同一 $\theta$ 作用于所有分解向量，保证分解之和等于完整激活输出（式 10-11）。

**分类头分解**：将最后编码层的 [CLS] 分解向量继续传入两层 FFN（池化层 + 分类层），得到每 token 对每类别 logits $y_c$ 的贡献 $y_{c\Leftarrow k}$，满足 $y_c = \sum_k y_{c\Leftarrow k}$（式 13-14），从而获得**正负贡献明确**的 per-label 解释。

## 实验与结果
**数据集**：SST-2（情感分析）、MNLI（NLI）、QNLI（QA）、HateXplain（仇恨检测）。
**模型**：Fine-tuned BERT-base-uncased、RoBERTa-base。
**评估指标**：AOPC（均值预测概率变化）、Accuracy（扰动后准确率）、Prediction Performance（充足性评估）。
**主要结果**：
- SST-2 AOPC：DecompX = **0.627**，对比 GlobEnc 0.307（+104%）、ALTI 0.416（+51%）。
- MNLI AOPC：DecompX = **0.703**，对比 GlobEnc 0.498（+41%）、ALTI 0.515（+37%）。
- 保留 Top-10% tokens 时，DecompX 准确率仅下降 **2.64%**，而 GlobEnc 和 Gradient×Input 分别下降 7.34% 和 15.6%。
**消融**：去除 FFN 使 AOPC 从 0.627 降至 0.494；去除分类头导致 AOPC 骤降至 0.288（影响最大）；偏置去除影响较小。

## 相关工作脉络
1. **Kobayashi et al. (2020, 2021)**：开创性地将向量范数引入 Transformer 归因分析，并纳入残差连接；DecompX 在此基础上推进到全组件分解与跨层传播。
2. **GlobEnc (Modarressi et al., 2022)**：在 Kobayashi 基础上加入第二残差连接和 LayerNorm，仍用 rollout 聚合；DecompX 以向量传播替代 rollout 以消除信息损失。
3. **ALTI (Ferrando et al., 2022)**：使用 L1 距离改进局部度量；但仍依赖 rollout 聚合且不包含分类头，DecompX 从根本上解决了这两点。
4. **Abnar & Zuidema (2020) Rollout**：早期递归注意力聚合方法，仅依赖标量；DecompX 明确指出标量聚合的信息损失问题并以向量传播替代。
5. **Gradient×Input / Integrated Gradients**：经典梯度基方法；DecompX 在 faithfulness 评估中显著优于两者，且无需反向传播，计算更高效。
6. **Mohebbi et al. (2023) Value Zeroing**：近期评估上下文混合的向量基方法，但其全局评估依赖 rollout，DecompX 的消融实验（附录 A.3）直接比较并展现更优性能。

## 局限性与未来方向
1. **未在大模型上验证**：由于资源限制，未在 GPT-2、T5 等大语言模型上测试。
2. **仅针对英语文本分类**：多语言场景及问答、生成等其它任务需额外验证。
3. **FFN 近似存在理论误差**：点态线性近似虽经验有效，但对强非线性激活可能存在近似偏差。

## 研究启发与可借鉴点
1. **非线性组件的线性近似策略**：点态斜率近似（激活输出/输入）可用于其他含非线性层的模型归因，是一个通用的近似框架。
2. **向量传播优于标量聚合**：在需要跨层传递信息的问题中，保持向量维度比先压缩为标量再聚合更能保留信息；这一思想可迁移到其他需要跨层归因的场景。
3. **将下游头纳入归因链路**：分类/解码头的纳入使解释直接与预测标签挂钩，这一设计模式可推广至 QA span prediction、序列生成等任务。
4. **偏置项的分解处理**：AbsDot 策略提供了一种在复杂计算图中分配全局偏置的可行方案，适用于其他含偏置的多层架构。
5. **faithfulness 评估协议的复用**：AOPC + Accuracy + Prediction Performance 三指标组合可有效验证解释方法的质量，可作为标准评估套件参考。

## 关键术语表
- **Decomposed Token Representation**：将每一层 token 的表示向量按来源输入 token 分解为 $N$ 个要素向量的和。
- **Propagation（传播）**：将分解后的向量在各层间逐层传递而不进行标量聚合，避免信息丢失。
- **Rollout**：早期全局归因方法，通过递归聚合标量化的局部注意力得到全局归因图；本文认为其会导致信息损失。
- **Faithfulness（忠实度）**：解释方法与模型实际决策行为的一致性，通过扰动输入并观测输出变化来评估。
- **AbsDot**：本文提出的偏置分解策略，按分解向量与偏置的点积绝对值比例分配偏置贡献。
- **FFN Approximation**：通过将非线性激活函数近似为零截距直线（斜率为逐元素激活输出/输入）实现 FFN 的向量分解。
- **AOPC（Average Over Proportion of Cases）**：在不同掩码比例下，模型预测概率变化的平均值，用于衡量解释的忠实度。
- **Per-label Attribution**：DecompX 输出的每个 token 对每个预测类别 logits 的具体贡献值，可区分正/负影响。

## 可复现要素
- **数据集**：SST-2、MNLI、QNLI、HateXplain（均为公开数据集）。
- **代码**：已开源，地址 https://github.com/mohsenfayyaz/DecompX。
- **模型**：Fine-tuned BERT-base-uncased 和 RoBERTa-base（来自 Hugging Face Transformers）。
- **关键超参**：Integrated Gradients 步长设为 0.1；扰动时用 [MASK] 替换而非删除 token。
- **评估实现**：基于 Hugging Face Transformers 库，具体训练超参论文未详细列出。
