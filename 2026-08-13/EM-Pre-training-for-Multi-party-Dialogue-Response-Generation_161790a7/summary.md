---
title: "EM-Pre-training-for-Multi-party-Dialogue-Response-Generation"
source: https://aclanthology.org/2023.acl-long.7.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:20:16"
field: "多_party对话理解与生成"
keywords: ["多_party对话", "响应生成", "预训练", "EM算法", "隐变量", "Ubuntu IRC"]
innovations: ["首次提出基于EM算法的多_party对话响应生成预训练方法", "将addressee建模为隐变量并通过Hard EM迭代优化", "证明期望完整对数似然为不完全对数似然的下界"]
benchmarks: ["Ubuntu IRC Benchmark"]
---

# 论文速读：EM-Pre-training-for-Multi-party-Dialogue-Response-Generation

## 一句话总结
本文首次研究了多_party对话响应生成（MPDRG）的预训练问题，针对大规模无标注addressee标签的多_party对话语料稀缺问题，提出了一种基于EM（Expectation-Maximization）算法的迭代预训练方法，将addressee建模为隐变量，通过交替执行E步（预测addressee分布）和M步（优化生成模型）逐步提升模型性能。

## 研究问题与动机
1. **现有方法局限于两-party对话**：当前预训练语言模型（PLMs）主要针对两-party或序列结构对话，而多_party对话具有树状响应关系，复杂度更高，缺乏针对性预训练方法。
2. **标注数据稀缺**：多_party对话语料库（如Ubuntu Dialogue Corpus）中addressee标注极为稀少，现有方法只能在小规模标注数据集上微调，无法利用海量无标注语料进行预训练。
3. **预训练与微调割裂**：前序工作（如GSN、HeterMPC）直接在小规模标注数据上微调PLM，未能充分发挥预训练迁移学习的潜力。
4. **隐变量建模需求**：如何在没有人工标注的情况下，有效建模addressee这一关键信息，是多_party对话生成任务的核心挑战。

## 核心贡献（创新点）
1. **首次提出多_party对话响应生成的预训练方法**：突破了以往仅能在小规模标注数据上微调的局限，实现了从无标注大规模语料中学习的能力。
2. **提出基于EM算法的隐变量建模策略**：将addressee作为离散隐变量$z_t$，通过迭代E步（预测addressee分布）和M步（优化生成模型）协同提升加essee预测准确性和响应生成质量。
3. **理论证明EM预训练的可行性**：证明了所构造的期望完整对数似然$\hat{\ell}$是不完全对数似然$\ell$的下界，最大化下界可间接优化目标函数。
4. **免费获得对话解析器作为副产品**：预训练过程中学到的addressee预测能力可直接用于推断对话话语结构，服务于响应选择、问答等下游任务。

## 方法详解
1. **任务定义**：给定对话历史$c_t = \{S_1:U_1 [SEP] S_2:U_2 [SEP] \ldots S_t:\}$和响应说话人$S_t$，以及addressee $z_t=j$，目标是生成响应$r_t=U_t$。
2. **Addressee建模**：采用简单有效的嵌入方式，在词嵌入和位置编码前加入addressee嵌入（2个entry的查找表，判断词是否属于addressee语句）。
3. **E步（Expectation）**：基于贝叶斯网络$p(c,z,r) = p(c)\cdot p(z|c)\cdot p(r|c,z)$，假设$p(z|c)$服从均匀分布，推导出$p(z|c,r) \propto p(r|c,z)$，利用当前生成模型计算每个候选addressee的概率分布。
4. **M步（Maximization）**：采用Hard EM策略，选择概率最高的$z_t^i$作为预测标签，使用自回归语言建模损失$\mathcal{L}_G = -\sum_{k=1}^{N}\sum_{i=1}^{n_k}\log p(w_i^k|w_{<i}^k,c_t^k,z_t^k;\theta)$优化模型参数。
5. **置信度过滤**：引入超参数$\alpha \in [0,1]$控制使用的训练数据比例，仅选择置信度最高的$\alpha \times N$个样本参与M步优化。
6. **初始化策略**：使用Shi & Huang (2019)的discourse parser预测无标注语料的addressee，获得初始训练集，避免随机初始化导致的收敛问题。

## 实验与结果
- **预训练数据**：Ubuntu Dialogue Corpus v2，过滤后764,373个对话（无addressee标注）。
- **微调/评估数据**：Ubuntu IRC benchmark，311,725个训练对话、5,000个测试对话（含完整addressee标注）。
- **评估指标**：BLEU-1~4、METEOR、ROUGE-L，以及人工评估（相关性、流畅性、信息量）。
- **最强结果（PF设置）**：BLEU-4达到2.45，METEOR达到5.52，ROUGE-L达到11.71，全面超越之前SOTA模型HeterMPC-BART（BLEU-4: 1.49, METEOR: 4.94, ROUGE-L: 11.20）。
- **预训练-only（PO）已可比肩SOTA**：仅使用无标注语料预训练即可达到BLEU-4: 1.41，接近HeterMPC-BART的1.49。
- **人工评估**：PF模型得分1.92，最佳响应比例达28%，与人类参考一致。
- **消融实验**：移除EM过程导致性能骤降；仅使用reply chain会损失侧信息；纯去噪预训练效果与直接微调相当。
- **迭代收敛**：约第6次迭代后BLEU-4和addressee预测准确率均达到峰值。

## 相关工作脉络
1. **DialoGPT（Zhang et al., 2020）**：基于GPT架构在Reddit两-party对话语料上预训练，关注序列对话历史；本文面向树状多_party对话，建模addressee隐变量。
2. **PLATO/DialogVED（Bao et al., 2020; Chen et al., 2022）**：引入离散/连续隐变量建模对话意图，但意图无实际实体对应；本文隐变量$z_t=j$具有明确语义（第j个话语为addressee）。
3. **GSN（Hu et al., 2019）**：构建对话图并使用GNN编码历史，在小规模标注数据上微调；本文利用无标注大规模语料预训练。
4. **HeterMPC（Gu et al., 2022）**：使用异构图神经网络建模六种边类型，结合Transformer解码器；本文方法更简单（仅addressee嵌入），但性能更优。
5. **硬EM方法（Min et al., 2019）**：弱监督问答中的硬EM策略被本文借鉴，用于addressee预测。
6. **话语解析器（Shi & Huang, 2019）**：提供初始addressee预测，用于EM算法的第一轮数据初始化。

## 局限性与未来方向
1. **实验数据集单一**：仅在Ubuntu IRC benchmark上验证，预训练也限于Ubuntu聊天领域，未在其他开放域数据集（如影视剧脚本）上测试。
2. **忽略对话历史的reply-to关系**：当前方法仅利用单轮addressee信息，未建模对话历史中话语间的reply-to依赖关系，限制了上下文理解能力。
3. **未评估跨域迁移能力**：虽声称方法可扩展到其他领域，但未提供实际跨域实验证据。
4. **隐变量假设简化**：假设$p(z|c)$服从均匀分布可能与实际分布有偏差。

## 研究启发与可借鉴点
1. **EM算法解决弱监督预训练问题**：将缺少的标注信息（addressee）作为隐变量，通过EM迭代逐步改进预测和生成，可迁移到其他缺少标注的生成任务。
2. **下界优化理论保证**：通过Jensen不等式构造期望完整对数似然作为不完全对数似然的下界，为方法可行性提供严谨理论支撑，值得在其他预训练场景中借鉴。
3. **Hard EM + 置信度过滤策略**：选择最高概率样本而非采样，结合动态$\alpha$控制数据质量，平衡了噪声容忍与训练效率。
4. **副产品复用思路**：预训练过程中自然获得的对话解析器可直接用于下游理解任务（如响应选择、问答），实现"一次训练，多种收益"。
5. **简单有效的addressee建模**：仅用2-entry嵌入表区分addressee词与非addressee词，却显著优于复杂的图神经网络基线，提示简洁设计的有效性。

## 关键术语表
- **MPDRG（Multi-party Dialogue Response Generation）**：多_party对话响应生成任务，指在多参与者对话中根据历史和指定addressee生成回复。
- **Addressee**：响应话语的接收者/目标对话者，是多_party对话中的关键结构信息。
- **EM算法（Expectation-Maximization）**：迭代优化算法，E步估计隐变量分布，M步优化模型参数。
- **Hard EM**：EM算法的变体，在E步直接选择概率最高的隐变量取值而非采样。
- **Ubuntu IRC Benchmark**：从Ubuntu对话语料中提取的带addressee标注的多_party对话评测基准。
- **BART**：基于Transformer的序列到序列预训练模型，采用去噪自编码目标。
- **Pseudo-labeling**：利用模型预测作为伪标签训练数据，本文通过EM迭代实现。
- **Incomplete Log-likelihood**：不完全对数似然，考虑隐变量的边缘概率，通常难以直接优化。

## 可复现要素
- **预训练数据集**：Ubuntu Dialogue Corpus v2，过滤后764,373个对话，公开可用。
- **微调/评估数据集**：Ubuntu IRC Benchmark，公开可用（Hu et al., 2019）。
- **代码开源**：是，官方实现位于https://github.com/EricLee8/MPDRG。
- **模型权重**：论文未明确声明权重是否开源，但代码已开源。
- **骨干模型**：BART-base，公开可用。
- **关键超参**：$\alpha \in [0,1]$（置信度过滤比例），动态设置为使验证集addressee预测准确率超过80%；EM迭代约6次收敛。
