---
title: "DIFFUSEMP-A-Diffusion-Model-Based-Framework-with-Multi-Grain"
source: https://aclanthology.org/2023.acl-long.158.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:50:07"
field: "可控文本生成与对话系统"
keywords: ["共情回复生成", "扩散模型", "可控文本生成", "多粒度控制", "对话系统"]
innovations: ["提出多粒度控制信号（CM/IT/SF）实现从话语到Token级的共情表达引导", "设计控制范围掩码策略将结构化控制嵌入扩散模型自注意力层", "构建情感增强匹配方法在推理时获取控制信号候选"]
benchmarks: ["EMPATHETICDIALOGUE"]
---

# 论文速读：DIFFUSEMP-A-Diffusion-Model-Based-Framework-with-Multi-Grain

## 一句话总结
本文提出 DIFFUSEMP，一个基于条件扩散语言模型的框架，通过引入通讯机制（CM）、意图（IT）和语义帧（SF）三种多粒度控制信号，解决共情回复生成中表达单调、泛化严重的问题。

## 研究问题与动机
- **共情回复生成单调性问题**：现有基于 MLE 的方法常生成"I'm sorry to hear that."等安全但浅层的泛化表达，缺乏对上下文的具体理解和深度共情。
- **多维度共情难以显式建模**：心理学中的共情是多维因素（情绪反应、解释理解、探索挖掘等）的综合体现，单一模型难以直接建模。
- **扩散模型在文本可控生成中的应用空白**：现有扩散语言模型（如 Diffusion-LM、DiffuSeq）主要面向无条件生成或粗粒度控制，尚未在细粒度共情表达控制上发挥作用。

## 核心贡献（创新点）
1. **提出多粒度显式控制信号体系**：首次将 CM（话语级）、IT（句子级）和 SF（Token 级）三种共情维度信号引入回复生成，实现从粗到细的共情表达引导。
2. **设计控制范围掩码（Control-Range Masking）策略**：在 Transformer 自注意力层中引入掩码矩阵，使不同粒度的控制信号精准作用于对应响应 token，替代传统全注意力带来的控制模糊问题。
3. **构建端到端可控扩散共情回复框架 DIFFUSEMP**：将对话上下文、多粒度控制信号与响应拼接输入扩散模型，结合部分噪声策略（partial noising）仅在响应部分加噪，实现上下文感知与可控生成。
4. **设计情感增强匹配（Emotion-Enhanced Matching）方法**：在推理阶段通过语义相似性与情感一致性加权匹配获取候选回复作为控制信号来源，弥补训练时标签已知的假设限制。

## 方法详解

### 整体架构
将对话上下文 $\mathbf{w}^u$、控制信号序列 $\mathbf{w}^c$ 和响应 $\mathbf{w}^y$ 拼接为输入 $\mathbf{w} = \mathbf{w}^u \oplus \mathbf{w}^c \oplus \mathbf{w}^y$，通过嵌入函数映射为连续表示 $\mathbf{x}_0$，送入基于 BERT-base 架构的扩散模型。

### 控制信号获取
- **CM（Communication Mechanism）**：使用三个 RoBERTa 分类器识别 Emotional Reaction (ER)、Interpretation (IP)、Exploration (EX)。
- **IT（Intent）**：在 EMPATHETICINTENT 数据集上训练的 BERT 分类器，标注如 Acknowledging、Consoling、Questioning 等意图。
- **SF（Semantic Frame）**：使用 open-SESAME 模型从 FrameNet 提取每个 Token 的语义帧标签（共 1222 类）。

### 前向扩散过程
采用 **部分噪声策略（partial noising）**，仅对响应部分 $\mathbf{y}_t$ 施加高斯噪声，上下文和控制信号保持不变：
$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$
噪声调度 $\beta_t$ 采用平方根调度（square-root noise schedule），共 2000 步。

### 反向去噪过程
使用 Transformer 预测均值和方差，引入 **控制范围掩码矩阵 $M$** 约束自注意力：
$$S^{i+1} = \text{softmax}\left(\frac{Q^{i+1}K^{i+1\top} + M}{\sqrt{d_k}}\right)$$
其中掩码规则：若 token $i$ 控制 token $j$，则 $M(i,j)=0$；否则 $M(i,j)=-\infty$。
- CM 信号控制整句话的所有 token
- IT 信号控制对应句子内所有 token
- SF 信号精确控制单个 token

### 训练损失
最小化变分下界（Variational Lower Bound）：
$$\mathcal{L}_{\text{vlb}} = \sum_{t=2}^{T} ||\mathbf{y}_0 - \tilde{f}_\theta(\mathbf{x}_t, t)||^2 + ||\text{EMB}(\mathbf{w}^y) - \tilde{f}_\theta(\mathbf{x}_1, 1)||^2 + \mathcal{R}(||\mathbf{x}_0||^2)$$

### 推理阶段的候选检索
通过情感增强匹配获取控制信号来源：
$$\text{Score} = \text{SIM}_{\text{semantic}} + \gamma \cdot \text{SIM}_{\text{emotional}} \quad (\gamma=0.2)$$
计算查询上下文与候选池上下文的语义相似度（BERT 句向量余弦）和情感分布相似度（情感分类器输出分布），加权排序后取最高分候选作为信号源。

## 实验与结果
- **数据集**：EMPATHETICDIALOGUE（24,850 条多轮对话，32 种情绪标签），按 8:1:1 划分。
- **评估指标**：相关性（BERTScore↑、MIScore↓）、可控性（ACC-CM↑、ACC-IT↑、F1-SF↑）、信息量（Dist-1/2/4↑、Self-BLEU↓）、长度（AvgLen↑）。
- **最强结果**：DIFFUSEMP 在可控性上显著领先——ACC-CM = 92.36%（+30% 以上 vs 基线）、ACC-IT = 84.24%、F1-SF = 52.79%，均大幅优于 Transformer 类、预训练模型类和 DiffuSeq 基线。
- **信息量优势**：Dist-2 = 29.25，Self-BLEU = 1.09，优于所有对比方法，证明多粒度控制有效提升了表达多样性。
- **人工评估**：在共情（3.68/5）、相关性（3.39/5）、信息量（4.63/5）三项均获最高分，且 A/B 测试中显著优于 Baseline。
- **Oracle 上界**：使用真实标签时 BERTScore = 0.7458，说明当前控制信号提取方法仍有提升空间。

## 相关工作脉络
- **CoMAE (Zheng et al., 2021)**：同样使用 CM、DA、EM 等多因子控制共情回复，但仅在话语级使用统一因子作用于所有解码位置；DIFFUSEMP 进一步细化到句子级和 Token 级，并通过掩码实现精准控制。
- **DiffuSeq (Gong et al., 2022)**：首个面向 Seq2Seq 的无条件扩散语言模型，采用部分噪声策略；本文在其基础上引入显式控制信号和掩码机制，从"无条件生成"迈向"细粒度可控生成"。
- **Diffusion-LM (Li et al., 2022b)**：通过额外训练的 classifier 控制情感/句法等属性，依赖分类器干预；本文直接将控制信号作为输入上下文的一部分，无需额外 classifier，且粒度更细。
- **MIME (Majumder et al., 2020) / EmpDG (Li et al., 2020)**：基于 Transformer + MLE 的经典共情回复方法；本文证明扩散模型在可控性和信息量上能超越此类架构。
- **TransferTransfo / BART**：大规模预训练模型微调基线；本文在参数量更少（91M vs 117M/140M）的情况下，在可控性和多样性指标上全面超越。

## 局限性与未来方向
- **控制信号标注依赖预训练工具**：CM/IT/SF 三类信号的准确性受限于 RoBERTa、BERT 和 open-SESAME 等标注工具性能，表格显示 CM-IP 的 F1 仅 62.60%，SF 标注存在误差传播风险。
- **推理阶段候选检索质量限制**：情感增强匹配方法相对简单，检索到的候选回复可能不够理想，导致控制信号不准确。
- **扩散模型计算开销大**：2000 步采样导致推理速度慢，GPU 资源消耗高，限制了实际部署可行性。
- **未来方向**：引入更强大的标注器和响应选择模型；探索 Diffusion Model 加速采样方法（如 DDIM、Analytic-DPM）；将框架扩展到其他可控文本生成任务。

## 研究启发与可借鉴点
1. **多粒度控制信号的思想**：将同一任务的控制信号按话语/句子/Token 三个层级拆解，可迁移至其他可控文本生成任务（如风格控制、观点生成），实现从粗到细的精准引导。
2. **控制范围掩码（Control-Range Masking）机制**：用掩码矩阵显式建模"信号→token"的控制关系，嵌入 Transformer 自注意力层，是一种不依赖额外模块即可实现结构化控制的设计，值得在其他条件生成任务中复用。
3. **情感增强匹配（双路相似度融合）**：将语义相似性与情感分布相似度加权结合用于检索辅助生成，比单一相似度更有效，可推广到其他需要保持情感一致性的生成任务。
4. **Oracle 上界分析范式**：通过提供真实标签计算理论上限，量化标注工具/检索模块的性能瓶颈，为后续改进提供明确方向。

## 关键术语表
- **DIFFUSEMP**：本文提出的基于扩散模型的多粒度可控共情回复生成框架。
- **Control-Range Masking**：一种嵌入 Transformer 自注意力的掩码机制，根据控制信号的粒度（话语/句子/Token）限制其影响范围。
- **Communication Mechanism (CM)**：话语级共情控制信号，分为情绪反应（ER）、解释理解（IP）和探索挖掘（EX）三类。
- **Intent (IT)**：句子级共情意图信号，描述听者在该句中的行为倾向（如安慰、提问、认可等）。
- **Semantic Frame (SF)**：Token 级语义帧信号，基于 FrameNet 标注每个词的语义角色类别。
- **Partial Noising**：仅对响应部分施加扩散噪声、保持上下文和控制信号不变的前向过程策略。
- **Emotion-Enhanced Matching**：结合语义相似度与情感分布相似度的加权检索方法，用于推理时获取控制信号候选。
- **MIScore**：基于最大互信息思想的上下文反推得分，衡量生成回复与对话上下文的相关性。

## 可复现要素
- **数据集**：EMPATHETICDIALOGUE（公开，CC-BY 4.0 许可），由 Rashkin et al. (2019) 构建，可通过 ACL Anthology 获取。
- **代码/权重**：论文未提及开源，基线方法使用官方代码。
- **关键超参**：扩散步数 T=2000，学习率 1e-4，batch size=128，dropout=0.1，最大输入长度=128，嵌入维度=128，情感-语义权重 $\gamma=0.2$，噪声调度为 square-root schedule。
