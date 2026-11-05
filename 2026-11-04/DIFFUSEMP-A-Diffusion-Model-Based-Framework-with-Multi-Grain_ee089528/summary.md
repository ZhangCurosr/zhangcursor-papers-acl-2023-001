---
title: "DIFFUSEMP-A-Diffusion-Model-Based-Framework-with-Multi-Grain"
source: https://aclanthology.org/2023.acl-long.158.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:08:31"
field: "可控文本生成"
keywords: ["diffusion model", "empathetic dialogue", "controllable generation", "multi-grained control", "language generation"]
innovations: ["提出多粒度控制信号（CM/IT/SF）引导共情对话生成", "设计控制范围掩码策略实现token级差异化注意力控制", "构建情感增强匹配的推理阶段信号获取方法"]
benchmarks: ["EMPATHETICDIALOGUE"]
---

# 论文速读：DIFFUSEMP: A Diffusion Model-Based Framework with Multi-Grained Control for Empathetic Response Generation

## 一句话总结
本文提出 DIFFUSEMP，一个基于条件扩散语言模型的共情对话响应生成框架，通过引入话语级（CM）、句子级（IT）和词元级（SF）多粒度显式控制信号，解决现有方法共情表达单调、泛化的问题，实现了从粗到细的精准共情引导生成。

## 研究问题与动机
1. **共情表达单调问题**：现有方法生成的共情响应倾向于通用安全表达（如反复使用"I'm sorry to hear that"），缺乏上下文关联性和信息量。
2. **控制信号粒度不足**：已有工作多在话语或句子层面建模共情因素，缺乏词元级别的精细控制，难以实现多样化的共情表达。
3. **扩散模型在可控文本生成中的应用空白**：现有扩散语言模型（DiffuSeq、Diffusion-LM）主要面向无条件或通用可控生成，尚未探索其在多粒度控制共情对话生成中的应用。
4. **上下文与共情信号的统一建模难题**：如何将对话上下文、多维共情属性和响应文本在生成过程中有效融合，仍缺乏系统性解决方案。

## 核心贡献（创新点）
1. **多粒度显式控制信号引入**：首次将 CM（话语级）、IT（句子级）、SF（词元级）三种粒度信号系统化引入共情响应生成，将任务从无条件生成转化为可控生成设置。
2. **DIFFUSEMP 扩散框架设计**：提出基于部分去噪（partial noising）的条件扩散语言模型，统一建模上下文与控制信号，实现非自回归的多粒度控制生成。
3. **控制范围掩码（Control-Range Masking）策略**：在 Transformer 自注意力层设计细粒度掩码矩阵，精确规定每类信号对响应词元的控制范围，确保生成逻辑连贯性与共情表达的层次性。
4. **情感增强匹配检索方法**：在推理阶段提出结合语义相似度与情感一致性的上下文-上下文匹配策略，为控制信号提供高质量候选源，弥补训练时无法获取真实响应的缺陷。
5. **与 CoMAE 的本质区别**：CoMAE 基于上下文预测各因素并统一作用于所有解码位置；DIFFUSEMP 直接使用与响应强相关的控制信号作为任务输入，并通过掩码实现token级差异化控制。

## 方法详解
**整体架构**：将对话上下文 $\mathbf{w}^u$、控制信号序列 $\mathbf{w}^c$、响应 $\mathbf{w}^y$ 拼接后通过嵌入函数映射为连续表示 $\mathbf{x}_0 = \mathbf{u}_0 \oplus \mathbf{c}_0 \oplus \mathbf{y}_0$，输入扩散模型。

**前向扩散过程**：采用部分去噪策略，仅对响应部分 $\mathbf{y}_t$ 添加噪声，上下文和控制信号保持不变：
$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$
噪声调度采用平方根方案，总步数 $T=2000$。

**反向去噪过程**：使用 Transformer 预测均值和方差，结合舍入步骤将连续表示映射回离散词表：
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_\theta(\mathbf{x}_t, t))$$
最后通过 softmax 分布 $p_\theta(w_i|x_i)$ 得到词元分布。

**控制范围掩码机制**：在自注意力层引入掩码矩阵 $M$，定义信号 token 对响应 token 的控制关系：
$$M(i,j) = \begin{cases} 0, & i \Rightarrow j \\ -\infin, & i \neq j \end{cases}$$
其中 $i \Rightarrow j$ 表示 token $i$ 控制 token $j$。CM 信号控制整个话语的所有 token，IT 信号控制对应句子的所有 token，SF 信号精确控制单个 token。同层级 token（包括 IT、SF 和响应 token）也相互可见以保证逻辑连贯性。

**控制信号获取**：
- **CM**：使用三个 RoBERTa 分类器识别 Emotional Reaction (ER)、Interpretation (IP)、Exploration (EX)。
- **IT**：在 EMPATHETICINTENT 数据集上训练 BERT 分类器识别 9 种共情意图。
- **SF**：使用 open-SESAME 模型从 FrameNet 提取 1222 种类别的语义帧标签。

**训练损失**：最小化变分下界（VB利）：
$$\mathcal{L}_{vbl} = \sum_{t=2}^{T} ||\mathbf{y}_0 - \tilde{f}_\theta(\mathbf{x}_t, t)||^2 + ||\text{EMB}(\mathbf{w}^y) - \tilde{f}_\theta(\mathbf{x}_1, 1)||^2 + \mathcal{R}(||\mathbf{x}_0||^2)$$

**推理检索策略**：情感增强匹配，加权计算语义相似度和情感分布余弦相似度：
$$Score = \text{SIM}_{semantic} + \gamma \cdot \text{SIM}_{emotional}, \quad \gamma=0.2$$

## 实验与结果
**数据集**：EMPATHETICDIALOGUE，包含 24,850 轮多轮对话，32 种情绪标签，按 8:1:1 划分。

**基线方法**：
- Transformer 类：TRS, MTRS, MoEL, MIME, EmpDG, CEM
- 预训练模型类：TransferTransfo (GPT-2), BART
- 扩散模型类：DiffuSeq

**评估指标**：相关性（BERTScore, MIScore）、可控性（ACC-CM, ACC-IT, F1-SF）、信息量（Dist-1/2/4, Self-BLEU）、响应长度（AvgLen）。

**核心结果**：
| 方法 | ACC-CM | ACC-IT | F1-SF | Dist-1 | Dist-2 | Self-BLEU↓ | AvgLen |
|------|--------|--------|-------|--------|--------|------------|--------|
| DiffuSeq | 59.23 | 28.58 | 17.26 | 1.79 | 26.97 | 1.29 | 10.30 |
| **DIFFUSEMP** | **92.36** | **84.24** | **52.79** | **2.84** | **29.25** | **1.09** | **14.12** |
| Oracle | 92.38 | 83.66 | 51.95 | 2.84 | 30.46 | 1.11 | 14.01 |
| Human | 100.00 | 100.00 | 98.40 | 19.49 | 43.55 | 0.85 | 13.04 |

**最强结果**：DIFFUSEMP 在可控性指标上大幅领先——ACC-CM 提升 33.13%，ACC-IT 提升 55.66%，F1-SF 提升 35.53%；信息量方面 Dist-1 达到 2.84，Self-BLEU 降至 1.09；Human 评估 empathy=3.68、relevance=3.39、informativeness=4.63，三项均为最高。

**消融结论**：
- 去除掩码（w/o Mask）导致 IT 准确率从 84.24 骤降至 73.80，证明掩码对句子级控制至关重要。
- 去除 SF 使 F1-SF 从 52.79 降至 22.48，同时 MIScore 从 626.92 升至 993.44，说明词元级控制对信息密度和上下文相关性均有显著贡献。

## 相关工作脉络
1. **EMPATHETICDIALOGUE 基准**（Rashkin et al., 2019）：首次构建共情对话数据集并提出任务定义，本文在此基础上进行可控生成研究。
2. **CoMAE**（Zheng et al., 2021）：在话语层面建模 CM/DA/EM 关系，但统一作用于所有解码位置；DIFFUSEMP 进一步细化至句子和词元级，并通过掩码实现差异化控制。
3. **DiffuSeq**（Gong et al., 2022）：条件扩散 seq2seq 模型，采用部分去噪但无控制信号机制；本文在此基础上引入多粒度控制与掩码策略。
4. **Diffusion-LM**（Li et al., 2022b）：通过额外训练分类器实现可控生成，但仅支持粗粒度属性控制；本文直接控制响应内容而非通过后置分类器。
5. **MIME/C EM 等 Transformer 基线**：依赖 MLE 训练，生成结果单调；本文通过扩散模型的分布建模能力提升多样性和信息量。
6. **EmpatheticIntent/FrameNet 标注体系**：本文复用既有共情意图和语义帧标注资源，扩展至生成任务的显式控制应用。

## 局限性与未来方向
1. **控制信号标注依赖预训练工具**：原数据集缺乏多粒度标注，依赖 RoBERTa/BERT/open-SESAME 等工具的准确性（Table 1 显示 SF 类别达 1222，F1 仅 86.55%），制约了可控性上限。
2. **推理阶段检索质量瓶颈**：控制信号来源于上下文匹配检索，当前情感增强匹配方法相对简单，未能充分利用 SOTA 对话响应选择模型。
3. **扩散模型计算开销**：2000 步去噪导致推理速度慢、GPU 资源消耗大，限制实际应用部署。
4. **未来方向**：①引入更多共情控制信号类型；②替换更强大的响应选择模型和标注工具；③探索扩散模型加速采样方法（如 DPM-Solver、latent diffusion）；④推广至其他可控文本生成任务。

## 研究启发与可借鉴点
1. **多粒度控制信号分层设计范式**：可将 CM/IT/SF 的分层思路迁移至其他可控文本生成任务（如情绪控制、风格迁移、知识注入），实现从粗到细的精细化引导。
2. **控制范围掩码机制的可复用性**：该掩码策略本质上是一种结构化的注意力约束，可推广至需要属性-内容对齐的控制生成场景（如属性控制翻译、条件摘要）。
3. **情感增强匹配检索的通用设计**：语义相似度+情感一致性加权检索策略可直接复用于其他需要上下文-响应匹配的对话系统。
4. **部分去噪条件扩散框架的扩展**：本文的 partial noising + 掩码架构为 Seq2Seq 扩散模型提供了可扩展模板，可与 Latent Diffusion、DDIM 加速等方法结合。
5. **Oracle 上界的启示**：Oracle 设置表明当前性能瓶颈主要在信号获取而非模型架构，提示后续工作应优先优化标注工具和检索模块。

## 关键术语表
**Diffusion Model（扩散模型）**：通过前向加噪和反向去噪过程学习数据分布的生成模型，已在图像生成中取得突破，本文首次将其系统应用于可控对话生成。
**Control-Range Masking（控制范围掩码）**：在 Transformer 自注意力中设计的结构化掩码，限定每类控制信号仅能影响其对应粒度范围内的响应词元。
**Communication Mechanism (CM)**：话语层面的共情分类，包含 Emotional Reaction、Interpretation、Exploration 三种机制，描述高层共情功能。
**Intent (IT)**：句子层面的共情意图标签（如 Acknowledging、Consoling、Questioning），反映说话者在特定句子中的行为目标。
**Semantic Frame (SF)**：词元层面的语义帧标签，基于 FrameNet 定义，表示每个词在通用事件/概念/关系范畴中的抽象类别。
**Partial Noising（部分去噪）**：仅对响应部分施加噪声而保持上下文和控制信号不变的扩散策略，适用于条件 Seq2Seq 生成。
**Emotion-Enhanced Matching（情感增强匹配）**：结合语义余弦相似度和情感分布相似度的检索打分策略，用于推理阶段获取控制信号来源。
**Oracle Setting**：使用真实标注控制信号作为输入的评测设置，代表方法性能的理论上限。

## 可复现要素
- **数据集**：EMPATHETICDIALOGUE（公开，CC-BY 4.0 许可）
- **代码**：论文未提及代码开源
- **权重**：论文未提及权重公开
- **关键超参**：扩散步数 $T=2000$，学习率 $1\times10^{-4}$，batch size=128，dropout=0.1，$\gamma=0.2$，最大输入长度=128，嵌入维度=128，AdamW 优化器
