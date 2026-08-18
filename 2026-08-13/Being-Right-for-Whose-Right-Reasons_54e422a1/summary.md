---
title: "Being-Right-for-Whose-Right-Reasons"
source: https://aclanthology.org/2023.acl-long.59.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:47:35"
field: "自然语言处理中的公平性与可解释性"
keywords: ["explainability", "fairness", "rationale alignment", "demographic bias", "model interpretability", "NLP fairness"]
innovations: ["构建首个人口统计学增强的 NLP rationale 标注语料库，覆盖6组年龄×族裔人群", "提出 group-model rationale 对齐度量框架，揭示性能平权下推理依据的群体偏差", "发现模型规模与跨人群 rationale 一致性呈负相关，蒸馏无法系统性改善公平性"]
benchmarks: ["DynaSent", "SST-2", "CoS-E"]
---

# 论文速读：Being Right for Whose Right Reasons?

## 一句话总结
本文首次构建了带有详细人口统计学信息（年龄、族裔）标注的 NLP 人工语料库，系统评估了 16 种 Transformer 模型在不同人群下的推理依据（rationale）一致性，发现模型在相同性能水平下对**年长者与白人的推理模式**显著更友好；同时发现模型规模增大与蒸馏压缩均不能改善公平性。

## 研究问题与动机
1. 可解释性方法常以"right for the right reasons"为评价标准，但现有工作默认"rationale（理由依据）"是客观统一的，忽视了什么是合理的推理依据在不同人群之间可能存在主观差异。
2. 公平性研究通常以预测性能的 group-level parity 来定义公平性，但存在性能平权的模型仍可能在推理逻辑上对不同人群不公——目前学界缺乏对"推理依据对齐"层面的公平性度量。
3. 年龄和族裔是已被证实影响 NLP 系统表现的两个关键属性（年龄相关的语法差异、AAVE 变体语言等），值得在解释性层面做跨人群对比分析。
4. 模型规模和知识蒸馏已被证明可提升性能，但是否能同时提升跨人群的解释公平性？现有证据不足，本研究提出实证检验。

## 核心贡献（创新点）
1. **首个人口统计学增强的语料库**：构建并公开了三个已有数据集（DynaSent、SST-2、CoS-E）的细粒度标注，每个标注覆盖六个年龄×族裔人群组合，为公平性与可解释性交叉研究提供了新资源。
2. **跨人群的 rationale 对齐度量框架**：将模型生成的 rationale 与不同人群的人工标注 rationale 进行对比，通过 token-level F₁ 和 IOU-F₁ 两种指标系统度量 group-model 一致性与 group-group 差异，填补了公平性指标中"理由公平"的空白。
3. **模型规模与 rationale 一致性的负相关发现**：与普遍预期相反，研究发现模型参数量与跨人群 rationale 对齐度呈显著负相关，且更大的模型并不比小模型更公平（min-max gap 与模型规模无关）。
4. **知识蒸馏未能改善群体间 rationale 公平性**：尽管蒸馏模型整体对齐分更高，但绝大多数情况下并未缩小群体间的 performance gap，仅 minilm-l6-h384-uncased 在两项指标上同时优于平均。
5. **揭示"性能平权≠推理平权"**：证明了即使组间预测性能相近，模型在推理依据层面仍可能存在系统性偏差，对公平性评估框架提出了必要补充。

## 方法详解
### 数据与标注流程
- **三个数据集**：DynaSent（情感分类，480 个实例×6 组）、SST-2（情感分类，263 个实例×6 组）、CoS-E（常识推理，500 个实例×6 组）。
- **六个人群组**：由族裔 {Black/African American, White/Caucasian, Latino/Hispanic} × 年龄 {Young (<36), Old (>37)} 的笛卡尔积生成：BO、BY、LO、LY、WO、WY。
- **标注任务设计**：让标注者自主选标签并自行框选支持词（extractive rationale），而非给定 label 后解释——避免先入为主地偏向某一人群的直觉。
- **排除与质量控制**：族裔自报与招募不符者、注意力检查失败者、重复参与者被剔除；所有参与者均等报酬，平均 7.1£/小时。
- 完全标签对齐（full label agreement）的实例数：DynaSent 209、SST-2 152、CoS-E 161。

### 模型与可解释性方法
- 微调 16 种 Transformer 模型（BERT、RoBERTa、DistilBERT、MiniLM 系列等），统一超参数、训练 3 个 epoch、选取验证集最高准确率的 checkpoint。
- 两种 post-hoc 解释方法：
  - **Attention Rollout (AR)**：通过累积 attention 近似各输入 token 的相对重要性。
  - **Layer-wise Relevance Propagation (LRP)**：梯度回传方法，从输出层向输入层传播"相关性"，产生更稀疏的归因分数。
- **Rationale 二值化**：将连续分数向量转为 top-kᵍᵈ 的二进制向量，其中 kᵍᵈ = 句子长度 × 该人群在该数据集上的平均 rationale 长度比例（RLR）。RLR 均值：SST-2（29.6%）、DynaSent（31.9%）、CoS-E（33.0%）。

### 评估指标
- **token-level F₁**（公式 1）：逐实例计算模型 rationale 与人工标注的精确率和召回率，取均值。
- **IOU-F₁**（公式 2–3）：基于交并比的二值化 F₁，IOU ≥ 0.5 计为 1。
- **公平性定义**：以各模型的 min-max token-F₁ gap（最高分人群与最低分人群之差）衡量，gap 越小表示越公平。
- **Spearman 相关系数 ρ** 用于量化模型规模与 token-F₁ 的相关趋势。

## 实验与结果
### Group-Group 对齐分析
- 全标签一致实例上，White 组（WO、WY）的 rationale 平均与其他组最接近；Black Young（BY）与其他组的差异最大，尤其在情感任务中。
- CoS-E 作为高主观性任务，跨组差异相对较小；DynaSent 对齐度最高，CoS-E 最低（即使标签一致）。

### Group-Model 对齐分析
- **主要发现**：在情感分析任务（DynaSent、SST-2）上，所有族裔组的模型 rationale 均与**年长者**的对齐度高于年轻人（SST-2 的 White Young 为唯一例外）。
- 模型规模与 token-F₁ 呈**负相关**：Spearman ρ 在 DynaSent、SST-2 和 CoS-E（AR 方法）上均显著为负；仅在 CoS-E + LRP 时为显著正相关。
- **min-max gap 与模型规模无关**：更大模型并未缩小群体间差距，没有证据表明增大模型参数有助于公平性。

### 知识蒸馏实验（Table 2）
| 模型 | token-F₁ | IOU-F₁ | min-max token-F₁ | min-max IOU-F₁ |
|------|----------|--------|-------------------|-----------------|
| minilm-l6-h384-unc. | .31 | .28 | .045 | .068 |
| minilm-112-h384-unc. | .27 | .21 | .045 | .083 |
| distilbert-base-unc. | .29 | .24 | .064 | .100 |
| distilroberta-base | .36 | .36 | .065 | .069 |
| **16 模型平均** | **.29** | **.24** | **.054** | **.081** |

- 蒸馏模型整体 token-F₁/IOU-F₁ 略高，但仅 minilm-l6-h384-uncased 在两项 gap 指标上同时优于平均（.045/.068 vs .054/.081）；其余蒸馏模型 gap 更大或持平，无证据表明蒸馏能改善公平性。

### 最强结果
- DistilRoBERTa 在所有组别中取得最高平均 token-F₁（.36）和 IOU-F₁（.36），但 min-max gap 为 .065/.069，并非最公平。
- minilm-l6-h384-uncased 是唯一在"性能+公平性"双维度上均优于平均的蒸馏模型。

## 相关工作脉络
1. **McCoy et al. (2019) "Right for the Wrong Reasons"**：首次提出用人类 rationale 检验模型是否"理由正确"，但将该标准视为统一的人类共识，未考虑人群差异——本文的核心扩展。
2. **DeYoung et al. (2019) ERASER 基准**：定义 token-F₁ 和 IOU-F₁ 等离散 rationale 评估指标，本文直接采用该框架。
3. **Zhang et al. (2021) Sociolectal Analysis**：证明语言模型在词预测任务中与年长/白人用户更好地对齐，本文将其结论推广至"推理依据对齐"层面。
4. **Balkir et al. (2022) TrustNLP 工作**：指出将可解释性用于提升公平性充满挑战，本文提供了首个面向 rationale 对齐的系统性实证。
5. **Sap et al. (2019) 种族偏见研究**：揭示了 hate speech 检测中的种族偏见，但仅关注预测输出，本文说明偏见可存在于解释层即便输出性能平权。
6. **Ali et al. (2022) XAI for Transformers**：比较 AR 与 LRP 在 Transformer 上的行为，本文据此选择两种方法进行对比验证。

## 局限性与未来方向
1. 仅分析非自回归 Transformer 模型，未覆盖 GPT 类等自回归架构的公平性。
2. 所有模型使用统一超参数微调，未做超参数搜索，可能影响部分模型的最终性能。
3. AR 与 LRP 提取的 rationale 为连续值，通过 top-kᵍᵈ 阈值转为二值化，主观性较强（论文引用 Jørgensen et al. (2022) 的讨论）。
4. 族裔类别仅覆盖 Prolific 平台提供的三个选项，未包含更多元身份，排除了跨种族多重身份认同的复杂性。
5. 未来方向：探索能同时优化性能与跨人群 rationale 公平性的方法；将分析扩展到更多语言和文化背景；研究其他可解释性方法（如 perturbation-based faithfulness）是否也存在类似偏差。

## 研究启发与可借鉴点
1. **multi-dimensional fairness 评估框架**：将"性能公平性"与"推理公平性"分离评估的思路，可为本团队在中文场景下的公平性诊断提供方法迁移路径。
2. **人口统计学增强的标注协议**：自主选标签+自主框 rationale 的设计避免了先入为主的 label bias，适用于任何需要跨人群评估的 NLP 任务。
3. **min-max gap 作为公平性代理指标**：用最高/最低分人群之间的绝对差距衡量 group-fairness，简洁且可解释，适合纳入模型的常规评估 pipeline。
4. **模型规模≠公平性保障**：提示后续研究不应盲目通过扩大模型规模来提升公平性，而应关注训练数据多样性与专门的公平性正则化。
5. **蒸馏作为潜在的公平性缓解手段（有限）**：尽管整体效果不显著，但 minilm-l6-h384 的成功提示特定蒸馏配置可能对某些人群更友好，值得进一步搜索。

## 关键术语表
- **Rationale（推理依据）**：模型或人类做出预测时所依赖的证据片段，通常为文本中支持结论的关键词/token。
- **Right for the Right Reasons**：模型不仅预测正确，且其推理依据与人类公认理由一致的高质量解释标准。
- **Token-level F₁**：以词级别为粒度的精确率-召回率调和均值，用于衡量模型 rationale 与人工标注 rationale 的覆盖率。
- **IOU-F₁**：基于交并比（Intersection-over-Union）的 binary F₁，将模型与人工的 rationale 集合转化为二值向量后计算。
- **Min-max Gap**：组间公平性度量，取各人群得分最高与最低之差，gap 越小表示跨人群越公平。
- **Attention Rollout (AR)**：一种基于累积 attention 机制的 Transformer 可解释方法，近似输入 token 对输出的相对重要性。
- **Layer-wise Relevance Propagation (LRP)**：通过梯度回传将输出层的"相关性"逐层传播到输入的归因方法，产生更稀疏的解释分数。
- **Socio-demographic Group**：由年龄和族裔等社会人口统计属性定义的子群体，本文定义为六个组合。

## 可复现要素
- **数据集**：DynaSent Round 2（测试集）、SST-2（Zuco 子集）、CoS-E（test set）——均为公开数据集；新增人口统计学增强标注已公开。
- **代码/权重**：论文注明公开了所有标注数据（包括被排除的原始数据）；模型微调代码基于原有公开库（footnote 6、8）。
- **关键超参**：训练 3 个 epoch，fine-tune，standard hyperparameter values for sentiment analysis（DeYoung et al. 2019）；验证集准确率最高 checkpoint 用于测试；top-kᵍᵈ 取 k = 句子长度 × RLR。
- **可解释性实现**：AR 和 LRP 均遵循 Ali et al. (2022) 规则；tokenization 与输入格式均已归一化处理。
