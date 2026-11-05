---
title: "Can-NLI-Provide-Proper-Indirect-Supervision-for-Low-resource"
source: https://aclanthology.org/2023.acl-long.138.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:07:30"
field: "生物医学自然语言处理"
keywords: ["Biomedical Relation Extraction", "Natural Language Inference", "Indirect Supervision", "Low-resource NLP", "Abstention Calibration", "InfoNCE Loss"]
innovations: ["首次将 NLI 间接监督引入生物医学关系抽取，验证跨域知识迁移有效性", "提出 ranking-based Abstention Calibration 损失，隐式处理 abstinent 类别不平衡", "证明通用 NLI 知识可有效迁移至生物医学 RE，低资源场景下边际收益显著"]
benchmarks: ["ChemProt", "DDI", "GAD", "BLURB"]
---

# 论文速读：Can NLI Provide Proper Indirect Supervision for Low-resource Biomedical Relation Extraction?

## 一句话总结
本文提出 NBR（NLI Improved Biomedical Relation Extraction）框架，将生物医学关系抽取重新表述为自然语言推理（NLI）任务，利用通用领域 NLI 知识的间接监督信号，在低资源场景下显著提升关系抽取性能。

## 研究问题与动机
- **标注稀缺与不完美**：生物医学关系抽取高度依赖专家手动标注，成本高昂，导致训练数据稀缺且覆盖不全，传统监督模型泛化能力差。
- **无法选择性拒绝**：现有方法将 RE 视为多分类任务，面对未见过的实例（abstinent instances，即标注为"无关系"的实例）时，模型会强行从已见过关系中猜测，而非主动拒绝，这在高风险医疗场景中尤为危险。
- **现有间接监督局限于通用领域**：已有 NLI 间接监督工作（如 Sainz et al., 2021; Li et al., 2022）主要探索通用领域任务，未研究跨域（通用 NLI → 生物医学 RE）的知识迁移效果。
- **生物医学 NLI 资源匮乏**：现存的生物医学 NLI 数据集（如 MedNLI）规模小、覆盖率低，难以直接支撑任务，需要验证通用 NLI 知识是否足够有效。

## 核心贡献（创新点）
- **首次将 NLI 间接监督引入生物医学 RE**：将关系标签 verbalize 为自然语言假设，利用 NLI 模型语义理解能力缓解标注稀缺问题；与已有工作的区别在于将跨域间接监督验证于生物医学这一特定高风险领域。
- **揭示跨域 NLI 知识的迁移有效性**：实验证明即使存在领域差距，通用领域 NLI 知识仍能为生物医学 RE 提供强间接监督信号；与已有工作的区别在于明确探索了跨域（general domain NLI → biomedical RE）的知识转移可行性。
- **提出基于排名的 Abstention Calibration 损失**：通过 InfoNCE 损失学习关系排序，同时引入 margin-based 正则项隐式校准 abstinent 实例的置信度；与已有工作的区别在于显式处理生物医学 RE 中普遍存在的"无关系"类别不平衡问题。
- **系统性验证低资源场景下的有效性**：在 ChemProt、DDI、GAD 三个基准上，full-shot 场景提升最高达 1.79 点 F1，低资源场景最高提升 34.25 点 F1；与已有工作的区别在于揭示了间接监督在数据极度稀缺时边际收益最大的规律。

## 方法详解
- **RE 重构为 NLI 查询**：给定输入句子 x 和实体 e₁, e₂，将 x 作为 premise，每个候选关系 y 通过模板 verbalize 为自然语言假设 ν(y)，形成 |𝒱|+1 个 NLI 查询。实体提及替换为类型掩码（如 @CHEMICAL$、@GENE$），避免预训练语料中罕见术语的低质量表示。
- **置信度计算**：s(y) = f_NLI(x [SEP] ν(y))，其中 f_NLI 为 Transformer-based NLI 模型，输出 premise 蕴含 hypothesis 的 logit。
- **InfoNCE 排序损失**：对每个正例 y，采样 n 个负例 {y₁,...,yₙ}，优化正例得分高于所有负例：
  L_NCE = Σ -ln(exp(s(y)/τ) / (exp(s(y)/τ) + Σᵢ exp(s(yᵢ)/τ)))，温度 τ 控制对 hard negative 的聚焦。
- **Abstention Calibration 损失**：处理 abstinent 实例类别不平衡问题。若 y = ⊥（无关系），压制 s(⊥) 使其低于其他关系得分；若 y ≠ ⊥，则要求 s(⊥) 低于 s(y)，margin 为 γ。
  L_AC 在两种情况间切换，最终总损失为 L_NCE + λ·L_AC。
- **显式拒绝检测器（EAD，可选）**：额外训练一个 NBR 变体，将关系标签简化为二元"有/无关系"，在 dev 集上确定阈值，仅在 EAD 预测为"无关系"时才由主模型 NBR 输出，否则直接采用 EAD 判断。
- **推理**：对所有关系假设计算蕴含得分，返回得分最高的关系作为预测。

## 实验与结果
- **数据集**：BLURB 基准中的 ChemProt（化学-蛋白相互作用，5 类关系 + ⊥）、DDI（药物-药物相互作用，4 类关系 + ⊥）、GAD（基因-疾病关联，2 类关系，无 ⊥）。评估指标为 micro F1（仅对非 abstinent 实例计算）。
- **基线方法**：BioLink-BERT、PubMed-BERT、Sci-Five、BLUE-BERT、Bio-BERT 等基于分类/生成的监督方法；Sainz et al. (2021)、LITE 等间接监督基线。
- **全量数据结果（Table 1）**：NBR_NLI+FT+EAD 在 ChemProt 达 81.10 F1（较最强直接监督 BioLink-BERT_large 提升 1.10 点）、DDI 达 85.14 F1（+1.79 点）、GAD 达 85.86 F1（+0.96 点），均为 SOTA。
- **低资源结果（Table 2）**：在 8-shot ChemProt 上，NBR_NLI+FT+EAD 获得 51.44 F1，较直接监督基线最高提升 34.25 点；1% 和 10% 设置下 NBR 各变体均稳定超越所有直接监督方法，零样本设置下分类方法完全失效。
- **消融结果（Table 3）**：去除 L_NCE 或 L_AC 均导致性能下降，尤其在 1% 设置下下降更显著（ChemProt 1% 从 60.17 降至 53.87）；使用单一负例排名损失（对应 LITE）进一步恶化（ChemProt 1% 降至 20.71）。用 MedNLI 替代 MNLI+SNLI 交叉域微调反而略有下降，归因于 MedNLI 规模小（35×）且实体覆盖率低（仅 11.77%）。
- **分析结论**：NLI 格式化本身在低资源下比直接监督更有效；生物医学领域知识与 NLI 知识缺一不可，二者结合产生最佳性能。

## 相关工作脉络
- **Sainz et al. (2021)**：首次将 NLI 间接监督用于低资源关系抽取（通用领域），本文将其扩展至生物医学领域并验证跨域迁移效果，同时引入了 abstention 处理。
- **Li et al. (2022) / LITE**：利用 NLI 间接监督进行实体细粒度分类（通用领域），使用 ranking loss sum；本文采用 InfoNCE 损失并在生物医学 RE 中专门处理 abstinent 类别不平衡。
- **Gu et al. (2021) / BLURB 基准**：提出生物医学 NLP 评估基准，本文在其框架下系统对比各类方法，并补充了低资源设置实验。
- **Romanov & Shivade (202 8) / MedNLI**：生物医学 NLI 数据集，本文实验表明其规模不足且覆盖有限，效果不如通用领域大 NLI 数据集（MNLI/SNLI）。
- **Yeh et al. (2022) / BioRE-Prompt**：使用 prompt tuning 进行生物医学 RE，本文从 NLI 视角提供另一种语义建模路径，实验表明 Descriptive Template 优于 Learned Prompt。
- **Peng et al. (2019) / BLUE-BERT**：在 PubMed + MIMIC-III 上预训练的模型，作为直接监督基线之一，本文验证了间接监督在低资源下超越此类直接微调方法的潜力。

## 局限性与未来方向
- **计算开销较高**：推理时需对每个关系执行一次前向传播（共 |𝒱|+1 次），相比标准单次前向的分类方法，训练和推理成本均提高 #label 倍，不利于实时应用和资源受限环境部署。
- **模板设计依赖人工**：当前 verbalizer 模板需人工设计，随关系数量增加，人力成本上升；缺乏客观指标评估模板质量，现有最佳实践（Descriptive Template）仍需人工经验。
- **领域知识门槛**：方法需要生物医学预训练语言模型（如 BioLinkBERT）和通用 NLI 微调，计算资源要求较高，限制了在资源受限场景的应用。
- **未来方向**：探索其他间接监督来源；研究基于 prompt learning 的自动关系模板生成；优化推理效率（如缓存、近似计算）。

## 研究启发与可借鉴点
- **间接监督思路可迁移**：将目标任务重构为另一种具有丰富预训练资源的源任务（NLI），利用间接监督信号弥补标注稀缺，这一范式可推广至其他低资源专业领域（如法律、金融）的关系抽取。
- **Abstention 类别处理技巧**：将"无关系"作为独立标签并用 margin-based 正则显式校准，避免模型过度自信，该方法可适用于任何存在"未知/不适用"类别的分类任务。
- **跨域知识迁移的价值**：即使源域（通用 NLI）与目标域（生物医学 RE）存在领域差距，源域知识仍能提供有效监督；这为"轻量级领域适配"策略提供了实证依据——不一定需要大量领域内 NLI 标注数据。
- **InfoNCE 在排序任务中的优势**：对比实验表明 InfoNCE 比简单 ranking loss sum 更有效，尤其在低资源设置下；可在其他排序/关系建模任务中尝试对比损失。
- **模板设计在低资源下的评估效率**：在 1% 子集上评估模板变体即可快速甄别优劣，避免了全量训练的昂贵代价，值得在模板驱动方法中推广。

## 关键术语表
- **Biomedical Relation Extraction（生物医学关系抽取）**：从生物医学文本中自动识别实体间预定义语义关系（如药物-药物相互作用）的任务。
- **Indirect Supervision（间接监督）**：利用资源丰富的源任务的监督信号来提升资源受限的目标任务性能，通过任务 reformulation 实现跨任务信号迁移。
- **Natural Language Inference（NLI，自然语言推理）**：判断给定 premise 与 hypothesis 之间的逻辑关系（entailment/contradiction/neutral）的任务，模型具备语义推理归纳偏置。
- **Abstinent Instance（缺席实例）**：标注为"无关系"（⊥）的实体对，表示上下文中实体间不存在预定义关系，是生物医学 RE 中常见的类别。
- **Verbalizer（verbalize）**：将离散标签（如关系名）转换为自然语言假设句子的模板函数，使标签携带语义信息供 NLI 模型理解。
- **InfoNCE Loss**：基于对比学习的排序损失，将正例得分推高使其优于所有负例，温度超参控制对 hard negative 的关注程度。
- **Abstention Calibration（拒绝校准）**：通过 margin-based 正则项隐式校准模型对 abstinent 实例的置信度，使模型学会在不确定时主动拒绝预测。
- **Explicit Abstention Detector（EAD，显式拒绝检测器）**：独立训练的二分类 NBR 变体，专门判断"是否存在关系"，与主模型组合以提升拒绝预测精度。

## 可复现要素
- **数据集**：ChemProt、DDI、GAD（来自 BLURB 基准），均为公开可用的生物医学 NLP 基准数据集。
- **代码/权重**：论文未明确声明代码开源，但使用了 HuggingFace 公开模型权重（BioLinkBERT、BioBERT 等），模型 checkpoint 可从 HuggingFace Hub 获取。
- **关键超参**：AdamW 优化器，学习率 1e-5；margin γ=0.7；温度 τ=0.01；calibration 强度 λ 在 {0.001, ..., 10} 范围内 sweep；训练 300 epochs，每 10 epochs 在 dev 集评估并保存最佳 checkpoint。
- **硬件**：Quadro RTX 8000 GPU。
