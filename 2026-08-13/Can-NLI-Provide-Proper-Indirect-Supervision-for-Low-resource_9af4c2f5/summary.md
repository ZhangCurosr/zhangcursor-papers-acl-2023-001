---
title: "Can-NLI-Provide-Proper-Indirect-Supervision-for-Low-resource"
source: https://aclanthology.org/2023.acl-long.138.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:49:36"
---

# 论文速读：Can-NLI-Provide-Proper-Indirect-Supervision-for-Low-resource

## 一句话总结
本文提出NBR框架，将生物医学关系抽取（RE）重构为自然语言推理（NLI）任务，通过关系标签的自然语言表述与排名对比损失，在标注稀缺的低资源场景下利用通用NLI知识提供间接监督，显著提升模型泛化能力与不确定样本拒答能力。

## 研究问题与动机
- 生物医学关系抽取高度依赖专家标注，导致训练数据稀缺且覆盖不全，现有监督方法在低资源设定下泛化性能急剧下降。
- 传统多分类方法将关系视为离散整数索引，无法捕捉关系语义，且在未见实例上倾向于盲目猜测已知关系，缺乏选择性拒答能力。
- 现有NLI间接监督研究多聚焦通用领域，跨领域迁移（通用NLI→生物医学）的有效性及机制尚未验证。
- 生物医学数据中大量“无关系（abstinent）”实例造成决策边界模糊，亟需细粒度的实例级置信度校准机制。

## 核心贡献（创新点）
- 首次将通用领域NLI的间接监督引入生物医学关系抽取，通过语义化假设替代离散标签，与仅依赖分类logits的传统方法形成本质差异。
- 证明即使存在领域鸿沟，通用NLI语料（MNLI/SNLI）微调仍能为生物医学下游任务提供强效间接监督信号，拓展了跨域知识迁移的认知边界。
- 提出融合InfoNCE对比损失与margin-based拒答校准（$\mathcal{L}_{\text{AC}}$）的联合训练目标，隐式学习实例级决策边界并指导不确定样本拒答。
- 设计显式拒答检测器（EAD）作为可选后处理模块，将“有无关系”与“具体关系”预测解耦，降低单一模型的多重优化负担。

## 方法详解
- **任务重构与模板化**：将输入句子作为前提（premise），每个候选关系标签$y \in \mathcal{Y} \cup \{\perp\}$通过模板转化为自然语言假设$\nu(y)$。实体提及替换为类型掩码（如@CHEMICAL$、@GENE$）以规避罕见医学术语的低质量表征。
- **蕴含分数计算**：使用Transformer类NLI模型$ f_{\text{NLI}} $对拼接后的前提-假设序列编码，输出蕴含 logits $s(y) = f_{\text{NLI}}(\mathbf{x}[\text{SEP}]\nu(y))$。
- **InfoNCE对比损失**：对每个正样本采样$n$个负类关系，优化正确关系的蕴含分数高于所有负类：
  $\mathcal{L}_{\text{NCE}} = \sum_{(\mathbf{x}, y) \in \mathcal{D}} -\ln \frac{\exp(s(y)/\tau)}{\exp(s(y)/\tau) + \sum_{i=1}^{n} \exp(s(y_i)/\tau)}$，温度$\tau$控制难负样本聚焦程度。
- **拒答校准损失（$\mathcal{L}_{\text{AC}}$）**：针对数据中大量的$\perp$实例引入margin排序正则项，当$y \neq \perp$时压制$s(\perp)$，当$y = \perp$时提升$s(\perp)$排名，使模型隐式学习实例感知的拒答阈值。总损失为$\mathcal{L}_{\text{NCE}} + \lambda \mathcal{L}_{\text{AC}}$。
- **推理与EAD集成**：推理时选择蕴含分数最高的关系输出。可选地接入独立训练的EAD模块：仅在EAD预测为$\perp$时接受拒答，否则信任NBR预测。

## 实验与结果
- **数据集与指标**：在BLURB基准的ChemProt（化学-蛋白）、DDI（药物相互作用）、GAD（基因-疾病）三个句子级RE数据集上评测，采用非拒答实例的micro F1。
- **全量数据SOTA**：$\mathrm{NBR}_{\mathrm{NLI+FT}}+\mathrm{EAD}$ 在ChemProt、DDI、GAD上分别达到81.10、85.14、85.86 F1，较直接监督最强基线（BioLinkBERT-large）分别提升1.10、1.79、0.96个点。
- **低资源显著提升**：在8-shot ChemProt上较直接监督基线提升高达34.25个F1点；在0/8/50-shot及1%/10%划分下，所有NBR变体均稳定优于分类基线，验证了间接监督在小样本下的优势。
- **消融结论**：移除InfoNCE或$\mathcal{L}_{\text{AC}}$均导致性能下滑；在通用NLI（MNLI+SNLI）上微调优于在生物医学MedNLI上微调；描述型模板（Descriptive Template）效果最佳。

## 相关工作脉络
- 生物医学RE主流工作（BioBERT、PubMed-BERT、BioLink-BERT等）采用直接监督分类，将关系编码为整数索引，缺乏语义交互建模与拒答机制。
- 间接监督早期方案将RE转化为机器阅读理解或多轮问答，依赖生成式目标，而非基于蕴含的语义匹配，难以直接利用NLI预训练先验。
- 现有NLI间接监督研究（如LITE、Sainz et al. 2021）聚焦通用领域实体类型抽取与关系抽取，未验证跨领域泛化性，也未处理生物医学中特有的高比例$\perp$实例。
- 本文定位：填补NLI间接监督在生物医学低资源场景的应用空白，确立“通用NLI语义先验+领域预训练底座”的迁移范式。

## 局限性与未来方向
- 推理成本随标签数量线性增长（需对每个假设独立前向传播），难以部署于实时或资源受限环境。
- 性能对关系描述模板高度敏感，当前依赖人工设计，缺乏自动化构建与客观评估机制。
- 依赖生物医学领域预训练底座（如BioLinkBERT），带来较高的计算资源门槛。
- 未来方向包括：基于prompt learning自动优化关系模板、探索其他任务的间接监督形式、研究低开销推理策略。

## 研究启发与可借鉴点
- **标签语义化重构**：将离散分类标签转化为自然语言假设，可有效注入预训练模型的语义先验，适用于任何具有明确语义定义的分类任务。
- **隐式拒答校准设计**：通过对比学习与margin约束联合优化，可在不增加显式二分类头的前提下实现实例级不确定性估计，值得迁移至Open-Vocabulary或长尾分类场景。
- **跨领域NLI微调范式**：在通用NLI语料上二次微调领域底座模型，能以较低成本激活蕴含推理能力，可作为领域自适应的标准流程。
- **模块化拒答检测器（EAD）**：将“有无关系”与“具体关系”预测解耦，通过简单启发式规则集成，可有效提升高风险领域模型的可靠性与可解释性。

## 关键术语表
- **Indirect Supervision（间接监督）**：将目标任务的训练/推理流程重构为资源丰富的源任务形式，从而借用源任务监督信号增强目标任务。
- **NLI（Natural Language Inference，自然语言推理）**：判断给定前提（premise）是否能蕴含（entail）、矛盾（contradict）或中立（neutral）于假设（hypothesis）的任务。
- **Abstinent Instances（拒答实例）**：数据集中实体之间不存在预设关系类别的样本，通常以$\perp$标记。
- **Verbalization（标签词化/表述化）**：将离散关系标签转换为自然语言句子的过程，此处通过模板生成假设。
- **InfoNCE Loss**：一种对比学习损失，通过softmax将正样本得分相对于负样本得分最大化。
- **EAD（Explicit Abstention Detector，显式拒答检测器）**：独立训练的辅助模块，专门用于判断实例是否为无关系样本。

## 可复现要素
- 数据集：ChemProt、DDI、GAD（BLURB基准公开，论文未提供私有数据集链接）。
- 代码/权重：论文未明确声明代码开源情况；使用HuggingFace公开的BioLinkBERT-large等权重。
- 关键超参：优化器AdamW，学习率1e-5，margin γ=0.7，温度τ=0.01，校准强度λ∈[0.001, 10] Sweep，训练300 epoch，单卡Quadro RTX 8000 GPU。

<!--META
{"keywords": ["Biomedical Relation Extraction", "Indirect Supervision", "Natural Language Inference", "Low-resource Learning", "Abstention Calibration", "Contrastive Learning"], "field": "生物医学自然语言处理 / 低资源关系抽取", "innovations": ["将生物医学关系抽取重构为NLI任务以实现间接监督", "提出InfoNCE与隐式拒答校准联合损失", "证明通用NLI知识可跨领域迁移至生物医学任务"],
