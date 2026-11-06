---
title: "Towards-Understanding-Chain-of-Thought-Prompting-An-Empirica"
source: https://aclanthology.org/2023.acl-long.153.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 22:26:35"
field: "大语言模型推理与提示工程"
keywords: ["Chain-of-Thought", "in-context learning", "ablation study", "reasoning evaluation", "LLM prompting"]
innovations: ["发现 CoT 示范推理的无效性对性能影响有限（保留80-90%）", "形式化提出relevance和coherence两个关键概念并系统性消融", "通过跨模型对比揭示LLM过度依赖预训练先验知识的现象"]
benchmarks: ["GSM8K", "Bamboogle"]
---

# 论文速读：Towards-Understanding-Chain-of-Thought-Prompting-An-Empirica

## 一句话总结
本文通过系统的消融实验，实证研究了 CoT prompting 中各要素的作用：发现即使示范示例中的推理步骤完全无效，模型仍能达到 CoT 性能 80-90%，真正关键的是推理步骤与查询的**相关性（relevance）**和步骤间的**一致性/顺序性（coherence）**。

## 研究问题与动机
- **核心问题**：CoT prompting 为何有效？已证明其能显著提升 LLM 多步推理能力，但对其内部机制缺乏系统理解——究竟示范示例的哪些方面对性能贡献最大？
- **现有认知盲区**：传统直觉认为示范示例的逻辑正确性（validity）是 CoT 有效的前提；若示范推理错误，模型应无法正确推理。然而 Min et al. (2022) 和 Webson & Pavlick (2022) 等研究表明 ICL 中标签/指令的正确性影响有限，这启发作者深入探究 CoT 的内在机制。
- **评估局限性**：已有工作主要依赖最终答案正确率（extrinsic evaluation），可能高估/低估实际推理质量；本文同时引入 bridging objects 的 Recall/F1 作为 intrinsic evaluation 指标，更忠实于推理过程质量。

## 核心贡献（创新点）
- **发现推理有效性并非 CoT 性能的决定因素**：即使示范示例的推理步骤完全逻辑无效（invalid reasoning），LLM 仍能达到 CoT 性能的 80-90%（inter. recall 保留超 90%），并能生成连贯合理的推理链。
- **形式化并提出"相关性"与"一致性"两个关键概念**：将 CoT 示例拆分为 bridging objects（关键对象/数值/实体）和 language templates（文本模板/谓词），并定义 relevance（与查询的信息关联）和 coherence（步骤间逻辑顺序）两个维度，为 CoT 分析提供了可操作的概念框架。
- **实证揭示了 relevance 比 coherence 更关键**：移除 relevance 导致性能断崖式下降（GSM8K Answer Acc. 从 48.5 降至 11.0），而仅破坏 coherence（尤其 bridging objects 的 coherence）影响相对较小。
- **揭示了 LLM 过度依赖预训练先验知识的风险**：Flan-PaLM 等强先验模型对所有消融几乎免疫，表明其推理能力主要来自预训练而非从示范中"学习"，这可能使模型忽略上下文中稀有但关键的任务语义信息。

## 方法详解
- **示例分解**：每个 CoT rationale 被分解为两个组件——**Bridging objects**（推理中必须推导的关键对象，算术推理为数值/方程，事实 QA 为主体/客体实体）和**Language templates**（衔接 bridging objects 的文本模板/谓词/关系描述）。
- **消融设计**：基于 CoT prompt 构造 7 组消融设置（①-⑦），分别破坏不同维度的属性：
  - **① Invalid Reasoning**：保持前提步骤（query 信息的复述），后续步骤全部改写为无逻辑关联的无效推理，但保留数字和实体等 bridging objects。
  - **② No coherence for bridging objects**：随机打乱 bridging objects 的顺序。
  - **③ No relevance for bridging objects**：随机替换 bridging objects 为无关数值/实体。
  - **④ No coherence for language templates**：随机打乱语言模板的顺序。
  - **⑤ No relevance for language templates**：从其他 query 的训练集示例中随机选取模板替换。
  - **⑥ No coherence**：同时打乱两类组件的顺序。
  - **⑦ No relevance**：同时替换两类组件为无关内容。
- **评估体系**：
  - **Extrinsic**：GSM8K 的 Answer Accuracy；Bamboogle 的 Answer F1。
  - **Intrinsic**：Bridging objects 的 Inter. Recall / Inter. F1（即推理过程中需推导出的中间对象的召回率与 F1，对 GSM8K 使用标注推理步骤中的衍生数值作为代理）。
- **主干模型**：以 InstructGPT-175B (text-davinci-002) 为主，同时在 text-davinci-003、PaLM、Flan-PaLM 上验证结论的鲁棒性；所有生成采用 greedy decoding（temperature=0）。

## 实验与结果
- **数据集**：GSM8K（算术推理，采样 800 测试样本）、Bamboogle（多跳事实 QA，全部 125 测试样本）。
- **主要结果（text-davinci-002）**：
  - **CoT**：GSM8K Inter. Recall 43.9 / Inter. F1 48.3 / Answer Acc. 48.5；Bamboogle Inter. Recall 45.2 / Answer F1 45.2。
  - **① Invalid Reasoning**：GSM8K Inter. F1 43.9（达 CoT 的 **91%**），Answer Acc. 39.5（达 CoT 的 **81.5%**）；Bamboogle Inter. Recall 44.4（达 CoT 的 **98.2%**）。
  - **⑦ No relevance（最严重消融）**：GSM8K Answer Acc. 11.0（**低于 STD 的 15.4**），Bamboogle Answer F1 23.9（接近 STD 的 20.6）。
  - **关键对比**：仅破坏 bridging objects 的 coherence（② Inter. F1 39.2）优于仅破坏其 relevance（③ Inter. F1 26.2）；而语言模板的 coherence 破坏（④ Inter. F1 28.3）影响远大于 bridging objects 的 coherence。
- **更强模型的补充结果**：
  - text-davinci-003：invalid reasoning 下 Inter. F1 53.5（甚至**超过** CoT 的 53.1）；⑦ No relevance 下 GSM8K Answer Acc. 16.2（仍低于 STD 的 15.2 附近）。
  - Flan-PaLM：所有消融对性能几乎无影响（Inter. F1 始终 ~72），进一步印证强先验模型从示范中"学习"的成分极少。
- **错误分析**：CoT 正确但 invalid reasoning 错误的 20 个样本，与反向情况的错误类型分布高度相似（计算错误 20%/20%，遗漏一步 35%/25%，语义理解 45%/55%），说明无效示范不会系统性改变错误模式。

## 相关工作脉络
- **Wei et al. (2022) CoT 原论文**：提出 Chain-of-Thought prompting 并在 GSM8K 等任务上验证有效性，但未分析示范示例各要素的具体贡献。本文定位为其背后的机制理解工作。
- **Madaan & Yazdanbakhsh (2022) Text and Patterns**：通过 counterfactual 扰动研究 CoT 各组件重要性，但其扰动方式较温和（仅改错计算或实体），且常因 query 同步调整而"恢复正确性"，与本文大幅破坏推理有效性的设计有本质区别。
- **Min et al. (2022) Rethinking the Role of Demonstrations**：发现 ICL 中 ground truth label 对分类任务影响有限，格式和分布更关键。本文将这一思路扩展到序列生成式的多步推理任务。
- **Webson & Pavlick (2022)**：发现 instruction model 在 irrelevant/misleading instructions 下仅轻微降级。与本文共同指向"LLM 不完全依赖示范语义"的现象。
- **Ye et al. (2022) Complementary Explanations**：同期工作，通过损坏 bridging objects/templates 研究 CoT，但消融设计与本文不同（仅 drop 而非完全替换/打乱），未覆盖 relevance/coherence 的系统性分解。
- **Saparov & He (2023)**：基于一阶逻辑合成 QA 数据集，对 CoT 进行形式化符号分析。本文与之互补：前者提供形式化理论视角，后者提供实证驱动的概念框架。

## 局限性与未来方向
- **任务覆盖有限**：仅在小语种/算术推理和多跳 QA 上实验，符号推理类任务（如 Last Letter Concatenation、Coin Flip）因步骤高度模板化、语言模板变化极小，消融设计难以适用。
- **无效推理的人工构造**：§4 的 invalid reasoning 示例为人工编写，自动合成仍具挑战；未来计划基于非形式逻辑谬误分类系统化处理。
- **Intrinsic 评估的局限**：基于 bridging objects 的 Recall/F1 虽与 language template 质量高度相关（经验验证），但需要人工标注 ground truth bridging objects，成本高且通用性受限；未来需探索无参考的推理质量评估方法。
- **单_run 实验**：所有结果仅报告单次运行，缺少误差棒统计，可能存在随机性偏差。

## 研究启发与可借鉴点
- **消融实验设计的精细化**：将 CoT rationale 分解为"组件×属性"的矩阵（bridging objects/language templates × relevance/coherence），并逐一正交消融，是分析 LLM prompt 机制的有效范式，可迁移至其他 prompt 技术研究（如 ReAct、Self-Consistency 等）。
- **Intrinsic vs. Extrinsic 联合评估**：用 bridging objects 的 Recall/F1 作为推理质量的内在指标，补充最终答案正确率的不足，为 CoT 类研究的评估体系提供了新标准。
- **"先验知识 vs. 上下文学习"的定量估测思路**：通过对比不同能力模型（text-davinci-002 / 003 / Flan-PaLM）对同一消融的敏感度差异，可间接量化模型的预训练先验程度，为 benchmark 设计提供新思路。
- **低资源场景下的 prompt 设计启示**：既然 relevance 是关键，那么在构建 few-shot 示例时，确保示例与 query 的主题/实体/数值高度对齐，可能比追求示例推理的"完美逻辑"更重要，这对节省示例标注成本有实际指导意义。
- **团队可结合方向**：可探索"在示例推理中故意引入噪声/误导性信息，但保持 relevance 和 coherence"的边界情况，进一步刻画 LLM 对示范内容的利用阈值。

## 关键术语表
- **Chain-of-Thought (CoT) Prompting**：通过在 few-shot 示范示例中加入中间推理步骤（rationale），引导 LLM 在输出最终答案前显式生成逐步推理过程的提示技术。
- **Bridging Objects**：CoT 推理链中必须推导出的关键中间对象（算术推理为数值/方程，事实 QA 为主体/客体实体），是连接前提与结论的"桥接"要素。
- **Language Templates**：承载推理关系和谓词的文本表达（如"So in total they had...", "After eating..."），引导模型按特定句式推导 bridging objects。
- **Relevance**：推理组件（bridging objects 或 language templates）与原始 query 之间的信息关联程度；具体表现为是否使用 query 中的实体/数值，或是否围绕 query 的主题展开。
- **Coherence**：推理步骤之间的逻辑顺序一致性；要求后续步骤可由前面步骤推导，且不存在因果倒置或前后矛盾的排列。
- **Intrinsic Evaluation**：不依赖最终答案正确性，而是评估生成的推理过程本身的质量（如 bridging objects 的 Recall/F1）。
- **Extrinsic Evaluation**：通过最终答案的正确率来间接评估推理质量的外部评估方式。

## 可复现要素
- **数据集**：GSM8K（Cobbe et al., 2021，公开）、Bamboogle（Press et al., 2022，公开）。
- **代码**：论文附录中包含所有 prompt 的完整列表（Table 9-24），但未提供独立开源代码仓库；示例数据可基于原始 CoT 和 QA 示例手工构造。
- **权重**：使用 InstructGPT-175B (text-davinci-002) 等商业模型的 API 接口，权重不公开。
- **关键超参**：greedy decoding（temperature=0）；示范示例数量未明确说明（基于原始 CoT 论文示例集）；GSM8K 采样 800 测试样本。
