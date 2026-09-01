---
title: "Plan-and-Solve-Prompting-Improving-Zero-Shot-Chain-of-Though"
source: https://aclanthology.org/2023.acl-long.147.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:56:41"
field: "大语言模型提示工程与推理"
keywords: ["Chain-of-Thought", "Prompting", "Zero-shot Reasoning", "Plan-and-Solve", "Large Language Models", "Mathematical Reasoning"]
innovations: ["提出 PS/PS+ 双阶段提示框架，通过显式计划生成减少步骤缺失和计算错误", "零样本 PS+ 性能接近 8-shot Manual-CoT，证明优质提示可替代人工示例"]
benchmarks: ["GSM8K", "SVAMP", "MultiArith", "AQuA", "CommonsenseQA", "StrategyQA", "Last Letter", "Coin Flip"]
---

# 论文速读：Plan-and-Solve-Prompting-Improving-Zero-Shot-Chain-of-Though

## 一句话总结
本文提出 Plan-and-Solve (PS) Prompting 和 PS+ Prompting，通过在零样本提示中引导大语言模型先制定解题计划再逐步执行，显著减少推理过程中的计算错误和步骤缺失，使零样本推理性能达到甚至接近 8-shot 人工 CoT 水平。

## 研究问题与动机
- **核心问题**：Zero-shot-CoT（仅添加"Let's think step by step"）在复杂多步推理任务中仍存在三类错误——计算错误（7%）、缺失步骤错误（12%）、语义理解错误（27%），如何在不依赖人工示例的情况下提升零样本推理质量？
- **现有方法不足**：Few-shot CoT 需要大量人工编写推理演示，成本高且可移植性差；Zero-shot-CoT 虽免去了人工成本，但推理过程缺乏结构化引导，容易遗漏中间步骤或产生计算失误；Program-of-Thought (PoT) 需调用代码解释器，依赖代码预训练能力，泛化受限。

## 核心贡献（创新点）
- **提出 PS Prompting 框架**：用"先制定计划再执行计划"的双阶段指令替代简单的"step by step"，本质区别在于通过显式分解任务结构来减少步骤缺失错误，而非仅鼓励逐步推理。
- **设计 PS+ 精细化提示策略**：在 PS 基础上增加"提取相关变量与数值""计算中间结果并注意常识"等指令，显著降低计算错误，这是通过指令细化直接干预推理质量的新路径。
- **系统性误差分析**：对 GSM8K 错误样本进行三类错误分类（计算/缺失步骤/语义理解），并建立变量定义、计划存在性与错误类型之间的相关性分析，为提示设计提供实证依据。
- **零样本超越部分少样本基线**：PS+ 在多个算术推理数据集上的平均准确率（76.7%）接近 8-shot Manual-CoT（77.6%），且优于 Auto-CoT（75.9%），证明优质提示可部分替代人工示例。

## 方法详解
- **双阶段流程**：
  - **Step 1（推理生成）**：将问题输入与触发指令拼接，LLM 输出包含变量提取、计划制定、逐步执行的推理文本及最终答案。
  - **Step 2（答案抽取）**：将推理文本与抽取指令拼接，要求 LLM 以阿拉伯数字格式输出最终答案。
- **PS Prompting 核心提示模板**：
  > "Q: [问题]. A: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."
- **PS+ Prompting 扩展指令**：在 PS 基础上增加三处细化：
  1. "extract relevant variables and their corresponding numerals"——强制显式提取数值变量，减少信息遗漏；
  2. "calculate intermediate results"——要求输出中间计算结果，便于验证；
  3. "(pay attention to calculation and commonsense)"——提醒注意计算准确性与常识一致性。
- **解码策略**：默认使用贪心解码（temperature=0），配合 Self-Consistency（N=10, temperature=0.7）可进一步提升性能。
- **提示设计原则**：通过控制变量定义、计划结构和计算步骤的存在性，引导 LLM 生成更完整的推理链，实验表明这些结构化元素与错误率呈负相关。

## 实验与结果
- **数据集**：10 个 benchmark，涵盖三类推理任务：
  - 算术推理（6 个）：GSM8K、SVAMP、MultiArith、AddSub、AQuA、SingleEq
  - 常识推理（2 个）：CommonsenseQA、StrategyQA
  - 符号推理（2 个）：Last Letter Concatenation、Coin Flip
- **模型**：GPT-3（text-davinci-003，175B），temperature=0（贪心解码）
- **主要结果（算术推理准确率）**：

| 方法 | MultiArith | GSM8K | AddSub | AQuA | SingleEq | SVAMP | Average |
|------|-----------|-------|--------|------|----------|-------|---------|
| Zero-shot-CoT | 83.8 | 56.4 | 85.3 | 38.9 | 88.1 | 69.9 | 70.4 |
| Zero-shot-PoT | 92.2 | 57.0 | 85.1 | 43.9 | 91.7 | 70.8 | 73.5 |
| **PS (ours)** | 87.2 | 58.2 | 88.1 | 42.5 | 89.2 | 72.0 | 72.9 |
| **PS+ (ours)** | **91.8** | **59.3** | **92.2** | **46.0** | **94.7** | **75.7** | **76.7** |
| Few-shot Manual-CoT | 93.6 | 58.4 | 91.6 | 48.4 | 93.5 | 80.3 | 77.6 |
| Few-shot Auto-CoT | 95.5 | 57.1 | 90.8 | 41.7 | 92.1 | 78.1 | 75.9 |

- **核心结论**：
  - PS+ 在所有算术数据集上大幅超越 Zero-shot-CoT（提升幅度 2.9%~8.0%），平均提升 6.3 个百分点；
  - PS+ 在 5/6 算术数据集上超越 Zero-shot-PoT，仅在 GSM8K 上略低；
  - PS+ 平均准确率 76.7% 与 8-shot Manual-CoT（77.6%）相当，且优于 Auto-CoT（75.9%）；
  - 常识推理：PS+ 在 CSQA（71.9% vs 65.2%）和 StrategyQA（65.4% vs 63.8%）上显著优于 Zero-shot-CoT；
  - 符号推理：PS+ 在 Last Letters（75.2%）上超越 Manual-CoT（70.6%），在 Coin Flip（99.6%）上接近 Manual-CoT（100%）；
  - 配合 Self-Consistency 后，PS+ 在 GSM8K 达到 73.7%（vs 无 SC 的 59.3%），SVAMP 达到 84.4%（vs 75.7%）。
- **误差分析**：PS+ 将计算错误从 7% 降至 5%，缺失步骤错误从 12% 降至 7%，语义理解错误保持 27% 不变。

## 相关工作脉络
- **Wei et al. (2022b) Few-shot CoT**：通过人工编写的多步推理示例激发 LLM 推理能力；本文定位为零样本替代方案，无需人工示例即可达到相近性能。
- **Kojima et al. (2022) Zero-shot-CoT**：仅添加"Let's think step by step"；本文在此基础上引入结构化计划机制，针对性解决步骤缺失问题。
- **Chen et al. (2022) Program-of-Thought (PoT)**：生成 Python 程序并执行以获得答案；本文无需代码执行，适用面更广，且在多数算术数据集上可比或超越 PoT。
- **Zhang et al. (2022) Auto-CoT**：自动聚类选择示例并生成推理链；本文完全零样本，避免了对数据集分布的依赖，且在算术推理上优于 Auto-CoT。
- **Wang et al. (2022b) Self-Consistency**：通过多次采样投票提升稳定性；本文与 SC 正交，结合后可进一步提升性能。
- **Zhou et al. (2022) Least-to-Most Prompting**：将问题分解为子问题依次求解；本文与 Least-to-Most 理念相似但实现更轻量，仅需提示词修改而不需多级调用。

## 局限性与未来方向
- **提示工程成本**：PS+/PS 提示需针对不同问题类型精心设计，GPT-3 对提示表达敏感，人工调优存在一定成本；
- **语义理解错误未解决**：PS+ 仅降低了计算错误（7%→5%）和缺失步骤错误（12%→7%），但语义理解错误（27%）保持不变，作者指出这需通过升级 LLM 能力或更深层提示策略解决；
- **未探索非推理任务**：作者提及 PS(+) 提示思想可扩展至非推理任务，但本文未进行验证；
- **计划质量未量化**：90% 的 PS 预测包含计划，但计划本身的合理性与最终答案正确性之间的关系尚待系统研究。

## 研究启发与可借鉴点
- **结构化计划引导是有效的零样本推理增强手段**：将"先规划后执行"的思想嵌入提示词，比单纯鼓励"逐步思考"更能减少步骤遗漏，该思路可迁移至代码生成、多步决策等需要结构化推理的场景；
- **指令细化策略具有通用性**："提取变量""计算中间结果""注意常识"等细化指令不仅适用于数学推理，也可推广至科学计算、法律推理、医学诊断等需要中间验证的领域；
- **误差分析与提示设计的闭环验证**：本文通过错误分类→相关性分析→提示迭代的设计路径，为后续研究提供了可复用的诊断框架，建议团队在提示优化中采用类似的"错误归因→针对性改进"流程；
- **零样本 vs 少样本的效率权衡**：PS+ 在算术推理上接近 8-shot CoT 但无需任何人工示例，提示团队在资源受限时优先考虑零样本结构化提示而非盲目增加示例数量；
- **Self-Consistency 与结构化提示的正交组合**：两者结合可进一步提升精度，建议在实际应用中同时采用"高质量提示+多采样投票"的双重保障策略。

## 关键术语表
- **Zero-shot-CoT**：仅通过"Let's think step by step"等触发语引导 LLM 生成推理步骤的零样本提示方法，无需示例。
- **Plan-and-Solve (PS) Prompting**：本文提出的零样本提示方法，要求 LLM 先制定解题计划再逐步执行，以减少步骤缺失错误。
- **PS+ Prompting**：在 PS 基础上增加变量提取、中间计算、常识注意等细化指令的增强版本，进一步降低计算错误。
- **Program-of-Thought (PoT)**：引导 LLM 生成 Python 程序而非自然语言推理链，通过代码解释器执行获得答案的方法。
- **Self-Consistency (SC)**：通过多次采样生成多个推理路径并取多数投票结果的解码策略，用于降低 LLM 输出的随机性。
- **Missing-Step Error**：推理过程中遗漏必要中间步骤导致的错误，PS/PS+ 主要通过结构化计划减少此类错误。
- **GSM8K**：Grade School Math 8K，包含 8000 道小学级数学应用题的算术推理数据集，是评估 LLM 多步推理能力的标准 benchmark。
- **Error Correlation Analysis**：分析生成内容中变量定义、计划结构等元素与三类错误（计算/缺失步骤/语义理解）之间的相关性，用于指导提示设计。

## 可复现要素
- **数据集**：GSM8K（MIT License）、SVAMP（未明确）、MultiArith（未明确）、AddSub（未明确）、AQuA（Apache-2.0）、SingleEq（未明确）、CommonsenseQA（未明确）、StrategyQA（Apache-2.0）、Last Letter（未明确）、Coin Flip（未明确）
- **代码开源**：是，地址 https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting
- **模型**：GPT-3 text-davinci-003（175B，通过 API 访问，非本地部署）
- **关键超参**：temperature=0（贪心解码）；Self-Consistency 时 temperature=0.7，N=10
- **Few-shot 示例数**：MultiArith/GSM8K/AddSub/SingleEq/SVAMP 用 8 个示例，AQuA/Last Letter 用 4 个，CSQA 用 7 个，StrategyQA 用 6 个
- **评估指标**：准确率（Accuracy），答案通过正则抽取阿拉伯数字或选项字母
