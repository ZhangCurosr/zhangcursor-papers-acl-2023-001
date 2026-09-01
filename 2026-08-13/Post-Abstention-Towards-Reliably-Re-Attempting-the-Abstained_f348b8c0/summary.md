---
title: "Post-Abstention-Towards-Reliably-Re-Attempting-the-Abstained"
source: https://aclanthology.org/2023.acl-long.55.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:57:19"
---

# 论文速读：Post-Abstention-Towards-Reliably-Re-Attempting-the-Abstained

## 一句话总结
本文提出了“后弃权（Post-Abstention）”任务，旨在对选择性预测系统拒绝回答的低置信度样本进行重尝试，以在不显著牺牲准确性的前提下提升系统覆盖率，并系统评估了基于 Paraphrase Ensembling、辅助分类模型（REToP）及人工干预的三种基线方法。

## 研究问题与动机
- **核心问题**：选择性预测（Selective Prediction）虽能通过拒答低置信度样本来维持高准确率，但会直接牺牲系统覆盖率，那么“弃权之后该如何处理这些被拒答的实例”？
- **现有方法不足**：
  1. 传统选择性预测研究仅聚焦于“何时拒答”的阈值控制，缺乏对拒答样本的后续处理机制。
  2. 若强制让原模型对所有样本作答，错误率将急剧上升，破坏系统可靠性。
  3. 缺乏针对“后弃权”阶段的统一数学形式化与公平评估指标，难以横向对比不同补救策略。

## 核心贡献（创新点）
1. **形式化定义 Post-Abstention 任务**：给出严格的数学表述与基于 Risk-Coverage 曲线的评估方法论。与以往选择性预测工作仅优化单一阈值曲线不同，本文聚焦于弃权后的二次决策阶段，填补了可靠性研究链条的缺口。
2. **提出 REToP 方法**：通过构建 (context, question, prediction) 三元组并训练辅助二分类器，学习区分 QA 模型正确/错误预测的细粒度表示。与 Calibraton 等方法相比，REToP 无需额外留出集（held-out dataset），且针对每个候选答案独立打分而非输出单点校准分数。
3. **系统探索并验证多种基线**：在 11 个 QA 数据集（含域内与域外）上全面评测了 Paraphrase Ensembling、REToP 与 Human Intervention，揭示了不同方法的覆盖重叠性、置信度分布规律及组合潜力，为后续可靠性研究提供了可复用的基准框架。

## 方法详解
- **任务形式化与评估**：给定选择性预测系统 $(f, g)$ 在阈值 $th$ 下的覆盖率 $cov_{th}$ 与风险 $risk_{th}$。后弃权方法对原本拒答实例重新预测，得到 $cov'_{th}$ 与 $risk'_{th}$。评估时，从原系统的 Risk-Coverage 曲线上读取 $cov'_{th}$ 对应的风险值，计算差值 $\Delta = risk(th, cov'_{th}) - risk'_{th}$。$\Delta > 0$ 表示方法有效，整体性能为所有阈值下 $\Delta$ 的聚合值（Total Risk Improvement）。
- **Ensembling using Question Paraphrases**：使用 BART-large 生成 10 个语义等价改写，分别输入原 QA 模型。聚合策略包括：
  - *Mean*：对每个候选答案取各改写版本置信度的平均值，超过阈值则输出最高均分候选。
  - *Max*：取最高置信度而非平均，旨在更易突破阈值，但会显著增加风险。
- **Re-Examining Top N Predictions (REToP)**：
  - *辅助模型训练*：用 QA 模型在训练集上生成 Top N 预测，构建 (上下文, 问题, 预测) 三元组，将正确预测标为 1、错误预测标为 0。刻意选用 Top N 错误预测作为负样本，以提供高信息量的 hard negatives。
  - *推理阶段*：对弃权实例的 Top N 候选，计算综合置信度 $c_p = \alpha \cdot s_q^p + (1-\alpha) \cdot s_a^p$（$s_q$ 为原模型概率，$s_a$ 为辅助模型预测正确性的 softmax 概率），选取 $c_p$ 最高者，超过阈值则输出。
- **Human Intervention (HI)**：
  - *全量多预测*：对弃权实例返回 $n$ 个候选答案供人工选择（覆盖率强制为 100%）。
  - *REToP-centric*：仅当 REToP 的综合置信度超阈值时才返回多预测，其余继续弃权。

## 实验与结果
- **数据集**：SQuAD 1.1（域内），以及 NewsQA、TriviaQA、SearchQA、HotpotQA、Natural Questions、DROP、DuoRC、RACE、RelationExtraction、TextbookQA（共 10 个域外数据集，均来自 MRQA shared task）。
- **实验设置**：使用 BERT-mini（11.3M 参数）配合 MaxProb 置信度估计；Ensembling 生成 10 个改写；REToP 考察 Top 10 预测，$\alpha \in [0.3, 0.7]$；HI 返回预测数 $n \in [2, 5]$。
- **主要结果**：
  - REToP 在域内 SQuAD 上取得 Total Risk Improvement 21.81，域外 TextbookQA 达 24.23，HotpotQA 达 21.54，整体表现最稳定。
  - Paraphrase Ensembling（Mean 策略）在多数数据集有效，但在 DuoRC 和 TBQA 等域外数据出现负提升（-1.69, -6.93）。
  - HI（n=2）数值最高（如 SQuAD 达 47.85），但因输出多预测，结果不直接与单预测方法可比。
  - 中等置信度阈值下 REToP 提升更显著（低阈值剩余样本难度过高）；不同方法的覆盖集合存在重叠（非互斥），提示可顺序组合使用。
- **最强结果**：REToP 在 TextbookQA 上实现 24.23 的风险改进，综合鲁棒性与稳定性最佳。

## 相关工作脉络
1. **Selective Prediction 方法**：如 Monte-Carlo Dropout、Calibration（Kamath et al., 2020; Varshney et al., 2022c）、Error Regularization（Xin et al., 2021）。本文定位：这些工作解决“事前过滤”，本文解决“事后补救”，两者正交可叠加。
2. **输入扰动与鲁棒性研究**：Jia & Liang (2017)、Belinkov & Bisk (2018) 指出 NLP 模型对微小语义保持扰动高度敏感。本文借鉴该现象，利用 Paraphrase Ensembling 稳定弃权样本的预测分布。
3. **Question Rewriting for QA**：Aliannejadi et al. (2021)、Anantha et al. (2021) 探索对话式问题重写以提升检索/问答性能。本文将其迁移至后弃权阶段，作为置信度提升的被动手段而非主动改题。
4. **Test-time Adaptation**：Chen et al. (2022)、Wang et al. (2021) 在测试时动态适配模型。本文指出该方法可作为未来 Post-Abstention 的潜在增强路径。
5. **Cascading Systems**：Varshney & Baral (2022)、Li et al. (2021) 研究模型级联推理。本文提出串行应用多个后弃权方法或更强模型是自然的延伸方向。

## 局限性与未来方向
- **局限性**：
  1. 引入额外计算开销（paraphrase 生成、辅助模型前向推理）与存储需求，虽对现代硬件可接受，但仍为额外负担。
  2. 人工干预（HI）依赖真人参与，难以完全自动化落地。
  3. 无法精确预知特定场景下后弃权方法能带来的具体提升幅度。
- **未来方向**：
  1. 探索 test-time adaptation、知识检索（knowledge hunting）及澄清问句生成作为新型后弃权手段。
  2. 研究多种后弃权方法的组合或串行应用（如先 REToP 过滤高置信补偿，再 Ensembling 覆盖剩余难例）。
  3. 将本文形式化框架与 REToP 范式推广至图像分类、语音识别等其他 NLP/CV 任务及更强基础模型。

## 研究启发与可借鉴点
1. **模块化可靠性设计**：将系统可靠性拆分为“事前过滤+事后补救”两阶段，提供了可扩展的架构思路，可迁移至任何带置信度输出的监督学习任务。
2. **Hard Negative 构造技巧**：REToP 刻意使用模型自身的 Top-N 错误预测作为负样本训练辅助分类器，有效避免了easy negatives，该策略可复用至其他模型纠错/重排序模块。
3. **Fair Evaluation Metric**：基于 Risk-Coverage 曲线在等覆盖率点上进行 pairwise risk 比较并聚合，比单纯使用 AUC 更能
