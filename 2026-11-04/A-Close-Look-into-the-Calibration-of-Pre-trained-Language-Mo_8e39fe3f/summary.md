---
title: "A-Close-Look-into-the-Calibration-of-Pre-trained-Language-Mo"
source: https://aclanthology.org/2023.acl-long.75.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:05:21"
---

# 论文速读：A-Close-Look-into-the-Calibration-of-Pre-trained-Language-Mo

## 一句话总结
本文系统考察了预训练语言模型（PLMs）在校准（Calibration）过程中的训练动态，发现PLMs并不会随训练自然学会校准，而是持续推高对错误预测的置信度；同时证明引入额外正确性判别任务的“可学习校准”方法能显著缓解过度自信，而单纯扩大模型规模或依赖预训练无法根本解决问题。

## 研究问题与动机
1. **PLMs是否在训练中自动学会校准？** 现有工作多基于固定训练与评估范式做静态 Benchmark，缺乏对训练全生命周期校准动态的细粒度观测。
2. **现有校准方法在实际应用中效果如何？** 传统不可学习方法（如温度缩放、标签平滑）可能仅做置信度正则化，未能真正教会模型识别自身错误；可学习方法的潜力未被充分剖析。
3. **既有结论是否存在评估偏差？** “越大模型越校准”“预训练提升校准”等共识可能源于评估数据集性能未饱和（欠拟合态）或未监测错误预测置信度（$\mathrm{CErr}_{neg}$），结论具有片面性。
4. **分布外（OOD）校准退化是否普适？** 需结合数据集难度相对关系与 Hard-to-Easy 偏移场景重新审视原有结论的边界条件。

## 核心贡献（创新点）
1. **提出训练动态的“欠拟合/过拟合”两态框架**：通过控制数据集难度、样本数、训练步数、可调参数量、模型规模与预训练六个因子，揭示PLMs在过拟合态下性能 plateau 而置信度持续攀升，导致 ECE 恶化与 $\mathrm{CErr}_{neg}$ 单调上升。
2. **反驳“大模型/预训练天然更校准”的既有结论**：证明这两类策略本质上只是增加模型容量；当评测集性能未饱和时观测到的是欠拟合假象，性能饱和后缩放仅加剧对错误预测的过度自信。
3. **系统验证可学习校准方法的优越性**：对比不可学习与可学习（外置/内置）方法，表明后者能通过正确性判别任务显著压低 $\mathrm{CErr}_{neg}$，而非像前者那样“一刀切”地同步压低正确与错误置信度。
4. **重新界定自我不确定性表达的成功机理**：对比相同架构下的外置（E-T5）与内置（I-Iter/I-Simul）校准器发现性能相当，指出该范式的核心在于校准任务的“可学习性”，而非PLMs的“自检能力”。
5. **修正OOD校准退化的普适性认知**：通过 Hard-to-Easy 偏移与任务无关输入实验证明，PLMs在OOD上的校准表现高度依赖ID/OOD数据集的难度梯度，原有退化结论仅在特定评测设置下成立。

## 方法详解
- **评估指标设计**：除 Accuracy、Average Confidence 与 ECE（100 bins, equal-mass）外，提出应用驱动指标：
  - $\mathrm{CErr}_{pos} = 1 - \mathrm{Conf}_{pos}$：衡量因低置信度错误拒绝正确预测的损失；
  - $\mathrm{CErr}_{neg} = \mathrm{Conf}_{neg}$：衡量因高置信度错误接受错误预测的风险。
- **六因子控制实验**：对 RoBERTa-base 与 T5-base，分别在 SST-2、MNLI、AG-News、Yahoo 上调节 K-shot 样本数、训练步数、Soft-prompt/Adapter 参数量、模型 Scale（Tiny~3B）与 Pretrained vs Random 初始化，绘制校准曲线。
- **校准方法分类与实现**：
  - *不可学习*：Vanilla、Temperature Scaling (TS)、Label Smoothing (LS)、EDA、Deep Ensemble。
  - *可学习*：利用验证集构建校准数据 $\mathbb{D}^* = \{(x_i, y_i^*, c_i)\}$（$c_i$ 表示原预测是否正确，类别平衡采样）。外置采用 MLP（E-MLP）或 T5（E-T5）；内置提出 I-Vanilla（持续训练）、I-Iter（交替训练）与 I-Simul（联合训练）以解决主任务性能退化问题。校准任务模板示例：`"<original input>, the model’s prediction is <prediction>, is the prediction True or False? It’s <mask>."`，以 `True` 类概率作为校准后置信度。
- **涌现校准（Emergent Calibration）分析**：独立研究校准任务的数据集规模（Small/Middle/Large）与校准器模型规模对最终校准性能的影响，刻画可学习方法的scaling行为。

## 实验与结果
- **数据集与基线**：涵盖情感分析（SST-2/Amazon/SST-5/SemEval/Dynasent）、NLI（MNLI/HANS/ANLI）、主题分类（Yahoo/AG-News）、有毒检测（Civil/Hate Speech/Implicit Hate）及任务无关输入（BookCorpus/Random Words），共15+数据集；对比5类不可学习与5类可学习/扩展方法。
- **ID/OOD 核心结果**：在标准分布偏移下，Vanilla T5 在 ANLI 上 Acc 仅 30.53% 但 Conf 高达 93.77%。可学习方法显著降低 $\mathrm{CErr}_{neg}$（如 MNLI 上从 ~84 降至 45–56），有效缓解过度自信；不可学习方法（TS/LS）虽改善 ECE，但 $\mathrm{CErr}_{pos}$ 与 $\mathrm{CErr}_{neg}$ 降幅相近，属粗略正则。
- **数据量与规模效应**：校准数据集从小增至中等时，各可学习方法呈现一致的**阶跃式性能提升**；校准器规模扩大（T5-small→base→large）在 ID 上进一步降低 $\mathrm{CErr}_{pos}$，但整体 Conf 仍随容量上升；OOD 表现受多因素干扰无明显单调趋势。
- **Hard-to-Easy 与任务无关输入**：当 ID 更难（Dynasent）而 OOD 更简单（Amazon/DSC）时，ECE 与 $\mathrm{CErr}_{pos}$ 反而下降，推翻“分布偏移必降校准”的单一结论；对 BookCorpus/Random Words 输入，可学习方法能给出更低的不合理高置信度。
- **最强结果**：在多项 ID/OOD 设置下，**I-Simul 与 E-T5** 在保持主任务 Acc 基本不变的前提下，将 $\mathrm{CErr}_{neg}$ 压降至各方法最低区间（如 Amazon 上降至 ~3.4–5.6，原 Vanilla 为 ~73–90），校准收益最为显著。

## 相关工作脉络
1. **Guo et al. (2017) / Minderer et al. (2021)**：奠定神经网络校准度量与大规模静态基准；本文通过细粒度控制实验填补训练动态视角空白。
2. **Desai & Durrett (2020)**：提出 PLMs 在 OOD 上校准退化结论；本文通过 Hard-to-Easy 偏移与难度敏感度分析指出该结论依赖评测设置，需分情况讨论。
3. **Hendrycks et al. (2019b)
