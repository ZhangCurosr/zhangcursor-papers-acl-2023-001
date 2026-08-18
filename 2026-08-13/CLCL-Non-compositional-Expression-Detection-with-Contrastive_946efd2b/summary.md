---
title: "CLCL-Non-compositional-Expression-Detection-with-Contrastive"
source: https://aclanthology.org/2023.acl-long.43.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:49:09"
field: "非组合性表达处理"
keywords: ["非组合性表达", "习语用法识别", "隐喻检测", "对比学习", "课程学习", "表示学习", "低资源NLP"]
innovations: ["首次联合对比学习与动态课程学习处理非组合性表达识别", "用对比目标动态度量样本难度并重排训练顺序", "在6个数据集上统一提升习语和隐喻检测性能"]
benchmarks: ["MAGPIE", "SemEval5B", "VNC", "VUA-18", "VUA-verb", "MOH-X"]
---

# 论文速读：CLCL-Non-compositional-Expression-Detection-with-Contrastive

## 一句话总结
本文提出 CLCL 框架，将对比学习（Contrastive Learning）与课程学习（Curriculum Learning）相结合，用于非组合性表达（习语和隐喻）的用法识别任务，有效解决了低资源场景下"如何学习好的表示"和"如何利用稀缺训练数据"双重挑战。

## 研究问题与动机
- 非组合性表达（如习语、隐喻）的意义无法从其组成部分推导，且同一表达可被字面或比喻使用，区分两种用法需精细的上下文感知表示。
- 现有方法主要聚焦设计复杂架构，忽视了在有限资源下如何建模非组合性表示，同时训练数据稀缺问题也未得到充分解决。
- 对比学习可通过将相同表达的不同用法表示拉开距离来提升表示质量；课程学习可让模型从易到难逐步学习，提升数据利用效率。
- 将两者结合可用于：用对比目标度量样本难度并动态调度训练顺序，首次统一处理习语用法识别与隐喻检测两个任务。

## 核心贡献（创新点）
- **首次联合结合对比学习与课程学习**：用对比目标衡量每个训练样本难度，并动态排序训练顺序，与传统静态课程学习形成本质区别。
- **统一处理习语用法识别与隐喻检测**：两类非组合性表达共享同一框架，相较以往仅聚焦单一任务的方案更具通用性。
- **动态课程调度策略**：每个 epoch 后根据模型当前能力重新计算难度并重新排序，优于固定顺序的课程学习。
- **在多个基准上取得最优**：在 6 个数据集（3 个习语 + 3 个隐喻）上超越此前 SOTA，且在 Type-based（未见表达）设置下提升尤为显著。
- **跨任务迁移分析**：揭示了隐喻→习语可正向迁移，而习语→隐喻难以迁移的现象，为非组合性表示学习提供了新视角。

## 方法详解
- **骨干模型**：以预训练的 RoBERTa Base（125M 参数）为特征提取器，句子表示经分类头预测字面/比喻标签。
- **三元组构造**：对每条训练样本 $Y_i$（锚点），随机采样同表达且同用法（同为字面或同为比喻）的 $Y_i^+$ 作为正样本，采样同表达但用法不同的 $Y_i^-$ 作为负样本，形成 triplet $<Y_i, Y_i^+, Y_i^->$。
- **对比损失**：
$$\mathcal{L}_{cts} = -\sum_{Y \in \mathcal{Y}} \log \frac{f(\boldsymbol{x}_i, \boldsymbol{x}_i^+)}{f(\boldsymbol{x}_i, \boldsymbol{x}_i^+) + f(\boldsymbol{x}_i, \boldsymbol{x}_i^-)}$$
其中 $f(\cdot,\cdot)$ 为相似度/距离函数，目的是拉近相同用法的表示、推远不同用法的表示。
- **分类损失**：$\mathcal{L}_{cls}$ 为标准交叉熵损失，基于真实标签（字面/比喻）。
- **总损失**：$\mathcal{L} = \mathcal{L}_{cts} + \mathcal{L}_{cls}$。
- **难度度量**：用对比目标值衡量样本难度，$d_\mathbf{M}(Y_i) = \frac{f(\boldsymbol{x}_i, \boldsymbol{x}_i^+)}{f(\boldsymbol{x}_i, \boldsymbol{x}_i^+) + f(\boldsymbol{x}_i, \boldsymbol{x}_i^-)}$；值越小表示越"容易"（正样本更近）。
- **动态调度**：每 epoch 训练结束后，重新计算所有样本难度，仅保留难度发生变化的样本重新排序，形成下一 epoch 的训练顺序，实现"由易到难"的动态课程。
- **Algorithm 1（CLCL 流程）**：初始化三元组 → 计算初始难度排序 → 每个 epoch 训练 → 更新难度并过滤变化样本 → 重新排序 → 循环直至结束。

## 实验与结果
- **数据集**：
  - 习语：MAGPIE、SemEval5B、VNC；每个数据集均报告 Random 和 Type-based 两种划分。
  - 隐喻：VUA-18、VUA-verb、MOH-X；使用官方 train/dev/test 划分。
- **评估指标**：习语任务用 Accuracy、F1、F1-fig（比喻为正类）；隐喻任务用 Accuracy、Precision、Recall、F1（比喻为正类）。
- **基线**：
  - 习语：vanilla RoBERTa、DISC（SOTA）。
  - 隐喻：vanilla RoBERTa、MelBERT、MisNet（SOTA）。
- **主要结果**：
  - **习语（MAGPIE Random）**：Acc 96.75（+1.72 vs vanilla）、F1-fig 97.82（+1.12）、F1 96.75（+3.24）；F1-fig 比 DISC 高 2.8。
  - **习语（MAGPIE Type-based）**：Acc 95.36（+2.50）、F1-fig 97.05（+2.26）、F1 94.20（+2.47）。
  - **习语（SemEval5B Random）**：F1-fig 96.56，比 DISC 高 0.76。
  - **习语（SemEval5B Type-based）**：F1-fig 92.65，比 DISC 高 **33.83**（显著提升）。
  - **习语（VNC Random）**：F1-fig 98.07，比 DISC 高 1.1。
  - **习语（VNC Type-based）**：F1-fig 96.16，比 DISC 高 7.14。
  - **隐喻（VUA-verb）**：Acc 84.7（+4.0 vs MelBERT）、F1 74.4（+3.4）；比 MisNet 高 5.6 Recall、2.0 F1。
  - **隐喻（MOH-X）**：Acc 84.3（+2.7 vs MelBERT）、F1 83.4（+2.3）。
- **消融结论**：
  - 去掉 CL 或 CTS 均导致性能下降，Type-based 下下降更明显。
  - CL 与 CTS 相互互补，联合使用效果最优。
- **t-SNE 可视化**：CLCL 在所有设置下均能将字面与比喻表示清晰分离，而仅 fine-tune 或仅加 CTS 存在误聚类。
- **跨任务迁移**：隐喻→习语迁移效果优于习语→隐喻；CLCL 在两方向均优于 MelBERT。

## 相关工作脉络
- **习语用法识别**：早期依赖规则/词汇特征（canonical form），中期引入 word embedding 和 RNN/CNN，近年转向 RoBERTa 等预训练模型；但多数工作聚焦架构复杂度，忽视低资源下的表示建模（本文填补该空白）。
- **隐喻检测**：早期使用 linguistic features（imageability、supersenses），中期用 CNN/LSTM，近期用 RoBERTa 并结合 POS/句法特征（MelBERT）；本文仅用纯语言模型即取得竞争力结果，无需额外特征。
- **对比学习在 NLP 中的应用**：已从 word/sentence embedding（CLEAR、DeCLUTR、CERT）扩展到预训练模型微调，本文首次用于非组合性表达的 sense-specific 表示学习。
- **课程学习**：从 Bengio et al. (2009) 提出，已在计算机视觉广泛验证；NLP 中主要用于神经机器翻译，本文首次将其引入习语/隐喻任务。
- **动态课程调度**：传统课程学习固定难度顺序，本文根据对比目标动态更新，更适应模型训练过程。
- **跨任务迁移**：本文发现隐喻知识可迁移至习语任务，但反向迁移困难，这一现象在以往工作中未被系统探讨。

## 局限性与未来方向
- **调度粒度较粗**：仅在每个 epoch 结束后重排训练样本，无法像 step-level 调度那样灵活适应训练过程。
- **跨任务迁移不对称**：习语→隐喻的迁移效果较差，说明当前框架对两类任务共享的非组合性抽象表示还不够强。
- **缺乏步级动态更新机制**：论文未探索 batch/step 级别的难度重估，可能存在更优的训练效率。
- **未尝试更大规模预训练模型**：当前使用 RoBERTa Base，RoBERTa Large 或 DeBERTa 等模型可能进一步提升性能。

## 研究启发与可借鉴点
- **对比学习构造三元组的天然优势**：同一表达的不同用法可直接作为正负样本，无需额外标注，这一思路可迁移到词义消歧、多义词表示学习等任务。
- **用对比目标度量课程难度**：将对比损失直接作为样本难度指标是一种新颖且自洽的做法，可推广到其他需要区分细粒度语义的任务。
- **Type-based 设置下的显著提升**：提示非组合性表达泛化能力的评估应重点关注未见表达，这一实验设定值得在其他表示学习任务中沿用。
- **跨任务迁移分析的价值**：本文揭示了隐喻→习语的可迁移性，后续可探索如何增强双向迁移，构建更通用的非组合性表达表示。
- **无需额外特征的纯语言模型方案**：CLCL 仅靠 RoBERTa 即超越需 POS/句法特征的基线，说明改进训练目标比增加特征更有效，这一理念可应用于其他低资源 NLP 任务。

## 关键术语表
- **非组合性表达（Non-compositional Expression）**：整体意义无法由其组成部分线性组合推导的语言表达，如习语和隐喻。
- **习语用法识别（Idiom Usage Recognition）**：判断给定语境中习语是被字面使用还是比喻使用的二分类任务。
- **隐喻检测（Metaphor Detection）**：识别句子中某词是否被用作隐喻的二分类任务。
- **对比学习（Contrastive Learning）**：通过在嵌入空间中拉近相似样本、推远不同样本来学习表示的自监督/半监督方法。
- **课程学习（Curriculum Learning）**：按从易到难的顺序安排训练样本，以提升模型学习效果的方法。
- **Type-based Split**：按表达类型（而非句子）划分训练/测试集，用于评估模型对未见表达的泛化能力。
- **F1-fig**：以比喻用法为正类的 F1 分数，用于衡量模型对非组合性用法的识别能力。
- **Contextualized Representation**：基于上下文生成的词/句子表示，区别于静态词向量。

## 可复现要素
- **代码**：已开源，地址 https://github.com/zhjjn/CLCL.git
- **数据集**：MAGPIE、SemEval5B、VNC、VUA-18、VUA-verb、MOH-X 均为公开数据集
- **基础模型**：RoBERTa Base（125M 参数）
- **超参**：batch size=32（习语）/16（隐喻），学习率=1e-5，训练 epoch=30，Adam 优化器，max length=128，5 次随机种子重复取均值
- **硬件**：双 NVIDIA V100 GPU（16GB）
- **实现框架**：Huggingface Transformers + PyTorch
