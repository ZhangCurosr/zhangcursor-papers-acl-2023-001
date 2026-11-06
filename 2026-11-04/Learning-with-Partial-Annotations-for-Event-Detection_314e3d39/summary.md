---
title: "Learning-with-Partial-Annotations-for-Event-Detection"
source: https://aclanthology.org/2023.acl-long.30.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 09:12:46"
field: "信息抽取"
keywords: ["事件检测", "部分标注", "对比学习", "Margin Softmax", "MC-Dropout", "伪负例", "半监督学习"]
innovations: ["提出trigger localization范式，用margin softmax替代硬分类以容忍部分标注噪声", "设计基于MC-Dropout的不确定性引导自校正机制，等价于EM算法的E/M步迭代", "重标注ACE 2005开发/测试集并发布为unbiased benchmark，揭示真实部分标注率约20%"]
benchmarks: ["ACE 2005", "MAVEN", "WikiEvents"]
---

# 论文速读：Learning-with-Partial-Annotations-for-Event-Detection

## 一句话总结
本文首次系统研究了事件检测（ED）中的部分标注（partial annotation）问题，提出了一种基于对比学习的**trigger localization**范式配合基于MC-Dropout的不确定性引导自校正机制，即使在训练集中超过90%的事件未被标注时，模型仍可在ACE 2005上达到F1>60%。此外，作者重新标注了ACE 2005的开发集和测试集并发布为公平基准。

## 研究问题与动机
- **真实数据存在大量隐性负例（false negatives）**：ACE 2005中近20%的事件未被标注，将部分标注数据当作完全标注数据训练会导致模型误学。
- **现有方法假设完整标注**：主流ED方法均依赖fully labeled训练集，缺乏对部分标注噪声的系统性建模与容忍机制。
- **当前无方法论层面的解决方案**：既有工作仅做错误分析或指出问题，未提出真正的学习框架；同时，在有部分标注的训练集上评估会造成不公平的对比。
- **标注缺失具有可识别模式**：重新标注后发现，未标注词多为light verbs（20.7%）、rare words（25.2%）、co-reference触发词（3%）等难以人工识别的类型，说明模型有能力补充这些遗漏。

## 核心贡献（创新点）
- **Trigger Localization范式**：将事件检测从硬分类重构为以事件类型Prompt引导的句子级trigger定位任务；与已有工作（如BERTQA、OneIE）的区别在于目标不是追求绝对最高F1，而是从根本上解决部分标注带来的噪声容忍问题。
- **Margin Softmax含正/负边距+高斯正则**：引入正边距$\lambda_+$拉大真触发的分数，负边距$\lambda_-$缓解"无事件句子"中隐藏负例的惩罚，高斯距离正则进一步处理多触发分布稀疏性；与标准softmax分类的本质区别是引入可调节的置信边界而非硬阈值。
- **不确定性引导自校正（MC-Dropout + EM视角）**：通过多次Dropout前向得到预测分布，采样生成修正标签迭代训练，等价于用MC-Dropout近似后验的EM算法；相比HiddenCRF等直接推断隐变量的方法，本方法无需额外图模型结构，直接在神经网络层面进行自我修正。
- **重标注ACE 2005开发/测试集**：结合自动伪负例识别（20种不同架构模型投票）和人工验证，揭示真实未标注率（开发集10.5%，测试集19.3%），提供 unbiased benchmark。
- **现实世界标注实验**：在WikiEvents上验证了仅在67%/52%标注覆盖率下仍可与完整标注模型竞争，证明方法的实际可用性。

## 方法详解
- **输入表示**：对每个事件类型$t$，将$[CLS]\ t\ [SEP]\ w_1\cdots w_N$输入BERT，得到联合表示$\boldsymbol{H}_{(t,X)}\in\mathbb{R}^{M\times d}$，通过共享向量$\boldsymbol{w}$映射为分数向量$\boldsymbol{s}^t=\boldsymbol{H}_{(t,X)}\boldsymbol{w}\in\mathbb{R}^M$。
- **Case 1（存在标注trigger）**：设标注trigger在位置$j$，使用正边距$\lambda_+$：
$$p_+(X,\tilde{Y},j)=\frac{\exp(s_{(j)}^t-\lambda_+)}{\exp(s_{(j)}^t-\lambda_+)+\sum_{m\neq j}\exp(s_{(m)}^t)}$$
并引入高斯正则$\hat{s}_{(m)}^t=s_{(m)}^t\times\mathcal{N}(|m-j|)$对远处词降权，缓解多trigger时的过惩罚。
- **Case 2（无标注trigger）**：以$[SEP]$标记为"无事件"指示器，使用负边距$\lambda_-$：
$$p_-(X,\tilde{Y})=\frac{\exp(s_\Delta^t+\lambda_-)}{\exp(s_\Delta^t+\lambda_-)+\sum_{m\neq\Delta}\exp(s_{(m)}^t)}$$
$\lambda_-$放松了$[SEP]$与其他词的差距，允许隐藏负例存在。
- **总损失**：
$$\mathcal{L}=-\sum_{(X,\tilde{Y})\in D}\sum_{t\in\mathcal{T}}\left[\delta_t\sum_{j:y_j=t}\log p_++(1-\delta_t)\log p_-\right]$$
其中$\delta_t=1$当且仅当存在type-$t$ trigger。
- **推理阶段**：计算$p^t=\text{softmax}(\boldsymbol{s}^t)$，阈值为$\tau=1/N$（$N$为句子长度），$p_{(i)}^t>\tau$的token即为trigger预测。
- **不确定性引导自校正**：在burn-in后，每步以MC-Dropout（K=20次）多次预测，构建categorical分布采样得$\tilde{Y}'$，再以此更新参数；交替标准训练与不确定性干扰阶段，可视为EM算法——E步用MC-Dropout近似后验$p(Z|D,\Theta^{(t)})$，M步最大化期望似然。

## 实验与结果
- **数据集**：ACE 2005（重标注dev/test）、MAVEN；额外在WikiEvents上做现实测试。
- **评估设置**：三种设置——full training（原始训练集）、data removal（随机移除正样本）、data masking（将部分label替换为O，更贴合部分标注场景）。
- **ACE 2005测试集（重标注集R）**：PromptLoc取得**F1=76.6**，显著优于FourIE（75.3）和OneIE（74.9）；在原测试集O上F1=73.9， Precision较低但 Recall较高（预测了未被标注的trigger）。
- **MAVEN**：PromptLoc dev F1=68.5*，test F1=68.9*，显著优于HiddenCRF（dev 67.4，test 68.1）和NegSPL（dev 67.1，test 68.2）。
- **极端部分标注（90% mask，p=10%）**：PromptLoc在ACE上F1>60%，而传统方法仅约20%，**绝对提升约40个百分点**。
- **重标注揭示**：开发集UL Rate=10.5%，测试集UL Rate=19.3%，说明部分标注问题是真实且严重的。
- **WikiEvents现实测试**：在67%和52%标注覆盖率下，传统方法F1下降超25%，PromptLoc与全量标注基线表现相当。

## 相关工作脉络
- **Hybrid / SeqBERT / BERTQA / OneIE / FourIE**：监督式ED主流方法，假设训练数据完全标注，未考虑false negative的影响；本文方法在相同架构上可无缝集成（如OneIE+PromptLoc）。
- **HiddenCRF（Jie et al., 2019）**：NER领域处理部分标注的开创性工作，将缺失标签视为隐变量并用CRF推断；本文将其思想迁移到ED并采用更轻量的神经网络不确定性估计，无需额外图模型。
- **NegSPL / Self-Pu**：正/未标记学习（PU learning）在NLP中的应用；NegSPL用负采样策略，Self-Pu是自训练 boosted 方法；本文与之不同的核心在于trigger localization范式从根本上避免了将O伪正例训练的难题。
- **Prompt-based IE（Wang et al., 2022a/b；Hsu et al., 2022）**：同样使用事件类型作为prompt，但目标是提升绝对性能；本文强调的是在部分标注场景下的鲁棒性，而非单纯准确率提升。
- **MEC-Dropout + EM框架**：与Gal & Ghahramani（2016）的不确定性建模和Dempster et al.（1977）的EM算法建立形式化联系，为自训练类方法提供了理论解释。

## 局限性与未来方向
- **仅针对ED任务**：未扩展到完整的事件抽取（EE）任务，尤其是事件论元抽取（argument extraction）中更复杂的标注缺失问题。
- **建模的是训练数据而非标注者本身**：未来可建模不同标注者的噪声水平差异，对低质量标注者降低信任度。
- **MC-Dropout的计算开销**：每次不确定性推断需K=20次前向，在大规模数据上可能较慢。
- **阈值$\tau=1/N$的简单性**：虽然有效，但缺乏更精细的自适应阈值机制探索。

## 研究启发与可借鉴点
- **Trigger Localization范式可迁移**：将关系抽取、NER等序列化标注任务重构为"定位式"而非"分类式"，同样能缓解部分标注噪声，值得在其他IE任务中验证。
- **正/负边距分离设计**：区分含trigger和不含trigger句子的loss构造方式（正边距push-up vs 负边距relax），是一种通用的噪声容忍损失设计思路，可推广到开放域信息抽取。
- **MC-Dropout自校正与EM的对应**：为自训练（self-training）类方法提供了清晰的理论框架，可用于分析其他弱监督学习场景中的迭代优化行为。
- **多模型投票识别潜在伪负例**：利用20种不同架构模型的共识来识别隐藏标注遗漏，是一种无需额外标注的廉价数据增强手段，值得在其他数据集上复现。
- **WikiEvents现实标注测试设计**：将实验室模拟（data masking）与实际有限标注场景（real-world annotation）结合评估，提供了更全面的方法验证范式。

## 关键术语表
- **Partial Annotation（部分标注）**：训练数据中仅部分事件实例被人工标注，其余事件虽存在但标签为O的设定。
- **Trigger Localization（trigger定位）**：将ED重构为在句子中定位trigger词的任务，而非传统的逐词硬分类，通过句子级softmax实现。
- **Margin Softmax**：在softmax中对正样本施加正边距$\lambda_+$、对负样本施加负边距$\lambda_-$，拉大类间间隔以提高噪声容忍度。
- **MC-Dropout（Monte Carlo Dropout）**：在推理/训练阶段多次激活dropout层进行多次前向传播，用于估计模型预测的不确定性。
- **False Negative（假负例/隐性负例）**：实际存在事件但未被人工标注的trigger，是部分标注问题的核心挑战。
- **Data Masking**：将训练集中部分事件的真实标签替换为O以模拟部分标注场景的实验设置。
- **Challenge Set / Control Set**：challenge set指多个ED模型均预测为trigger但原标签为O的词（高争议）；control set指少数模型预测为trigger的词（低争议）。
- **Positive and Unlabeled Learning（PU Learning）**：仅使用正例和无标签数据进行学习的半监督学习范式，本文将其思想应用于ED。

## 可复现要素
- **数据集**：ACE 2005（原始训练集；重标注dev/test集，代码仓库有说明）；MAVEN（公开）；WikiEvents（公开）。
- **代码**：已开源，地址 https://github.com/jianliu-ml/partialED
- **权重**：论文未提及独立权重发布。
- **关键超参**：BERT-large（ACE）、BERT-base（MAVEN）；batch size=10（ACE）、20（MAVEN）；learning rate=1e-5；$\lambda_+=10$，$\lambda_-=1$；MC-Dropout次数K=20；推理阈值$\tau=1/N$。
