---
title: "Improving-Pretraining-Techniques-for-Code-Switched-NLP"
source: https://aclanthology.org/2023.acl-long.66.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:39:35"
field: "语码转换NLP"
keywords: ["code-switching", "masked language modeling", "pretraining", "mBERT", "language identification", "residual connection", "GLUECoS", "multilingual NLP"]
innovations: ["提出SWITCHMLM和FREQMLM两种语言边界感知掩码策略替代随机MLM", "设计RESBERT架构通过中间层残差连接增强低层语言信息传递", "引入辅助LID损失与L2正则化联合优化以显式强化switch-point编码"]
benchmarks: ["GLUECoS QA (HI-EN)", "GLUECoS SA (HI-EN, ES-EN, TA-EN, ML-EN)", "L3CUBE Hindi-English"]
---

# 论文速读：Improving-Pretraining-Techniques-for-Code-Switched-NLP

## 一句话总结
论文针对语码转换（code-switched）文本稀缺的场景，提出了基于语言边界感知的掩码语言建模（SWITCHMLM/FREQMLM）与残差连接架构（RESBERT）相结合的预训练方法，在有限的语码转换数据下显著提升下游QA和SA任务性能。

## 研究问题与动机
1. **语码转换文本数据稀缺**：多语言预训练模型（如mBERT、XLM-R）直接从多语言语料训练，在处理语码转换输入时表征效果不佳，而专门用语码转换文本进行中间预训练虽有效，但高质量标注数据难以大规模获取。
2. **现有MLM策略未利用语言边界信息**：标准MLM（STDMLM）随机掩码15% token，未考虑语码转换句中语言切换点（switch-point）的特殊性，未能引导模型编码语言边界信息。
3. **LID标注难以获取**：语码转换句子中的词级语言识别（LID）标签通常需人工标注或专用标注器，且对转写（Romanized）形式效果差，限制了边界感知方法的适用性。
4. **如何从有限语码转换数据中榨取更大预训练价值**：在数据量受限的约束下，如何设计更有效的预训练目标和架构改动以提升下游性能是关键问题。

## 核心贡献（创新点）
1. **SWITCHMLM**：提出仅在语言边界周围token上进行掩码的新MLM目标，迫使模型学习编码switch-point信息；与STDMLM的本质区别在于掩码策略从随机变为基于语言切换位置的感知选择。
2. **FREQMLM**：提出基于单语语料相对频率的LID代理方法，无需真实LID标注即可近似SWITCHMLM；与SWITCHMLM的本质区别是用统计频率估计替代标注数据获取语言身份。
3. **RESBERT**：引入从中间层到MLM head的残差连接（带dropout正则化），为低层语言信息提供直达通道；与标准mBERT的本质区别是通过跳跃连接显式增强低层语言特征传播。
4. **辅助LID损失（$\mathcal{L}_{aux}$）**：在残差连接源层额外添加MLP预测LID标签的辅助损失，并用L2正则化约束表征偏移；与RESBERT单目标的区别在于同时优化语言识别辅助任务以显式强化边界感知。

## 方法详解

### 2.1 基于语言边界的掩码策略

**SWITCHMLM**：定义掩码选择向量$S \in \{0,1\}^n$，其中$S_i=1$仅当token $x_i$位于语言边界两侧（即相邻token语言不同）。例如句子"Laptop mere bag me rakha hai"（EN HI EN HI HI HI），则$LID=[EN,HI,EN,HI,HI,HI]$，边界处为位置1-4（"Laptop","mere","bag","me"），故$S=[1,1,1,1,0,0]$，仅对这些位置执行标准MLM掩码。

**FREQMLM**：当无LID标注时，通过单语语料相对频率估计LID。对每个token $x$，计算其在嵌入语言（固定为英语）语料$\mathcal{C}_{en}$和矩阵语言语料$\mathcal{C}_{lg}$中的负对数似然$nll_{en}$和$nll_{lg}$（若词未出现或计数极低则设为-1）。判定规则：
- $nll_{en}=-1$且$nll_{lg}=-1$ → OTHER
- 仅$nll_{en}\neq -1$ → EN；仅$nll_{lg}\neq -1$ → LG
- 差值超过$\ln 10$阈值判定归属
- 否则为AMB-EN或AMB-LG（实验中映射为对应语言）

后续提出**X-HIT**作为补充：对质量差的单语语料（如含噪声的ML语料），仅检查词是否存在于词汇表中而非计数，避免NLL方法在低质量语料上的失效。

### 2.2 架构修改：RESBERT与辅助损失

**RESBERT（Residual BERT）**：在mBERT最后一层输出与MLM head之间加入来自中间层$x$（$x\in\{1,...,10\}$）的残差连接，并施加$p=0.5$的dropout作为正则化：
$$h_{MLM\_input} = h_{last} + \text{Dropout}(h_x)$$
实证发现：STDMLM在较深层（如layer 9-10）加残差最优；SWITCHMLM在较浅层（如layer 2-3）最优。

**辅助LID损失**：从残差源层$x$的token表示送入MLP预测LID边界标签$y_i\in\{0,1\}$，并结合所有层的L2正则化防止表征偏离原始嵌入：
$$\mathcal{L}_{aux} = \alpha \sum_{i=1}^{n} -\log \text{MLP}(x_i) + \beta \sum_{j=1}^{L} ||\bar{\mathbf{W}}^{j} - \mathbf{W}^{j}||^2$$
其中$\alpha=5\times10^{-2}$，$\beta=5\times10^{-4}$为固定超参（论文未搜索）。

## 实验与结果

### 数据集
- **预训练语料**：HI-EN 185K句、ES-EN 66K句、TA-EN 118K句、ML-EN 34K句（均含LID标注）；另使用L3CUBE（52.9M句）子集185K句验证数据质量影响。
- **下游评估**：GLUECoS基准，含HI-EN/ES-EN的SA任务（15K/2.1K样本）及TA-EN/ML-EN的SA任务（9.6K/3.9K样本），以及HI-EN QA任务（295 train/54 test，无dev）。
- **单语参考语料**：英语（OPUS-100，55M句对）、西班牙语（26.5M句）、印地语/泰米尔语/马拉雅拉姆语（Dakshina+Samanantar组合，分别约10M/5.9M/5.2M句）。

### 主要结果（mBERT，F1分数）

**QA（HI-EN）**：
- Baseline（无预训练）：62.1±1.5（20ep）
- STDMLM：64.8±2.0（+2.7）
- SWITCHMLM：**69±3.7**（+6.9，最优；30ep时68.9±4.2）
- FREQMLM：68.6±4.5
- STDMLM+RESBERT₉：66.8±3.0
- SW/FREQMLM+RESBERT₂：**68.8±3.1**（30ep 68.9±3.0）
- 最终版SW/FREQMLM+RESBERT₂+$\mathcal{L}_{aux}$：30ep达**68.9±3.2**，40ep达**69.8±3.0**
- **相对STDMLM最大提升约5.8 F1**

**SA（多语言对）**：
- HI-EN：STDMLM 67.7±0.6 → SWITCHMLM+RESBERT₂+$\mathcal{L}_{aux}$ **69.1±0.4**（+1.4）
- TA-EN：STDMLM 76.4±0.3 → FREQMLM+RESBERT+L_aux **77.2±0.4**（+0.8）
- ML-EN：STDMLM 76.7±0.1 → FREQMLM+RESBERT+L_aux **77.2±0.4**
- ES-EN：STDMLM 62.2±1.5 → SWITCHMLM+RESBERT₂+$\mathcal{L}_{aux}$ **63.7±1.5**（+1.5）
- **SA相对最大提升约2.7 F1（TA-EN）**

**XLM-R对比**：FREQMLM在QA上反而降低XLM-R性能（推测因XLM-R参数是mBERT的5倍+小训练集+LID噪声），但在较大SA数据集（HI-EN 15K）上无此现象。

**L3CUBE高质量语料实验（Table 4）**：在更干净的HI-EN语料上，SWITCHMLM+RESBERT₂+$\mathcal{L}_{aux}$达69.66 max F1，HINGMBERT（全量52.9M预训练）达72.36，验证数据规模的重要性。

**Probing实验**（Section 4.2）：线性探测与条件探测（conditional probing，用soft Hamming距离度量）表明，SWITCHMLM模型最终层及RESBERT中间层均编码了更多switch-point信息，解释了下游性能提升的表征机制。

## 相关工作脉络

1. **Yamaguchi et al. (2021)**：提出替代MLM的简单预训练目标；本文差异化在于不仅修改目标，还结合架构改动并专门针对语码转换场景。
2. **Clark et al. (2020) / Wettig et al. (2022)**：讨论掩码比例（15% vs 80%）；本文关注掩码位置的语义/语言感知选择而非比例调整。
3. **Yang et al. (2022)**：时间不变自适应掩码策略；本文策略为静态但语言学直觉驱动的边界感知掩码。
4. **Gautam et al. (2021); Gupta et al. (2021); Tarunesh et al. (2021a)**：通过生成模型合成语码转换文本做数据增强；本文聚焦于如何在真实有限数据上改进预训练本身而非数据扩充。
5. **Khanuja et al. (2020b); Santy et al. (2021b)**：合并真实与合成语码转换文本预训练；本文与之正交——在给定真实语料上优化预训练目标与架构。
6. **Winata et al. (2019)**：证明多语言模型直接表征对语码转换任务无效；本文在此基础上探索中间预训练阶段的改进路径。

## 局限性与未来方向

**论文自述局限**：
1. FREQMLM在语言差异小的语言对（如英西）上失效，因共享词汇量大导致频率信号模糊。
2. RESBERT的最佳残差层选择需遍历训练10个模型再验证集选择，计算代价高；计划未来用门控函数参数化层选择。
3. 若嵌入式语言出现率极低，边界处可掩码token数可能不足以达到15%总掩码量（实验中通过掩码25%-35%的可掩码token补偿）。

**可合理推断的不足**：
4. X-HIT方法依赖高质量词汇表构建，对未登录词和复杂形态处理有限。
5. 辅助损失超参固定未搜索，可能存在调优空间。
6. 仅验证了4种语言对，跨脚本/低资源语言泛化性待检验。

**未来方向**（论文提及）：探索如何利用合成语码转换文本进行高效预训练。

## 研究启发与可借鉴点

1. **边界感知掩码策略可迁移**：SWITCHMLM的"仅在语言/领域切换边界处掩码"思路可迁移至其他混合语言或跨域预训练场景（如中英混合、代码-自然语言混合）。
2. **残差连接层位置的经验规律**：STDMLM适合从深层取残差（保留高层语义），SWITCHMLM适合从浅层取残差（增强低层语言信号）——这一发现为多任务预训练中残差设计提供启发。
3. **辅助任务损失的正则化价值**：$\mathcal{L}_{aux}$通过L2约束防止表征漂移，同时注入显式语言识别信号，可作为通用"主任务+辅助知识注入"范式的参考。
4. **FREQMLM的X-HIT变体**：对于噪声大或规模小的单语语料，基于存在性（而非计数）的LID方法更鲁棒，这一经验对低资源语言对极具参考价值。
5. **Probing+Conditional Probing组合分析**：论文用线性探测验证信息编码+条件探测量化增量贡献，为预训练方法分析提供了可复用的评估流程。

## 关键术语表

**Code-switching / Code-mixing**：语码转换/语码混合，指说话者在同一对话或句子中交替使用两种或以上语言的现象。

**SWITCHMLM**：语言边界感知掩码语言建模，仅在句子中语言切换点附近的token上进行掩码预测的预训练目标。

**FREQMLM**：基于单语语料词频估计的LID代理方法，通过相对负对数似然推断token语言身份以实现无标注的SWITCHMLM近似。

**RESBERT**：引入中间层到MLM head残差连接（带dropout正则化）的mBERT变体架构，增强低层语言信息的传递。

**辅助LID损失（$\mathcal{L}_{aux}$）**：在RESBERT中间层附加的MLP语言识别损失，结合L2正则化显式强化switch-point编码。

**GLUECoS**：语码转换NLP评测基准，包含多语言对（HI-EN、ES-EN等）的QA和SA任务。

**Conditional Probing**：基于可用信息理论的探测方法，通过比较"基线+新模型表征拼接"与"仅基线"的性能差来量化新增信息。

**L3CUBE**：从Twitter爬取的52.9M句大规模印地语-英语语码转换语料库。

## 可复现要素

- **数据集**：预训练语料（HI-EN 185K、ES-EN 66K、TA-EN 118K、ML-EN 34K）及L3CUBE子集；下游GLUECoS基准。**部分公开**（论文链接：https://github.com/csalt-research/code-switched-mlm）。
- **代码**：论文声明将在接受后以MIT许可证开源（Appendix C4）。
- **基线模型**：mBERT（178M参数，12层）、XLM-R（278M参数，24层）；单语语料：OPUS-100、Cañete et al.西班牙语语料、Dakshina+Samanantar印地/泰米尔/马拉雅拉姆语料、Aksharantar马拉雅拉姆语词汇。
- **关键超参**：dropout rate=0.5；pretrain batch size=4（mBERT）/8（XLM-R）；gradient accumulation=20（mBERT）/4（XLM-R）；pretrain epochs=4；SA fine-tune lr=5e-5（mBERT）/5e-6（XLM-R）；QA lr=5e-5；α=5e-2，β=5e-4（辅助损失）；GPU=A100 Tensor Core。
