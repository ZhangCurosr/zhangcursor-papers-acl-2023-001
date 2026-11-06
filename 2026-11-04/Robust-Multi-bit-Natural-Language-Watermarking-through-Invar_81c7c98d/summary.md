---
title: "Robust-Multi-bit-Natural-Language-Watermarking-through-Invar"
source: https://aclanthology.org/2023.acl-long.117.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:05:46"
field: "自然语言水印与内容安全"
keywords: ["natural language watermarking", "robust watermarking", "infill model", "invariant features", "syntactic dependency", "multi-bit watermark"]
innovations: ["提出基于语义关键词和句法依存不变特征的水印锚定框架", "设计抗扰动 infill 模型（reverse KL + 稀疏 top-k 分布对齐）", "揭示并量化水印系统 Phase 1 与 Phase 2 的鲁棒性鸿沟"]
benchmarks: ["IMDB", "Wikitext-2", "Dracula", "Wuthering Heights"]
---

# 论文速读：Robust-Multi-bit-Natural-Language-Watermarking-through-Invariant-Features

## 一句话总结
本文提出一种基于语言**不变特征**（语义关键词与句法依存关系）的鲁棒多比特自然语言水印方法，并设计了显式抗扰动的 infill 模型，在四种数据集、三种扰动类型和两种扰动比例下，相比 ContextLS 平均 BER 提升 16.8 个百分点，同时保持可比的语义质量。

## 研究问题与动机
- **多比特嵌入与鲁棒性难以兼顾**：现有神经水印方法（如 AWT）嵌入高比特时提取易出错；ContextLS 虽零污染 BER=0，但未考虑扰动下提取，鲁棒性差。
- **文本水印缺少"感知显著成分"**：图像水印可将水印嵌入感知显著分量，而文本中"修改单个词"比"修改单个像素"更显眼，需找到类不变特征作为锚点。
- **Phase 2（iFill 阶段）是鲁棒性瓶颈**：即使通过不变特征恢复了相同掩码位置（$\mathcal{R}_{g_1}$ 高），原始 BERT infill 模型对局部上下文敏感，导致有效水印集合恢复率 $\mathcal{R}_{g_2}$ 显著下降。
- **自动语义评估指标存在盲区**：NLI 模型在并列依存（coordination）替换和实体遮蔽场景下给出虚高分数，不能真实反映语义变化。

## 核心贡献（创新点）
1. **提出基于不变特征的水印框架**：利用语义关键词和句法依存关系作为水印锚点，使掩码位置在轻微扰动下保持稳定；与 ContextLS 本质区别在于后者完全随机选定替换位置，无不变性保障。
2. **构建按 NLI 蕴含分排序的句法依存优先级策略**：通过 infill + NLI 打分自动确定各类依存关系的嵌入优先级（将蕴含影响小的依存优先用于水印）；与已有工作相比，首次将 NLI 排序与不变特征框架联合使用。
3. **设计抗扰动的 infill 模型（+RI）**：在鲁棒训练目标上，以去梯度的原始分布为 target，对 top-$k_1$ token 构造稀疏目标分布并使用 reverse KL 损失最小化扰动前后的输出分布差异；与 AWT 端到端 encoder-decoder 学习范式根本不同。
4. **揭示并量化 Phase 1 vs Phase 2 的鲁棒性鸿沟**：系统分析表明，即使在最优不变特征下，$\mathcal{R}_{g_2}$ 仍显著低于 $\mathcal{R}_{g_1}$，明确了鲁棒 infill 的必要性。

## 方法详解
- **整体框架**（两阶段盲水印）：
  - **Phase 1**：$S = g_1(X)$，从原文中提取不变特征（关键词/句法依存），确定掩码位置集合 $S$。
  - **Phase 2**：$g_2(X, S)$，用 infill 模型对每个掩码位置生成 top-$k_1$ 候选词，过滤后取 top-$k_2$ 并按字母序排列，构建笛卡尔积得到候选水印集合 $\mathbb{X}$，再筛选满足 $g_1(X_{wm}) = g_1(X)$ 的合法水印集并分配 bit 消息。
- **不变特征选取**：
  - **关键词组件**：NER 识别专有名词 + YAKE 无监督关键词抽取，锚定相邻词作为掩码位。
  - **句法依存组件**：用 Spacy 构建依存树，对每种依存类型遮罩后由 infill 填充，再以 NLI 模型（RoBERTa-Large-NLI）计算蕴含分数排序，优先选择蕴含影响小的依存类型进行掩码，直至达到目标掩码数；确保关键词不被掩码。
- **鲁棒 infill 模型训练（+RI）**：
  - 损失函数采用 **reverse KL 散度**：
    $$\mathcal{L}_{con} = \sum_{i \in S} \mathrm{KL}(\tilde{p}_i \| p_i)$$
    其中 $\tilde{p}_i = P(\tilde{X}_{\backslash i}|\theta)$（扰动输入），$p_i = P(X_{\backslash i}|\mathrm{FREEZE}(\theta))$（冻结的原始输入分布，去梯度）。
  - 目标分布仅在 top-$k_1$ token 上非零并归一化，避免在 30000+ 词表上扩散。
  - 训练时沿用 §3.1 的掩码策略（而非预训练随机掩码），使训练/测试分布对齐。
- **扰动模拟**：词级插入/删除/替换，扰动率 CR=2.5% 和 5%，并用 sentence transformer（all-MiniLM-L6-v2）过滤余弦相似度<0.98 的样本。

## 实验与结果
- **数据集**：IMDB（影评）、Wikitext-2（百科）、Dracula 和 Wuthering Heights（小说，Project Gutenberg）。
- **评估指标**：BPW（比特/词）、BER（比特误差率）、ES（蕴含分数）、SS（语义相似度）、人工评估（流利度与语义保持）。
- **主要结果**（Table 3，最终方法 = Syntactic + RI）：
  - **IMDB，CR=0.025，删除**：ContextLS BER=0.219 → Ours +RI BER=0.074（绝对提升 **+14.5pp**）。
  - **IMDB，CR=0.05，删除**：0.392 → 0.200（**+19.2pp**）；四数据集三扰动类型两比例下**平均 BER 绝对提升 +16.8pp**。
  - **Dracula，CR=0.025，删除**：0.201 → 0.030（**+17.1pp**）。
  - BPW 与 ContextLS 相近（0.083~0.100），Ours 可达 0.144（IMDB），payload 更高。
  - **语义质量**：ES/SS 与 ContextLS 相当（IMDB ES=0.975 vs 0.985），显著优于 AWT 和传统方法。
- **效率**：嵌入 1000 句耗时 4 分钟（vs ContextLS 81 分钟），推理速度快 **20 倍**。
- **人工评估**（Table 5）：流利度变化 Δ=0.26±0.4，语义相似 SS=3.90±0.8，与 ContextLS 几乎一致，显著优于 AWT（Δ=1.32，SS=2.97）。
- **最强结果**：Dracula @CR=0.025 删除，BER=0.030（接近无损）；IMDB @CR=0.025 删除，BER=0.074。

## 相关工作脉络
- **ContextLS（Yang et al., 2022）**：算法嵌入+infill+NER过滤，零污染 BER=0，但不考虑扰动鲁棒性；本文通过不变特征和鲁棒 infill 解决其鲁棒性短板，同时保持更高层 payload（更大 $k_2$）。
- **AWT（Abdelnabi & Fritz, 2021）**：端到端 encoder-decoder 学习水印，高 payload 下 BER 显著，且常出现词重复等 artifacts；本文与之定位不同：无学习提取器、确定性提取、质量更可控。
- **传统规则水印（Topkara et al., 2006; Hao et al., 2018）**：基于同义词替换和算术编码，BPW 低且语义质量差；本文方法全面超越。
- **图像鲁棒水印（Cox et al., 1997; Zhu et al., 2018）**：启示本文"在感知显著分量嵌入"思想，但文本不可直接迁移，本文转化为"语义/句法不变特征"。
- **语言模型水印（Kirchenbauer et al., 2023; He et al., 2022）**：零比特检测类方法；本文支持多比特信息嵌入，适用场景不同。
- **隐写术（Fang et al., 2017; Ziegler et al., 2019）**：注重隐蔽性而非鲁棒性，可容忍覆盖文本任意生成；本文严格约束语义保持。

## 局限性与未来方向
- **语义自动评估存在偏差**：NLI 对并列依存替换和实体替换给出虚高分，实际语义损害被低估（§5.2）。
- **对删除扰动最脆弱**：句法依存组件在 CR=10% 删除时鲁棒性急剧下降（Appendix Fig.3）。
- **未测试释义（paraphrase）攻击**：可能大幅改变句法结构，但缺乏细粒度可控性。
- **对抗扰动的近似**：基于 TextAttack 的自动扰动未必模拟真实对手行为，与人类主观感知存在 gap。
- **与语义质量的 trade-off**：更强的水印必然干扰原文，这是不可回避的根本权衡。

## 研究启发与可借鉴点
1. **不变特征锚定策略可迁移**：将"找不变量→以之为锚嵌入"的思想迁移到其他模态或任务（如代码水印、语音水印）具有通用价值。
2. **Reverse KL 抗扰分布对齐技巧**：以冻结原始分布为 target、对 top-k 构造稀疏分布并使用 reverse KL 损失，避免 zero-forcing，可推广至其他需抗扰动的生成任务。
3. **NLI 排序辅助特征选择**：用蕴含模型自动量化不同语言特征的语义影响程度并据此排序，是一种无需人工标注的特征评估方法，可用于其他文本修改任务。
4. **Phase 1/Phase 2 分层鲁棒性分析框架**：将水印系统拆分为定位阶段和嵌入阶段分别度量鲁棒性，有助于精准定位瓶颈，方法论本身具有参考价值。
5. **与 LLM 输出水印结合的机会**：本文方法支持多比特嵌入且效率高（推理快 20 倍），可与 Kirchenbauer 等 LLM 水印思路结合，实现带所有者的可追溯 LLM 输出水印。

## 关键术语表
- **BPW（Bits Per Word）**：每词嵌入比特数，衡量水印 payload 容量。
- **BER（Bit Error Rate）**：提取消息与实际消息的比特错误率，越低表示鲁棒性越好。
- **Invariant Feature（不变特征）**：对轻微文本扰动保持稳定的语言特征，用作水印锚点。
- **Infill Model**：基于掩码语言模型（如 BERT-masked）在上下文条件下填补缺失词的模型。
- **Reverse KL Divergence**：$\mathrm{KL}(q\|p)$，以预测分布加权，防止目标分布过宽时模型输出 zero-forcing。
- **NLI（Natural Language Inference）**：自然语言蕴含任务，用于评估文本修改后的语义保持程度。
- **Syntactic Dependency（句法依存）**：句子中词与词之间的有向语法关系（如主谓、修饰等），构成依存树。
- **Corruption Ratio（CR）**：被扰动的词数占句子总词数的比例，本文取 2.5% 和 5%。

## 可复现要素
- **数据集**：IMDB、Wikitext-2、Dracula、Wuthering Heights（均为公开数据集，Project Gutenberg / HuggingFace）。
- **代码/权重**：论文未提及开源代码与权重（ACL 2023，2023 年发表时未附 GitHub 链接）；依赖第三方工具：Spacy（en-core-web-sm）、TextAttack、RoBERTa-Large-NLI、all-MiniLM-L6-v2、stsb-RoBERTa-base-v2。
- **关键超参**：top-$k_1$=32，top-$k_2$=3~4（依数据集调整），学习率 5e-5，AdamW，linear warmup 0.1，训练 100 epochs，取最后一个 checkpoint；鲁棒训练使用 separate train split（ novels 取前 40% 作训练集）。
- **硬件**：单卡 Titan RTX 或 RTX 3090。
