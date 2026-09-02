---
title: "Z-ICL-Zero-Shot-In-Context-Learning-with-Pseudo-Demonstratio"
source: https://aclanthology.org/2023.acl-long.129.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:11:58"
field: "上下文学习与提示工程"
keywords: ["zero-shot in-context learning", "pseudo-demonstrations", "copying effect", "retrieval-augmented prompting", "large language models"]
innovations: ["提出Z-ICL零样本ICL框架，通过原始语料库构造伪演示实现无标注少样本学习", "发现并定义复制效应，证明演示样例与测试输入过近会导致模型直接复制标签", "引入物理邻居和同义词标签两项技术有效抑制复制效应"]
benchmarks: ["CR", "Amazon Reviews", "Yelp", "TweetEval", "MR", "SST2", "SST5"]
---

# 论文速读：Z-ICL: Zero-Shot In-Context Learning with Pseudo-Demonstrations

## 一句话总结
论文提出Z-ICL，一种零样本上下文学习方法，通过从原始文本语料库检索近邻句子并构造伪演示样本来弥合零样本与少样本学习的性能差距；该方法利用"物理邻居"和"同义词标签"两项技术有效抑制了模型对演示样例的复制效应，使零样本性能达到与k-shot ICL相当的水平。

## 研究问题与动机
- 大型语言模型在无演示样例（zero-shot）时性能显著低于有演示样例（few-shot ICL）的场景，但近年研究表明ICL的主要作用是指定领域和格式而非提供显式训练信号，暗示zero-shot真实能力可能被低估。
- 现有nearest ICL方法在演示样例与测试输入过于接近时，模型会因"复制效应"（copying effect）直接沿用演示样例的标签，导致随机标签下的性能大幅下滑。
- 已有方法依赖带标签的训练数据构建演示，而真实场景中往往无法获得标注数据。
- 需探索不依赖任何标签信息的零样本演示构造方式，以更准确地估计模型的zero-shot能力。

## 核心贡献（创新点）
1. **提出Z-ICL零样本ICL框架**：从纯文本语料库检索近邻并配对随机标签构造伪演示，无需任何标注数据即可实现类ICL推理。
2. **发现并定义"复制效应"（Copying Effect）**：系统性地证明当演示输入与测试输入极度相似时，模型预测会被演示中的标签主导；相同输入存在时模型超过90%会直接复制其标签。
3. **物理邻居（Physical Neighbor）检索策略**：不直接使用最近邻句子，而是选取其在语料中物理相邻的句子，在保持分布相似性的同时拉开与测试输入的相似度距离，有效削弱复制效应。
4. **同义词标签（Synonym Labeling）技术**：在伪演示中使用标签的同义词（如用good/bad代替great/terrible），在保留标签语义空间信息的同时避免模型直接复制测试标签词汇。

## 方法详解
Z-ICL分为三步：

**Step 1 — 检索相关句子**：给定测试输入 $x$，使用SimCSE计算其与原始文本语料库 $\mathcal{C}$ 中所有句子的余弦相似度，取Top-$k$近邻 $\mathcal{N}_k(x) = \{c_1, \cdots, c_k\}$。为避免复制效应，不直接使用这些近邻，而是选取它们在语料中物理相邻的句子作为 $x_1, \cdots, x_k$，使其保持相近分布但距离测试输入足够远。

**Step 2 — 构造伪演示**：对每个 $x_i$，均匀随机采样一个标签 $y_i \in \mathcal{Y}$，并将其替换为该标签的手动选定同义词 $\tilde{y}_i$，形成配对 $(x_i, \tilde{y}_i)$。测试时仍使用原始标签集 $\mathcal{Y}$ 进行推理。

**Step 3 — 推理**：将$k$个伪演示样例与测试输入拼接为prompt：$(x_1, \tilde{y}_1), \cdots, (x_k, \tilde{y}_k), x$，输入LM后通过argmax预测：$\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y | x_1, \tilde{y}_1, \cdots, x_k, \tilde{y}_k, x)$。

关键设计原则：(a) 伪演示需告知正确的输入分布和标签空间；(b) 需减少模型对演示样例的直接复制。

## 实验与结果
- **数据集**：9个单句文本分类数据集（CR、Amz、Amz5、Yelp、Yelp5、Tweet-Eval、MR、SST2、SST5），其中6个领域被检索语料库（Demix，16个领域共约3000万句）覆盖，3个（MR、SST2、SST5）未被覆盖。
- **模型**：GPT-J（6B）、GPT-NeoX（20B）、GPT-3（175B）。
- **主要结果**：Z-ICL在覆盖语料的数据集上与ICL-gold（oracle）表现相当；在未覆盖数据集上虽有差距但仍大幅优于no-demos。整体较先前zero-shot方法提升5–30%绝对值。
- **GPT-J channel推理最强结果示例**：CR 80.1 vs ICL-gold 84.4；Amz 88.9 vs 90.9；Yelp 88.4 vs 91.0；未覆盖的MR 81.9 vs 86.9、SST2 82.6 vs 88.8。
- **消融结论**：(1) 物理邻居和同义词标签两个技术缺一不可，两者共同作用才能使伪演示达到与k-shot相当的性能；(2) 语料库规模和领域覆盖率均正向影响性能；(3) input-label pair格式本身（而非仅相关性文本）对性能提升至关重要。

## 相关工作脉络
- **Brown et al. (2020)**：开创ICL范式，证明LLM可通过输入-标签对进行少样本学习；本文在此基础上探索无标注条件下的零样本版本。
- **Min et al. (2022)**：证明ICL的收益主要来自正确的输入分布和标签空间而非标签正确性；Z-ICL直接利用这一洞察，用随机标签构造伪演示。
- **Liu et al. (2021)**：从训练数据检索近邻构建ICL演示；本文区别在于从原始无标注语料库检索，且不使用真实标签。
- **Olsson et al. (2022)**：发现ICL中模型会复制演示中的token模式；本文进一步指出复制效应在输入-标签配对格式下尤为显著，并给出缓解方法。
- **Reynolds and McDonell (2021)**：证明better template可显著提升zero-shot性能；本文不依赖模板工程，而是通过伪演示暴露模型内在能力。
- **Liu et al. (2022) Semantic-oriented Unlabeled Priming**：同样从无标注数据构建演示；本文首次系统研究复制效应并提出针对性缓解技术。

## 局限性与未来方向
- 仅评估单句分类任务，未扩展到自然语言推理等多句任务。
- 仅涉及分类任务，多选项生成任务和开放生成任务尚未探索。
- 同义词标签依赖人工选择，存在主观性，可探索自动同义词生成或更大范围的标签词变体。
- 检索质量高度依赖语料库的领域覆盖度，域外数据集上仍有提升空间。

## 研究启发与可借鉴点
- **零样本演示构造新思路**：在缺乏标注数据时，可利用外部无标注语料库+随机标签配对来构造高质量伪演示，为低资源场景提供新范式。
- **复制效应检测机制**：通过故意插入与测试输入完全相同的演示样例并统计匹配比例，可量化模型的复制倾向；这一诊断工具可用于评估其他ICL方法的可靠性。
- **同义词标签策略的通用性**：在需要避免模型直接拷贝标签的任何prompt工程中，均可考虑使用标签的同义/近义表达，兼顾语义引导与去偏。
- **语料库覆盖度的低成本提升**：仅增加2%规模的IMDB语料即可显著改善未覆盖数据集表现，提示领域扩展的性价比极高，可作为后续优化方向。

## 关键术语表
- **Z-ICL**：Zero-shot In-Context Learning，通过从原始语料库构建伪演示来实现零样本上下文学习的方法。
- **In-Context Learning (ICL)**：大语言模型仅凭prompt中提供的输入-标签示例（而不更新参数）完成新任务的能力。
- **Copying Effect（复制效应）**：当ICL演示样例的输入与测试输入极度相似时，模型预测被演示样例的标签主导的现象。
- **Physical Neighbor（物理邻居）**：选取语料中与检索到的近邻句子在文本中物理相邻的句子作为伪演示输入，以平衡分布相似性与距离足够的双重需求。
- **Synonym Labeling（同义词标签）**：在伪演示中使用标签的同义词而非原始测试标签，以阻止模型直接复制标签词汇。
- **No-demos**：完全不使用任何演示样例的纯零样本推理基线。
- **ICL-gold / ICL-random**：分别使用金标准标签和随机标签的k-shot ICL oracle基线，用于评估演示正确性的影响。
- **Channel Inference**：Min et al. (2021) 提出的噪声信道提示方法，通过比较各候选标签的条件概率进行预测。

## 可复现要素
- **数据集**：9个公开分类数据集（CR、Amz、Yelp、Tweet-Eval、MR、SST2/5）+ Demix原始文本语料库（Gururangan et al., 2021，16个领域）。
- **代码/权重**：论文声明"open-source code will point to the license"；模型使用GPT-J、GPT-NeoX开源权重，GPT-3通过API访问。
- **关键超参**：k=16（演示样例数），SimCSE嵌入+FAISS检索，max 256 tokens/示例、concat max 1024 tokens，int8量化运行GPT-NeoX，40GB A100。
- **相似度函数**：SimCSE余弦相似度。
- **模板**：使用Zhao et al. (2021) minimal templates，无额外prompt工程。
