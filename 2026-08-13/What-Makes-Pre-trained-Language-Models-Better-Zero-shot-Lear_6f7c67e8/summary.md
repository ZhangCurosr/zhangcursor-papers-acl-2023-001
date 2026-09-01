---
title: "What-Makes-Pre-trained-Language-Models-Better-Zero-shot-Lear"
source: https://aclanthology.org/2023.acl-long.128.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:52:12"
field: "零样本自然语言处理"
keywords: ["zero-shot learning", "prompt learning", "perplexity", "template selection", "text classification", "cloze-style prompting"]
innovations: ["提出 Perplection 方法，在无标注条件下通过困惑度自动选择最优 prompt 模板", "建立语言差异与困惑度的关联假设，揭示模板有效性可被语言模型本身评估", "在真实零样本场景下实现 annotation-free 的模板筛选，超越依赖开发集的现有方法"]
benchmarks: ["DOUBAN", "WEIBO", "WAIMAI", "ECOMMERCE", "FewCLUE", "SST-2", "TweetEval", "AG News"]
---

# 论文速读：What-Makes-Pre-trained-Language-Models-Better-Zero-shot-Lear

## 一句话总结
本文提出了 Perplection（基于困惑度的模板选择方法），在零样本文本分类中无需任何人工标注数据，仅通过计算候选 prompt 模板的困惑度即可自动筛选出最有效的模板，实现了真正现实场景下的 zero-shot prompt learning。

## 研究问题与动机
- 现有 prompt learning 方法在 zero-shot 场景下依赖开发集（human-annotated data）进行事后模板筛选，这在实际零样本场景中不切实际。
- cloze-style prompt learning 对模板设计高度敏感：不同模板在相同数据集上性能波动巨大（如 ECOMMERCE 数据集中，"[very/not] pleased." 准确率达 73.12%，而"[yellow/green] black." 仅 50.49%）。
- 预训练语言模型与下游任务之间存在"语言差异"（language discrepancy）和"目标差距"（objective gap），前者是阻碍 zero-shot 泛化的关键因素。
- 已有方法缺乏在无标注数据下"预测"模板性能的机制，无法在真实零样本场景中进行模板优选。

## 核心贡献（创新点）
- 提出 Perplection 方法：利用困惑度作为语言差异的代理指标，在无标注条件下预测并选择最优 prompt 模板。
- 与已有工作的本质区别：Zero-PET、NSP-BERT 等方法依赖开发集进行 post-hoc 模板选择，而 Perplection 完全 annotation-free，无需任何标注数据。
- 建立假设与实证关联：首次系统论证"困惑度越低→语言差异越小→zero-shot 性能越好"的相关性，并通过 UMAP 可视化展示 task-relevant 模板能构建更分离的特征空间。
- 方法具有模型无关性：在 BERT 和 RoBERTa 两种架构、中英文多数据集上均验证有效。
- 开源代码与模板库：提供可复现代码及中英文手动/自动模板集合，便于社区扩展至 NLG 任务。

## 方法详解
- **核心假设**：cloze-style prompt 将下游任务重构为 MLM 目标，缩小了 objective gap；剩余性能差异主要由 pre-training corpus 与 downstream data 之间的语言差异（language discrepancy）决定。
- **困惑度定义**：采用双向语言模型的 masked language modeling 版本计算 perplexity：
  $$\operatorname{PPL}(x)=\exp \left\{-\frac{1}{t} \sum_{i}^{t} \log p_{\theta}\left(x_{i} \mid c\right)\right\}$$
  其中 $c$ 为除第 $i$ 个 token 外的完整上下文，适用于 BERT/RoBERTa 等双向模型。
- **模板评分流程**：对每个候选模板，将 label words（如"很"和"不"）分别填入 [MASK] 位置生成两条 prompt 文本，计算各自困惑度后取均值作为该模板的得分。
- **模板选择**：选择困惑度最低的模板用于零样本预测（将 label words 替换回 [MASK]）。
- **实现变体**：
  - MPerplection：基于手动设计模板的 Perplection
  - APerplection：基于 LM-BFF 自动生成的模板的 Perplection
  - MRandom / ARandom：随机选择模板的对照基线

## 实验与结果
- **数据集**：中文 8 个（DOUBAN、WEIBO、WAIMAI、ECOMMERCE、EPRSTMT、TNEWS、CSLDCP、IFLYTEK）+ 英文 3 个（SST-2、TweetEval、AG News）。
- **最强结果**：MPerplectionR 在 ECOMMERCE 上达到 85.12% 准确率，显著提升 MRandomR（72.49%）+12.63%；在 WAIMAI 上达 75.49%，对比 MRandomR（66.43%）+9.06%。
- **对比 SOTA**：MPerplectionR 大幅超越 Zero-PET（除 TNEWS 外），与 NSP-BERT 在 DOUBAN 上持平（60.74% vs 60.85%），且后者依赖开发集。
- **英文验证**：在 SST-2、TweetEval、AG News 上均稳定优于随机基线，平均提升约 1.6-2%。
- **深入分析**：困惑度选择倾向选中第二佳模板（60% 概率）或最佳模板（约 10% 概率），完全排除无判别力的模板。
- **自动模板局限**：自动生成的模板多样性不足导致困惑度标准差小（Table 5），Perplection 效果受限，未来需改进模板生成质量。

## 相关工作脉络
- **Zero-PET**（Schick & Schütze, 2021）：基于 cloze 问题的零样本文本分类方法，需开发集进行模板筛选。
- **NSP-BERT**（Sun et al., 2022）：利用 next sentence prediction 预训练任务的 prompt 方法，依赖 post-hoc 模板选择。
- **P-tuning / Prefix-tuning**（Liu et al., 2022; Li & Liang, 2021）：连续 prompt 优化方法，参数效率高但需微调。
- **LM-BFF**（Gao et al., 2021）：基于 T5 自动生成 prompt 模板的方法，使用训练标签排序模板。
- **In-context Learning**（Brown et al., 2020）：GPT 系列通过前缀示例实现零样本学习，与本文 cloze-style 范式不同。
- **定位差异**：本文聚焦 truly annotation-free 的零样本场景，填补了无开发集条件下的模板选择空白。

## 局限性与未来方向
- 主要在 BERT 家族模型上验证，扩展至 decoder-only 架构（如 LLaMA、GLM）及 NLG 任务需进一步研究。
- 困惑度对长文本存在偏好偏差，迫使模板设计需保持长度一致，未来需探索 length-agnostic 度量。
- 自动模板生成质量受限（多样性不足），影响 Perplection 效果，需改进模板生成策略。
- 若模型在预训练阶段未接触过特定概念，依赖困惑度选择可能导致次优性能，可探索无标注自监督训练或 few-shot 扩展。
- 在多领域细粒度模板选择场景中表现受限（如 IFLYTEK 数据集），未来可引入 domain-relevant 模板池。

## 研究启发与可借鉴点
- **困惑度作为模板评估指标**：将语言差异量化为困惑度，为 prompt 选择提供无标注评估路径，可迁移至其他语言任务。
- **UMAP 可视化验证假设**：通过特征空间分离度直观验证模板有效性，为后续工作提供可复用的可视化分析手段。
- **Manual vs Automatic 模板对比分析**：揭示模板多样性对 Perplection 效果的关键影响，提示未来工作应优先关注高质量模板库构建。
- **长度归一化策略**：通过将数据集子采样为固定长度句子（14-15 词）来消除困惑度长度偏差，为类似研究提供实用技巧。
- **中英文跨语言验证**：同时在中文和英文数据集上验证方法泛化性，增强结论可信度，值得在跨语言研究中借鉴。

## 关键术语表
- **Cloze-style prompt**：将分类任务重构为完形填空形式，通过 [MASK] 占位符引导模型预测标签词。
- **Language discrepancy**：预训练语料与下游数据之间的语言差异（词汇、句法、频率等），是阻碍 zero-shot 泛化的核心因素。
- **Objective gap**：预训练目标（如 MLM）与下游任务目标（如序列分类）之间的差异，prompt 可部分弥合此差距。
- **Perplexity (PPL)**：语言模型对序列的预测不确定度度量，越低表示模型对该文本越"熟悉"。
- **Label word (Verbalizer)**：将类别标签映射到具体词汇（如"好"→正类，"差"→负类）的映射函数。
- **Zero-shot prompt learning**：在无标注数据条件下，通过 prompt 激活预训练模型的零样本能力。
- **Pre-trained feature space**：prompt 激活的、有利于下游任务分类的特征表示空间。

## 可复现要素
- **数据集**：中文数据集（DOUBAN、WEIBO、WAIMAI、ECOMMERCE、FewCLUE 基准）及英文数据集（SST-2、TweetEval、AG News）均为公开数据。
- **代码开源**：论文声明提供代码（"We make our code available for replication"）。
- **模型权重**：使用 Chinese-RoBERTa-wwm-ext、BERT-base-uncased、RoBERTa-base 等开源预训练模型。
- **关键超参**：模板长度控制在 14-15 词（中文）、label words 为对称二元集（如"很/不"）、随机种子 5 次运行取平均。
- **计算资源**：Tesla V100 GPU（32GB 显存）。
