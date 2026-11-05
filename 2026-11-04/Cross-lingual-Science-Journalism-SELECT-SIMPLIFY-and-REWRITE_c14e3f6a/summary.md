---
title: "Cross-lingual-Science-Journalism-SELECT-SIMPLIFY-and-REWRITE"
source: https://aclanthology.org/2023.acl-long.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:08:31"
---

# 论文速读：Cross-lingual-Science-Journalism-SELECT-SIMPLIFY-and-REWRITE

## 一句话总结
本文首次将跨语言科学新闻写作（CSJ）定义为文本简化与跨语言科学摘要的下游任务，提出 SSR（SELECT-SIMPLIFY-REWRITE）三阶段流水线架构，通过层级图抽取关键句、强化学习段落简化、mBART跨语言重写，为英文科学文献自动生成面向非专业读者的德语科普摘要，在 WIKIPEDIA 与真实科普杂志 SPEKTRUM 数据集上显著超越现有基线。

## 研究问题与动机
- 现有跨语言科学摘要（CLS）模型仅关注摘要生成，未针对非专家读者进行语言降复杂处理，输出仍充斥专业术语与长难句。
- 科学文献篇幅极长（平均 2000–5000 词），而传统模型受限于 500/2048 tokens 输入窗口，直接处理易引发事实不一致与幻觉。
- 科学文本 discourse 呈“沙漏型”，核心信息分散于全文各章节；现有抽取模型受新闻“倒金字塔”先验影响，仅依赖 lead tokens，无法有效捕获全局关键句。
- 缺乏贴近真实科学记者工作流（标记重点→通俗化→跨语言成稿）的自动化基准与模块化可解耦方案。

## 核心贡献（创新点）
1. **首次形式化 CSJ 任务**：将跨语言科学新闻写作定位为文本简化与跨语言科学摘要的下游任务，填补非专家导向跨语言科普生成的研究空白。
2. **提出 SSR 分治流水线架构**：将复杂任务拆解为 SELECT、SIMPLIFY、REWRITE 三个独立组件，支持即插即用与单模块替换，区别于端到端黑盒模型。
3. **层级语篇感知图排序选择器**：基于 HIPORANK 构建非对称边加权机制，结合句边界与段边界函数精准定位分散于科学文献各部分的核心句子。
4. **RL 多目标段落简化器**：引入 KEEP-IT-SIMPLE 框架，联合优化简洁性、流畅性、显著性与护栏奖励，有效抑制简化过程中的信息幻觉与过度压缩。
5. **真实场景零样本迁移评测**：在 WIKIPEDIA 训练后直接于高质量真实杂志数据集 SPEKTRUM 上进行人类评估与多维度可读性分析，验证模型工程实用价值。

## 方法详解
- **整体信息流**：英文科学原文 → SELECT 抽取关键句 → SIMPLIFY 英语内段落简化 → REWRITE 跨语言抽象生成德语摘要。
- **SELECT（基于 HIPORANK 的层级图抽取）**：将文档建模为图 $G=(V,E)$，节点为句子。句边界函数 $s_b(v_i^I) = \min(x_i^I, \alpha(n^I - x_i^I))$ 与段边界函数 $d_b(v^I) = \min(x^I, \alpha(N - x^I))$ 为靠近文档/段落边界的节点赋予更高权重，通过非对称边加权区分局内与局际关联，最终综合局部与全局中心性 $c(v_i^I) = \mu \cdot c_{inter} + c_{intra}$ 进行贪婪抽取。
- **SIMPLIFY（基于 KIS 的 RL 简化）**：采用 k-SCST 训练策略，总奖励为四维度乘积：① **Simplicity**：FKGL 估算句法复杂度，结合插入/删除词的 Zipf 频移计算 $L_{score}$；② **Fluency**：GPT-2 生成器与 RoBERTa 判别器对抗，基于语言模型似然差计算 $LM_{score}$；③ **Salience**：Transformer 填空模型还原原文掩码实词的准确率；④ **Guardrails**：基于长度压缩比 $C=L_2/L_1$ 与 NER 实体一致性过滤幻觉。损失函数 $\mathcal{L} = \sum_{j=1}^k (\bar{R}^S - R^{Sj}) \sum_{i=0}^N \log p(w_i^{Sj}|w_{<i}^{Sj}, P)$。
- **REWRITE（基于 mBART 的跨语言生成）**：采用 mBART-large-50，Encoder 语言设为英语、Decoder 为德语。利用 Self-attention 捕获简化后文本内部依赖，Cross-attention 实现跨语言对齐，条件生成 $p(y|x,\theta) = \prod_{t=1}^m p(y_t|y_{<t}, x, \theta)$，beam size=4，max length=200。
- **训练配置**：SELECT 使用预训练权重；SIMPLIFY 初始化 GPT-2-medium（LR=1e-6, batch=4, k=4）与 RoBERTA-base（LR=1e-5, batch=4），训练 14 天；REWRITE 微调 mBART-large-50（LR=5e-5, warmup=100, epochs≤30），训练 6 天。全部基于 PyTorch/HuggingFace/Apex，单卡 Tesla P40。

## 实验与结果
- **数据集**：WIKIPEDIA（50,132 英德科学文章对，平均 1572→100 词，80/10/10 划分）；SPEKTRUM（1,510 篇真实科普杂志对，2337→361 词，作零样本外部验证）。
- **基线**：抽取式+翻译（LEAD, TextRank, ORACLE, HR+SB）；从头训练 CLS（S2S, PGN, TRF）；预训练
