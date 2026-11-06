---
title: "Vision-Meets-Definitions-Unsupervised-Visual-Word-Sense-Disa"
source: https://aclanthology.org/2023.acl-long.88.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 22:27:15"
---

# 论文速读：Vision-Meets-Definitions-Unsupervised-Visual-Word-Sense-Disa

## 一句话总结
本文提出了一种无需额外训练的无监督视觉词义消歧（VWSD）方法，通过贝叶斯边缘化将外部词汇知识库的sense definition引入预训练ITM模型，并结合基于GPT-3的上下文感知释义生成（CADG）解决OOV问题，显著提升了多义词配图选图性能。

## 研究问题与动机
- 无监督VWSD要求根据上下文从候选图中选出最贴合目标词词义的图片，但现有SOTA零样本ITM模型（如CLIP）对多义词理解有限，常因缺乏显式词义约束而选错（如将“Angora city”误选为猫图片）。
- 传统流水线式WSD+ITM方案易产生错误级联（error cascading），且需大量标注数据微调，部署成本高。
- 词汇知识库（LKB）中的释义能有效刻画词义，但VWSD设定下正确答案未知，无法直接挑选单条释义注入模型。
- 约14.33%的目标词在WordNet中无记录（OOV），需借助大语言模型动态补全释义信息以覆盖长尾词汇。

## 核心贡献（创新点）
- **贝叶斯推理融合知识库释义**：将候选释义作为潜变量进行边缘化，把后验概率分解为C2D（上下文→释义）与D2I（释义→图像）两项乘积求和，无需微调即可适配CLIP/FLAVA；与直接拼接图文的传统ITM本质区别在于显式引入词汇语义约束并避免单点消歧依赖。
- **上下文感知释义生成（CADG）**：针对WordNet缺失的OOV词，利用GPT-3结合目标词所在句法语境与词性生成专属释义；与仅输入词本身的生成方法（DG）相比，CADG将生成任务从“静态词典补全”转变为“上下文驱动的动态消歧”，语义对齐更精准。
- **免训练联合概率建模框架**：整个流程不更新任何模型参数，仅通过推理逻辑重构得分计算方式，避免了流水线系统的错误级联，性能可与微调过的生成式WSD模型（$T5_{SemCor}$）媲美甚至超越。

## 方法详解
- **任务形式化**：将无监督VWSD建模为多分类问题，目标为 $\hat{v} = \arg\max_{v \in V^t} P(v|c, t)$，其中 $V^t$ 含1张正例与9张干扰图。
- **贝叶斯释义融合**：引入LKB释义集合 $D^t$，利用链式法则展开为 $P(v|c,t) = \sum_{i=1}^{|D^t|} P(v|D_i^t, c, t) P(D_i^t|c, t)$。右侧 $P(D_i^t|c,t)$（C2D）衡量上下文与第 $i$ 条释义的匹配度；左侧 $P(v|D_i^t, c, t)$（D2I）衡量候选图像与“释义+上下文”的跨模态对齐度。
- **编码与模板设计**：将释义与上下文联合输入文本编码器，模板固定为 `{context}: {ith sense's definition}`，并在提示中附加目标词及词性（如 `angora (n): A Angora is a variety...`）。C2D与D2I均通过对应编码器的内积+Softmax转化为概率分布，最终加权求和得出图像得分。
- **CADG生成策略**：对OOV词构造提示 `Define "{target word}" in {context}. {target word} ({POS}):`，调用GPT-3（Davinci，temperature=1.0）生成释义，与WordNet释义混合使用（记为WN+CADG）。

## 实验与结果
- **数据集与基线**：SemEval-2023 Task 1 (SE23) VWSD数据集（12,896样本，14.33% OOV），基线模型为CLIP与FLAVA；对比源包括WordNet（WN）、Malkin等(DG)、本文CADG及混合策略（WN+CADG），并引入微调WSD模型 $T5_{SemCor}$ 作pipeline对比。评估指标为Hits@1与MRR，采用Student's t-test验证显著性。
- **主要结果**：CLIP基线Hits@1为73.00，引入WordNet后跃升至81.98（+8.98%p, $p<1e-10$）；结合CADG后进一步提升至Hits@1 83.08、MRR 89.60，达到最佳。FLAVA呈现相同趋势但生成释义带来的增益略低。所有提升均通过统计显著性检验。
- **关键分析**：在单义与多义词上均有显著提升；CADG在OOV样本上远优于DG；增加采样释义数量（n=2,3）并未带来性能提升；对比pipeline系统，本文方法Hits@1持平但MRR更高，验证了联合概率建模对错误级联的有效抑制。

## 相关工作脉络
- **Word Sense Disambiguation (WSD)**：传统方法依赖LKB词汇相似度或KG连通性；基于BERT的Glossbert等通过cross/bi-encoder融合释义。本文同源但面向多模态，且采用贝叶斯边缘化而非监督微调。
- **Visual Verb Sense Disambiguation (VVSD)**：如Gella等人基于
