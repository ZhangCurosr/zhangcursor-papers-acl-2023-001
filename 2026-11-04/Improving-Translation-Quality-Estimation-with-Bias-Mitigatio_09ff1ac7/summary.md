---
title: "Improving-Translation-Quality-Estimation-with-Bias-Mitigatio"
source: https://aclanthology.org/2023.acl-long.121.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:39:31"
field: "机器翻译质量评估"
keywords: ["Quality Estimation", "Bias Mitigation", "Contrastive Learning", "Denoising Reconstruction", "Machine Translation Evaluation", "Pre-trained Models"]
innovations: ["提出对比正则化QE框架，通过正负样本对比学习缓解单语特征偏差", "设计去噪重建负样本生成策略，结合知识蒸馏对齐解码空间", "在保持QE精度的同时有效缓解偏差，优于此前以性能退化为代价的去偏方法"]
benchmarks: ["MLQE-PE", "WMT19 QE"]
---

# 论文速读：Improving-Translation-Quality-Estimation-with-Bias-Mitigatio

## 一句话总结
提出一种基于对比学习的偏差缓解方法（ConRegQE），通过对平行句对目标端引入噪声并重建生成负样本，联合对比学习损失与回归损失训练 QE 模型，在降低对单语特征过度依赖的同时显著提升了翻译质量估计精度。

## 研究问题与动机
- 当前最先进的 QE 模型存在明显偏差：过度依赖单语特征（如目标句流畅度、源句复杂度），而未能真正建模双语语义对齐，导致 fluent but inadequate 的翻译被错误赋予高分。
- 现有偏差缓解方法（如 Behnke et al., 2022 的多任务架构）虽然能在一定程度上减轻偏差，但均以牺牲 QE 估计精度为代价，整体性能出现退化。
- QE 任务数据稀缺（通常仅数千训练样本），简单剔除单语特征会同时移除有用信息，需要在偏差缓解与保持估计精度之间取得平衡。
- 先前对比学习在 QE 中的应用（Huang et al., 2021）仅在 zero-shot 场景下进行，未将其用于偏差缓解目的。

## 核心贡献（创新点）
- **提出对比正则化 QE（ConRegQE）框架**：以对比学习作为正则化器，迫使 QE 模型在关注单语特征的同时聚焦双语语义对齐，与 Behnke et al. 的"硬性剔除偏斜特征"思路形成本质区别。
- **设计去噪重建（Denoising Reconstruction）负样本生成策略**：先通过手工规则（masking、insertion、deletion、infilling、replacement）破坏参考译文，再用预训练重建器恢复，生成既自然又与源句语义不对齐的负样本；区别于以往基于规则或 MT 解码的负样本生成方法（模式明显、易被模型识别）。
- **引入知识蒸馏将待评估 MT 模型的解码空间迁移至重建器**：使生成负样本的噪声分布更贴近真实 MT 输出，进一步提升负样本质量；这是前序工作未涉及的设计。
- **在 MLQE-PE 和 WMT19 QE 数据集上验证了方法的有效性**：在所有语言对和标注类型上均实现显著提升，同时通过两个对抗测试集详细分析了 QE 偏差，证明方法在偏差缓解与估计精度间取得了良好折衷。

## 方法详解
- **整体架构**：以预训练多语言模型 XLM-RoBERTa（$\Phi$）为骨干，配备两个独立的 fully-connected 头——对比学习头 $FC_{CL}$ 和回归头 $FC_{reg}$，避免两任务互相干扰。
- **负样本生成（去噪重建）**：对平行句对的参考译文施加五种手工扰动规则（masking、replacement、insertion、deletion、infilling），再用预训练重建器（encoder-only 如 BERT，或 encoder-decoder 如 BART/CPT）恢复，得到含噪声且与源句不对齐的负样本；每个正样本对应 20 个负样本。
- **知识蒸馏**：从待评估的 NMT 模型采样 500k 句平行文本进行翻译，用生成的目标序列对重建器进一步微调，使重建器的解码空间与待评估 MT 模型对齐，噪声分布更一致。
- **对比学习损失（InfoNCE）**：
  $$L_{CL} = \frac{e^{s(q, k^+)/\tau}}{e^{s(q, k^+)/\tau} + \sum_{i=1}^{n} e^{s(q, k_i^-)/\tau}}$$
  其中 $\tau=0.3$，$n=20$，$s(q,k) = FC_{CL}(\Phi(q,k))$ 为 QE 模型输出的 logit。
- **回归损失（MSE）**：
  $$L_{MSE} = ||FC_{reg}(\Phi(q,k)) - l(q,k)||_2$$
  其中 $l(q,k)$ 为人工标注质量分数。
- **联合训练**：
  $$L_{total} = L_{MSE} + \lambda \times L_{CL}$$
  $\lambda$ 控制对比损失的权重，实验表明过大的 $\lambda$ 会损害单语特征建模从而降低估计精度，需取适中值以达成折衷。

## 实验与结果
- **数据集**：MLQE-PE（7 个语言对：EN-DE、EN-ZH、RO-EN、ET-EN、RU-EN、SI-EN、NE-EN，涵盖高/中/低资源）和 WMT19 QE（EN-DE、EN-RU）。标注类型包括 DA（Task1）和 HTER（Task2）。
- **评估指标**：Pearson 相关系数（PCC）和 Spearman 秩相关系数（SRCC），五次随机种子平均。
- **基线对比**：TransQuest、OpenKiwi、COMET（均基于相同预训练模型公平比较），以及 Behnke et al. (2022) 的四种偏差缓解策略（bilingual、augmented、adversarial、focal）。
- **主要结果（MLQE-PE Task1 PCC，avg）**：ConRegQE 达到 0.602，优于 TransQuest（0.615→注意此处 baseline 数值因复现差异略有不同）、OpenKiwi（0.588）、COMET（未报告 avg）；Behnke et al. 的四种方法 avg 分别为 0.614、0.606、0.496、0.572，多数出现性能退化。ConRegQE 在 EN-ZH Task2 上 PCC 达 0.372（baseline TransQuest 为 0.336），提升显著。
- **WMT19 QE 结果**：ConRegQE（XLMR-base）avg PCC 为 0.5102，优于 TransQuest（0.4766）、OpenKiwi（0.4309）、COMET（0.4584）；mBERT+TLM 设置下达 0.5654，超过 Kepler et al. 的夺冠系统（0.5520）。
- **交叉标注鲁棒性**：在 Task1 上训练、Task2 上测试时，ConRegQE PCC 达 0.3827（Test20），显著高于 TransQuest（0.3331）和 COMET（0.3406），证明模型泛化能力更强。
- **最强结果**：WMT19 QE mBERT+TLM 设置下 EN-DE PCC=0.5386、EN-RU PCC=0.5654，avg=0.5654，超越 WMT19 夺冠系统。

## 相关工作脉络
- **Sun et al. (2020)**：首次揭示 QE 模型对单语特征的偏差依赖（partial input bias），并提出用 adequacy 指标作为标签来缓解；但后续 Behnke et al. 指出其新标注数据中偏差依然显著。本文在此基础上提出更有效的缓解方案。
- **Behnke et al. (2022)**：提出四种多任务偏差缓解策略（bilingual、augmented、adversarial、focal），但均导致 QE 性能退化；本文定位为"在不损失精度的前提下缓解偏差"，通过与 Behnke 方法的对比实验证明了折衷策略的有效性。
- **Huang et al. (2021)**：首次将对比学习应用于 QE，但仅在 zero-shot 场景下验证，未用于偏差缓解；本文将其扩展为带监督的对比正则化训练，并设计了专门的负样本生成策略。
- **TransQuest（Ranasinghe et al., 2020）/ OpenKiwi（Kepler et al., 2019）/ COMET（Rei et al., 2020）**：当前 QE 的主流基线系统，均基于多语言 PTM；本文方法可叠加于这些基线之上进一步提升性能。
- **Fomicheva et al. (2020)**：提出 MLQE-PE 数据集，构成 WMT21 QE 评测基础；本文在此数据集上进行系统性实验验证。
- **Poliak et al. (2018)**：揭示 NLI 任务中的 hypothesis-only 偏差，与 QE 中的 partial input bias 现象类似，均为模型过度依赖单侧输入的体现。

## 局限性与未来方向
- 仅有 Behnke et al. (2022) 一个直接相关的偏差缓解工作可作为基线对比，该领域研究尚不充分。
- 未采用 WMT QE 评测中复杂的 data augmentation 和 model ensemble 技术，因此结果无法与 WMT 挑战赛最优系统直接竞争。
- 方法需要参考译文作为正样本，对于无参考的 QE 标注数据难以直接应用。
- 未来方向：更深入地探究 QE 偏差的本质，提升模型在真实应用场景中的鲁棒性和泛化能力。

## 研究启发与可比借鉴点
- **对比学习作为正则化的思想可迁移至其他 NLP 评测任务**：凡是以预训练模型为骨干、存在偏斜特征依赖问题的任务（如摘要质量评估、语义相似度评测），均可借鉴"正负样本对比+回归"的双任务框架。
- **去噪重建负样本生成策略具有通用性**：通过"扰动+重建"生成语义不对齐但语言自然的负样本，比纯规则扰动或 MT 解码更具多样性且难以被模型识别，可推广至其他需要对比学习的文本任务。
- **知识蒸馏对齐解码空间的设计值得借鉴**：将待评估模型的解码特性迁移至辅助模块，使合成数据的噪声分布更贴近真实场景，这一思路可用于提升数据增强的有效性。
- **对抗测试集的构建方法可用于偏差诊断**：test-adv1（相邻样本交换源句）和 test-adv2（用 post-edit 替换 MT）两种对抗集能有效量化模型对单语特征的依赖程度，可作为 QE 模型诊断工具。
- **λ 参数的折衷分析提供了方法论启示**：偏差缓解不是"越彻底越好"，过度抑制单语特征会损害有用信息的利用；这一"软约束而非硬剔除"的思路对其它去偏研究具有指导意义。

## 关键术语表
- **Quality Estimation (QE)**：在无人工参考译文的情况下，自动预测机器翻译质量的任务，通常在句子级别或词级别进行。
- **Contrastive Learning（对比学习）**：通过将正样本对拉近、负样本对推开来学习表征的方法，本文用 InfoNCE 损失实现。
- **Denoising Reconstruction（去噪重建）**：先对文本施加随机扰动（mask/replace/insert/delete），再用预训练模型恢复，本文用于生成负样本。
- **Partial Input Bias（部分输入偏差）**：QE 模型过度依赖单一输入侧（如仅目标句）进行预测的偏差现象。
- **InfoNCE Loss**：对比学习中常用的归一化温度缩放损失函数，通过softmax形式计算正负样本的区分度。
- **Direct Assessment (DA)**：一种翻译质量人工标注尺度，评委对翻译质量打分。
- **Human-Targeted Error Rate (HTER)**：基于编辑率的翻译质量评估指标，衡量从机器翻译到人工校订文本所需的最小编辑量。
- **Knowledge Distillation（知识蒸馏）**：将目标模型（MT 系统）的解码行为迁移至辅助模型（重建器），使后者生成的噪声分布更接近真实 MT 输出。

## 可复现要素
- **数据集**：MLQE-PE（公开，https://github.com/smtAugus/MLQE-PE）和 WMT19 QE（通过 WMT 共享任务获取）；论文已声明数据集可公开获取。
- **代码/权重**：论文未明确声明代码开源；基线系统（TransQuest、OpenKiwi、COMET）均有开源实现。
- **关键超参**：温度系数 $\tau=0.3$，负样本数 $n=20$，骨干模型 XLM-RoBERTa-base 或 Multilingual BERT，重建器使用 BERT（英文）和 CPT（中文），知识蒸馏采样 500k 句（英文用 WikiMatrix，其他语言用 CC100），五次随机种子取平均。
