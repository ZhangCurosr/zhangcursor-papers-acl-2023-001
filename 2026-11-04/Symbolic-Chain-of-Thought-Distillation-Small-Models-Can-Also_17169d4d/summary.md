---
title: "Symbolic-Chain-of-Thought-Distillation-Small-Models-Can-Also"
source: https://aclanthology.org/2023.acl-long.150.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 09:17:07"
field: "小模型推理能力提升"
keywords: ["Chain-of-Thought", "Knowledge Distillation", "Small Language Models", "Reasoning", "Few-shot Learning"]
innovations: ["提出SCoTD方法使小模型通过多样本CoT蒸馏学会逐步推理", "发现样本数量比质量过滤更重要，随机下采样接近高级过滤策略", "验证CoT蒸馏可显著提升小模型在contrast sets和跨任务迁移上的鲁棒性"]
benchmarks: ["CommonsenseQA", "OpenBookQA", "QuaRel", "IMDB contrast set", "SST-2"]
---

# 论文速读：Symbolic-Chain-of-Thought-Distillation-Small-Models-Can-Also-Think

## 一句话总结
本文提出 Symbolic Chain-of-Thought Distillation (SCoTD)，通过将大模型（GPT-3）对每个输入采样多条思维链（CoT）作为训练数据，使参数量小两个数量级以上的学生模型（125M–1.3B）也能学会逐步推理，在常识问答等基准上实现显著提升，且生成的 CoT 经人工评估与大模型相当。

## 研究问题与动机
- **核心问题**：Chain-of-Thought 提示（如 "Let's think step-by-step"）仅在大模型（>50B 参数）上有效，小模型无法通过提示获得推理能力。
- **现有方法不足**：
  - 传统监督微调仅利用标签，未利用中间推理过程，导致小模型在复杂/边界问题上泛化能力弱。
  - 已有将大模型解释蒸馏给小模型的工作（如 Li et al., 2022a）通常每样本仅采样 1 条 rationale，数据多样性不足。
  - 缺乏对小模型如何有效利用 CoT 蒸馏的系统性分析（样本数量、多样性、teacher likelihood 等）。

## 核心贡献（创新点）
1. **提出 SCoTD 框架**：通过从大教师模型采样多条思维链+标签构建训练语料，使小模型学会自解释推理；与 Li et al. (2022) 的单样本方法本质区别在于强调多样本采样与 self-consistency 解码。
2. **发现样本数量是关键因素**：每样本采样 N=30 条 rationale 优于单条，且随机下采样表现接近过滤后的子集，说明数据量本身是提升主因。
3. **系统对比蒸馏策略**：在 few-shot 与 fully supervised 两种设置下验证 SCoTD，并探索 self-consistency 解码在小模型蒸馏后的有效性。
4. **验证泛化与鲁棒性**：证明 SCoTD 在 contrast sets（IMDB 对比集）和 unseen out-of-domain tasks（SST-2）上显著优于无 CoT 基线，支持"解释促进更稳健泛化"的假设。

## 方法详解
- **流程**：
  1. 手动撰写少量 labeled CoT 示例构成 prompt set $\mathcal{P}$。
  2. 对训练集每个无标签输入 $x_i$，用教师模型 $\mathcal{T}$ 采样 $N$ 条 (rationale, label) 对 $(\tilde{y}_i^k, \tilde{z}_i^k)_{k=1}^N$。
  3. 构建语料库 $\mathcal{C}$，在 supervised 设置下过滤掉教师预测错误的样本。
  4. 以学生模型标准 language modeling loss 在 $\mathcal{C}$ 上 fine-tune：$\mathbb{E}_{(x,\tilde{y},\tilde{z}) \sim \mathcal{C}}[\log S(\tilde{y}, \tilde{z}|x)]$。
- **解码策略**：
  - Greedy/beam search 取最可能的 $(\tilde{z}, \tilde{y})$。
  - Self-consistency：采样多条 reasoning paths 后对预测 label 多数投票。
- **超参数**：GPT-3 采样温度 T=1.0，每样本 N=30；OPT 微调 batch size=32，lr=$2 \times 10^{-5}$，训练于单卡 48GB GPU。

## 实验与结果
- **数据集**：CommonsenseQA (CSQA)、OpenBookQA (OBQA)、QuaRel；IMDB contrast set 用于鲁棒性测试；SST-2 用于跨任务迁移。
- **基线**：Label-Only（仅监督标签）、Greedy-CoT（每样本单条 CoT）。
- **主要结果**（Table 2）：
  - Few-shot：SCoTD 在 QuaRel 上达 73.0%，显著高于 Label-Only 的 65.6% 和 Greedy-CoT 的 64.7%。
  - Full supervised：SCoTD 在 QuaRel 上达 83.8%，Label-Only 仅 59.0%，提升 +24.8%。
  - OBQA 提升明显：Full SCoTD 67.0 vs Label-Only 60.2。
- **最强结果**：SCoTD + Self-Consistency 在 CSQA 达 64.7%（+4.5 相对 No-SCoTD 60.2），OBQA 达 57.8%（+13.4）。
- **人工评估**：OPT-1.3B+SCoTD 的 CoT 质量与 GPT-3 无显著差异（p>0.01），学生在 correctness-controlled 设置中 51% 胜率与教师持平。
- **对比集测试**：IMDB contrast set 上 SCoTD 92.0% vs Label-Only 81.6%（+10.4%）。
- **跨任务迁移**：SCoTD 模型在 SST-2 上 few-shot 准确率达 79.6%，而 MetaICL 基线仅 0%。

## 相关工作脉络
- **Chain-of-Thought Prompting** (Wei et al., 2022b; Kojima et al., 2022)：本文扩展 CoT 到小模型蒸馏场景，而非仅依赖 prompt。
- **Li et al. (2022a)**：同样用 GPT-3 生成解释教小模型，但仅每样本采样 1 条 rationale；本文证明多样本采样+self-consistency 更有效。
- **Self-Consistency** (Wang et al., 2022b)：本文将 majority vote 策略引入蒸馏后的小模型推理。
- **Symbolic Knowledge Distillation** (West et al., 2022)：本文延续"大模型生成训练语料"范式，但聚焦 CoT 而非常识知识。
- **Learning with Explanations** (Camburu et al., 2018; Narang et al., 2020)：本文验证多 rationale 样本比单一 rationale 更能提升小模型推理能力。
- **Contemporaneous work** (Huang et al., 2022; Magister et al., 2022; Ho et al., 2022)：同期研究同样探索小模型 CoT 能力，但本文系统分析样本数量与多样性影响。

## 局限性与未来方向
- **局限性**（论文自述）：
  1. 仅涉及英文语言与任务。
  2. 依赖封闭源 GPT-3，训练集未知。
  3. 学生模型仅使用 OPT 系列，未验证其他架构。
  4. CoT 可能产生"看似合理但结论错误"的幻觉推理，存在自动化偏见风险。
- **未来方向**（论文提出）：
  1. 将 SCoTD 扩展到生成任务（当前仅分类任务）。
  2. 增加 source task 数量以提升跨任务泛化。
  3. 进一步探索 CoT 样本中哪些特征更重要（通过过滤/下采样实验）。

## 研究启发与可借鉴点
1. **多样本蒸馏策略**：每样本采样多条 noisy CoT 并配合 self-consistency 解码，可有效弥补小模型推理能力不足，此策略可迁移到其他需要中间推理的任务（如数学推理、代码生成）。
2. **数据量优先于质量过滤**：随机下采样接近高级过滤策略的效果，暗示在 CoT 蒸馏中"量"比"质"更重要，实验设计时可直接扩大采样规模而非过度过滤。
3. **Contrast set 作为鲁棒性评估**：使用 contrast sets 验证模型泛化能力值得借鉴，可作为评估 CoT 蒸馏效果的补充指标。
4. **跨任务 few-shot 迁移实验**：在源任务上训练、目标任务上 few-shot 评估的设计，可推广到评估小模型是否真正习得推理能力而非记忆。
5. **人工评估 CoT 质量**：除 accuracy 外进行 pairwise human evaluation，可从质量角度补充定量评估的不足。

## 关键术语表
**Symbolic Chain-of-Thought Distillation (SCoTD)**：一种知识蒸馏方法，从大语言模型采样多条思维链及预测标签作为训练数据，使小模型学会逐步推理。

**Chain-of-Thought (CoT)**：通过提示词（如 "Let's think step-by-step"）引导语言模型显式输出中间推理步骤的提示策略。

**Self-Consistency**：对同一输入采样多条推理路径，对预测结果进行多数投票以 aggregating 最终答案的解码策略。

**Contrast Sets**：通过对原始测试样本进行有意义扰动并翻转 gold label 构建的评估集，用于检验模型的局部决策鲁棒性。

**Symbolic Knowledge Distillation**：用大语言模型生成训练语料（而非 soft logits）来训练小模型的蒸馏范式。

**Rationale**：模型生成的中间推理陈述，解释从输入到预测的决策过程。

## 可复现要素
- **数据集**：CommonsenseQA、OpenBookQA、QuaRel、IMDB（含 contrast set）、SST-2；均为公开标准数据集。
- **代码开源**：https://github.com/allenai/cot_distillation
- **语料开源**：论文声明将发布采样的 CoT 语料。
- **关键超参数**：温度 T=1.0，每样本采样数 N=30，batch size=32，lr=$2 \times 10^{-5}$。
- **硬件要求**：单卡 48GB GPU 可复现主实验。
