---
title: "Prompts-Can-Play-Lottery-Tickets-Well-Achieving-Lifelong-Inf"
source: https://aclanthology.org/2023.acl-long.16.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:51:01"
field: "持续学习与通用信息提取"
keywords: ["Lifelong Learning", "Prompt Tuning", "Information Extraction", "Parameter-Efficient Fine-Tuning", "Catastrophic Forgetting", "Lottery Ticket Hypothesis"]
innovations: ["提出LPT框架，通过共享soft prompt与任务感知二值mask在线剪枝实现参数高效的终身UIE", "引入straight-through estimator联合优化prompt与mask，避免迭代剪枝过程", "在seen/novel tasks上统一评估遗忘防止与few-shot/zero-shot迁移，BWT=0且显著超越L2P"]
benchmarks: ["13 IE Datasets (ACE04/05, CoNLL03/04, SciERC, NYT, CASIE, SemEval-14/15/16)", "Few-shot/Zero-shot Adaptation on 4 Novel IE Tasks"]
---

# 论文速读：Prompts-Can-Play-Lottery-Tickets-Well-Achieving-Lifelong-Inf

## 一句话总结
论文提出了 **Lottery Prompt Tuning (LPT)**，一种参数与部署高效的终身信息提取方法：通过共享 soft prompt 向量并结合任务感知的二值 mask 在线剪枝，实现知识正向迁移与灾难性遗忘的双重保障，在 seen/novel tasks 上均取得 SOTA。

## 研究问题与动机
1. 现实场景中 IE 训练数据以流式到达，跨任务/跨领域逐步涌现，现有 UIE 方法假设全量数据可访问，难以支撑实际部署。
2. 人类可从少量样本甚至零样本泛化，UIE 系统应具备 few-shot/zero-shot 快速适应能力，避免为每个新任务重新训练庞大 PLM。
3. 现有 PEFT 类 CL 方法（AdapterCL、C-PT）在知识迁移上受限：AdapterCL 完全隔离无共享；C-PT 虽渐进初始化但未显式处理跨任务参数复用与剪枝。
4. 已有 lottery ticket 探索（如 Xprompt）依赖迭代 retrain-prune-rewind，计算开销大，无法适配 continual learning 的在线增量场景。

## 核心贡献（创新点）
1. **构建首个面向终身 UIE 的基准**：要求系统不仅在接受序列任务上维持性能，还要在 novel tasks 上实现 few-shot/zero-shot 泛化，覆盖遗忘防止与知识扩展两项指标。
   - *区别*：以往 lifelong IE 工作多聚焦单一任务（NER/RE/EE），本文统一在 generative UIE 框架下评估跨任务类型与域的多阶段学习。
2. **提出 LPT（Lottery Prompt Tuning）框架**：单次训练同时优化 soft prompt 与 task-aware binary mask，通过 top-c% 重要性得分在线剪枝直接得到 lottery prompt，无需额外剪枝阶段。
   - *区别*：与 Xprompt 的 token/piece 级剪枝及迭代 rewinding 不同，LPT 在 parameter 级做一次性联合学习，显著降低 continual 场景下的时间与存储成本。
3. **设计参数隔离 + 复用机制统一遗忘与迁移**：前向传播允许 reuse 之前任务选中的 prompt 参数实现知识迁移；反向传播通过累积 mask `(1-M_{k-1})` 冻结历史参数防止灾难性遗忘。
   - *区别*：不同于 L2P 的距离选择或 C-PT 的独立 prompt 初始化，LPT 共享同一组软提示并依靠 mask 动态解耦，实现了更高的参数复用率与更低的增量存储。

## 方法详解
1. **任务形式化**：将所有 IE 任务统一为 text-to-structure 生成，输入拼接为 `x = [s; t]`（schema prompt + raw text），输出为结构化提取语言 SEL，模型为冻结参数的 T5 encoder-decoder。
2. **Prompt 拼接**：在输入最前端追加可学习的连续 soft prompt 向量 `p`，输入变为 `x = [p; s; t]`，仅训练 prompt 参数 `θ_p`，损失函数为标准的负对数似然：
   $$\mathcal{L} = \sum_{(x,y) \in \mathcal{D}_k} -\log p(y|x; \theta_p)$$
3. **在线剪枝（Lottery Prompt）**：引入与 prompt 同形的可学习重要性得分 `s_k`，经 indicator 函数 `h(.)` 得到二值 mask：
   $$\mathbf{m}_k = h(\mathbf{s}_k), \quad \hat{\boldsymbol{\theta}}_p^k = \boldsymbol{\theta}_p \odot \mathbf{m}_k$$
   使用 straight-through estimator 绕过不可导操作直接回传梯度，避免迭代重训。
4. **遗忘隔离**：训练第 k 个任务时，累积历史 mask $\mathbf{M}_{k-1} = \bigvee_{i=1}^{k-1}\mathbf{m}_i$，梯度更新限制在未分配区域：
   $$\boldsymbol{\theta}_p \leftarrow \boldsymbol{\theta}_p - \eta \left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_p} \odot (\mathbf{1} - \mathbf{M}_{k-1})\right)$$
5. **Novel task mask 选择**：提出两种轻量策略——基于各 mask 在输入上的 perplexity 选取最低者；或基于梯度 one-shot 代理系数 $\alpha_i$ 计算 entropy 梯度选最高者。

## 实验与结果
1. **数据集与协议**：13 个 IE 数据集覆盖 NER（ACE04/ACE05-Ent/CoNLL03）、RE（CoNLL04/ACE05-Rel/SciERC/NYT）、EE（CASIE/ACE05-Evt）、ABSA（SemEval-14/15/16），按 5 种随机任务序评估，留 4 个不同任务类型数据集作 novel tasks。
2. **评测指标**：平均 F1、BWT（遗忘）、FWT（正向迁移）；novel task 采用 10-shot 与 zero-shot 设置。
3. **Seen tasks 结果**：LPT 平均 F1 = **76.914**，较次优 CL 方法 L2P（73.610）提升 **+3.304**；BWT = **0**（完全无遗忘），FWT = **9.414**。
4. **Novel task 结果**：Few-shot 平均 F1 = **54.58**（优于 L2P 51.25）；Zero-shot 平均 F1 = **32.69**（显著高于 L2P 20.88）。
5. **参数效率**：增量可训练参数仅 **0.097%**，额外存储 **0.302%**，与 multi-task prompt tuning（76.774）相当但无需并行多任务数据。
6. **消融**：剪枝比例 0.7 为最优折中；mask 相关性可视化显示 LPT 既复用相似任务参数又自动探索未选参数。

## 相关工作脉络
1. **AdapterCL / C-PT**：均为 per-task 独立模块方案，知识隔离强但迁移弱；LPT 共享同一 prompt 池并通过 mask 动态选择，实现了参数级共享而非模块级隔离。
2. **L2P**：基于 prompt pool + 距离选择，存在轻微负向 BWT（-0.039）；LPT 通过硬性二值 mask 完全阻断历史参数更新，BWT 严格归零。
3. **Xprompt**：探索 prompt 剪枝的 lottery ticket，但仅在 token/piece 级操作且需迭代重训；LPT 在 parameter 级在线学习，适配 continual 场景。
4. **Regularization/Rehearsal 基线（EWC/ER）**：EWC 需权衡新/旧任务导致 Average 较低；ER 需存储 50 条历史样本引发隐私与存储负担；LPT 无需记忆与正则化，纯参数隔离。
5. **Multi-task baselines**：Multi-task FT/PT/AT 被视为 CL 上界；LPT 在极少参数开销下逼近 Multi-task prompt tuning（76.77 vs 76.91），证明终身学习可与多任务性能对齐。

## 局限性与未来方向
1. 最优 sparsity（top-c%）仍需人工设定，缺乏自适应确定机制，影响训练效率与泛化稳定性。
2. 仅在 UIE 场景验证，未拓展至 multi-task learning、prompt ensembling 等其他 PEFT 应用场景。
3. 未与 Adapter tuning、LoRA 等其他参数高效微调方法做组合实验，兼容性与叠加收益待探究。

## 研究启发与可借鉴点
1. **在线二值 mask 联合学习**可直接移植到其它 PEFT 方法（如 LoRA adapter 的 low-rank 选择），实现"一次训练、动态剪枝"的持续学习范式。
2. **straight-through estimator 跳过不可导操作**的设计思路可推广至任何含离散选择（top-k、routing）的 continual 模块。
3. **BWT=0 + FWT>0** 的双重指标体系为后续工作提供清晰的评估契约，建议纳入团队 UIE/CL 项目的标准评测集。
4. **PPL-based 与 gradient-based 两种 novel-task mask 选择策略**实现极简、无需微调，可作为 few-shot 冷启动的通用启发式基线。
5. 本文揭示 **多任务并行训练的负干扰** 可通过参数隔离缓解，提示后续可将 multi-task pretrain 与 continual mask 选择解耦设计。

## 关键术语表
1. **Lifelong Learning**：单模型依次学习多个任务并持续积累知识、防止遗忘的持续学习范式。
2. **Prompt Tuning**：冻结 PLM 主体参数，仅学习前置连续 soft prompt 向量的参数高效微调方法。
3. **Lottery Ticket Hypothesis**：过参数化网络中存在子结构（winning ticket），单独训练即可达到全参数同等性能。
4. **Catastrophic Forgetting**：在学习新任务过程中对旧任务性能急剧下降的现象。
5. **BWT (Backward Transfer)**：后续学习任务对先前任务 F1 的平均变化，负值表示遗忘程度。
6. **FWT (Forward Transfer)**：已学任务对当前/新任务 F1 的平均增益，正值反映知识迁移效果。
7. **Straight-Through Estimator**：在反向传播中忽略非线性离散操作的梯度，直接用前向值近似传递梯度的技巧。
8. **Text-to-Structure**：将各类 IE 子任务统一为从 schema + 原文到结构化抽取语言 SEL 的生成任务范式。

## 可复现要素
- 数据集：13 个公开 IE 数据集（ACE04/05、CoNLL03/04、SciERC、NYT、CASIE、SemEval-14/15/16），划分方式遵循原 UIE 论文。
- 代码/权重：论文未提供开源仓库；权重基于 UIE-large checkpoint（Lu et al., 2022）。
- 关键超参：pruning ratio top-c% = 0.7；prompt length = 20；每任务 epochs = 30；batch size = 24；8×NVIDIA A100 GPU。
- 任务顺序：5 种随机排列（见 Appendix Table 4），主要结果取平均。
