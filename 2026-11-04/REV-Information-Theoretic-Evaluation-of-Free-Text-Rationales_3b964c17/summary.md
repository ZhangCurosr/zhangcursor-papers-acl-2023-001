---
title: "REV-Information-Theoretic-Evaluation-of-Free-Text-Rationales"
source: https://aclanthology.org/2023.acl-long.112.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:05:28"
field: "可解释自然语言处理"
keywords: ["可解释NLP", "自由文本解释评估", "条件V信息", "信息论评估", "链式思维", "Reasoning Evaluation"]
innovations: ["提出基于条件V信息的REV指标，自动量化自由文本解释中超出输入/标签的新增信息量", "引入占位基线解释(vacuous rationale)作为参照，首次区分重复性解释与信息性解释", "揭示Chain-of-Thought生成解释对预测有支持但不保证正确性的现象"]
benchmarks: ["ECQA", "CoS-E", "QuaRTz", "e-SNLI"]
---

# 论文速读：REV-Information-Theoretic-Evaluation-of-Free-Text-Rationales

## 一句话总结
论文提出 **REV**（Rationale Evaluation with conditional $\mathcal{V}$-information），一种基于信息论视角的自由文本解释评估指标，用于量化解释中**超出输入和标签本身**的新颖且与标签相关的信息量；实验表明 REV 比现有指标（LAS、RQ）更能区分解释质量，且与人类判断一致度更高。

## 研究问题与动机
1. **自由文本解释评估的缺失维度**：现有自动评估指标（LAS、RQ）仅关注"解释是否能辅助代理模型预测标签"，无法衡量解释提供了多少**新的、超出输入的**理由性信息。
2. **空心解释（vacuous rationale）难以识别**：例如直接将输入+标签 declaratively 组合的文本（如"个人蘑菇可以保存在冰箱里"），现有指标会给出高分，但实际未提供任何推理依据。
3. **准确率度量与信息度量的混淆**：一个解释可能只是重复输入内容仍被判定为"好"，因为准确率仅反映标签-解释关联，不反映新颖信息。

## 核心贡献（创新点）
1. **提出基于条件 $\mathcal{V}$ 信息（CVI）的 REV 指标**：将信息论中的条件互信息适配到 NLP 解释评估场景，形式化定义"解释对标签的新增可用信息"。
2. **首次自动量化解释的新信息含量**：相比 LAS/RQ 只关注标签-解释联合预测准确率，REV 引入"占位基线解释"（vacuous baseline rationale, B）作为参照，通过差值衡量真正的新增信息。
3. **揭示 Chain-of-Thought 推理的内在局限**：在 GPT-3 和 LaMDA 上进行 Few-shot / CoT 推理分析发现，生成的中间解释虽对预测有正支持（REV > 0），但并不能保证预测正确性（准确率差异显著）。

## 方法详解
### 2.1 信息论框架
- 目标是量化 $I(Y; R | B)$：在给定占位基线 $B$（输入+标签的 declarative 组合）条件下，解释 $R$ 对标签 $Y$ 的新增信息量。
- 直接计算互信息需要真实分布，不可行；引入 $\mathcal{V}$-information（Xu et al., 2020），通过限定模型族 $\mathcal{V}$ 估算可用信息。
- 条件 $\mathcal{V}$-信息（CVI，Hewitt et al., 2021）：
  $$I_{\mathcal{V}}(R \to Y \mid B) = H_{\mathcal{V}}(Y \mid B) - H_{\mathcal{V}}(Y \mid R, B)$$
  其中 $H_{\mathcal{V}}(\cdot)$ 为条件 $\mathcal{V}$-熵，通过最小化负对数似然实现。
- 逐样本点估计：
  $$\text{REV}(x, y, r) = -\log g'[b](y) + \log g[r, b](y)$$
  $g'$ 仅以基线 $b$ 预测 $y$，$g$ 以 $[r, b]$ 联合预测 $y$。

### 2.2 分数含义
- $\text{REV} > 0$：解释提供了新增信息支持标签（如 $r_1^*$）
- $\text{REV} = 0$：解释未提供超出基线的新信息（如 $\hat{r}_{1,a}$）
- $\text{REV} < 0$：解释不支持标签或与标签相悖（如 $\hat{r}_{1,b}$）

### 2.3 占位基线构造
- CQA：用 T5-3B 将 `(question, answer)` 转换为 declarative 陈述句（Demszky et al., 2018）。
- NLI：先用模板"premise implies/contradicts/is not related to hypothesis"生成，再用预训练模型改写避免模板模式被评估模型过拟合。

### 2.4 评估模型训练
- 使用 T5 Large，以数据集的 gold 标签和人工标注解释构建训练集 $\mathcal{D}_{train}$。
- 目标函数为条件交叉熵：最大化 $g[r^*, b^*](y^*)$ 和 $g'[b^*](y^*)$ 的对数似然。
- 评估阶段，给定任意 $(x, y, r)$（$y$ 可以是模型预测，不一定是 gold）。

## 实验与结果
### 数据集与任务
- **CQA**：ECQA、CoS-E（v1.11）、QuaRTz
- **NLI**：e-SNLI
- 所有数据集均含人工标注的自由文本解释。

### 实验设置
- 任务模型：T5 Large，三种生成设置：$\mathrm{XY^* \to R}$、$\mathrm{X \to YR}$、$\mathrm{X \to RY}$；以及人工标注 $(\mathrm{Y^*; R^*})$。
- 评估模型：T5 Large；对比基线：LAS（Hase et al., 2020）、RQ（Wiegreffe et al., 2021）。

### 关键结果
1. **跨数据集 REV 排名一致性**（表 2）：
   - ECQA：$(\mathrm{Y^*; R^*})$ 0.7943 > $\mathrm{XY^*\to R}$ 0.7806 > $\mathrm{X\to YR}$ 0.5840 > $\mathrm{X\to RY}$ 0.5599
   - CoS-E 的人工标注得分（0.2415）明显低于 ECQA，符合已知数据质量问题。
   - e-SNLI 所有类型得分极低（最高 0.0752），因大量解释为标签专用模板。
2. **与人类判断一致性**（§4.2）：
   - REV 排序：$\mathrm{Y^*; R^*} > \mathrm{XY^*\to R} > \mathrm{X\to YR} > \mathrm{X\to RY}$
   - 人类判断排序相同；而 LAS/RQ 将 $\mathrm{X\to RY}$ 排在 $\mathrm{X\to YR}$ 之前，与人类判断不符。
3. **噪声敏感性测试**（§4.3）：
   - 对输入加入高斯噪声后，REV 和 RQ 敏感度相似，但 LAS 对噪声不敏感（因代理模型在退化样本上过拟合）。
4. **Few-shot 提示与 Chain-of-Thought**（§4.4）：
   - GPT-3 对 gold label 的 few-shot 解释：REV 与人类判断 80% 一致；LAS/RQ 支持率仅 35%/23%，56%/60% 得分为零。
   - CoT 场景：GPT-3（准确率 77%）与 LaMDA（59%）的平均 REV 接近（0.92 vs 0.99），说明**解释对预测有支持，但不保证正确性**。

## 相关工作脉络
1. **LAS（Hase et al., 2020）**：通过代理模型比较含/不含解释时预测正确标签的能力，但未考虑信息新增量，对占位解释评分偏高。
2. **RQ（Wiegreffe et al., 2021）**：类似 LAS 但以 gold label 训练代理模型，仍无法区分"重复输入"与"提供新信息"的解释。
3. **$\mathcal{V}$-information（Xu et al., 2020）**：信息论框架，将可用信息定义为模型族可提取的信息量；本文将其扩展至条件情形（CVI）。
4. **CVI / Conditional Probing（Hewitt et al., 2021）**：引入条件 $\mathcal{V}$-信息概念，用于衡量表示中超出基线的可用信息；本文将其适配到自由文本解释评估。
5. **ERASER 基准（DeYoung et al., 2020）**：系统性评估可解释 NLP，涵盖提取式和自由文本解释，但未包含信息新增维度的度量。
6. **Sun et al.（2022）**：人类研究显示自由文本解释确实能提供额外知识，但仍是手工评估；本文首次提出自动量化方案。

## 局限性与未来方向
1. **可能奖励错误预测的解释**：只要解释提供了新信息支持某标签，即使标签错误也会得到正分；未来需结合预测正确性形成综合指标。
2. **不考虑解释的事实性**：当前 REV 不惩罚虚构或错误事实陈述，未来应引入事实核查机制。
3. **基线构造方式单一**：仅使用了 declarative 句型，不同构造方式对分数有影响，尚未系统分析。
4. **依赖训练数据质量**：评估模型质量受 crowd-sourced 解释数据影响，低质量训练数据（如 CoS-E）会降低度量敏感度。
5. **评估模型架构影响分数绝对值**：使用不同架构（T5 Base / BART Large / GPT-2 Large）会得到不同分数，排序一致但数值有差异（附录 B.3）。

## 研究启发与可借鉴点
1. **"占位基线"设计思想可迁移**：将 $(x, y)$ 转为无新信息的 declarative 语句作为 baseline，这一思路可推广到其他解释评估场景（如多模态、对话解释）。
2. **条件 $\mathcal{V}$-信息作为通用评估工具**：本文证明了 CVI 在解释评估中的有效性，未来可扩展至程序证明、数学推理链等其他可解释场景。
3. **Chain-of-Thought 分析的启示**：通过 REV 发现 CoT 生成解释虽然支持预测，但与预测正确性解耦——这对理解大模型推理行为具有重要参考价值，可进一步探索如何提升 CoT 的可靠性和事实一致性。
4. **多源评估互补思路**：REV 与传统准确率指标正交，可结合使用以获得更全面的可解释性洞察；团队可在后续研究中设计联合评测框架。
5. **敏感性与鲁棒性测试范式**：通过逐步注入噪声测试指标对退化样本的敏感度，是评估解释指标稳健性的有效手段，可借鉴到团队其他指标的评测流程中。

## 关键术语表
- **自由文本解释（Free-text rationale）**：以自然语言形式生成的、解释模型预测决策过程的文本。
- **条件 $\mathcal{V}$-信息（Conditional $\mathcal{V}$-information, CVI）**：在给定基线变量条件下，衡量某变量对目标变量的新增可用信息量，由模型族 $\mathcal{V}$ 限制下的最优预测性能界定。
- **占位基线解释（Vacuous rationale, B）**：仅将输入和标签 declaratively 组合而不提供新信息的解释文本。
- **$\mathcal{V}$-信息（$\mathcal{V}$-information）**：在限定可建模函数族 $\mathcal{V}$ 下，从一个随机变量可提取的关于另一个变量的信息量上界估计。
- **LAS（Leakage-Adjusted Simulatability）**：通过代理模型评估解释质量的已有指标，比较含/不含解释时的预测正确性差异，并对标签泄露进行校正。
- **RQ（Rationale Quality）**：类似 LAS，但使用 gold label 训练代理模型，不区分标签泄露情况。
- **Chain-of-Thought（CoT）**：提示大语言模型先生成中间推理步骤再给出最终答案的推理范式。

## 可复现要素
- **数据集**：ECQA、CoS-E (v1.11)、QuaRTz、e-SNLI，均为公开数据集。
- **代码/权重**：论文未提供开源代码链接（来源为 ACL Anthology，但 URL 仅指向 PDF）。
- **关键超参**：任务模型与评估模型均为 T5 Large（770M 参数）；训练 20 epochs，学习率 5e-6，batch size 8；单张 NVIDIA RTX 8000 GPU。
- **GPU 时间**：约 12 小时（主实验）。
