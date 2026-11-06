---
title: "ThinkSum-Probabilistic-reasoning-over-sets-using-large-langu"
source: https://aclanthology.org/2023.acl-long.68.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 17:06:19"
field: "大语言模型推理与方法"
keywords: ["LLM reasoning", "ThinkSum", "probabilistic inference", "BIG-bench", "chain-of-thought", "prompt engineering", "latent variable model"]
innovations: ["提出ThinkSum两阶段推理：Think并行检索+Sum外部概率聚合", "系统化Think/Sum操作族与多种聚合策略（混合/乘积/似然比/潜在变量）", "在BIG-bench 13个任务上刷新SOTA，小模型GPT-2 XL超越大模型直接提示"]
benchmarks: ["BIG-bench Lite"]
---

# 论文速读：ThinkSum-Probabilistic-reasoning-over-sets-using-large-langu

## 一句话总结
提出**ThinkSum**两阶段概率推理范式：第一阶段（Think）通过并行LLM查询提取集合关联信息，第二阶段（Sum）在外部执行结构化概率聚合以得出最终预测。该方法在BIG-bench的13个难题上取得新SOTA，用远小于GPT-3的GPT-2 XL即可超越 davinci 直接提示效果。

## 研究问题与动机
- **LLM慢思考能力不足**：先进LLMs擅长零样本/少样本联想匹配（System 1），但在需要多对象/多事实并行推理与序列逻辑推导的任务（如BIG-bench中逻辑推导、概念组合）上差距显著。
- **CoT/提示工程的脆弱性**：链式思维等方法把推理压缩为自回归生成，易受提示顺序、格式偏差、recency bias等干扰，且对prompt设计高度敏感。
- **外部推理未被充分挖掘**：现有工作大多依赖LLM自身自注意力完成全部"推理"，未将结构化概率推断作为独立控制器引入。
- **小模型潜力未被释放**：直接提示下小模型（GPT-2 Small~XL）在复杂推理任务上远低于人类平均水平，期望通过范式改进"延寿"已有模型。

## 核心贡献（创新点）
1. **提出ThinkSum两阶段推理范式（Think + Sum）**，将LLM的联想检索与外部概率推断解耦，避免将推理完全交由线性文本生成。
2. **系统化整理Think操作族与Sum聚合族**：包括子stitution/归一化、示例/列表/事实验证、翻译，以及多数票、似然比、混合平均、乘积聚合、潜在变量模型等，形成可复用模块库。
3. **在BIG-bench 13个任务上刷新SOTA**：InstructGPT在10个任务、GPT-2 XL在3个任务上取得最佳，LOGICAL DEDUCTION(N=5)达77%超越平均人类（67%）。
4. **引入 latent variable model + EM** 解释"找不同"类任务，提升可解释性并展示结构化的概率推理潜力。

## 方法详解

**Think阶段（快速联想/知识检索）**：通过并行LLM查询构造字符串集合，核心操作包括：
- **字符串操控**：Order inversion（答案前置）、Premise erasure（擦除前提）、Substitution + 句法规范化。
- **示例/列表生成**：Prompt如"A bam is a simple dwelling. Examples: 1."→解析为集合$S_1$；"List of words: $w_i$, "→计算$p(w_j|...)$得语义相似度矩阵。
- **事实验证**：Prompt如"List facts about cats. 1."→生成可替代的事实集合。
- **翻译**：自然语言↔符号结构（如逻辑不等式）、跨语言。

**Sum阶段（慢速推理/外部概率聚合）**：
- **Mixture（平均）聚合**：$\displaystyle p_j = \frac{\frac{1}{|S_1||S_2|}\sum_{w_1,w_2} p_{LLM}(s_j\langle w_1,w_2\rangle)}{p_{LLM}(s_j)}$，选$\arg\max_j p_j$（INVENTED WORDS）。
- **Product 聚合**：对各元素似然相乘，适用于"必须满足所有概念"的场景（NOVEL CONCEPTS、LOGICAL DEDUCTION）。
- **Minority vote / Log-product**：ODD ONE OUT中$\displaystyle i = \arg\min_i \prod_j P_{ij}$。
- **Likelihood ratio**：消除答案先验偏差。
- **Latent Variable Model + EM**：假设物品与事实由隐类$c\in\{0,1\}$生成，$P(i,f)=\sum_c P(c)P(i|c)P(f|c)$，用EM迭代200次求解聚类，用于增强ODD ONE OUT解释性。

**组合示例（LOGICAL DEDUCTION）**：Translation(自然语言→不等式) → Substitution(替换对象为数字) → Product(所有不等式联合似然) → Mixture(对所有排列求和)。

## 实验与结果

- **数据集**：BIG-bench Lite（13个任务），含INVENTED WORDS、ODD ONE OUT、LOGICAL DEDUCTION、PHRASE RELATEDNESS、CODENAMES、KNOWN UNKNOWNS、SPORTS UNDERSTANDING、EMOJI MOVIE、MISCONCEPTIONS RUSSIAN、NOVEL CONCEPTS、CODE LINE DESCRIPTION、LANGUAGE IDENTIFICATION、PARSINLU等。
- **模型**：GPT-3 davinci 175B、InstructGPT (text-davinci-002)、GPT-2 XL/Small/Medium/Large。
- **主要结果**：

| 任务 | 人类均值 | davinci 3-shot | InstructGPT | GPT-2 XL |
|---|---|---|---|---|
| INVENTED WORDS | N/A | 0.21 | **0.71** | 0.29 |
| ODD ONE OUT | 0.80 | 0.23 | 0.80 | **0.84** |
| FIVE OBJECTS | N/A | 0.32 | — | **0.77** |
| LOGICAL DEDUCTION (N=5) | 0.67 | 0.32 | — | **0.77** |
| PHRASE RELATEDNESS | 0.74 | 0.59 | **0.87** | 0.79 |

- **最强结果**：ODD ONE OUT 用GPT-2 XL达**0.84**（超越davinci 0.23、InstructGPT 0.80）；LOGICAL DEDUCTION N=5达**0.77**（超越人类均值0.67）。
- **对比CoT/auxiliary knowledge**：ThinkSum在全部对比任务上优于CoT与auxiliary knowledge（见表3、4），且对prompt格式扰动鲁棒。

## 相关工作脉络
1. **Chain-of-Thought (Wei et al., 2022; Kojima et al., 2022)**：让LLM生成中间推理token；ThinkSum将推理移出LLM到外部概率聚合，避免recency bias与prompt敏感性。
2. **Auxiliary knowledge / Generated-knowledge prompting (Shwartz et al., 2020; Liu et al., 2022a)**：将检索到的知识拼接进prompt；本文证明拼接会因长度/顺序退化，而ThinkSum并行聚合更稳定。
3. **Dual-system (System 1/System 2) 深度学习 (Nye et al., 2021b; Bengio, 2017)**：启发ThinkSum的快/慢思考分离思想，但本文用显式概率聚合而非隐式权重压缩实现System 2。
4. **Bayesian/inference-as-reasoning (Xie et al., 2021; Dohan et al., 2022)**：形式化in-context learning为贝叶斯推断；ThinkSum将其推广到任意外部集合概率推理。
5. **Decomposed prompting / Self-consistency (Khot et al., 2022; Wang et al., 2022a,b)**：模块化推理与多采样表决；ThinkSum强调**并行**Think调用与**确定性**Sum聚合的结合。
6. **Order inversion / Prompt design (Min et al., 2022; Holtzman et al., 2021)**：本文将其纳入通用Think操作族并配合混合/乘积聚合。

## 局限性与未来方向
- **计算开销随对象数线性增长**：Sum阶段枚举所有替换/排列导致LLM调用数上升；作者建议用外部算法或剪枝优化。
- **提示工程仍需人工**：为每个新任务需挑选Think/Sum操作组合，可结合自动化prompt engineering（Zhou et al., 2022b）缓解。
- **评测局限于GPT-3系列**：尚未在Flan-PaLM、GPT-4等更强指令微调模型上验证。
- **LLM生成格式不稳定**：某些模型无法可靠输出规定格式（如GPT-3 davinci在LOGICAL DEDUCTION失败），影响可复现性。
- **不等式一致性问题**：LLM可能在Think阶段产生不一致的翻译（如不等式方向反转），需在Sum阶段引入隐变量校正。

## 研究启发与可借鉴点
1. **Think/Sum解耦思路可迁移**：任何需要"检索中间事实→概率组合"的任务（如多跳QA、知识图谱推理）均可套用此范式。
2. **乘积/混合聚合是通用工具**：Product aggregation适用于"全部约束必须同时满足"场景，Mixture适用于"任一实例均可"场景，值得在后续工作中广泛尝试。
3. **Latent variable + EM增强可解释性**：对分类/异常检测类任务，可引入隐类模型并用EM拟合，既提升性能又提供决策依据。
4. **Prompt工程设计经验**：Order inversion、Premise erasure、Substitution+句法归一化作为原子操作，可与任何LLM pipeline组合。
5. **小模型复现路径**：GPT-2 XL在ODD ONE OUT上超越InstructGPT，提示结构化推理能显著"延寿"小模型，适合资源受限场景。

## 关键术语表
**ThinkSum**：两阶段推理范式，Think负责并行检索关联信息，Sum负责外部概率聚合得出预测。
**Chain-of-Thought (CoT)**：通过在prompt中插入LLM生成的中间推理token激发推理能力的方法。
**Mixture aggregation**：对一组同构提示的LLM输出似然取平均的聚合方式。
**Product aggregation**：对一组必须同时成立的约束的LLM似然取乘积的聚合方式。
**Latent Variable Model (LVM)**：假设观测数据由隐类生成并用于聚类的概率图模型，本文用EM求解。
**Order inversion**：将问题与选项顺序对调以校正答案先验偏差的prompt操作。
**Average human performance**：BIG-bench中的人类基准准确率，用作难度参照。
**Recency bias**：LLM对prompt末尾内容赋予过高权重的已知现象，是CoT脆弱的原因之一。

## 可复现要素
- **数据集**：BIG-bench Lite（开源，来自 Srivastava et al., 2022）。
- **代码/权重**：模型通过OpenAI API调用；GPT-2实验使用PyTorch + Hugging Face Transformers；代码未单独开源。
- **关键超参**：max_tokens=100（示例生成）/1000（事实验证）；temperature=0（greedy）或0.5（采样）；$N_e$=2~4；EM迭代200次；threshold=0.01（SPORTS UNDERSTANDING）。
