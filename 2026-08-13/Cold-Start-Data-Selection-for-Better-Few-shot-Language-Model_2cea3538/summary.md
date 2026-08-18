---
title: "Cold-Start-Data-Selection-for-Better-Few-shot-Language-Model"
source: https://aclanthology.org/2023.acl-long.141.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:49:48"
field: "高效语言模型微调"
keywords: ["cold-start active learning", "few-shot fine-tuning", "data selection", "prompt-based learning", "uncertainty propagation"]
innovations: ["基于prompt的不确定性传播方法，解决冷启动场景下预测偏差问题", "分区-重写（PTR）策略，动态平衡样本信息量与跨簇多样性", "统一框架验证于vanilla finetuning、prompt-based learning、半监督学习与多轮AL四种设置"]
benchmarks: ["IMDB", "Yelp-full", "AG News", "Yahoo! Answers", "DBPedia", "TREC"]
---

# 论文速读：Cold-Start-Data-Selection-for-Better-Few-shot-Language-Model

## 一句话总结
论文提出了PATRON，一种针对预训练语言模型（PLM）少样本微调的冷启动数据选择方法，通过在零标注数据场景下利用prompt生成伪标签并设计不确定性传播与分区-重写（PTR）策略，有效解决了标注数据稀缺时训练集选取的难题。

## 研究问题与动机
1. **核心问题**：在冷启动（cold-start）场景下，PLM微调缺乏初始标注数据，需要设计有效的数据选取策略从大量无标签语料中查询最具信息量的样本。
2. **现有不确定性方法的不足**：传统基于不确定性的主动学习依赖于已训练模型，而冷启动场景下PLM对类别的预测存在偏差，导致不确定性估计不准确，甚至不如随机选择策略。
3. **多样性控制的挑战**：冷启动数据选择需要更强的样本多样性保障，已有聚类方法仅能控制簇内距离，无法控制不同簇间样本的距离，难以获得最优多样性。
4. **预训练与微调的差距**：直接利用预训练嵌入或MLM损失辅助数据选择时，预训练目标与下游任务的不匹配会损害方法效果。

## 核心贡献（创新点）
1. **提出PATRON冷启动数据选择框架**：专为零标注数据场景设计，利用prompt将分类任务转化为cloze风格任务，生成任务感知的伪标签，桥接预训练与下游任务差距，本质区别于依赖预训练嵌入或MLM损失的已有方法。
2. **设计基于prompt的不确定性传播方法**：通过核相似度测量样本间相关性并传播预测不确定性，只有当样本自身及其邻居的不确定性均较高时才赋予高传播不确定性，解决了直接 uncertainty 估计易受离群点干扰的问题。
3. **提出分区-重写（PTR）策略**：结合K-Means聚类与kNN邻域图，通过正则化项惩罚相邻簇间过近样本，动态调整每个簇内选定样本，相比仅依赖聚类中心的方法能更好地控制跨簇样本距离。
4. **系统性验证与适配**：在6个文本分类数据集上验证PATRON较最强冷启动基线平均提升3.4%-6.9%，并证明其与prompt-based learning、半监督学习及多轮主动学习的兼容性。

## 方法详解
PATRON方法包含三个核心模块：

**1. 基于Prompt的不确定性估计**
- 使用预定义模板τ和verbalizer将输入x包装为cloze问题，通过MLM预测[MASK]位置的概率分布：
  $$p(y|x) = p([MASK] = \mathcal{V}(y) | \mathcal{T}(x))$$
- 为解决校准问题（label words出现频率偏差），构建支持集S（每类取top-k高概率样本），计算contextualized prior：
  $$P(v) \approx \frac{1}{|S|} \sum_{x \in S} P_\mathcal{M}([MASK] = v | x)$$
- 校准后的伪标签：
  $$\widehat{y_i} = \left(\frac{p(y_i|x)}{P(\mathcal{V}(y_i))}\right) / \left(\sum_{j=1}^C \frac{p(y_j|x)}{P(\mathcal{V}(y_j))}\right)$$
- 使用熵衡量不确定性：$u(x) = -\sum_{i=1}^C \widehat{y_i} \log \widehat{y_i}$

**2. 不确定性传播（Uncertainty Propagation）**
- 使用SimCSE生成样本嵌入$\mathbf{z} = g(x;\theta)$
- 计算RBF核相似度：$\kappa(x_i, x_j) = \exp(-\rho \|\mathbf{z}_i - \mathbf{z}_j\|_2^2)$
- 传播不确定性：
  $$\widehat{u}_{prop}(x) = u(x) + \frac{\sum_{x_i \in \mathcal{X}_{KNN}(x)} \kappa(x, x_i) \cdot u(x_i)}{|\mathcal{X}_{KNN}(x)|}$$

**3. 分区-重写（PTR）策略**
- **初始化阶段**：用K-Means将无标签池划分为b个簇（b为查询批量大小），贪心选择：
  $$q_i = \arg\max_{x_j \in \mathcal{C}_i} \left(\widehat{u}_{prop}(x_j) - \beta \|\mathbf{z}_j - \overline{\mathbf{z}}_i\|_2^2\right)$$
- **重写的正则化项**：构建簇级kNN图，添加惩罚项防止相邻簇样本过近：
  $$\widetilde{q}_i = \arg\max_{x_j \in \mathcal{C}_i} \left(\widehat{u}_{prop}(x_j) - \beta \|\mathbf{z}_j - \overline{\mathbf{z}}_i\|_2 - \gamma \sum_{q_k \in \mathcal{X}_{c-KNN,i}} [m - \|\mathbf{z}_j - \mathbf{z}_k\|_2]_+\right)$$
- 迭代2-3轮直至收敛

## 实验与结果
**数据集**：IMDB（2类）、Yelp-full（5类）、AG News（4类）、Yahoo! Answers（10类）、DBPedia（14类）、TREC（6类）

**评估设置**：标注预算B ∈ {32, 64, 128}，单轮数据选择为主实验，10次随机种子取均值

**主要结果**（Table 1）：
- **平均提升**：PATRON在3个预算设置下较最强基线（BERT-KM/TPC）分别提升6.9%、5.0%、3.4%
- **绝对性能**：仅用128个标签（<0.5%全量数据），PATRON达到全监督性能的91.0%（vanilla finetuning）和92.1%（prompt-based learning）
- **类别数影响**：多分类数据集（如TREC、Yahoo!）提升更显著
- **稳定性**：18组实验中14组标准差低于基线

**Prompt-based Learning实验**（Table 2）：
- LM-BFF平均比vanilla finetuning提升12.5%，PATRON仍超最佳基线2.0%-4.5%

**其他设置**：
- **多轮AL**：预算512、每轮64标签，PATRON表现竞争性；IMDB在标签>256时uncertainty方法反超，提示可设计hybrid策略
- **半监督学习**：PATRON结合UDA/ST后在多数数据集上取得最优结果

**标签效率分析**：512标签多轮AL后，PATRON达到全监督性能的95%，相当于随机采样的3倍标签量

## 相关工作脉络
1. **Active Learning（AL）**：传统AL需大量初始标注训练模型，依赖uncertainty或gradient estimation；本文聚焦冷启动场景（零初始标签），与ALPS、CAL等方法定位不同
2. **Cold-Start AL**：Yuan et al. (2020) 使用MLM loss作为不确定性代理；本文利用prompt生成任务感知伪标签，弥补预训练-微调差距
3. **Embedding-based Selection**：Chang et al. (2021) 利用预训练嵌入聚类选样；本文结合prompt不确定性+SimCSE嵌入，更全面利用PLM知识
4. **Diversity-promoting Methods**：Coreset、BERT-KM、TPC等；本文PTR策略显式控制跨簇距离，优于仅考虑簇内多样性的方法
5. **Prompt-based Learning**：LM-BFF等方法关注微调策略；本文证明可与其结合，且prompt不确定性天然适合辅助数据选择
6. **Weakly-supervised Selection**：仅利用类别关键词；本文拓展至更系统的冷启动数据选择范式

## 局限性与未来方向
1. **模型类型限制**：仅针对MLM风格预训练模型（如BERT/RoBERTa），未考虑discriminative PLMs（如ELECTRA）；未来可结合prompting for discriminative models
2. **任务类型**：当前仅验证文本分类，可扩展至NLI、关系抽取等任务
3. **高预算场景**：论文明确不声称在高预算下超越传统AL方法；未来可与uncertainty-based方法结合设计hybrid策略
4. **微调方法兼容性**：未与前沿few-shot微调技术（如LoRA、pattern exploiting training）充分结合；理论兼容但实验待验证

## 研究启发与可借鉴点
1. **Prompt用于伪标签生成**：利用prompt将分类任务转化为cloze形式，桥接预训练与下游任务，解决冷启动下uncertainty估计偏差问题；可迁移至其他需要伪标签的任务
2. **不确定性传播机制**：通过核相似度加权邻居不确定性，避免离群点干扰；可作为通用的uncertainty refinement模块嵌入其他active learning pipeline
3. **PTR策略的工程价值**：分区+邻域图+正则化的两阶段选择框架，兼顾信息量与多样性；可复用于视觉等领域的batch selection
4. **SimCSE vs RoBERTa Embedding**：消融实验揭示RoBERTa embedding存在degeneration问题，SimCSE显著提升性能；提示在数据选择任务中应优先使用对比学习嵌入
5. **弱监督信号的系统化利用**：仅用类别关键词即可驱动数据选择；为资源受限场景下的数据-centric方法提供新思路

## 关键术语表
**Cold-start Data Selection**：零初始标注数据场景下的训练集选择，是主动学习的极端低资源变体
**PATRON**：论文提出的Prompt-based Uncertainty PROPagation和Partition-Then- Rewrite方法框架
**Prompt-based Learning**：通过template和verbalizer将下游任务转化为cloze-style MLM预测，缓解预训练-微调gap
**Uncertainty Propagation**：基于样本嵌入空间核相似度，将局部不确定性信息传播到邻居节点的方法
**Partition-then-Rewrite (PTR)**：先聚类分区初始化选样，再通过邻域图正则化迭代优化多样性的两阶段策略
**Verbalizer**：将离散标签映射到预训练词表surface words的函数，如positive→"good", negative→"terrible"
**SimCSE**：Simple Contrastive Sentence Embeddings，通过对比学习生成高质量句子表示的预训练模型
**Label Distribution Divergence (LDD)**：衡量所选样本类别分布与原始分布KL散度的指标，值越低越均衡

## 可复现要素
- **数据集**：6个公开数据集（IMDB、Yelp-full、AG News、Yahoo! Answers、DBPedia、TREC），均提供HuggingFace下载链接
- **代码开源**：https://github.com/yueyu1030/Patron
- **主干模型**：RoBERTa-base（125M参数）
- **关键超参**：ρ∈{0.05, 0.1}（RBF带宽）、β∈{0.5, 1, 5, 10}（簇内多样性权重）、γ∈{0.1, 0.3, 0.5}（跨簇惩罚权重）、m=0.5（margin）、k=1000（支持集大小）、K=50/10（KNN邻居数）
- **训练设置**：AdamW optimizer，lr∈{1e-5, 2e-5, 5e-5}，batch size∈{4, 8, 16}，epochs=15
