---
title: "Improving-Continual-Relation-Extraction-by-Distinguishing-An"
source: https://aclanthology.org/2023.acl-long.65.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:24:40"
field: "持续关系抽取 / 类比关系区分"
keywords: ["continual relation extraction", "catastrophic forgetting", "analogous relations", "memory-based continual learning", "focal knowledge distillation", "contrastive learning"]
innovations: ["memory-insensitive prototype 缓解存样过拟合", "focal knowledge distillation 聚焦难区分类比对", "memory augmentation 实体替换+句拼接将存样量×4"]
---

# 论文速读：Improving-Continual-Relation-Extraction-by-Distinguishing-Analogous-Semantics

## 一句话总结
本文针对持续关系抽取（Continual RE）中类比关系难以区分和存样过拟合两大核心问题，提出 CEAR 模型：通过记忆不敏感原型与记忆增强缓解过拟合，通过整合训练与焦点知识蒸馏提升类比关系区分能力。在 FewRel 和 TACRED 两个基准上均取得 SOTA。

---

## 研究问题与动机

1. **过拟合被长期忽视**：既有记忆型 continual RE 工作（EA-EMR、EMAR、CRL、CRECL、KIP-Framework 等）只关注灾难性遗忘，几乎无人显式建模"反复重放少量典型样本会导致过拟合"这一事实（Verwimp et al., 2021; Lange et al., 2022）。
2. **类比关系遗忘是灾难性遗忘的关键诱因**：作者实证研究发现，关系原型的最大余弦相似度越高，该关系的准确率越低、遗忘 drop 越大——例如关系 "location" 在学到新关系 "country of origin" 后准确率从 0.98 跌至 0.6。
3. **现有工作的因果分析过于表层**：大多数工作将遗忘归因于"新知识冲刷旧知识"，但未深入剖析"为什么某些旧关系学得越好反而忘得越狠"。
4. **低资源 / 小存样场景下问题更严重**：memory size 越小，存样过拟合与类比区分难度同步上升，然而现有方法对小 memory 极其敏感。

---

## 核心贡献（创新点）

1. **首次显式建模 continual RE 中的过拟合问题**，提出 memory-insensitive relation prototype 与 memory augmentation 两项机制来缓解——与 ACА（Wang et al., 2022）仅从对抗样本角度强化表征不同，本文直击"存样重放过拟合"本身。
2. **通过实证发现"类比关系相似度突增 → 准确率骤降"的因果关联**，提出 integrated training（对比学习 + 线性分类联合优化）和 focal knowledge distillation（Focal-Loss 式加权蒸馏），从表征空间与决策边界双通路强化类比区分——ACА 用 adversarial class augmentation 构造硬负类，本文则通过 loss 权重直接聚焦难区分样本对。
3. **设计记忆增强（Memory Augmentation）将存样量 ×4**：实体替换 + 句子拼接两种策略生成合成样本，且明确区分"用于训练"与"不用于原型生成"，避免噪声干扰原型稳定性——这是 memory-based continual RE 中首次引入显式数据增强通道。
4. **端到端可复现**：源码已在 GitHub 开源（https://github.com/nju-websoft/CEAR），在 FewRel/TACRED 10 次任务序列平均下全面超越 CRL、CRECL、KIP-Framework。

---

## 方法详解

### 4.1 整体流程
对新任务 $T_k$：① 用 $D_k^{\text{train}}$ 训练模型；② k-means 选每关系 $m$ 个典型样本入 $M^r$；③ 计算 memory-insensitive 原型 $\mathbf{p}_r$；④ 记忆增强生成 $\hat{M}_k$（$|\hat{M}_k|=4|\tilde{M}_k|$）；⑤ 整合训练 + 焦点知识蒸馏做记忆回放，更新参数。

### 4.2 新任务训练
编码器 BERT + 特殊 token 标记实体头尾；句表示 $\mathbf{h}_x = \text{LayerNorm}(\mathbf{W}_1[\mathbf{h}_x^{11};\mathbf{h}_x^{21}] + \mathbf{b})$；线性 softmax 分类 $P(x;\theta_k)=\text{softmax}(\mathbf{W}_2\mathbf{h}_x)$；交叉熵损失 $\mathcal{L}_{\text{new}}$。

### 4.3 记忆样本选择
对每个关系样本聚类（簇数=memory size $m$），选取到 medoid 最近的样本存入 $M^r$，累积内存 $\tilde{M}_k=\bigcup_r M^r$。

### 4.4 Memory-Insensitive Relation Prototype
关键公式：$\mathbf{p}_r = (1{-}\beta)\mathbf{p}_r^{\text{static}} + \frac{\beta}{|M^r|}\sum_{x_i \in M^r} \mathbf{h}_{x_i}$。
- $\mathbf{p}_r^{\text{static}}$：首学该关系时所有训练样本的平均表示（"历史全景"）；
- 动态项：存样的均值（"当前适应"）；
- $\beta$ 控制两者比重（FewRel: 0.5，TACRED: 0.2）。
区别于 KIP-Framework 依赖外部知识的 prompt 生成原型，本方法纯数据驱动、零额外开销。

### 4.5 Memory Augmentation
两种增强：
1. **实体替换**：从 $M^r$ 另取 $x_j^r$，把 $x_i^r$ 的头尾实体换成 $x_j^r$ 的对应实体，得到 $x_{ij}^r$；
2. **句子拼接**：从 $\tilde{M}_k \setminus M^r$ 各取无关句 $x_m, x_n$ 拼到 $x_i^r, x_{ij}^r$ 尾部，得到 $x_{i\text{-}m}^r, x_{ij\text{-}n}^r$。
增强样本**仅用于训练，不参与原型生成**。最终 $|\hat{M}_k|=4|\tilde{M}_k|$。

### 4.6 记忆回放
**整合训练**：$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{c\_cls}} + \mathcal{L}_{\text{l\_cls}}$。
- 对比损失（InfoNCE + Triplet）：$\mathcal{L}_{\text{c\_cls}} = -\frac{1}{|\hat{M}_k|}\sum \log \frac{\exp(\mathbf{z}_{x_i}\cdot\mathbf{z}_{y_i}/\tau_1)}{\sum_r \exp(\mathbf{z}_{x_i}\cdot\mathbf{z}_r/\tau_1)} + \frac{\mu}{|\hat{M}_k|}\sum \max(\omega - \mathbf{z}_{x_i}\mathbf{z}_{y_i} + \mathbf{z}_{x_i}\mathbf{z}_{y_i'}, 0)$，其中 $y_i'$ 是余弦最相似的错误标签。
- 线性损失（交叉熵）：$\mathcal{L}_{\text{l\_cls}}$ 同 (9)。

**焦点知识蒸馏**：借鉴 Focal Loss，难样本与高相似度样本对赋高权。
- $s_{x_i,r_j} = \text{softmax}_r(\text{sim}(\mathbf{h}_{x_i},\mathbf{p}_r)/\tau_2)$；
- $w_{x_i,r_j} = s_{x_i,r_j}(1{-}P(y_i|x_i;\theta_k))^\gamma$；
- $\mathcal{L}_{\text{fkd}} = -\frac{1}{|\hat{M}_k|}\sum_{x_i}\sum_{r_j} w_{x_i,r_j} P(r_j|x_i;\theta_{k{-}1}) \log P(r_j|x_i;\theta_k)$。
总回放损失：$\mathcal{L}_{\text{replay}} = \mathcal{L}_{\text{cls}} + \lambda_1\mathcal{L}_{\text{c\_fkd}} + \lambda_2\mathcal{L}_{\text{l\_fkd}}$。

### 4.7 推理
$\hat{y}_i^* = \arg\max_{r} ((1{-}\alpha)P_c(x_i^*) + \alpha P_l(x_i^*))$。

---

## 实验与结果

### 数据集
- **FewRel**：80 关系 ×700 样本，分 10 个 disjoint task。
- **TACRED**：41 关系（去掉 no_relation），106,264 样本，分 10 个 task；类不平衡、每关系样本少，更难。

### 基线
EA-EMR、EMAR (BERT)、CML、RP-CRE、CRL、CRECL、KIP-Framework（含外部知识）。全部相同任务划分、memory size=10、5 次随机序列平均。

### 主结果（准确率 %，全部已见关系）
| 数据集 | 最佳任务 | 末期 ($T_{10}$) | 说明 |
|---|---|---|---|
| FewRel | $T_2$: **95.8** | **84.2 ±0.4** | SOTA，超越 KIP-Framework (82.5) |
| TACRED | $T_1$: 97.7 | **79.1 ±1.1** | SOTA，超越 KIP-Framework (78.6) |

**最强提升**：FewRel $T_2$ 较 CRECL (94.9) +0.9pt；末期较 CRL (83.1) +1.1pt。

### 关键子实验
- **类比关系**（原型相似度 >0.85）：Ours 71.1% Acc / 18.7% Drop（FewRel），70.4%/18.3%（TACRED），均最低 drop。
- **消融**（TACRED $T_6\text{-}T_{10}$ 均值）：w/o FKD -0.7pt、w/o MA -0.7pt、w/o CM -0.8pt、w/o DP -0.7pt、w/o SP -0.8pt，各模块必要。
- **Memory size 敏感性**：size=2/5/15 均 SOTA 或接近；size 越小优势越大，鲁棒性明显。
- **ACA 组合**：Ours + ACA 在 FewRel 略微下降（-0.3pt），作者解释为二者目标重叠。
- **不相似关系**（最大相似度 <0.7）：Ours 92.4%/4.1% drop（FewRel），SOTA。

---

## 相关工作脉络

1. **EA-EMR (Wang et al., 2019)**：最早引入记忆回放 + embedding aligned，关注表征失真但未考虑类比区分与过拟合。
2. **EMAR (Han et al., 2020)** / **CRL (Zhao et al., 2022)** / **CRECL (Hu et al., 2022)**：三类对比学习导向方法，以原型相似度做分类；CEAR 明确指其"预测对原型敏感、类比易混"，并以整合训练+FKD 补强。
3. **RP-CRE (Cui et al., 2021)** / **KIP-Framework (Zhang et al., 2022)**：线性分类 + 原型增强路线；KIP 依赖外部知识 prompt，CEAR 不引入任何外部信号。
4. **CML (Wu et al., 2021)**：课程学习 + 元学习降低顺序敏感性，仍无存样过拟合建模。
5. **ACA (Wang et al., 2022)**：唯一显式考虑类比遗忘的 adversarial class augmentation；CEAR 通过 loss 层面 Focal 加权解决，二者互补性有限（实验显示组合后反而略降）。
6. **通用 CL 回顾**：正则化派（EWC、SI）、动态架构派（PackNet、BNS）、记忆派（GEM、iCaRL）；CEAR 定位在记忆派内部、针对 RE 任务的过拟合 × 类比双重挑战。

---

## 局限性与未来方向

1. **存储依赖**：需存典型样本 + 静态原型，性能受存储容量上限约束。
2. **存样质量敏感**：k-means 选出的"典型样本"若质量差（如边缘噪声），会拖累原型与增强效果。
3. **未探索大模型**：最近 LLM 进展可能缓解遗忘与过拟合，本文未涉及。
4. **与 ACA 存在重叠**：组合 ACA 后性能轻微下降，说明类比处理策略有冗余。
5. **未来方向**：迁移至 few-shot RE、探索与其他增强方法的组合、引入 LLM 辅助。

---

## 研究启发与可借鉴点

1. **"类比关系相似度突增 → 准确率骤降"的实证范式**可作为 continual RE 诊断工具：任何新方法都应在 "sim>0.85 关系上的 Acc/Drop" 报告，便于横向对比。
2. **记忆增强（实体替换 + 无关句拼接）的思路可泛化**到其他 memory-based CL 场景（如 continual NER、dialogue state tracking），只要保持"增强样本不入原型"的隔离原则。
3. **Focal-KD 的权重设计**（相似度 softmax × $(1-P)^\gamma$）是一个即插即用的"难样本聚焦"模块，可直接嫁接至任意蒸馏-based continual 方法。
4. **静态/动态原型加权** $(1{-}\beta)\mathbf{p}^{\text{static}} + \beta\mathbf{p}^{\text{dynamic}}$ 提供了"历史知识锚定 + 当前特征适应"的清晰权衡——在 relation-level 任务中比直接在 sample 层做 regularization 更稳定。
5. **整合训练（对比 + 线性）的联合优化**比单独使用更能拉开类比原型的间距（Figure 3 可视化直观），是 RE 类任务值得默认基线。

---

## 关键术语表

- **Continual Relation Extraction (CRE)**：关系集合随时间不断扩展的关系抽取任务，要求在学习新关系的同时保持旧关系性能。
- **Catastrophic Forgetting**：在新任务学习过程中，模型对已学任务知识的急剧遗忘。
- **Analogous Relations**：语义相近、原型余弦相似度高的关系（文中以 >0.85 为阈值），是遗忘重灾区。
- **Memory-Insensitive Prototype**：静态（首轮全部样本均值）与动态（存样均值）的加权和，降低对少量存样的敏感度。
- **Memory Augmentation**：实体替换 + 无关句拼接生成合成存样，使回放样本量 ×4，且不参与原型计算。
- **Integrated Training**：InfoNCE 对比损失与交叉熵线性损失联合优化，兼顾特征空间对齐与任务特异决策边界。
- **Focal Knowledge Distillation**：借鉴 Focal Loss，按"原型相似度 × 分类置信度补"加权蒸馏，迫使模型聚焦难区分类比对。
- **Drop**：从该关系首次学完到当前任务结束的平均准确率下降幅度，衡量遗忘程度。

---

## 可复现要素

- **数据集**：FewRel、TACRED 均为公开基准；作者从 RP-CRE 开源代码获取相同任务划分。
- **代码**：已开源 https://github.com/nju-websoft/CEAR。
- **关键超参**（Appendix B）：FewRel $\alpha{=}0.5,\beta{=}0.5,\tau_1{=}0.1,\mu{=}0.5,\omega{=}0.1,\tau_2{=}0.5,\gamma{=}1.25,\lambda_1{=}0.5,\lambda_2{=}1.1$；TACRED $\alpha{=}0.6,\beta{=}0.2,\tau_1{=}0.1,\mu{=}0.8,\omega{=}0.15,\tau_2{=}0.5,\gamma{=}2.0,\lambda_1{=}0.5,\lambda_2{=}0.7$；memory size=10，5 次随机序列平均。
- **环境**：Python 3.9.7、PyTorch 1.11.0、单卡 NVIDIA RTX A6000 48GB。
- **编码器**：BERT-base（公开权重）。

---
