---
title: "Improving-Pretraining-Techniques-for-Code-Switched-NLP"
source: https://aclanthology.org/2023.acl-long.66.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:37:58"
---

# 论文速读：Improving-Pretraining-Techniques-for-Code-Switched-NLP

## 一句话总结
本文针对代码切换（code-switched）预训练语料稀缺的问题，提出感知语言边界的改进型 MLM 目标（SWITCHMLM/FREQMLM）与残差架构优化（RESBERT + 辅助 LID 损失），在多种印欧语言对的情感分析与问答任务上实现显著性能提升，并借助探针实验验证了模型对语言切换信息的显式编码能力。

## 研究问题与动机
1. 多语言预训练模型（mBERT、XLM-R）直接提取的表征在代码切换任务上效果有限，中间预训练可改善该问题，但高质量代码切换无标注文本获取困难。
2. 标准 MLM（STDMLM）采用固定比例随机遮盖 token，完全忽略代码切换句子中固有的语言边界（switch-point）结构信息。
3. 现有代码切换 NLP 工作多依赖生成式数据增强或混合真实/合成语料，缺乏对“如何在有限真实数据上让预训练目标与架构协同感知语言边界”的系统性探索。

## 核心贡献（创新点）
1. **SWITCHMLM 目标**：仅对语言边界附近的 token 施加 [MASK]，强制模型学习切换位置信息；与 STDMLM 的本质区别在于从随机遮盖转向基于语言学结构的定向遮盖。
2. **FREQMLM 替代方案**：在缺乏人工 LID 标注时，通过双语单语语料库的词频/NLL 统计推断 token 语言身份；与神经 LID 分类器方案的区别在于无需额外训练标注器，直接复用频率先验。
3. **RESBERT 架构改造**：在 mBERT 中间层与 MLM head 之间引入带 Dropout 的残差连接；与标准 Transformer 堆叠结构的区别在于打通底层语言表征到预测头子的直接通路。
4. **辅助 LID 损失 $\mathcal{L}_{aux}$**：在残差源层增加 MLP 预测语言边界，并施加 L2 正则防止表征偏移；与纯 MLM 预训练的区别在于显式注入边界识别监督信号，形成多任务预训练范式。

## 方法详解
- **遮蔽函数定义**：输入句子 $X$ 与 LID 指示向量 $S=\{0,1\}^n$，标准遮蔽函数为 $X_{\mathrm{mlm}} = \mathcal{M}(X, S, f)$，STDMLM 中 $S=\{1\}^n$（全 token 可遮）。
- **SWITCHMLM**：依据 token 的 LID 标签构造 $S$，仅当相邻 token 语言不同时将其及其邻域标记为可遮（如示例 `Laptop EN / mere HI / bag EN / me HI` 对应 $S=[1,1,1,1,0,0]$）。
- **FREQMLM**：对 token $x$ 计算其在嵌入语 $\mathcal{C}_{\mathrm{en}}$ 与矩阵语 $\mathcal{C}_{\mathrm{lg}}$ 中的负对数似然 `nll_en`/`nll_lg`（缺失或极低频置为 -1）。按阈值规则分配 LID（EN/LG/AMB-EN/AMB-LG/OTHER），将 AMB 映射回对应语言后执行边界遮蔽。
- **RESBERT 残差连接**：从第 $x$ 层（$x \in \{1,\dots,10\}$）引出路径，经 Dropout ($p=0.5$) 后加至最后一层输出，再送入 MLM head。经验表明：STDMLM 配合较深层（如 layer 9）收益最大，SWITCHMLM 配合较浅层（如 layer 2）更优。
- **辅助损失公式**：
  $$
  \mathcal{L}_{\mathrm{aux}} = \alpha \sum _{i=1} ^{n} - \log \mathrm {MLP} ( x_ {i} ) + \beta \sum _{j=1} ^{L} | | \bar { \mathbf { W } } ^{j} - \mathbf { W } ^{j} | |^{2}
  $$
  第一项为边界预测交叉熵，第二项为各层嵌入矩阵的 L2 正则；默认 $\alpha=5\mathrm{e}{-2}$，$\beta=5\mathrm{e}{-4}$。

## 实验与结果
- **数据集**：HI-EN (185K 句子 + L3CUBE-185k)、ES-EN (66K)、TA-EN (118K)、ML-EN (34K)；下游任务采用 GLUECoS 的 QA（HI-EN，295 train / 54 test）与 SA（四语言对）。
- **基线**：mBERT/XLM-R 零预训练、STDMLM、SWITCHMLM、FREQMLM、RESBERT 各变体。
- **主要结果**：
  - **QA (HI-EN, mBERT)**：SWITCHMLM 达 69.0±3.7 F1（20 epochs），相对 STDMLM（64.8）提升约 **+5.8 F1**；引入 L3CUBE 数据与 RESBERT+L_aux 后最佳达 **69.8±3.0 F1**（40 epochs）。
  - **SA (TA-EN)**：SW/FREQMLM+RESBERT 达 68.9±0.4 F1，相对 STDMLM（67.3）提升约 **+2.7 F1**。
  - **SA 全语言对最优**：SW/FREQMLM + RESBERT + $\mathcal{L}_{\mathrm{aux}}$ 在四对语言上均居首（HI-EN 69.1、ES-EN 63.7、TA-EN 68.9、ML-EN 77.2 F1）。
  - **统计显著性**：SWITCHMLM vs STDMLM 在 QA/SA 上 Wilcoxon Signed Rank test $p<0.05$。
  - **Probing 验证**：线性探测与条件探测均证实，改进方法使各层表示携带更多语言边界信息；STDMLM+RESBERT 从深层（layer 9）获取边界信息最多，SWITCHMLM 因浅层已富含该信息，残差收益相对平缓。
  - **FREQML
