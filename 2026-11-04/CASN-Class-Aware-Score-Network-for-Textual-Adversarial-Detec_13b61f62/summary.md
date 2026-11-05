---
title: "CASN-Class-Aware-Score-Network-for-Textual-Adversarial-Detec"
source: https://aclanthology.org/2023.acl-long.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 21:57:45"
field: "文本对抗安全检测"
keywords: ["textual adversarial detection", "score-based generative model", "denoising score matching", "supervised contrastive learning", "Langevin dynamics", "adversarial robustness"]
innovations: ["提出class-aware score network结合denoising score matching与监督对比学习，缓解分布重叠问题", "通过多步去噪漂移距离（余弦相似度累积）作为对抗置信度评分指标", "利用predictor-corrector SDE求解提升去噪质量，检测性能接近100%"]
benchmarks: ["SST-2", "IMDB", "AG-NEWS"]
---

# 论文速读：CASN-Class-Aware-Score-Network-for-Textual-Adversarial-Det

## 一句话总结
本文提出CASN（Class-Aware Score Network），一种基于score-based生成模型的文本对抗检测方法，通过多步Langevin去噪过程估计数据分布对数密度梯度，并结合监督对比学习保留不同类别数据的各向异性，显著提升对抗样本检测性能。

## 研究问题与动机
1. 预训练语言模型（PLMs）易受文本对抗攻击威胁，现有对抗防御（如对抗训练）会牺牲原始准确率且限制攻击空间。
2. 主流密度估计方法（如RDE、MDRE）假设对抗样本远离正常数据流形，但研究表明对抗样本往往贴近非对抗数据流形边缘，导致分布重叠问题严重削弱检测性能。
3. 文本对抗样本语义相似度高（如TextFooler-adj使用严格同义约束），传统基于特征密度或词频的方法难以区分细微语义差距。
4. 需要一种既能避免分布重叠缺陷、又能充分利用标签信息指导分布建模的检测框架。

## 核心贡献（创新点）
1. 提出基于去噪score matching的新范式，通过Langevin动力学多步去噪刻画对抗样本到正常数据流形的漂移距离，避免显式密度估计的重叠问题。
2. 在score network训练中引入监督对比学习，强制模型利用标签信息区分不同类别的条件分布，防止坍塌到单一数据流形并保留分布各向异性。
3. 设计基于去噪过程余弦相似度的对抗置信度评分机制，融合预测器-校正器（predictor-corrector）的SDE求解方式提升去噪质量。
4. 在三个文本分类数据集上对四种先进攻击算法的实验显示，CASN平均超越之前SOTA方法15.2 F1分数，多数场景下达到接近100%准确率。

## 方法详解
**1. 编码表示获取**：冻结预训练编码器E（BERT/RoBERTa），对文本x获取句级表示$h = E(x)$，作为score network的输入。

**2. Denoising Score Matching训练**：
- 对表示$h$施加多级高斯噪声扰动：$p_\alpha(\tilde{h}|h) = \mathcal{N}(\tilde{h}|\sqrt{\alpha}h, (1-\alpha)I)$，其中$\alpha$从1递减至0。
- 训练score network $s_\theta(\tilde{h},\alpha)$近似$\nabla_h \log p(h)$，优化目标为：
$$L(\theta)_\alpha = \frac{1}{T}\sum_{i=1}^T (1-\alpha_i) \cdot \frac{1}{2}\mathbb{E}\left[\left\|s_\theta(\tilde{h},\alpha_i) + \frac{\tilde{h}-\sqrt{\alpha_i}h}{1-\alpha_i}\right\|^2\right]$$

**3. 监督对比学习正则化**：
- 正样本对：原始表示$h_i$与其轻微噪声扰动$\tilde{h}_i$
- 负样本对：不同标签的所有表示（含/不含噪声扰动）
- 对比损失：$L(\theta)_{cons} = -\sum_{i\in I}\frac{sim(h_i,\tilde{h}_i)}{\sum_{a\in A(i)}[sim(h_i,h_a)+sim(h_i,\tilde{h}_a)]}$
- 总损失：$L(\theta) = L(\theta)_\alpha + \lambda L(\theta)_{cons}$

**4. 基于去噪的检测**：
- 采用predictor-corrector形式交替求解SDE和Langevin动力学：
  - Predictor（ODE）：$h_i \leftarrow (2-\sqrt{1-\beta_{i+1}})h_{i+1} + \frac{1}{2}\beta_{i+1}s_{\theta^*}(h_{i+1},\beta_{i+1})$
  - Corrector（Langevin）：$h_i \leftarrow h_i + \epsilon_i s_{\theta^*}(h_i,\beta_i) + \sqrt{2\epsilon_i}z$
- 累积去噪各步的余弦相似度作为对抗置信度：$\text{confidence} = \sum_{i=\text{start}}^{0} \cos\langle h_i^{\text{mean}}, h_{\text{start}}^{\text{mean}}\rangle$
- 对抗样本表现为较大的特征漂移（较低相似度），通过阈值过滤检测。

## 实验与结果
**数据集**：SST-2（二分类情感，短句）、IMDB（二分类情感，长句）、AG-NEWS（四分类主题）。
**Victim模型**：BERT-base-uncased、RoBERTa-base。
**攻击算法**：BAE、PWWS、TextFooler、TextFooler-adj（共4种，使用TextAttack默认设置）。
**评估指标**：F₁、AUROC、Accuracy（ACC）。

**主要结果**（BERT victim，Scenario 1）：
- SST-2 + TextFooler-adj：CASN F₁=80.8，超越RDE（72.3）8.5分；BAE攻击下F₁=97.2、ACC=97.1。
- IMDB + TextFooler-adj：CASN F₁=97.8/AUROC=99.7，显著超越RDE（82.2）。
- AG-NEWS：对BAE/TextFooler/PWWS均接近100% AUROC和ACC。
- **平均提升**：较之前SOTA（RDE）平均提高**15.2 F₁分数**。

**关键结论**：CASN对强约束语义攻击（TextFooler-adj）仍保持高检测率，验证了去噪漂移指标相比密度估计的优越性。

## 相关工作脉络
1. **RDE（Yoo et al., 2022）**：基于多元高斯分布建模clean样本特征密度，低密度区域判为对抗样本——CASN通过梯度估计+去噪漂移规避了显式密度重叠缺陷。
2. **MD（Lee et al., 2018）**：基于类条件高斯分布的Mahalanobis距离检测OOD和对抗样本——同为密度估计思路，CASN改用以score network估计的梯度场为指引。
3. **MDRE（Liu et al., 2022）**：利用局部内在维度（Local Intrinsic Dimensionality）建模表示分布——CASN利用更精细的多步去噪轨迹代替单点局部维度假设。
4. **FGWS（Mozes et al., 2021）**：基于词频特征检测低频词替换攻击——仅对特定攻击有效，CASN为通用检测方法。
5. **DISP（Zhou et al., 2019）**：基于扰动判别器+同义词替换的防御框架——依赖语义可替换性假设，CASN不依赖攻击类型先验。

## 局限性与未来方向
1. **推理速度慢**：多步Langevin迭代（起始步k通常为90-120步）导致检测耗时较高（3000样本约需1小时），难以满足实时需求。
2. **跨域泛化差**：score network高度依赖源域数据特征，跨域迁移（如IMDB→AG-NEWS）检测性能显著下降（AUROC常低于70%），限制了通用部署。
3. **未来方向**：探索少步快速去噪采样策略以加速推理；引入域不变性表征学习提升跨域迁移能力。

## 研究启发与可借鉴点
1. **Score-based方法用于检测任务的迁移价值**：将score network从图像生成拓展至NLP对抗检测，证明了去噪漂移距离作为"对抗置信度"的有效性和普适性。
2. **监督对比学习在去噪训练中的正则化作用**：通过在score matching中引入对比学习约束，可有效防止不同类别流形坍塌合并，这一思路可迁移至其他分布建模任务。
3. **Predictor-corrector去噪框架的借鉴**：交替使用SDE ODE解和Langevin动力学校正，可提升单步去噪质量，适用于任何需要高质量去噪轨迹的检测/净化场景。
4. **实验设计可借鉴**：同时评估Scenario 1（检测全部对抗样本）和Scenario 2（仅检测成功攻击），更全面反映实际部署风险。

## 关键术语表
**Score Matching**：通过最小化score function（数据分布对数梯度的估计）与真实梯度之间的距离来拟合未归一化概率分布的训练方法。
**Denoising Score Matching（DSM）**：Vincent提出的改进版score matching，通过添加可计算的高斯噪声替代复杂的配分函数项，使高维数据上的score估计可行。
**Langevin Dynamics**：利用score function驱动的随机微分方程采样过程，通过多步迭代从噪声分布逐步恢复至数据分布。
**Class-Aware Score Network（CASN）**：本文提出的score network，结合denoising score matching与监督对比学习，建模类条件数据分布梯度。
**Supervised Contrastive Learning**：Khosla等人提出的对比学习变体，利用样本标签构建正负对，拉近同类样本、推远异类样本的表示。
**Predictor-Corrector Sampling**：Song等人提出的扩散模型采样策略，交替执行ODE预测步和Langevin校正步以提升生成质量。
**Adversarial Confidence Score**：本文提出的检测指标，通过去噪过程中表征漂移的累积余弦相似度衡量样本的对抗可能性。
**Anisotropy（各向异性）**：指不同类别数据分布在空间中呈现的方向性差异，监督对比学习旨在增强这种差异以避免流形坍塌。

## 可复现要素
- **数据集**：SST-2、IMDB、AG-NEWS均为公开数据集（论文附录A.3提供了具体规模：SST-2训练集67349/验证集872；IMDB训练/测试各25000；AG-NEWS训练集120000/测试集7600）。
- **代码/权重**：论文未提及代码开源声明。
- **关键超参**：
  - 噪声级别数$T=1000$；$\bar{\beta}_{min}=0.1, \bar{\beta}_{max}=20$
  - 对比损失系数$\lambda$：SST-2取0.15-0.2，IMDB取0.1，AG-NEWS取0.2
  - 去噪起始步$k$：SST-2取120，IMDB/AG-NEWS取90
  - 训练：AdamW，lr=$2\times10^{-5}$，dropout=0.1，batch size=64，20 epochs
  - Score network backbone：XLNET
