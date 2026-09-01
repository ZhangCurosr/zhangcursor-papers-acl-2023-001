---
title: "HyPe-Better-Pre-trained-Language-Model-Fine-tuning-with-Hidd"
source: https://aclanthology.org/2023.acl-long.182.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:24:23"
field: "预训练语言模型高效微调"
keywords: ["PLM 微调", "隐表示扰动", "过拟合缓解", "GLUE", "表示坍缩", "正则化"]
innovations: ["在 Transformer 各层隐状态间直接注入无偏随机噪声以增强微调鲁棒性", "证明 HyPe 与 R-Drop/R3F/ChildTuning/NoisyTune 等方法正交可叠加"]
benchmarks: ["GLUE", "advGLUE", "SNLI", "SciTaiL", "SICK"]
---

# 论文速读：HyPe-Better-Pre-trained-Language-Model-Fine-tuning-with-Hidden-Representation-Perturbation

## 一句话总结
本文提出 HyPe（Hidden Representation Perturbation），一种通过在 Transformer 各层隐藏表示中注入无偏随机噪声来缓解预训练语言模型微调过拟合与表示坍缩的轻量级微调增强技术，在 GLUE 等基准上稳定超越 Vanilla 及 R-Drop 等 SOTA 方法，且计算开销极小。

## 研究问题与动机
- **微调不稳定**：PLM 微调后在下游任务上性能方差较大，常出现**过拟合（over-fitting）**或**表示坍缩（representation collapse）**，尤其在低资源场景下更严重（Dodge et al., 2020; Aghajanyan et al., 2021）。
- **现有噪声方法局限**：已有工作（如 NoisyTune、R3F、ChildTuning、R-Drop）仅对输入嵌入、参数或梯度加噪，或需要 KL 散度/两次前向等额外计算；注入隐藏表示层间噪声的研究相对缺失。
- **层间语义多样性被忽视**：Tenney et al. (2019) 指出不同 Transformer 层编码不同类型的语言信息，隐藏表示承载着比输入 embedding 更丰富的语义，值得直接扰动以提升各层鲁棒性。

## 核心贡献（创新点）
1. **提出 HyPe 隐表示扰动微调框架**：在各 Transformer 层输入处添加参数无关的随机噪声（N 或 U 分布），无需额外正则项，显著提升微调稳定性与泛化。
2. **与 SOTA 方法对比更全面且开销更低**：在 GLUE 小数据集上平均超越 R-Drop 0.15，同时仅需 ~11GB GPU 显存，远低于 R-Drop/R3F 的 16GB。
3. **发现 Dropout 与 HyPe 的功能重叠并给出替代策略**：HyPe-N 关闭 Dropout 效果更佳，二者作用机制类似（离散 vs 连续噪声），HyPe 可作为 Dropout 的优质连续噪声替代。
4. **系统验证跨维度增益**：证实 HyPe 在**模型缩放**（Base→XXL）、**对抗鲁棒性**（advGLUE 最高 +8.20）、**任务/领域泛化**（NLI 跨域 probe）及**方法兼容性**（与 R-Drop/R3F/ChildTuning/NoisyTune 叠加持续增益）上均有效。

## 方法详解
- **基本设定**：PLM 前向映射 $f_\theta(x) = g_{\theta^n} \circ g_{\theta^{n-1}} \circ \cdots \circ g_{\theta^1}(x)$，其中 $g_{\theta^i}$ 为第 i 个 Transformer 层。
- **HyPe 前向过程**（Algorithm 1）：
  1. 输入 token 序列经 Embedding 得 $h^1$。
  2. 对每一层 $i$：从 $\mathcal{N}(0,\sigma^2)$（HyPe-N）或 $\mathcal{U}(-\sigma,\sigma)$（HyPe-U）采样噪声 $\varepsilon^i$，令 $h^i \leftarrow h^i + \varepsilon^i$，再送入 $h^{i+1} = g_{\theta^i}(h^i)$。
  3. 最终分类头输出 $\hat{y} = c_\psi(h^n)$。
- **训练目标**：与传统微调一致，$\mathcal{L}^{\text{HyPe}}(\theta,\psi) = \mathcal{L}(c_\psi(f_\theta^{\text{HyPe}}(x)), y)$，无需新增正则损失。
- **超参数**：噪声尺度 $\sigma \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$，实践中 $\sigma=10^{-5}$ 或 $10^{-4}$ 最佳；使用 AdamW(lr 网格搜索 $1\sim4\times10^{-5}$)，batch size 16/32，3 个随机种子取平均。

## 实验与结果
- **数据集**：GLUE 基准（STS-B、CoLA、MRPC、RTE 四个小数据集 + SST2、QNLI、QQP、MNLI 大任务）；低资源子集（各任务 1k 样本）；advGLUE 对抗集；NLI 跨域泛化（SNLI、SciTaiL、SICK）。
- **最强结果**（BERT-large，小数据集平均）：
  - HyPe-N：**80.75**（STS-B 90.37 / CoLA 66.26 / MRPC 91.98 / RTE 74.37），超越 Vanilla 1.60，超越 R-Drop 0.15。
  - HyPe-U：**80.60**。
  - 在 RoBERTa-large 小数据集平均提升 **0.89**；XLNet 提升 **7.45**，ELECTRA 提升 **5.80**（因 XLNet/ELECTRA Vanilla CoLA 方差极高导致均值偏低，HyPe 显著稳定收敛）。
- **低资源（1k 样本，RoBERTa-large）**：HyPe-N/U 平均分别提升 **+7.37 / +7.96**，RTE 最高提升达 **+13.00 / +16.97**。
- **参数缩放（DeBERTa Base→XXL）**：各规模均稳定提升，平均增益 +0.72 / +0.29 / +0.63 / +0.39，证明与模型规模正交互补。
- **对抗鲁棒（advGLUE, BERT-large）**：HyPe-N 在 advRTE 提升 +8.10，advQNLI 提升 +8.20。
- **方法兼容性**：与 R-Drop/R3F/ChildTuning_D/NoisyTune 组合均在 MRPC/STS-B/CoLA/RTE 四任务上带来额外增益（如 HyPe-N + ChildTuning_D on RoBERTa 平均 +1.10）。
- **Token 表示各向同性分析**：HyPe 使高层隐藏表示的 token 间余弦相似度更低（分布更 isotropic），缓解表示坍缩。

## 相关工作脉络
- **R-Drop (Liang et al., 2021)**：通过两组 dropout mask 的 KL 散度正则化提升泛化；HyPe 无需额外前向，仅需一次前向加噪，显存节省 ~5GB。
- **R3F (Aghajanyan et al., 2021)**：在输入嵌入上加对抗噪声并配合 KL 正则；HyPe 直接扰动中间层隐状态，语义信息更丰富且开销更低。
- **ChildTuning (Xu et al., 2021)**：通过梯度掩码仅更新部分参数；HyPe 全参训练但通过隐状态扰动间接正则化，更通用且无需 Fisher 矩阵预计算。
- **NoisyTune (Wu et al., 2022)**：在预训练参数上加权噪声；HyPe 将扰动位置移至网络内部表示层，解耦了参数扰动与表征扰动的贡献。
- **SMART / FreeLB (Jiang et al., 2020; Zhu et al., 2019)**：基于信任域/对抗训练的输入扰动方法；HyPe 无需梯度反传求对抗扰动，只需采样简单分布噪声。
- **Top-K Tuning / Mixout / RecAdam**：约束微调参数偏离预训练值的正则化思路；HyPe 不约束参数空间，而在表征空间加噪，属于正交视角。

## 局限性与未来方向
- **数据充足时提升边际**：在大规模数据集（Table 2）上平均增益仅 0.17–0.22，HyPe 主要价值集中在低资源场景。
- **引入超参数搜索负担**：需额外调优噪声分布形式与尺度 $\sigma$（不同任务最优组合不同，如 CoLA 宜用 N 小尺度，MRPC 宜用 U 小尺度）。
- **未来方向**：探索更优的噪声分布/结构（非独立同分布）、仅对高敏感层加噪的策略、结合梯度信息自适应调整 $\sigma$，以及扩展至生成/多模态任务。

## 研究启发与可借鉴点
1. **层间噪声注入作为通用正则化手段**：HyPe 的思路可迁移至编码器-解码器架构（仅在 Encoder 层加噪）或多模态 Transformer，作为 Dropout 的连续替代。
2. **表示各向同性（isotropy）作为微调质量指标**：论文通过 token 间余弦相似度量化表示退化，该方法可直接用于诊断其他微调策略的效果。
3. **"关闭 Dropout + 加连续噪声"的组合策略**：揭示了两类正则手段的功能重叠，提示未来设计应优先选择机制正交的增强模块。
4. **与现有 SOTA 微调技术无缝叠加**：HyPe 的纯前向注入方式不与 R-Drop/KL 正则、梯度掩码等冲突，可作为即插即用组件加入现有 pipeline。

## 关键术语表
- **HyPe（Hidden Representation Perturbation）**：在 Transformer 各层输入隐状态上加随机噪声的微调增强方法。
- **Representation collapse（表示坍缩）**：微调导致 PLM 隐藏表示分布退化至低维锥内、多样性下降的现象。
- **Isotropic distribution（各向同性分布）**：表示在隐藏空间中均匀分散而非集中在狭窄锥内的理想性质，与更好泛化正相关。
- **Token cosine similarity**：样本内各 token 隐藏表示间的平均余弦相似度，用于衡量表示各向异性程度。
- **advGLUE**：针对 GLUE 五个任务构建的文本对抗扰动评测基准。
- **R-Drop**：通过双重 dropout 输出的 KL 散度正则化 PLM 微调的方法（NeurIPS 2021）。
- **R3F**：在输入嵌入上施加对抗扰动的鲁棒微调方法（ICLR 2021）。

## 可复现要素
- **数据集**：GLUE（公开，CC-BY-4.0）、advGLUE（公开）、SNLI/SciTaiL/SICK（各有对应许可证）；低资源实验为作者自行 subsample 的 1k 子集，未单独发布。
- **代码/权重**：代码已开源 → https://github.com/Yuanhy1997/HyPe；使用 Huggingface Hub 上的 BERT/RoBERTa/XLNet/ELECTRA/DeBERTa 官方预训练权重。
- **关键超参**：学习率 $\{1,2,3,4\}\times10^{-5}$ 网格搜索；$\sigma \in \{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$；batch size 16（BERT/RoBERTa）或 32（ELECTRA/XLNet）；3 epochs / 3 个随机种子；关闭 Dropout；AdamW（$\beta=(0.9,0.99), \epsilon=10^{-5}$，weight decay 0.1），线性 warmup 10% steps；序列长度截断至 128 tokens。
