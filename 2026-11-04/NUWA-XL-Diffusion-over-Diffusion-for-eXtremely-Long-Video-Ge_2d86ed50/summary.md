---
title: "NUWA-XL-Diffusion-over-Diffusion-for-eXtremely-Long-Video-Ge"
source: https://aclanthology.org/2023.acl-long.73.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-06 02:42:33"
field: "视频生成与编辑"
keywords: ["长视频生成", "扩散模型", "Diffusion over Diffusion", "Coarse-to-fine", "并行推理", "视频生成"]
innovations: ["首次提出 Diffusion over Diffusion 层级架构实现极长视频并行生成", "首次在 3376 帧长视频上直接训练以消除训练-推理差距", "推理速度提升 94.26%（1024 帧从 7.55min 降至 26s）"]
benchmarks: ["FlintstonesHD", "Avg-FID", "B-FVD"]
---

# 论文速读：NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation

## 一句话总结
本文提出 NUWA-XL，一种"Diffusion over Diffusion"架构，通过"粗到细"的层级扩散过程实现极长视频生成；该模型首次直接在长达 3376 帧的长视频上训练，消除了训练-推理差距，同时支持并行推理，在生成 1024 帧时将推理时间从 7.55 分钟降至 26 秒（提速 94.26%）。

## 研究问题与动机
- **训练-推理差距**：现有方法在短片段（如少于 16 帧）上训练，却在推理时生成上千帧，导致生成的长视频出现不真实的镜头切换和长期不一致。
- **串行推理效率低**："Autoregressive over X"架构因滑动窗口的依赖限制，无法并行生成，如 TATS 生成 1024 帧需 7.5 分钟，Phenaki 需 4.1 分钟。
- **数据与计算资源瓶颈**：直接训练长视频需要大量算力，而现有数据集视频长度短、分辨率低、注释粗糙，难以支撑长视频生成研究。
- **长程一致性与镜头变化建模缺失**：短视频训练模型无法学习真实长视频中的镜头切换、人物动作连续性等模式。

## 核心贡献（创新点）
1. **"Diffusion over Diffusion"架构**：首次将长视频生成视为"粗到细"的层级过程，用全局扩散生成关键帧，再用局部扩散递归填充中间帧，区别于串行自回归范式。
2. **首次在 3376 帧长视频上直接训练**：打破"短训长推"的传统，从根本上消除训练-推理分布差距，这是已知首个直接训练于极长视频的模型。
3. **并行推理加速 94.26%**：全局扩散后各局部扩散可并行执行，在 1024 帧生成任务上将推理时间从 7.55 分钟降至 26 秒。
4. **构建 FlintstonesHD 基准数据集**：提供密集注释的长视频数据集，含 166 集《摩登原始人》卡通（平均 38000 帧/集，分辨率 1440×1080），填补长视频生成评测空白。
5. **提出 Mask Temporal Diffusion (MTD)**：设计支持条件帧输入与无条件输入的统一扩散骨干网络，通过多尺度注入与对称注入策略有效融合视觉条件。

## 方法详解
- **Temporal KLVAE (T-KLVAE)**：在预训练图像 KLVAE 的空间模块基础上，增加时序卷积与时序注意力层，并采用恒等初始化（权重初始化为恒等映射），使空间特征不受影响的同时引入时序建模能力；编码器输出紧凑潜码 $x_0 \in \mathbb{R}^{b \times L \times c \times h \times w}$。
- **Mask Temporal Diffusion (MTD)**：骨干为 Mask 3D-UNet，支持两种输入模式——全局扩散时无条件帧（全零），局部扩散时首尾帧作为视觉条件、中间帧被掩码；损失函数为 $\mathcal{L}_\theta = ||\epsilon - \epsilon_\theta(x_t, p, t, x_0^c)||_2^2$，其中 $x_0^c$ 为条件帧的潜码表示。
- **UpBlock 条件注入机制**：可视化条件 $x_0^c$ 通过零初始化卷积层转换为 scale ($w^c, w^m$) 和 shift ($b^c, b^m$)，以线性投影方式注入 hidden state；同时引入 Multi-scale Injection（多尺度注入）与 Symmetry Injection（对称注入， DownBlock 与 UpBlock 均注入条件）。
- **Diffusion over Diffusion 层级生成**：给定 L 个提示词，全局扩散首先生成 L 个关键帧（粗粒度剧情骨架）；随后每一层局部扩散以相邻帧为条件填充中间帧，视频长度按 $O(L^m)$ 指数增长（m 为扩散深度）；各局部扩散可并行执行。
- **推理公式**：标准 DDPM 采样步：$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\cdot)\right) + \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}\epsilon$，最终经 T-KLVAE 解码器还原为像素视频。

## 实验与结果
- **数据集**：FlintstonesHD，166 集卡通，平均 38000 帧/集，分辨率 1440×1080，使用 GIT2 模型逐帧生成密集 caption 并经人工过滤。
- **评估指标**：Avg-FID（帧级质量）、B-FVD-16（分段视频质量）、推理时间。
- **与 SOTA 对比（FlintstonesHD，128 分辨率）**：
  - 生成 1024 帧时：NUWA-XL/128 Avg-FID=35.79，显著优于 Phenaki (48.56) 和 FDM* (43.24)；B-FVD-16=572.86，优于 Phenaki (622.06) 和 FDM* (618.42)；推理时间仅 26 秒，比 Phenaki (259s) 快 89.96%，比 FDM* (453s) 快 94.26%。
  - 生成 256 帧时：NUWA-XL/128 Avg-FID=35.68，推理时间 17 秒（比 Phenaki 65s 快 85.09%）。
- **关键结论**：NUWA-XL 质量不随视频长度增长而下降（不同于 AR 方法的误差累积），且推理速度优势随长度增加更显著。
- **消融实验**：
  - T-KLVAE 恒等初始化（FID=4.35，FVD=11.88）优于随机初始化（FID=5.44，FVD=12.75）和基础 KLVAE（FID=4.71，FVD=28.07）。
  - MTD 完整版（含 MS+SI）FID=35.95，B-FVD=520.19，优于去掉多尺度注入或对称注入的变体。
  - 扩散深度 m=3、局部长度 L=16 为较优配置；L=32 时 OOM，L=8 时质量下降明显。

## 相关工作脉络
- **Phenaki (Villegas et al., 2022)**：AR over AR 架构，C-ViViT 编码+MaskGiT 骨干，在少于 16 帧上训练却可生成上千帧，存在严重训练-推理差距，串行推理耗时 4.1 分钟（1024 帧）。
- **TATS (Ge et al., 2022)**：时间无关 VQGAN+时间敏感 Transformer，滑动窗口生成，7.5 分钟/1024 帧，同样面临短期训练长期推理的不匹配问题。
- **FDM (Harvey et al., 2022)**：AR over Diffusion，基于 DDPM 的长视频完成模型，串行推理耗时 453 秒（1024 帧），需单独实现文本条件版本以公平比较。
- **NUWA-Infinity (Wu et al., 2022a)**：AR over AR 的无限视频生成，局部自回归生成 patch、全局自回归建模 patch 间一致性，本工作与之形成"Diffusion over Diffusion"的对照范式。
- **MCVD (Voleti et al., 2022)**：随机独立掩码过去/未来帧的视频扩散模型，支持多种视频生成任务，但与 NUWA-XL 的层级并行策略不同。
- **LDM / Imagen Video / Make-A-Video**：短视频/图像扩散模型的先驱工作，NUWA-XL 在它们的基础上引入时序扩展与层级扩散思想，面向极长视频场景。

## 局限性与未来方向
- **数据集领域受限**：仅在卡通《摩登原始人》上验证，缺乏电影、电视剧等开放域长视频，作者正构建开放域长视频数据集计划扩展。
- **数据需求巨大**：直接训练长视频对数据量和数据质量要求极高，限制了模型在更小数据集上的适用性。
- **并行推理依赖充足 GPU 资源**：加速收益需要合理的并行 GPU 算力支持，在资源受限场景下优势可能打折。
- **分辨率与生成长度的 trade-off**：L=32 时出现 OOM，说明在固定显存下生成长度与分辨率之间存在约束，需进一步优化显存效率。

## 研究启发与可借鉴点
- **"粗到细"层级生成范式可迁移**：将复杂生成分解为全局骨架+局部填充的两阶段策略，可借鉴到长序列视频补全、视频超分、时序插帧等任务中。
- **恒等初始化时序扩展预训练权重**：在冻结的空间模块基础上追加时序层并以恒等初始化，是一种安全高效的视频化方案，可推广至其他预训练图像模型的视频化改造。
- **条件帧的多尺度+对称注入设计**：MTD 中将条件信息同时注入 DownBlock 和 UpBlock 的多尺度策略，对条件扩散模型的条件融合具有通用参考价值。
- **B-FVD 分段评估指标的提出**：将长视频分割为短片段计算 FVD 再取平均，为长视频生成评估提供了一种简洁有效的量化方案，值得在后续工作中沿用。
- **并行推理的时间-质量权衡**：NUWA-XL 展示了并行化可带来数量级加速，启发团队在视频生成任务中探索非串行的并行生成架构设计。

## 关键术语表
- **Diffusion over Diffusion**：由全局扩散生成关键帧、局部扩散递归填充中间帧的层级生成架构，有别于传统的自回归串行范式。
- **T-KLVAE (Temporal KLVAE)**：在预训练图像 KLVAE 基础上增加时序卷积与时序注意力（恒等初始化）的视频压缩变分自编码器。
- **MTD (Mask Temporal Diffusion)**：支持条件帧输入的统一扩散骨干，通过掩码机制区分全局/局部扩散的不同输入模式。
- **Avg-FID**：生成视频所有帧的平均 Fréchet Inception Distance，衡量帧级图像质量。
- **B-FVD (Block FVD)**：将长视频分段后计算每段的 FVD 再取平均，衡量局部时序连贯性。
- **Multi-scale Injection (MS)**：将视觉条件同时注入 Mask 3D-UNet 的多层 DownBlock 和 UpBlock 的条件融合策略。
- **Symmetry Injection (SI)**：将视觉条件对称地同时注入 DownBlock 和 UpBlock 而非仅注入 UpBlock 的设计。
- **Coarse-to-fine**：先粗粒度生成关键帧、再细粒度填充中间帧的"由粗到细"生成流程。

## 可复现要素
- **数据集**：FlintstonesHD（由作者构建，论文未声明开源状态）；原片来源为公共卡通《The Flintstones》。
- **代码/权重**：论文未提及代码和模型权重是否开源；主页链接为 https://msra-nuwa.azurewebsites.net/。
- **关键超参**：全局/局部扩散步数 T 未明确给出；扩散深度 m=3、局部长度 L=16 为最优配置；生成分辨率 128/256；T-KLVAE 时序层 kernel size 奇数、恒等初始化。
