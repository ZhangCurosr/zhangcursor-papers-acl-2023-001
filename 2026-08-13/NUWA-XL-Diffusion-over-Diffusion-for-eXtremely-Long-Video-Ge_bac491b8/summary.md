---
title: "NUWA-XL-Diffusion-over-Diffusion-for-eXtremely-Long-Video-Ge"
source: https://aclanthology.org/2023.acl-long.73.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:40:24"
field: "视频生成"
keywords: ["long video generation", "diffusion model", "coarse-to-fine", "parallel inference", "training-inference gap"]
innovations: ["提出Diffusion over Diffusion架构，通过粗到细层次化过程直接训练长视频", "首次直接在3376帧长视频上训练，消除训练-推理差距", "支持并行推理，1024帧生成时间从7.55分钟降至26秒（提升94.26%）"]
benchmarks: ["FlintstonesHD"]
---

# 论文速读：NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation

## 一句话总结
NUWA-XL 提出了一种 "Diffusion over Diffusion" 架构，通过"粗到细"的层次化过程直接在长视频（3376 帧）上训练，消除训练-推理差距，并支持并行推理；在生成 1024 帧时推理时间从 7.55 分钟降至 26 秒（提升 94.26%），同时显著改善长程一致性和镜头切换真实感。

## 研究问题与动机
1. **训练-推理差距（Training-Inference Gap）**：现有方法（如 Phenaki、TATS）仅训练于短视频（<16 帧），却需推理长达 1024 帧，导致长程不一致和不真实镜头切换。
2. **序列生成效率低下**：滑动窗口/自回归架构无法并行，TATS 生成 1024 帧需 7.5 分钟，Phenaki 需 4.1 分钟。
3. **数据集局限**：现有视频数据集长度短、分辨率低、标注稀疏，缺乏适合长视频生成研究与评测的数据。
4. **计算资源挑战**：直接在像素空间训练长视频扩散模型计算成本极高。

## 核心贡献（创新点）
1. **提出 "Diffusion over Diffusion" 架构**：将长视频生成视为"粗到细"过程，全局扩散生成关键帧，局部扩散递归填充中间帧；与"Autoregressive over X" 的本质区别在于消除序列依赖，支持并行推理并直接在长视频上训练。
2. **首个直接在长视频（3376 帧）上训练的模型**：通过层次化设计，NUWA-XL 首次实现直接训练长视频，从根本上缩小训练-推理差距；与现有方法仅训练短片的本质区别在于模型能学习长程依赖和真实镜头切换模式。
3. **并行推理加速 94.26%**：全局扩散生成关键帧后，所有局部扩散可并行执行；与自回归方法的本质区别在于推理耗时不再随视频长度线性增长。
4. **构建 FlintstonesHD 数据集**：提供 166 集动画（平均 38000 帧/集，1440×1080 分辨率）及密集帧级标注，填补长视频生成评测基准空白。

## 方法详解
### 1. Temporal KLVAE（T-KLVAE）
- 在预训练图像 KLVAE 基础上，每个空间卷积后插入时间 1D 卷积，每个空间注意力后插入时间注意力。
- **初始化策略**：时间卷积核 $W^{conv1d} \in \mathbb{R}^{c_{out} \times c_{in} \times k}$ 初始化为零，仅中心位置 $W^{conv1d}[i,i,(k-1)//2]=1$（恒等映射）；时间注意力输出投影 $W^{att\_out}=0$，确保初始化等价于原始 KLVAE。
- 编码器输出潜码 $x_0 \in \mathbb{R}^{b \times L \times c \times h \times w}$。

### 2. Mask Temporal Diffusion（MTD）
- **统一框架**：全局扩散仅输入文本提示 $p$（视觉条件全零）；局部扩散输入文本提示 + 首尾帧作为视觉条件 $v_0^c$（中间帧被 mask）。
- **前向扩散**：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。
- **训练损失**：$\mathcal{L}_\theta = ||\epsilon - \epsilon_\theta(x_t, p, t, x_0^c)||_2^2$。
- **Mask 3D-UNet 设计**：
  - 多尺度注入（Multi-scale Injection）：视觉条件经下采样后注入各层 DownBlock 和 UpBlock。
  - 对称注入（Symmetry Injection）：条件同时注入 DownBlock 和 UpBlock。
  - 零初始化卷积层生成 scale/shift 参数，通过线性投影注入隐藏状态。
  - 分离的空间/时间卷积与注意力：空间轴 $hw$ 视为 batch，时间轴 $L$ 视为序列长度。

### 3. Diffusion over Diffusion 架构
- **全局扩散**：输入 $L$ 个提示 $p_1$，生成 $L$ 个关键帧 $v_{01}$。
- **局部扩散迭代**：第 $i$ 层局部扩散以相邻帧为条件，生成中间 $L-2$ 帧，提示 $p_i$ 时间间隔逐步缩小。
- **视频长度扩展**：$m$ 层深度可扩展至 $O(L^m)$ 帧；例如 $L=16, m=3$ 可生成 $16^3=4096$ 帧。
- **并行性**：每层所有局部扩散任务相互独立，可并行执行。

### 4. 推理过程
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, p, t, x_0^c)\right) + \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}\epsilon$$
最终由 T-KLVAE 解码为视频像素。

## 实验与结果
### 数据集
**FlintstonesHD**：166 集《摩登原始人》动画，平均 38000 帧/集，分辨率 1440×1080，每帧由 GIT2 生成密集 caption 并经人工过滤。

### 评估指标
- **Avg-FID**：生成帧的平均 Fréchet Inception Distance。
- **B-FVD-X**：将长视频切为 X 帧片段，计算各片段 FVD 的平均值。
- **推理时间**。

### 主要结果（Tab. 1）
| 方法 | 1024f Avg-FID↓ | 1024f B-FVD-16↓ | 1024f Time↓ |
|------|---------------|-----------------|-------------|
| Phenaki | 48.56 | 622.06 | 259s |
| FDM* | 43.24 | 618.42 | 453s |
| **NUWA-XL/128** | **35.79** | **572.86** | **26s** |
| **NUWA-XL/256** | **32.07** | 642.87 | 51s |

- NUWA-XL/128 在 1024 帧生成上比 FDM 提速 **94.26%**（453s → 26s），比 Phenaki 提速 **90.0%**（259s → 26s）。
- 随帧数增加，NUWA-XL 质量几乎不下降（Avg-FID 稳定在 35-36），而 AR 方法显著恶化。
- 256 帧时比 FDM 提速 **85.09%**（114s → 17s）。

### 消融实验（Tab. 2）
- **T-KLVAE**：恒等初始化（T-KLVAE）优于随机初始化（T-KLVAE-R）和原始 KLVAE。
- **MTD 设计**：多尺度注入（MS）显著提升质量；对称注入（S）略有增益。
- **深度 m**：m=3 在 256/1024 帧上 B-FVD 最佳（542.26/572.86）。
- **局部扩散长度 L**：L=16 效果最佳，L=32 因显存不足（OOM）无法训练。

## 相关工作脉络
1. **Phenaki（Villegas et al., 2022）**：AR over AR，训练 <16 帧，推理 1024 帧需 4.1 分钟；NUWA-XL 通过直接训练长视频和并行推理解决其训练-推理差距和效率问题。
2. **TATS（Ge et al., 2022）**：AR over Diff，滑动窗口生成 1024 帧需 7.5 分钟；NUWA-XL 放弃序列依赖，推理提速 94%。
3. **MCVD（Voleti et al., 2022）**：随机 mask 历史/未来帧进行条件生成；NUWA-XL 采用结构化 mask（中间帧 mask、首尾帧作为条件），更适合层次化长视频生成。
4. **FDM（Harvey et al., 2022）**：DDPM 框架生成长视频补全；NUWA-XL 通过"Diffusion over Diffusion"实现端到端长视频生成而非补全，且支持并行。
5. **NUWA-Infinity（Wu et al., 2022a）**：AR over AR 生成无限视频；NUWA-XL 转向 Diffusion over Diffusion，避免自回归误差累积，支持并行。
6. **Imagen Video（Ho et al., 2022a）**：级联视频扩散模型；NUWA-XL 采用层次化关键帧+局部填充策略，扩展性更强（$O(L^m)$）。

## 局限性与未来方向
1. **数据集受限**：仅在卡通《Flintstones》上验证，缺乏开放域长视频（电影、电视剧）数据；团队正在构建开放域长视频数据集，计划未来扩展。
2. **数据依赖挑战**：直接训练长视频对数据质量和规模要求极高。
3. **GPU 资源需求**：并行推理需合理 GPU 资源支持，大规模并行可能受限。
4. **分辨率与长度权衡**：L=32 时 OOM，需在质量和显存间平衡。

## 研究启发与可借鉴点
1. **"粗到细"层次化生成范式**：将长序列生成分解为关键帧生成+中间帧填充，可迁移至长文本生成、3D 场景生成等任务。
2. **预训练权重迁移策略**：在预训练图像模型基础上添加时间模块并恒等初始化，是一种高效、稳定的视频模型初始化方法，可复用于其他多模态预训练场景。
3. **Mask 条件注入机制**：通过 mask 区分条件和噪声帧，统一全局/局部扩散训练，设计简洁且灵活，可推广至图像修复、视频插值等任务。
4. **B-FVD 评估指标**：将长视频分段计算 FVD 再平均，为长视频生成提供标准化评测手段，可作社区通用基准。
5. **并行推理设计**：消除序列依赖以实现并行，对需要低延迟的视频/时序生成应用具有重要参考价值。

## 关键术语表
- **Diffusion over Diffusion**：由全局扩散模型生成关键帧、局部扩散模型递归填充中间帧的多层扩散生成架构。
- **T-KLVAE**：在预训练图像 KLVAE 基础上添加时间卷积和注意力层的视频压缩编码器，时间模块初始化为恒等映射。
- **Mask Temporal Diffusion (MTD)**：支持文本提示和可选首尾帧条件的统一扩散模型，通过 mask 机制区分条件帧与待生成帧。
- **B-FVD-X**：将长视频切为 X 帧片段，计算各片段 FVD 后取平均的长视频质量评估指标。
- **FlintstonesHD**：作者构建的长视频基准数据集，含 166 集《摩登原始人》动画，平均 38000 帧/集，1440×1080 分辨率，带密集帧级 caption。
- **Autoregressive over X**：训练于短片段、推理时通过滑动窗口/自回归扩展至长视频的架构范式，X 可为 AR 模型或扩散模型。
- **Training-Inference Gap**：模型训练分布（短视频）与推理分布（长视频）不一致导致的性能退化现象。

## 可复现要素
- **数据集**：FlintstonesHD，论文称公开可用（homepage: https://msra-nuwa.azurebsites.net/）；原始动画来源为公开卡通。
- **代码/权重**：论文未明确提供 GitHub 链接，仅给出项目主页；权重开源情况未提及。
- **关键超参**：全局扩散关键帧数 $L$（16/128/256）、扩散深度 $m$（1/2/3）、时间步数 $T$（未明确）、分辨率 128/256。
- **训练设备**：论文未明确提及。
- **评估代码**：FID、FVD 计算方式遵循标准实现。
