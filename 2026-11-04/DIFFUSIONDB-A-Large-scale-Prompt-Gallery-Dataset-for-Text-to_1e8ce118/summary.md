---
title: "DIFFUSIONDB-A-Large-scale-Prompt-Gallery-Dataset-for-Text-to"
source: https://aclanthology.org/2023.acl-long.51.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:09:00"
field: "文本到图像生成与提示工程"
keywords: ["text-to-image generation", "prompt engineering", "diffusion models", "dataset", "Stable Diffusion", "deepfake detection"]
innovations: ["首个大规模开放文本到图像提示词数据集（1400万图像/180万提示词/完整超参数）", "CLIP嵌入距离驱动的提示-图像语义对齐误差分析方法", "超参数对生成错误影响的系统化归因分析"]
benchmarks: ["DIFFUSIONDB", "DIFFUSIONDB-2M"]
---

# 论文速读：DIFFUSIONDB-A-Large-scale-Prompt-Gallery-Dataset-for-Text-to

## 一句话总结
本文提出了 **DIFFUSIONDB**，这是首个大规模文本到图像提示词数据集，包含 1400 万张由 Stable Diffusion 生成的图像、180 万个唯一提示词及完整超参数。论文通过提示词语法/语义分析和超参数误差分析，揭示了常见提示词模式、模型错误原因及潜在有害使用场景，为提示工程、可解释生成和深度伪造检测等研究方向提供了开放基准。

## 研究问题与动机
1. 用户对文本到图像模型如何响应不同提示词、什么是最优提示词缺乏系统性认知，提示词编写常被视为"随机且无原则"的黑盒过程。
2. 提示工程在文本到文本生成中已较成熟（如 PromptSource），但在文本到图像生成领域尚无大规模真实用户驱动的数据集可供研究。
3. 此前 Lexica 等平台虽有搜索功能，但未公开内部数据库，阻碍了学术界对提示词-模型交互关系的深入分析。
4. 大规模真实使用数据可为理解模型错误模式、检测深度伪造和设计人机交互工具提供前所未有的基础。

## 核心贡献（创新点）
1. **首个大规模开放文本到图像提示词数据集（DIFFUSIONDB）**：包含 1400 万张图像、180 万个唯一提示词和全部超参数（seed、step、CFG scale、sampler、image size），采用 CC0 1.0 许可证开源。
2. **系统化的提示词语法与语义分析方法**：通过 Spacy 提取命名实体和名词短语构建层级结构，结合冻结 CLIP ViT-L/14 提取语义嵌入并用 UMAP+KDE 可视化，揭示 art/photography 等主题分布。
3. **超参数与提示词风格的误差归因分析**：发现负 CFG scale、小 step、小尺寸和高对比度 aspect ratio 会显著导致提示-图像语义偏离；短提示、非英语提示和 emoji 提示也易引发错误。
4. **揭示模型的潜在有害使用证据**：检测到涉及政治人物的负面描述提示（如 Trump/Biden 相关超 6.5 万/4.8 万张图像）及可能生成虚假信息（如"科学家将微芯片植入疫苗"）的提示。

## 方法详解
1. **数据收集**：使用 DiscordChatExporter 从 Stable Diffusion 官方公开 Discord 服务器下载聊天消息 HTML 文件，收集用户通过 bot 生成的图像及对应的 prompt、hyperparameters（seed、CFG scale、step、sampler、width、height）。
2. **元数据提取与处理**：用 Beautiful Soup 解析 HTML，通过 Pillow 将拼贴图（collage）拆分为单张图像并赋予唯一 UUID filename；所有图像以无损 WebP 压缩存储。
3. **NSFW 检测**：
   - 提示词：使用 Detoxify 多语言毒性模型，取 toxic 和 sexually explicit 概率的最大值作为 NSFW score。
   - 图像：使用 EfficientNet 分类器预测 drawing/hentai/neutral/sexual/porn 五类概率之和；另用 Laplacian 核（阈值 10）检测已模糊图像并赋分 2.0。
4. **语法分析**：按逗号分割提示词，用 Spacy 提取命名实体（NE）和名词短语（NP），构建短语层级树，并以圆圈打包（circle-packing）交互式可视化展示频率分布。
5. **语义分析**：用冻结的 CLIP ViT-L/14 文本编码器将提示词转为 768 维向量，UMAP 降维后用 KDE 估计密度生成等高线图；对图像采用相同 CLIP 投影进行对比。
6. **误差分析**：计算每对 (prompt, image) 的 CLIP 余弦距离，选取距离 > μ + 4σ 且未被模糊的 13,411 个"坏对"，用逻辑回归检验超参数与错误概率的关系（p < 0.0001）。

## 实验与结果
- **数据集规模**：1400 万张图像，180 万个唯一提示词，总大小 6.5TB。
- **语言分布**：98.3% 提示词为英文；非英语前五为德语（5.2k）、法语（4.6k）、意大利语（3.2k）、西班牙语（3k），共识别出 34 种语言。
- **提示词长度**：最常见为 6–12 tokens；在 75 tokens 处出现显著尖峰（Stable Diffusion tokenizer 截断上限）。
- **超参数影响**：CFG scale、step、width、height 四个超参数均与错误生成概率呈负相关（p < 0.0001）；sampler 分布在坏对中显著不同（χ² = 40873.11, p < 0.0001）。
- **NSFW 检测器性能**：提示词检测器 precision=0.3604, recall=0.9565, adjusted recall=0.6661；图像检测器 precision=0.315, recall=0.9722, adjusted recall=0.3037。
- **语义分布差异**：CLIP 空间中提示词与生成图像分布不一致，如提示词中"movie"区域对应图像的"portrait"区域。
- **有害内容发现**：含"Donald Trump"提示词生成超 65k 张图像；含"Joe Biden"超 48k 张；发现涉及非自愿色情和虚假信息生成的示例。

## 相关工作脉络
1. **PromptSource**（Bach et al., 2022）：2k 条文本到文本提示词数据集及开发框架；本文聚焦文本到图像，规模达 1400 万对，且含完整超参数。
2. **Lexica**（Shameem, 2022）：支持 500 万张 Stable Diffusion 图像搜索的创业公司产品；未公开内部数据库，本文提供完全开源版本。
3. **Oppenlaender (2022)**：通过民族志研究归纳 6 类 prompt modifier；本文提供大规模真实用户使用数据验证这些模式。
4. **Liu & Chilton (2022)**：基于 1296 个提示词的实验提出设计指南；本文规模为其约 1000 倍，覆盖更广泛的真实用户行为。
5. **Stable Diffusion**（Rombach et al., 2022）：本文核心研究对象的开源模型，基于 latent diffusion 和 CLIP 文本编码器。
6. **LAION-5B**（Schuhmann et al., 2022）：大规模图像-文本配对训练数据；本文是其下游用户生成数据的补充，提供使用侧视角而非训练侧视角。

## 局限性与未来方向
1. **数据源偏差**：Discord 早期用户多为 AI 艺术爱好者，其提示风格不能代表新手用户；也未必能泛化至医学等专业领域。
2. **图像质量度量有限**：CLIP 嵌入距离仅反映语义对齐度，不能衡量整体图像质量；作者曾考虑 entropy/variance/color 等指标但未采用。
3. **泛化性受限**：不同模型（如 DALL-E 2、Midjourney）可能需要不同提示模式（如逗号分隔在 SD 中常见但 DALL-E 2 中较少），本文结论不宜直接移植。
4. **NSFW 检测器精度偏低**：adjust recall 仅约 0.67（提示词）和 0.30（图像），存在漏检风险。
5. **未来方向**：可扩充多语言覆盖；结合人类评分提升质量评估；探索跨模型提示词迁移；设计提示词自动补全、语义搜索和人机交互工具。

## 研究启发与可借鉴点
1. **CLIP 余弦距离作为语义对齐代理指标**：可用于快速筛选 prompt-image 错配样本，成本远低于人工标注，可迁移至其他多模态模型的错误分析。
2. **交互式可视化辅助数据探索**：圆圈打包（prompt 短语层级）和等高线图（CLIP 嵌入分布）的设计值得借鉴，可为团队内部数据集探索提供 UI 范式。
3. **超参数影响的系统归因分析框架**：通过逻辑回归检验超参数与错误概率的关系，可用于评估其他扩散模型或生成模型的超参数敏感性。
4. **开源数据集的伦理治理机制**：CC0 许可证、用户名哈希化、NSFW 分数标注、用户申诉删除表单的设计可作为负责任开源的参考模板。
5. **跨领域可迁移性**：该数据集和方法论可直接应用于提示词自动补全、deepfake 检测训练数据构建、以及多模态模型可解释性研究。

## 关键术语表
- **DIFFUSIONDB**：首个大规模文本到图像提示词数据集，含 1400 万张 Stable Diffusion 生成图像及 180 万唯一提示词。
- **Stable Diffusion**：基于 latent diffusion 的开源文本到图像生成模型，由 Stability AI 开发，使用 CLIP 文本编码器。
- **CFG scale（Classifier-Free Guidance scale）**：控制生成图像与提示词对齐程度的超参数，值越高越遵循提示，负值会导致偏离。
- **CLIP embedding distance**：基于 CLIP 模型计算的提示词与图像语义空间余弦距离，用于衡量语义对齐度。
- **NSFW score**：通过 Detoxify 和 EfficientNet 模型计算的提示词/图像不适宜内容概率分数。
- **Prompt engineering**：系统性地构造和优化提示词以引导生成模型产生期望输出的过程。
- **Circle-packing visualization**：一种层级数据可视化技术，用嵌套圆圈大小表示短语频率并展示其从属关系。
- **UMAP**：Uniform Manifold Approximation and Projection，一种高维数据降维技术，用于将 768 维 CLIP 嵌入投影至 2D 可视化。

## 可复现要素
- **数据集**：DIFFUSIONDB，包含 1400 万图像及元数据，6.5TB，可通过 https://poloclub.github.io/diffusiondb 获取；另提供 DIFFUSIONDB-2M（200 万采样子集）。
- **代码**：数据收集、处理和分析代码已开源（GitHub）。
- **许可证**：CC0 1.0（数据集）、Stability AI Terms of Use。
- **关键超参**：Stable Diffusion v1（默认超参未详细列出）；CLIP ViT-L/14 文本编码器；Detoxify 多语言毒性模型；EfficientNet NSFW 分类器；UMAP 超参 n_neighbors=60, min_dist=0.1。
- **NSFW 检测器评估**：基于 5k 图像 + 2k 提示词手动标注验证。
