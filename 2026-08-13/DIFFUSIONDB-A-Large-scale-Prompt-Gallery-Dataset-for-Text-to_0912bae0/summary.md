---
title: "DIFFUSIONDB-A-Large-scale-Prompt-Gallery-Dataset-for-Text-to"
source: https://aclanthology.org/2023.acl-long.51.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:50:09"
field: "多模态生成模型与提示工程"
keywords: ["text-to-image", "diffusion model", "prompt engineering", "dataset", "Stable Diffusion", "CLIP", "deepfake detection"]
innovations: ["首个公开的大规模文本到图像提示词数据集（1400万图像/180万提示词）", "系统性揭示提示词句法模式、语义分布与模型错误关联", "提供NSFW分级评分与有害使用实证分析"]
benchmarks: ["无专用评测基准，以数据集分析为主"]
---

# 论文速读：DIFFUSIONDB-A-Large-scale-Prompt-Gallery-Dataset-for-Text-to-Image-Generative-Models

## 一句话总结
本文提出了DIFFUSIONDB，第一个大规模文本到图像生成提示词数据集（6.5TB，1400万张图像、180万个唯一提示词及超参数），用于系统研究提示工程、模型可解释性、深度伪造检测等研究方向。

## 研究问题与动机
- **提示工程缺乏系统性理解**：现有文本到图像扩散模型虽受欢迎，但用户需通过大量试错才能写出有效提示词，且对模型如何响应不同提示词缺乏认知。
- **缺乏大规模真实用户数据**：prompt engineering 在文本生成领域已有研究（如PromptSource），但针对文本到图像提示词的大规模数据集长期缺失。
- **模型行为与安全隐患不明**：生成式模型被用于制造虚假信息和非自愿色情内容等问题尚未被系统记录和分析。
- **现有工具不可复现**：类似Lexica的搜索工具积累了海量数据但未开源，阻碍了学术研究。

## 核心贡献（创新点）
- **构建首个大规模提示词数据集**：DIFFUSIONDB包含1400万张Stable Diffusion生成图像及对应提示词、超参数（seed、CFG scale、sampler等），总规模6.5TB；不同于此前小样本或封闭数据集，这是首个公开可用的百万级人类操作数据集。
- **系统性提示词句法与语义分析**：通过命名实体识别、依赖解析及CLIP嵌入可视化，揭示了提示词常用短语模式、语言分布（98.3%英文）及语义聚类特征。
- **发现模型错误模式与有害使用证据**：通过CLIP嵌入距离定位了与提示词严重不对齐的"坏"图像（如负CFG scale、小step、极小尺寸导致失败），并发现用于生成政治 misinformation 和非自愿色情内容的提示词实例。
- **提供NSFW检测评分工具**：对每条提示词和图像计算NSFW得分，支持研究者按需过滤不安全数据，降低数据集使用风险。

## 方法详解
- **数据采集**：使用DiscordChatExporter从Stable Diffusion官方公共Discord服务器抓取聊天消息HTML文件，聚焦用户通过Bot命令生成图像的频道。
- **元数据提取**：用Beautiful Soup解析HTML，将图像与提示词、超参数、seed、时间戳、用户Discord用户名关联；对拼图图像（collage）用Pillow拆分为单张图像并分配独立元数据。
- **NSFW检测**：
  - 提示词：使用Detoxify多语言毒性分类模型，取toxic与sexually explicit概率最大值作为NSFW得分。
  - 图像：使用EfficientNet分类器（Schuhmann et al., 2022），累加hentai/sexual/porn三类概率；同时对已被Stable Diffusion模糊处理的图像用Laplacian核检测并赋分2.0。
- **句法分析**：用spaCy对180万提示词进行命名实体（NE）和名词短语（NP）提取，构建短语层级关系，并开发交互式圆形填充图（circle packing）可视化。
- **语义分析**：使用Stable Diffusion内部相同的冻结CLIP ViT-L/14文本编码器提取768维提示词嵌入，经UMAP降维后用KDE生成密度等高线图，并标注各区域高频TF-IDF关键词。
- **错误分析**：计算所有提示词-图像对的CLIP余弦距离，选取均值+4标准差以上的13,411对作为"坏"样本，用逻辑回归分析超参数（CFG scale、step、尺寸、sampler）与错误的关联性。

## 实验与结果
- **数据集规模**：6.5TB，1400万张图像，180万个唯一提示词，覆盖34种语言（Top4：德语5.2k、法语4.6k、意大利语3.2k、西班牙语3k）。
- **提示词长度分布**：最短6-12 token最常见；75 token处出现峰值（因Stable Diffusion截断限制），反映用户提交超长提示词的问题。
- **语言分布**：98.3%为英文提示词，反映训练数据LAION-2B(en)的英语主导特性。
- **NSFW检测器性能**（手动标注验证）：提示词检测器precision=0.36、recall=0.96、调整后recall=0.67；图像检测器precision=0.32、recall=0.97、调整后recall=0.30。
- **模型错误分析关键发现**：
  - CFG scale为负值、step过小（如step=2）、尺寸过小（如64×512）或大长宽比均显著增加生成失败概率（p<0.0001）。
  - 非英文提示词、极短提示词、emoji主导提示词是另一个主要错误来源。
  - 提示词与图像的CLIP嵌入分布存在系统性偏移（如"movie"类提示在图像嵌入中转为"portrait"聚类）。
- **有害使用证据**：发现超6.5万张涉及"Donald Trump"、4.8万张涉及"Joe Biden"的图像，部分以负面形象呈现；发现含"vaccine microchip conspiracy"等misper information提示词。

## 相关工作脉络
- **PromptSource (Bach et al., 2022)**：包含2k文本提示词的NLP提示词数据集；本文与之对比定位不同——专注于文本到图像领域，且规模达1400万提示-图像对。
- **Lexica (Shameem, 2022)**：允许用户搜索500万张Stable Diffusion图像的工具；核心差异在于Lexica未开源其内部数据库，而DIFFUSIONDB完全公开。
- **Oppenlaender (2022)**：通过民族志研究归纳6类提示词修饰符；本文提供了更大规模实证数据支撑其分类的量化验证。
- **Liu & Chilton (2022)**：基于1296个提示词的文本到图像提示工程设计指南；本文提供180万提示词的宏观统计视角作为补充。
- **CLIP/Radford et al. (2021)**：文本-图像联合嵌入基础模型，本文将其作为核心分析工具用于语义可视化和错误检测。
- **LAION-5B (Schuhmann et al., 2022)**：大规模图像-文本配对训练数据集；本文与LAION的差异在于聚焦生成过程的元数据（超参数、时间序列）而非训练语料。

## 局限性与未来方向
- **数据源偏差**：Discord早期用户多为AI艺术爱好者，提示风格不能代表普通新手用户；且可能不适用于医学影像等专业领域。
- **图像质量评估有限**：CLIP嵌入距离只能衡量提示-图像对齐度，无法直接评估整体图像质量；熵、方差等备选指标效果不佳，需人工标注。
- **NSFW检测器精度问题**：调整后recall偏低（提示词0.67、图像0.30），存在漏检风险，需更优检测模型。
- **泛化性限制**：不同模型（如DALL-E 2、Midjourney）的提示模式不同，本文发现未必适用于其他生成模型。
- **未来方向**：可扩展至其他生成模型（DiffusionDB-2）、引入质量标注、研究跨语言提示词、探索提示词自动生成与推荐系统。

## 研究启发与可借鉴点
- **交互式可视化设计**：圆形填充图（circle packing）和UMAP等高线图展示了大数据集可视化的有效方法，可迁移至其他提示词/嵌入分析任务。
- **提示词-图像联合嵌入分析**：利用CLIP距离定位生成错误的方法简洁有效，可推广至其他扩散模型的幻觉检测与训练数据清洗。
- **超参数-错误关联分析**：通过逻辑回归量化CFG scale/step/sampler等与错误率的关系，为后续设计智能超参数推荐系统提供了方法论参考。
- **NSFW分级策略**：同时针对文本和图像分别计算不安全得分，并为已模糊图像单独标记，该分层处理思路可复用于其他安全敏感数据集构建。
- **人机协同数据治理**：提供用户申诉删除通道（Google Form）+定期更新机制（双月更新），为数据集伦理治理提供了可借鉴的运营框架。

## 关键术语表
- **DIFFUSIONDB**：首个大规模文本到图像提示词数据集，包含1400万张Stable Diffusion生成图像及其提示词、超参数和用户元数据。
- **Prompt Engineering（提示工程）**：系统性地设计和优化输入提示词以控制生成模型输出质量的研究方向。
- **CFG Scale（Classifier-Free Guidance Scale）**：控制生成图像与提示词匹配程度的超参数，值越高图像越忠实于提示，负值会导致偏离。
- **NSFW（Not Suitable For Work）**：指不适合工作场所的内容（如色情、暴力），本文用分类器为每条数据计算NSFW得分。
- **CLIP Embedding Distance**：基于CLIP模型的文本-图像联合嵌入空间中计算的余弦距离，用于衡量提示词与生成图像的对齐程度。
- **Circle Packing Visualization**：一种层次数据可视化技术，用嵌套圆环表示短语层级和频率分布。
- **U-MAP**：Uniform Manifold Approximation and Projection，一种高维数据降维可视化方法，用于将768维CLIP嵌入投影到2D空间。

## 可复现要素
- **数据集**：DIFFUSIONDB公开可用（https://poloclub.github.io/diffusiondb），CC0 1.0许可；代码开源（https://github.com/poloclub/diffusiondb）。
- **关键超参**：Stable Diffusion默认CFG scale=7.5、step=50；CLIP ViT-L/14文本编码器；Detoxify毒性模型；EfficientNet图像分类器。
- **NSFW阈值**：论文建议score>0.5为NSFW预测，图像处理中已模糊图像赋分2.0；Laplacian核阈值=10。
- **数据分割**：论文未提供固定训练/测试分割，研究者可按需自由选取子集。
- **语言检测器**：Joulin et al. (2017) Bag of Tricks预训练语言检测模型。
- ** tokenizer**：Stable Diffusion内置tokenizer，截断长度75 token（不含特殊token）。
