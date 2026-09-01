---
title: "UniLG-A-Unified-Structure-aware-Framework-for-Lyrics-Generat"
source: https://aclanthology.org/2023.acl-long.56.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:51:34"
field: "音乐信息处理与自然语言生成交叉"
keywords: ["歌词生成", "结构感知", "复合模板", "循环一致性损失", "音乐生成"]
innovations: ["提出融合文本与音乐信息的复合模板实现统一多条件歌词生成", "设计两阶段Pipeline通过中间模板桥接节奏源与歌词", "引入循环一致性损失增强音乐信息对结构建模的促进"]
benchmarks: ["Song8k", "Lyric-Template Dataset"]
---

# 论文速读：UniLG: A Unified Structure-aware Framework for Lyrics Generation

## 一句话总结
本文提出 UniLG，一个统一的结构感知歌词生成框架，通过设计融合文本与音乐信息的复合模板（compound template）来建模歌词结构（如主歌/副歌），并能统一处理多种生成条件（音乐谱、音频、部分歌词等），无需重新训练模型即可适应不同信号源。

## 研究问题与动机
1. **结构建模缺失**：现有歌词生成方法大多忽略歌词背后的音乐属性和歌词结构（chorus/verse），导致生成歌词缺乏音乐可呈现性。
2. **标注成本高**：已有显式结构建模方法需要额外的人工标注（如句子级 chorus/verse 标签），成本高昂。
3. **条件扩展性差**：大多数工作仅聚焦于特定生成条件（如仅基于音乐谱或部分歌词），难以统一框架扩展到其他信号。
4. **形式约束局限**：SongNet 等方法采用预设格式（如音节数）或语言标签，无法直接表达音乐概念（如主歌/副歌的旋律重复性）。

## 核心贡献（创新点）
1. **提出 UniLG 统一框架**：设计两阶段 pipeline（Input-to-Template + Template-to-Lyric），通过复合模板作为桥梁连接节奏源（音频/乐谱）与歌词。
2. **复合模板设计**：融合语义（Masked Lyric/Lyric）、音乐（Beat/Bar）和文本（Segment/Intro-Position）五类符号，提供判别性表示以实现结构建模。
3. **循环一致性损失（CCL）**：引入从生成歌词重建音乐信息的 CCL，增强音乐信息对结构建模的促进作用。
4. **统一多条件生成能力**：通过模板接口实现无需重新训练即可支持音乐谱、音频、部分歌词等多种生成条件的统一处理。

## 方法详解
**复合模板（Compound Template）**：
- 由五类符号组成：Masked Lyric $M$、Lyric $L$、Beat $B$、Bar $A$、Segment $S$、Intro-Position $P$
- **音乐信息**：Beat 表示小节内局部音乐信息（$\text{Beat}_0 \sim \text{Beat}_3$），Bar 表示全局小节信息（$\text{Bar}_0 \sim \text{Bar}_{511}$）
- **文本信息**：Segment 表示句子位置（$\text{Seg}_0 \sim \text{Seg}_{255}$），Intro-Position 表示句内局部位置（$\text{Pos}_0 \sim \text{Pos}_{31}$）

**两阶段 Pipeline**：
1. **Input-to-Template**：将歌词/输入信号转换为复合模板 $\mathbb{T} = (t_i)_{i=1}^n = (<m_i, b_i, a_i, s_i, p_i>)_{i=1}^n$
   - Masked Lyric 通过随机遮盖 85% 非特殊 token 构建
   - Beat 通过 Lyric-to-Beat 模型（基于 MT5 的 Seq2Seq）从歌词提取
   - Bar 根据拍号从 Beat 推导；Segment 和 Intro-Position 可从任意分量构建

2. **Template-to-Lyric**：基于预训练编码器-解码器 MT5
   - Encoder 输入：$H_E^0 = \text{LN}(E_M + E_B + E_A + E_S + E_P)$
   - Decoder 输入：$H_D^0 = \text{LN}(E_L + E_B + E_A + E_S + E_P)$
   - 训练损失：$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{T2L}} + \alpha \cdot \mathcal{L}'_{\text{L2B}}$
     - $\mathcal{L}_{\text{T2L}}$：从模板生成歌词的负对数似然
     - $\mathcal{L}'_{\text{L2B}}$：循环一致性损失（从生成歌词重建 Beat）
     - $\alpha = 0.03$（验证集调优）

**推理过程**：
- Beat Construction：根据输入 X（beat序列/歌词/MIDI/音频/None）构造 Beat B
- Masked Lyric Construction：基于关键词 K 构造 M
- Components Construction：构建完整模板 T，然后自回归生成歌词 L

## 实验与结果
**数据集**：
- **Lyric-Template Dataset**：从 249,007 首中文流行歌曲中提取，8000 首验证集、8000 首测试集
- **Song8k**：8000 首标注了句子级 chorus/verse 标签的歌曲，用于结构评估

**基线**：MT5（预训练 Transformer 语言模型）、SongNet（格式控制文本生成）

**主要结果（测试集，Table 1）**：

| 模型 | Dataset | PPL(↓) | Integrity(↓) | Format F1(%) ↑ | Beat F1(%) ↑ | Structure F1(%) ↑ |
|------|---------|--------|--------------|----------------|--------------|-------------------|
| MT5 | T-L | 1.96 | 1.92 | 77.08 | 14.63 | - |
| SongNet | T-L | 2.62 | 2.39 | 86.36 | 31.19 | - |
| **UniLG** | T-L | 2.41 | 2.11 | **87.39** | **32.88** | - |
| MT5 | S8 | 1.99 | 2.10 | 76.11 | 14.37 | 50.02 |
| SongNet | S8 | 2.68 | 2.66 | 85.79 | 31.56 | 50.68 |
| **UniLG** | S8 | 2.19 | 2.14 | **88.91** | **34.25** | **53.71** |

- UniLG 在 Format F1、Beat F1、Structure F1 上全面超越基线
- Structure F1 提升显著：相比 MT5 提升 +3.69%，相比 SongNet 提升 +3.03%

**消融实验（Table 2，S8 数据集）**：
- 移除 Bar&Beat：Beat F1 从 34.25% 降至 31.65%，Structure F1 从 53.71% 降至 51.08%
- 移除 Seg&Pos：Structure F1 降至 50.98%
- 移除 CCL：Structure F1 降至 52.34%，Beat F1 降至 32.42%
- **关键发现**：音乐信息（Bar&Beat）对结构建模贡献最大；CCL 进一步提升了 Beat F1 和 Structure F1

**主观评估（Table 5）**：
- UniLG 在 Coherence、Fluency、Correlation、Fascination 四项指标上均优于 MT5 和 SongNet
- Correlation（结构/语义相似性）提升最明显：3.19 vs 3.11 (SongNet) vs 3.03 (MT5)

**Beat Generation 策略对比（Table 4）**：
- Sample 策略（基于数据统计采样）在所有主观指标上最优，表明音乐信息的真实性能显著提升歌词质量

## 相关工作脉络
1. **SongNet (Li et al., 2020)**：采用预设格式（如音节数）和语言标签控制文本生成，但无法直接表达音乐概念（如副歌重复性），UniLG 通过引入 Beat/Bar 符号实现更精细的结构建模。
2. **Songmass (Sheng et al., 2021)**：使用预训练与对齐约束进行歌曲创作，但未统一处理多种生成条件；UniLG 通过复合模板实现统一接口。
3. **Deeprapper (Xue et al., 2021)**：针对说唱歌词生成，建模押韵和节奏；UniLG 关注更通用的流行歌词结构（主歌/副歌），适用范围更广。
4. **TeleMelody (Ju et al., 2021)**：基于模板的两阶段歌词-旋律生成；UniLG 借鉴其模板思想，但将模板扩展为融合五类符号的复合结构。
5. **Pre-trained Language Models (GPT-2, MT5)**：作为骨干网络，但通用 LM 未考虑歌词结构；UniLG 在预训练模型基础上注入结构和音乐信息。

## 局限性与未来方向
1. **数据依赖性强**：结构学习是数据驱动的，高度依赖数据质量，可能难以泛化到小众风格。
2. **拍号限制**：Lyric-to-Beat 模型假设所有歌曲为 4/4 拍，非 4/4 拍需重新训练该模块。
3. **Beat Construction 较简单**：仅探索了基于规则的随机/采样方法，未来可用语言模型生成 beat 序列。
4. **多模态融合有限**：不能同时利用多个模态（如同时使用音频和 MIDI），本质仍是"给定 beat 生成歌词"。
5. **复现门槛**：使用预训练 MT5 需至少 20GB 显存的 GPU，且需从头训练或微调。

## 研究启发与可借鉴点
1. **复合模板设计范式**：五类符号（语义/音乐/文本）融合的结构化表示方法，可迁移到其他需要结构感知的生成任务（如诗歌生成、剧本写作）。
2. **循环一致性损失**：通过交叉任务（歌词→beat→歌词）验证信息保留，思路可应用于音乐生成、图像-文本生成等跨模态任务。
3. **统一接口设计**：将不同输入信号（音频/乐谱/文本）转换为统一中间表示（复合模板），实现了"一次训练、多条件生成"，对多模态生成系统有借鉴价值。
4. **Beat 统计采样策略**：基于训练数据统计采样 beat 序列的方法比随机/规则生成效果更好，提示了在生成任务中利用数据分布先验的重要性。
5. **无标注结构建模**：通过音乐信号（beat/bar）隐式学习歌词结构，避免了昂贵的结构标注，为其他结构化生成任务提供了低成本方案。

## 关键术语表
**Compound Template（复合模板）**：融合文本和音乐信息的五元组序列，作为连接节奏源与歌词的中间表示。
**Beat（节拍）**：表示乐小节内的局部音乐信息，取值 $\{\text{Beat}_0, \text{Beat}_1, \text{Beat}_2, \text{Beat}_3\}$。
**Bar（小节）**：表示全局小节编号，用于捕捉歌词句子的全局音乐位置。
**Cycle-Consistency Loss（循环一致性损失）**：从生成歌词重建 beat 序列的辅助损失，强制音乐信息保留。
**Structure F1（结构 F1）**：通过预训练的 Lyric-to-Structure 模型预测生成歌词的 chorus/verse 标签，与人工标注计算 F1 分数。
**Lyric-to-Beat Model**：基于 MT5 的 Seq2Seq 模型，从歌词预测 beat 序列，准确率达 92.18%。
**Integrity（完整性）**：评估句子完整性的指标，计算 separation token 的平均概率。
**Format F1 / Beat F1**：分别评估生成歌词与模板中 Segment/Intro-Position 和 Beat 的一致性。

## 可复现要素
- **数据集**：Lyric-Template Dataset（249,007 首歌曲）、Song8k（8,000 首标注歌曲）；论文未提及公开，受版权保护
- **代码/权重**：论文未提供开源代码链接；使用 HuggingFace 的 MT5-small 作为初始化
- **关键超参**：CCL 权重 $\alpha = 0.03$；学习率 0.0001；warmup 8,000 步；epoch 5；dropout 0.1（Lyric-to-Beat）/0.2（Lyric-to-Structure）
- **训练环境**：GeForce RTX 3090，batch size 32，max tokens 4096
