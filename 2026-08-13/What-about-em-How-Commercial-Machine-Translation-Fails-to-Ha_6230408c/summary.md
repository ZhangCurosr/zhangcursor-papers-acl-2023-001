---
title: "What-about-em-How-Commercial-Machine-Translation-Fails-to-Ha"
source: https://aclanthology.org/2023.acl-long.23.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:45:21"
field: "包容性自然语言处理"
keywords: ["机器翻译", "新式代词", "身份包容性", "性别偏见", "非二元群体", "跨语言评估"]
innovations: ["首次系统性评估商业MT对新式代词（xe/ey/vam/emojiself/numberself等）的翻译处理能力", "揭示新式代词引入导致语法正确率最高下降16pp、语义一致性最高下降47pp的系统性损害", "通过n=49非二元群体调查证明用户代词翻译偏好高度多样化，不存在统一共识"]
benchmarks: ["WinoMT", "Google Translate", "Microsoft Bing", "DeepL Translator"]
---

# 论文速读：What-about-em-How-Commercial-Machine-Translation-Fails-to-Ha

## 一句话总结
本研究对 Bing、DeepL、Google Translate 三大主流商业 MT 系统进行"现实检验"，系统评估其在英→丹/波斯/法/德/意及丹→英双向翻译中对性别人称代词（含新式代词 neo-pronouns）的处理能力，并结合覆盖非二元群体的最大规模调查显示了用户在代词翻译策略上的高度多样化偏好。

## 研究问题与动机
1. **现有性别偏见研究局限于二值性别**：已有 MT 偏见工作（如 WinoMT 系列）将性别视为 male/female 二元变量，忽视了非二元（non-binary）、跨性别等多元身份群体的代表性与伤害（misgendering）。
2. **新式代词现象缺乏系统性评估**：NLP 研究普遍忽视近年来英语等语言中涌现的大量新式人称代词现象（xe/xem、ey/em、vam/vamp、emojiself、numberself 等），商业 MT 如何应对这些形式尚无数据支撑。
3. **跨语言不对应带来的翻译困境**：当源语言代词在目标语言中无直接对应物时（如英语 they 译入德语/意大利语、丹麦语 hen 译入英语），MT 系统的处理策略及其潜在伤害未被充分探究。
4. **用户偏好高度分歧，缺乏统一方案**：非二元群体内部对"如何翻译自己的代词"存在显著差异，现有系统默认策略可能导致系统性错称。

## 核心贡献（创新点）
1. **首次系统性评估商业 MT 对新式代词的处理**：构建涵盖 8 类代词集（性别代词、性别中立 they、传统新式 xe/ey、nounself vam、emojiself、numberself 0）的测试集，覆盖 5 个语法格，在 6 种语言/3 个引擎上进行量化评估。*与已有工作仅关注 they 或传统性别代词的本质区别在于将评估范围扩展至新兴的多元代词家族。*
2. **揭示新式代词对翻译质量的系统性损害**：发现性别中立/新式代词引入后，语法正确率最高下降 16pp，语义一致性（含代词）最高下降 47pp，并展示大量错称与语义不一致的具体案例。*区别于以往仅报告 BLEU/COMET 指标的工作，本文通过细粒度人工标注揭示 identity-harming 错误模式。*
3. **发起最大规模非二元群体 MT 代词偏好调查（n=49，预研究 n=149）**：通过 IRB 批准调查揭示了用户对四种翻译策略（避免代词、复制、翻译匹配、列出多代词）的多样化选择，证明不存在共识。*这是首次将受影响群体声音纳入 MT 评估框架的经验研究。*
4. **提出三条面向包容性 MT 的实践建议**：① 将代词视为开放词类进行系统开发与测试；② 尽可能提供个性化配置选项；③ 在个性化不可行时优先采用性别中立策略避免错称。*区别于纯学术评估，本文落脚于可操作的工程与政策建议。*

## 方法详解
**数据来源与构造**：
- 基于 WinoMT（Stanovsky et al., 2019）提取 5 个语法格（主格 N、宾格 A、依附领属 PD、独立领属 PI、反身 R）共 10 个模板句，另人工补充 10 个简单句（共 164 句 EN 源句）。
- 用 8 组代词填充占位符，覆盖性别代词（he/she）、性别中立代词（they/them）、传统新式代词（xe/xem/xyr）、ey/em/eir、nounself（vam/vamp/vamps/vampself）、emojiself（🏳️‍🌈self 系列）、numberself（0/self 系列）等。

**自动翻译**：
- 使用 Google Translate、Microsoft Bing、DeepL 三个商业引擎，将 EN 译入 DA/FA/FR/DE/IT；另做反向实验：DA→EN（用 han/hun/hen 填充模板，共 48 句）。

**人工标注三维度评估**：
1. **语法正确性（Grammatical Correctness）**：判断译文 B 是否符合目标语语法。
2. **语义一致性（Semantic Consistency）**：分两档——忽略代词时是否保义，以及计入代词后是否保义。
3. **代词处理策略（Pronoun Translation Behavior）**：标注为 omitted/copied/translated；对 translated 进一步标注是否为目标语常见代词、其性与数。

**用户调查设计**：
- 四选项翻译策略（避免/复制/翻译匹配/列出多代词）+ 自定义选项，覆盖英/德/丹/意/俄/波斯母语者，年龄 14–43 岁，经 IRB 审批。

**可靠性保障**：DE/IT 双 annotator 计算 Krippendorff's α=0.73/0.69，FA 重标 intra-annotator α=0.86。

## 实验与结果
**数据集与基线**：
- 测试集：164 句 EN 模板句 + 48 句 DA 反向测试句；引擎：Google Translate、Bing、DeepL；语言：DA、EN、FA、FR、DE、IT。
- 基线对比隐含在 gendered vs. gender-neutral 的横向比较中。

**核心结果数字**：
- **语法正确率**：以 gendered 为基准，新式代词引入后最大下降 **16pp**（emojiself）。
- **语义一致性（代词计入）**：最大下降 **47pp**（aggregated across all languages）；FA nounself 从 79% 降至 45%（下降 34pp）。
- **代词处理策略**：DeepL 翻译比例最高（65%），GTranslate 复制比例最高（43%）；性别代词翻译率 89%，they 翻译率 90%，numberself 复制率 74%，emojiself 复制率 68%。
- **错称率**：传统新式代词被翻译后 **56%** 映射为目标语性别代词，emojiself 中 50% 输出 male-associated、23% female-associated 的代词（共 73% 存在错称风险）。
- **DA→EN 反向**：gendered 语法正确率 75%，hen（中性）仅降 4pp 至 71%；但 hen 译入 EN 时**从未**选择 they，约 **72%** 译为 he。

**最强结果**：DeepL 在代词翻译方面表现最积极（65% 翻译率），但在保守语言（DE/IT）中性代词处理上仍存在显著错称风险。

## 相关工作脉络
1. **Stanovsky et al. (2019) WinoMT**：建立基于职业性别刻板印象的 MT 偏见评估基准，为本文模板构建基础。*本文将其扩展到非二元/新式代词评估，突破二值性别局限。*
2. **Cho et al. (2019)**：评估 EN→KO 翻译中韩语性别中立代词的处理，提出性别中立保留度量。*本文覆盖更广语言对（5 种目标语）及新式代词族，且新增用户偏好维度。*
3. **Dev et al. (2021)**：调查 queer 群体感知到的 NLP 偏见伤害，指出 MT 是非二元用户面临 representation/allocative harm 风险最高的应用。*本文为该论断提供实证支撑。*
4. **Lauscher et al. (2022)**：系统梳理英语第三人称代词多样性现象（新式代词家族）。*本文为该语言现象在 MT 领域的直接延伸应用。*
5. **Saunders et al. (2020)**：提出通过 inflection tag（如 non-binary tag）引导 MT 翻译中性实体。*本文结果为其可行性提供现实检验：当前系统缺乏此类标签机制。*
6. **Brandl et al. (2022)**：分析 LM 在丹麦语/英语/瑞典语中处理性别中立代词的能力（NLI 与共指消解）。*本文聚焦 MT 场景，填补跨语言翻译维度的空白。*

## 局限性与未来方向
1. **代词覆盖有限**：仅测试 8 组代词（每类 1–2 个实例），无法穷尽丰富多元的代词生态。
2. **句级测试**：仅评估单句翻译，未考察长文本/篇章级上下文对代词翻译的影响。
3. **仅限高资源语言**：测试语言（DA/EN/FA/FR/DE/IT）均为资源丰富的语言，低资源语言的对等性问题未涉及。
4. **用户样本量有限**：主研究 n=49、预研究 n=149，虽为同类最大，但仍不足以代表全球非二元群体全貌。
5. **未来方向**：扩展到低资源语言、篇章级评估、开发支持个性化代词配置的可交互 MT 系统、探索基于 inflection tag 的技术路线在真实系统中的落地效果。

## 研究启发与可借鉴点
1. **"以用户为中心"的评估范式可迁移**：将受影响群体调查纳入技术评估流程的做法，对于其他身份包容性 NLP 任务（如 co-reference resolution、sentiment analysis 中的 slur 处理）具有直接借鉴价值。
2. **代词处理策略分类框架**：omitted/copied/translated + 细粒度性别标注的三层标注体系，可作为通用评测协议推广到其他 open-class word 的跨语言处理评估。
3. **反向翻译实验设计**：DA→EN 反向实验揭示了"单向友好"的系统偏差（有中性代词的语言→无中性代词的语言时表现更差），这一对比实验设计值得在其他不对等语言对推广中使用。
4. **个性化配置建议的工程启示**：在 LLM 时代的 MT 系统中，可通过 prompt 或 API 参数暴露"代词处理偏好"配置，为后续研究提供可复现的个性化 NMT 架构思路。
5. **结合 inflection tag 的改进路径**：本文结果证实当前系统完全缺失对非二元代词的显式建模，未来可与 Saunders et al. 的 tag 方案结合，探索可插拔的身份标注接口。

## 关键术语表
**Neo-pronouns（新式代词）**：相对于传统 he/she/they 而言新兴的第三人称代词形式，如 xe/xem、ey/em 等，用于表达非二元或多元性别身份。
**Misgendering（错称）**：因语言系统错误地将代词映射为与个体自我认同不符的性别形式，从而造成对非二元群体的身份否定与伤害。
**Nounself pronouns**：由名词衍生的人称代词形式（如 vam/vamp），用于指代个体身份的特定面向，属于新式代词的一种。
**Numberself pronouns**：以数字为词根的代词形式（如 0/zero），体现使用者与数字符号的身份联结。
**Emojiself pronouns**：以 emoji 符号为词根的代词形式（如 🏳️‍🌈self），是最新出现的一类身份表达手段。
**WinoMT**：Stanovsky et al. (2019) 提出的基于 Winogender 范式的 MT 性别偏见评估数据集，包含职业-代词共现的测试句。
**Pro-drop language（代词省略语言）**：允许主语代词省略的语言，如意大利语和波斯语，本文观察到在这类语言中代词 omission 比例更高。
**Gendered language（有语法性别的语言）**：拥有 grammatical gender 系统的语言，如德语（三性）、法语/意大利语（两性），与性别中立语言（如波斯语）形成对比。

## 可复现要素
- **数据集**：测试句子基于 WinoMT（MIT License）构造并新增，作者声明将在同等许可下公开其句子选择；部分目标语言（FA 无 DeepL 覆盖）受限于商业引擎 API。
- **代码/权重**：论文未提及代码或模型权重开源；仅提及自建标注界面（HTML/JavaScript，托管于 AMT Sandbox）。
- **关键超参**：未涉及模型训练，使用现成商业引擎 API；标注报酬为 15€/小时，高于意大利最低工资；Krippendorff's α 用于一致性评估。
