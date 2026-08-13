# MoralDial: A Framework to Train and Evaluate Moral Dialogue Systems via Moral Discussions

Hao Sun<sup>1</sup>, Zhexin Zhang<sup>1</sup>, Fei Mi<sup>2</sup>, Yasheng Wang<sup>2</sup>, Wei Liu<sup>3</sup>, Jianwei Cui<sup>3</sup>, Bin Wang<sup>3</sup>, Qun Liu<sup>2</sup>, Minlie Huang<sup>1</sup>∗

<sup>1</sup>The CoAI group, DCST; <sup>1</sup>Institute for Artificial Intelligence; <sup>1</sup>State Key Lab of Intelligent Technology and Systems;

<sup>1</sup>Beijing National Research Center for Information Science and Technology; <sup>1</sup>Tsinghua University, Beijing 100084, China.

<sup>2</sup>Huawei Noah’s Ark Lab. <sup>3</sup>Xiaomi AI Lab.

h-sun20@mails.tsinghua.edu.cn, aihuang@tsinghua.edu.cn

## Abstract

Morality in dialogue systems has raised great attention in research recently. A moral dialogue system aligned with users’ values could enhance conversation engagement and user connections. In this paper, we propose a framework, MORALDIAL to train and evaluate moral dialogue systems. In our framework, we first explore the communication mechanisms of morality and resolve expressed morality into three parts, which indicate the roadmap for building a moral dialogue system. Based on that, we design a simple yet effective method: constructing moral discussions between simulated specific users and the dialogue system. The constructed discussions consist of expressing, explaining, revising, and inferring moral views in dialogue exchanges, which makes conversational models learn morality well in a natural manner. Furthermore, we propose a novel evaluation method under the framework. We evaluate the multiple aspects of morality by judging the relation between dialogue responses and human values in discussions, where the multifaceted nature of morality is particularly considered. Automatic and manual experiments demonstrate that our framework is promising to train and evaluate moral dialogue systems.<sup>1</sup>

## 1 Introduction

Morality is described as “principles concerning the distinction between right and wrong or good and bad behaviors” (English, 1976). In recent years, aligning AI with human values, morality, ethics, and social norms has become a hot topic in research (Moor, 2006; of the President et al., 2016; Siau and Wang, 2020; Hendrycks et al., 2020; Jiang et al., 2021). As an important application of AI, open-domain dialogue systems, which directly interact with users, requires the nature of morality more urgently (Shum et al., 2018; Qiu et al., 2021). A moral open-domain dialogue system can practice social norms and gain users’ trust more easily (Pereira et al., 2016). Moreover, moral dialogue systems further promote dialogue safety, mitigating immoral speeches and behaviors (Sun et al., 2021; Dinan et al., 2021).

To analyze text-based morality, related works introduce Rules ofthumb (RoTs) (Forbes et al., 2020; Jiang et al., 2021; Ziems et al., 2022), the basic conceptual units to study social norms and morality (e.g. you shouldn’t slap or punch others’face). Adopting RoTs to model morality is proved effective. For example, Jiang et al. (2021) train Delphi on RoTs judgment corpora and find that machine has the potential to make ethical judgments. However, to the best of our knowledge, taking advantage of RoTs to improve the morality of open-domain dialogue systems is yet to be explored.

There are three challenges to building a moral dialogue system. Firstly, morality is a biological attribute of human-beings (Ayala, 1987), thus how to understand and express morality by explicitly interacting with users is a great challenge. Exploring the communication mechanisms of morality is necessary. Secondly, RoTs are often in the form of sentence descriptions rather than conversation, making it difficult to make use of RoTs through conversations. Lastly, moral evaluation is another important challenge to building moral dialogue systems. Lacking an evaluation standard hinders a lot the development of moral dialogue systems.

To address these challenges, we design a framework named MORALDIAL to train and evaluate moral conversational models in §2. In this framework, we explore the communication mechanisms of morality by surveying many multidiscipline pieces of research. We resolve morality into three sub-modules: (1) Standpoint Sentences/Phrases (sentence-level), (2) Discussion State (conversation-level), and (3) Discusser Behavior (utterance-level), which provides more detailed requirements that the conversational models should understand and capture.

![](images/31ee6643bda653fffc4f8ca93d097f798918cc187fb8fb8e283e8e98220e218d.jpg)  
Figure 1: The proposed framework to model the communication mechanisms in moral discussion. The framework includes three parts to express morality. When acting moral explanation and moral revision, the discusser would use the expression of basic RoTs (marked in the same color). In summary, To express morality, a person or dialogue system is supposed to (1) understand the expression of basic RoTs; (2) appropriately deal with possible mora conflict; (3) explain its moral views; and (4) revise its moral views if necessary.

For training a conversational model to satisfy the above requirements, we propose a simple yet effective method by constructing corresponding moral discussions, which embeds morality standpoints (RoTs) into a conversation. In the constructed discussions, the dialogue system and the simulated users are pre-set to have respective moral views. Then we design some dialogue flows including moral answering, moral explanation, moral revision, and RoT inference learning. The dialogue flows also correspond to our proposed framework. We adopt multi-task learning and make conversational models learn the skills simultaneously. By expressing, explaining, and revising moral views in dialogue exchanges, conversational models learn morality well in a natural manner.

We also adopt this framework to evaluate moral dialogue systems. It is quite difficult to directly judge morality due to its subjectivity, topicbroadness, and open-endedness. Instead, we evaluate morality from the decomposed sub-modules, including moral answering, explanation, revision, and inference. Furthermore, we transform this complex moral evaluation problem into an agreement judgment between one’s response and moral values, which is computationally and quantitatively feasible. In this procedure, we consider the moral values of the user, the chatbot, and the general population at the same time, which emphasizes the multifacetedness of morality.

We apply our proposed framework and methods on popular conversational models (i.e. DialoGPT (Zhang et al., 2019) and Blenderbot (Roller et al., 2020)). The automatic and human experimental results demonstrate that each sub-module in our framework is indispensable and our framework is promising to train and evaluate a moral dialogue system.

In summary, our contributions are threefold.

• We propose a framework named MORAL-DIAL to describe and model moral discussions, which also explores the communication mechanisms of expressed morality.

• Inspired by the framework, we construct moral discussions from the sentence-formal RoTs to train moral dialogue systems.

• We present a novel evaluation method to evaluate the moral performance of conversational models based on the framework.

## 2 Framework of Expressed Morality

We propose a framework (illustrated as Figure 1) named MORALDIAL to capture, describe, and model moral discussions. It consists of three submodules: (1) Standpoint Sentences/Phrases, (2) Discussion State, (3) Discusser Behavior. This framework uncovers the communication mechanisms of expressed morality and inspires us the roadmap to build a dialogue system to understand and express text-based morality. We sequentially introduce the parts in this section.

Standpoint Sentences/Phrases Morality is an implicit property of human-beings while expressing moral views or standpoints is explicit. Expressing a moral view is to form “a judgment” of “an action”, which “makes a general rule and still provides enough detail” (Forbes et al., 2020; Ziems et al., 2022). Standpoint sentences/phrases are those basic expression elements in a moral discussion. These elements are often applied in statements and explanation. Learning to understand and utilize the expression of basic RoTs helps dialogue systems build some principles and generalize to more scenarios.

Discussion State The discussion state describes whether the two sides in the discussion get moral conflict or moral harmony, which means that the standpoints of the discussers are in alignment or not. Discussion state embodies that morality is multifaceted. For the same issue, the views can be totally different based on different moral foundations (Haidt, 2012) <sup>2</sup>. Besides, moral standards vary widely across cultures, regions, and even individuals (Joyce, 2007; Talat et al., 2021). We pay more attention on moral conflict because moral conflict is more likely to spur a deeper discussion and encourage discussers to exchange moral views. The discussion state can be changed to “harmony” when one discusser is persuaded and makes revision.

Discusser Behavior Discusser behavior means the intention or dialogue act of each utterance in the discussion. Moral explanation and moral revision are two dominant behaviors in moral discussions. Moral explanation is to give some explanations for her/his own answers from the perspective of the human values, which concerns the ability of reasoning about social and moral norms. A deep and essential explanation could directly reflect high moral level of a dialogue system. Moral revision works when one discusser makes mistakes or mismatches the other one’s values with respect to morality. Modifying the previous opinion to be in accord with the other side is an error correction mechanism to learn from constructive feedback and form better morality. Other behaviors like greeting and questioning are not considered in this moral framework because these behaviors also occur in general discussions.

## 3 Methodology

The proposed framework inspires us to train dialogue systems toward the required sub-modules. In order to meet the requirements, we design a simple yet effective method to make conversational models learn from data naturally. Intuitively, training on the dialogue flows which embody some certain moral ability could enhance the corresponding ability of conversational models. Therefore, our goal is to construct discussions carrying moral view expression, moral conflict, moral explanation, and moral revision. We will introduce the discussion prototype in §3.1 and specific construction implementation in §3.2 and §3.3.

## 3.1 Moral Discussion Prototype

Discussion Settings We have a hypothetical scenario where a chatbot and a user are exchanging and arguing opinions regarding a morality-related question. Meanwhile, the user has a corresponding rule of thumb based on her/his life experience, which guides her/him to develop an internal perspective on the question.

Discussion Flow As illustrated in Figure 1, we apply the ideas to design discussion flow. Before the discussion really starts, the chatbot is supposed to pre-learn the Expression of basic RoTs in order to understand and output moral standpoints in advance. At the beginning of the moral discussion, the user first throws a morality-related question and the chatbot answers the question. At this stage, Moral Conflict may happen between the answer and the user’s values (or those universal values). Note moral conflict does not mean that this discussion fails. Instead, we claim that it is important to tolerate mismatched opinions and moral views for users and machines, and logic self-consistence is much more important than never making mistakes. Continuing the discussion, the user may further ask the reason by a sentence like “Why do you say that?” and expect a deep Moral Explanation from the chatbot. Also, the user may debate the chatbot if the previous answer violates the user’s values where the user would point out her/his own standpoint to develop a deeper discussion. If the chatbot is persuaded, it is supposed to make a Moral Revision and give a new answer which is grounded by the user’s values.

<table><tr><td></td><td>Dialogue Flow</td><td>Modeling</td><td># Turns</td><td># Samples</td><td>Length (C/R)</td></tr><tr><td>MA</td><td> $Q  A$ </td><td> $P ( A | Q )$ </td><td>2</td><td>147,305</td><td>19.3/15.9</td></tr><tr><td>ME</td><td> $Q  A ^ { \prime }  W  R$ </td><td> $P ( R | Q , A ^ { \prime } , W )$ </td><td>4</td><td>179,397</td><td>39.8/8.8</td></tr><tr><td>MR</td><td> $Q  A  R  A ^ { \prime }$ </td><td> $P ( A ^ { \prime } | Q , A , R )$ </td><td>4</td><td>43,049</td><td>53.8/15.9</td></tr><tr><td>RIL</td><td> $\mathbf { M E / M R } {  } Q _ { n e w }  A _ { n e w }$ </td><td> $P ( A _ { n e w } | \mathbf { M E } / \mathbf { M R } , Q _ { n e w } )$ </td><td>6</td><td>14,198</td><td>71.0/11.0</td></tr><tr><td>Overall</td><td></td><td> $P ( { \mathrm { R e s p o n s e } } | { \mathrm { C o n t e x t } } )$ </td><td>3.3</td><td>383,949</td><td>34.6/12.4</td></tr></table>

Table 1: The statistics of our constructed discussion dataset. Length (C/R) denotes the mean utterance length in context/response. We model the probability of response conditioned on context.

We admit the constructed moral discussions are limited to specific scenarios and distinct from daily dialogues. However, the discussions embed the RoTs and the parts in our framework in a quite natural manner. We expect that chatbots become more moral by learning the communication mechanisms in our framework and then generalize to more generic scenarios.

## 3.2 Moral Views Pre-training

For enhancing the chatbot’s ability to express the moral views in discussions, we extract the RoTs in Social Chemistry 101 dataset (Forbes et al., 2020). The dataset collects and annotates about 300k RoTs, which cover lots of topics and scenarios such as ethical commonsense, social norms, codes of conduct, etc. The judgment in RoTs for the same action may change under different situations. For example, it is bad to interrupt your neighbor v.s. it is okay to interrupt your neighbor given that you are in an emergency. Inspired by Jiang et al. (2021), we integrate the fields {situation} and {judgment} in Social Chemistry 101 dataset (Forbes et al., 2020) to form more diverse and situational statement-format RoTs. The basic format is {Judgment}{Action}{whenconj.}{Situation} where "when-conj." denotes the phrases like “when”,“if”, etc. We train conversational models on the RoTs by standard language modeling.<sup>3</sup>

## 3.3 Moral Discussion Construction

Ziems et al. (2022) releases MIC dataset. In MIC dataset, there are four main parts in each sample: a collected question Q, an answer A by a chatbot, a related RoT R, and a revised answer A′ written by crowd-workers. Meanwhile, the RoT attributes are annotated including the alignment for answer, global consensus, severity of violation, and moral foundation. We construct the moral discussions based on this meta dataset.

Moral Answer (MA) Generation We first train the basic ability: moral answer generation to a given question. We simply concatenate the question and answer (or revised answer) (i.e. Q A and $Q  A ^ { \prime } )$ . For avoiding chatbots learning immoral answers, we filter out (1) the answers that violate the corresponding RoTs, and (2) the revised answers when the corresponding RoTs are in a low consensus degree. The second rule is based on the finding that some RoTs are controversial, which may degrade the morality performance of chatbots.

Moral Explanation (ME) Generation Moral explanation requires that when asked why, the chatbot generates an RoT-like sentence, which reveals the potential moral principle of its last-turn answer. We construct dialogue flow $Q  A ^ { \prime }  W  R ,$ where W denotes “why-question”, which is manually written to inquire the reason of answer A′ (e.g. Why? or What is the reason?).

Moral Revision (MR) Generation If a user receives an unsatisfactory answer and then presents her/his RoT, the chatbot is expected to revise its original answer and generate a new answer grounded on human values. We construct dialogue flow $Q  A  R  A ^ { \prime }$ . This flow is constructed only when A does not align with R in the MIC dataset.

RoT Inference Learning (RIL) We design another flow RIL for two reasons (1) to confirm that the chatbot really understands the RoT in ME and MA, then generalize it to other similar scenarios; (2) to make chatbots learn to keep consistently practicing the previous RoT. We append a new pair of QA to the back of the above flows. The new QA and the original QA are based on the same RoT. The flows include $Q  A ^ { \prime }  W  R $ $Q _ { n e w } \  \ A _ { n e w }$ and $Q  A  R  A ^ { \prime } $ $Q _ { n e w }  A _ { n e w } .$

Data Statistics After constructing MA, ME, MR, and RIL dialogue flows, we list some important statistics of the dataset as Table 1. To make the whole dialogue more fluent, we insert some conjunctions into the dialogue flows (refer to Appendix A). Each dialogue flow has different modeling goals. We adopt multi-task learning and simultaneously model the probabilities in Table 1.

## 4 Morality Evaluation

Automatic open-domain dialogue evaluation is pretty difficult due to the essence of one-to-many mapping. Traditional reference-based methods do not well evaluate our open-ended moral generation tasks. We propose a reference-free method to evaluate the ability of answering, explanation, revision and inference under our framework based on dynamic interacting. This method primarily learns a trainable metric to measure the agreement between an answer and a RoT given a question. This section is going to introduce how we build the answer-RoT agreement scorer and the moral metrics based on the agreement score.

## 4.1 Answer-RoT Agreement Scorer

Dataset MIC dataset (Ziems et al., 2022) provides the annotation of agreement between the answer and the RoT, which has three labels including “Agree”, “Neutral”, and “Disagree”. We formulate this task as a 3-way text classification task. In addition, we do some data augmentation to enhance the generalization of the dataset and make it better fit in real test scenarios (refer to Appendix B.1 for details).

Models It has been proven in recent years that the pre-trained models with Transformer-like architecture (Vaswani et al., 2017) dominantly perform the best on text classification tasks. Thus, we conduct experiments on multiple popular models including vanilla BERT (Devlin et al., 2018), ALBERT (Lan et al., 2019), and RoBERTa (Liu et al., 2019). We all choose the base versions of them.

<table><tr><td>Model</td><td>Input</td><td>Acc.</td><td>F1</td></tr><tr><td>BERT</td><td>Q&amp;A&amp;RoT</td><td>76.1</td><td>70.6</td></tr><tr><td>ALBERT</td><td>Q&amp;A&amp;RoT</td><td>75.4</td><td>70.1</td></tr><tr><td>RoBERTa</td><td>Q&amp;A&amp;RoT</td><td>78.4</td><td>73.8</td></tr><tr><td>RoBERTa</td><td>A&amp;RoT</td><td>72.8</td><td>66.7</td></tr></table>

Table 2: The 3-way agreement classification results. The question Q provides important context information.

Classification Results The classification results are shown in Table 2. RoBERTa with extra question input performs the best on the task. Therefore, we use the fine-tuned RoBERTa as the following answer-RoT agreement scorer.

Agreement Score Definition Given the input, we adopt the weighted output probability of labels to compute the final agreement score. That is,

$$
\begin{array} { r } { \mathrm { A S } ( Q , A , R ) = P ( y = \mathrm { A g r e e } | Q , A , R ) } \\ { - P ( y = \mathrm { D i s a g r e e } | Q , A , R ) } \end{array}\tag{1}
$$

The final AS score range is 1  1 (from disagree to agree).

## 4.2 Metrics

In test time, we first set the user RoT $R _ { u s e r }$ in advance, which is unseen by the chatbot. We test the chatbot by interacting in real time and first ask a question $Q .$ Then we follow the same dialogue flows as described in §3.3 and measure the scores as follows. These scores comprehensively take the RoTs of the user, the chatbot, and the common population into consideration.

Safety (MA) Score We illustrate the diagram to compute the safety score in Figure 2. In moral answer generation, we detect those immoral or unsafe answers by measuring the agreement between the generated answer A and “safety RoTs”. We define “safety RoTs” as those RoTs with the highest global consensus and severity of violation in MIC dataset (Ziems et al., 2022) and SOCIAL-CHEM 101 dataset (Forbes et al., 2020). Notably, safety RoTs have nothing to do with the user’s RoT $R _ { u s e r }$ and it is okay that A violates $R _ { u s e r }$ because we consider moral conflict is common and acceptable. In the implementation, we first retrieve top-k related safety RoTs by semantic matching using Sim-CSE (Gao et al., 2021), and we only compute the agreement between answer and the retrieved top-k RoTs $\{ R _ { 1 } , \cdots , R _ { k } \}$ for computational efficiency. Refer to Appendix B.2 for more details. The safety score is defined as

![](images/0bc606731f00d2bf495e81258ea10c993e1002c5064a3b0196562936edfd60b6.jpg)  
Figure 2: The illustration to compute safety score $S _ { M A }$

$$
S _ { M A } = \operatorname* { m i n } _ { i = 1 , \cdots , k } \{ \mathrm { A S } ( Q , A , R _ { i } ) \}\tag{2}
$$

The safety score is the primary standard to evaluate morality because this score directly reflects the extent to which the generated responses conform with the most accepted social norms.

ME Score In moral explanation generation, we check the logic self-consistency of the chatbot. After getting the chatbot’s answer A, we ask why and the chatbot gives the moral reason $R _ { b o t }$ . We measure the agreement between A and $R _ { b o t }$ . Note that this metric is independent of $R _ { u s e r }$ . Formally, ME score is formulated as

$$
S _ { M E } = \mathrm { A S } ( Q , A , R _ { b o t } )\tag{3}
$$

MR Scores In moral revision generation, we first measure the agreement $S _ { M R 1 }$ between the generated answer A and user RoT $R _ { u s e r }$ . If A violates $R _ { u s e r }$ , then the chatbot revises its answer to $A ^ { \prime }$ after getting $R _ { u s e r }$ . We compute the agreement score $S _ { M R 2 }$ between $A ^ { \prime }$ and $R _ { u s e r }$ . We record the gap $\triangle S _ { M R }$ between them. Besides, if $S _ { M R 1 }$ and $S _ { M R 2 }$ are both lower than a threshold $\lambda = - 0 . 3 5$ it means that the chatbot performs poorly on moral revision. I( ) denotes indicate function. Formally,

$$
\begin{array} { r l } & { S _ { M R 1 } = \mathrm { A S } ( Q , A , R _ { u s e r } ) } \\ & { S _ { M R 2 } = \mathrm { A S } ( Q , A ^ { \prime } , R _ { u s e r } ) } \\ & { S _ { \triangle { M R } } = S _ { M R 2 } - S _ { M R 1 } } \\ & { ~ S _ { M R } = 1 - \mathrm { I } ( S _ { M R 1 } < \lambda , S _ { M R 2 } < \lambda ) } \end{array}\tag{4}
$$

RIL Score RIL evaluation happens after ME or MR. In the dialogue flow of RoT inference learning, given the new question, we check whether the new answer generated by the chatbot violates the RoT mentioned in the previous context. To put it clearer, this score measures whether the chatbot keeps practicing the previous RoT (RoT consistency) after ME or MR. Different from other scores, RIL score is measured in a static setting where the context is given in advance. The reason is that we find it hard to control the dialogue flow to develop to where we expect. We define RIL score as

$$
S _ { R I L } = \mathrm { A S } ( Q _ { n e w } , A _ { n e w } , R _ { u s e r } )\tag{5}
$$

## 5 Experiments

To verify the effectiveness of our proposed framework, we conduct experiments to train a moral dialogue system and use the metrics proposed in §4 to evaluate.

## 5.1 Experimental Setup

We use the popular open-source conversational models for our experiments: DialoGPT-medium (DGPT) (Zhang et al., 2019) and Blenderbot-400M (BBot) (Roller et al., 2020). We first pre-train (PT) them on RoTs, which is described in §3.2.

Then as illustrated in §3.3, we do a multi-task training and train the conversational models on our constructed discussion dataset including MA, ME, MR, and RIL. Considering the catastrophic forgetting problem in deep learning (Kirkpatrick et al., 2017), we mix the discussion dataset with the general dialogue (GD) corpora including BST (Smith et al., 2020) and Daily Dialogue (Li et al., 2017). This is to confirm the general conversational ability other than morality. We name our proposed models trained on full tasks as Moral DGPT (BBot). We split train, dev, test sets based on meta dataset splits. There is no same question between train and dev/test sets and the overlap rate of RoTs in dev/test set to train set is 13%/12%.

After training, we primarily use the metrics introduced in $\ S 4$ to measure the moral performance of conversational models by interacting in real time. We take out the questions in dev and test sets as the discussion openings.

## 5.2 Main Experimental Results

Our experimental results are shown in Table 3. We compare the original conversational model with our proposed moral model (DGPT v.s. Moral DGPT,

<table><tr><td rowspan="2">Models&amp;Settings</td><td colspan="2"> $S _ { M A }$ </td><td colspan="2"> $S _ { M E }$ </td><td colspan="2"> $S _ { \triangle M R }$ </td><td colspan="2"> $S _ { M R }$ </td><td colspan="2"> $S _ { R I L }$ </td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>DGPT</td><td>-25.0</td><td>-25.5</td><td>-8.5</td><td>-10.2</td><td>20.6</td><td>19.1</td><td>94.0</td><td>93.6</td><td>19.3</td><td>20.6</td></tr><tr><td>DGPT+GD</td><td>-15.5</td><td>-16.7</td><td>6.4</td><td>3.3</td><td>33.8</td><td>33.2</td><td>94.8</td><td>95.2</td><td>34.2</td><td>24.4</td></tr><tr><td>Moral DGPT</td><td>7.2</td><td>7.3</td><td>67.4</td><td>66.0</td><td>20.9</td><td>20.1</td><td>96.1</td><td>96.5</td><td>46.4</td><td>35.1</td></tr><tr><td>BBot</td><td>-2.2</td><td>-1.1</td><td>46.7</td><td>44.9</td><td>33.3</td><td>31.7</td><td>94.9</td><td>95.0</td><td>47.8</td><td>46.4</td></tr><tr><td>BBot+GD</td><td>-3.8</td><td>-4.3</td><td>53.8</td><td>54.9</td><td>40.3</td><td>38.5</td><td>95.0</td><td>95.1</td><td>38.3</td><td>33.5</td></tr><tr><td>Moral BBot</td><td>13.9</td><td>12.5</td><td>68.2</td><td>68.3</td><td>37.8</td><td>37.7</td><td>96.9</td><td>97.0</td><td>50.9</td><td>47.5</td></tr><tr><td>w/o PT</td><td>12.2</td><td>10.8</td><td>72.6</td><td>71.0</td><td>36.7</td><td>34.8</td><td>97.1</td><td>97.1</td><td>61.1</td><td>55.2</td></tr><tr><td>w/o MA</td><td>4.5</td><td>2.0</td><td>61.5</td><td>61.0</td><td>43.9</td><td>43.9</td><td>97.1</td><td>97.4</td><td>49.4</td><td>52.2</td></tr><tr><td>w/o ME</td><td>9.3</td><td>10.1</td><td>48.5</td><td>48.2</td><td>40.0</td><td>38.5</td><td>96.9</td><td>97.2</td><td>47.3</td><td>40.7</td></tr><tr><td>w/o MR</td><td>11.2</td><td>11.8</td><td>69.5</td><td>68.2</td><td>43.1</td><td>42.1</td><td>96.1</td><td>96.3</td><td>51.5</td><td>46.1</td></tr><tr><td>w/o RIL</td><td>12.5</td><td>11.8</td><td>67.3</td><td>67.1</td><td>32.2</td><td>31.5</td><td>96.6</td><td>96.9</td><td>46.4</td><td>40.3</td></tr></table>

Table 3: The experimental results of different models and settings. The metric $S _ { M A }$ (or safety score) is our primary standard to evaluate morality. “GD” denotes general dialogue corpora including BST and Daily Dialogue. We remove each component of our dataset to do ablation studies. Each number is multiplied with 100 for better display.

BBot v.s. Moral BBot). It is found that all the metrics get very significant improvement especially the most important metrics $S _ { M A }$ and $S _ { M E } ,$ . By training based on our proposed framework, DialoGPT and Blenderbot are thus equipped with much stronger power of moral answering, moral explanation, moral revision and moral inference.

Besides, for controlling variables, we add experiments where we only train the models on GD. This proves (1) general dialogue corpora indeed helps morality performance, which indicates that morality is embodied in multiple scenarios (e.g. empathy in BST dataset) and could be enhanced implicitly; (2) The vast major improvement of scores of moral models is still attributed to the discussion datasets based on our framework, instead of GD.

Meanwhile, we also notice that Moral DGPT and BBot perform poorly in the metric $S _ { \triangle M R }$ , which measures the agreement (to the user’s RoT) gap between the first and the second answers. The result is in line with our expectations. When the first answer gets a low score, it would be easier to get a high score of $S _ { \triangle M R }$ . However, training on MA and ME tasks makes the first answer of the models often good enough. The ablation study in the row “w/o MA” also verifies that from the other side. Therefore, we consider it acceptable that our proposed moral models have a low score of $S _ { \triangle M R }$

At last, our experimental results also verify some findings by previous studies. For example, experimental results show that Blenderbot outperforms DialoGPT in all metrics, which is in accord with previous works (Roller et al., 2020; Xu et al., 2020). This also confirms that the proposed metrics are of

practical significance.

## 5.3 Ablation Studies

For exploring how each task affects respectively in our method, we conduct ablation studies on Blenderbot. In this experiment, we remove PT step or remove each component of our mixed dataset (shown as the last 5 rows in Table 3).

Firstly, the experimental results suggest that the PT step and the four tasks MA, ME, MR, RIL are all beneficial to the safety performance. The score $S _ { M A }$ substantially decreases if missing any task, especially the MA task. Meanwhile, when we remove any module, the corresponding metric score would drop significantly. For example, the model without ME task gets a quite low score $S _ { M E }$ . These results support that each task as well as each part in our framework is indispensable. Our multi-task paradigm makes the final model perform balanced across MA, ME, MR, and RIL tasks, achieving the best overall results.

Secondly, we find that MA task and ME task can enhance each other by joint training. In the row “w/o MA”, the ME score decrease by about 10%. The similar thing happens in the row “w/o $\mathbf { M E } ^ { \prime \prime }$ . The two tasks improve the performance upper bound of each other’s task. As for deep reasons, we conjecture that conversational models better organize its answer by learning to reason about morality. On the contrary, the conversational models also learn the implicit reasons in the moral answer generation tasks because many answers contain the reasons behind (e.g, I won’t kill anyone because killing people is wrong.).

<table><tr><td>Model</td><td>Emb.</td><td>Moral.</td><td>Sens.</td><td>Spec.</td></tr><tr><td>BBot</td><td>0.63</td><td>3.05</td><td>0.75</td><td>0.87</td></tr><tr><td>Moral BBot</td><td>0.86</td><td>3.55</td><td>0.75</td><td>0.88</td></tr></table>

Table 4: Human interactive evaluation results. The number represents the mean score of each criteria.

Thirdly, we discover that the advantages and the disadvantages of PT step coexist. On the one hand, pre-training on large-scale RoTs makes dialogue systems understand and learn to output the moral views in advance, helpful for the safety performance. On the other hand, we pre-train in the format of sentence rather than natural conversations, which degrades other conversational abilities like explanation and inference learning. The results reveal that pre-training has much room to improve towards its format inconsistency in our future work.

## 5.4 Human Interactive Evaluation

We conduct human interactive experiments to verify that (1) our proposed metrics in §4 are in accord with the golden metric, i.e. human evaluation results; (2) by learning in limited moral discussions, the moral models can generalize to more generic scenarios. We let the crowd-workers interact with models in real-time and do not limit moral topics and dialogue flows. Meanwhile, for each sentence generated by conversational models, the crowdworkers are asked to annotate (1) whether the sentence embodies morality (Embodiment, 1: yes, 0: no), and (2) If it does, how much proportion of people would accept the moral standpoint (Morality, from 1: none to 5: all). Following Adiwardana et al. (2020), we also evaluate Sensibleness and Specificity of each sentence, which measures the general dialogue ability (1: yes, 0: no). Refer to Appendix E for the detailed process and guideline of human interactive experiments. We compare BBot and Moral BBot and the human evaluation results are shown as Table 4.

Morality Comparison Human experimental results suggest that our proposed Moral BBot is better at making its sentence embody morality under the unconstrained topics, which indicates that morality may have been internalized. Besides, Moral BBot more conforms to the accepted social norms because it gets a higher morality score. Therefore, we conclude that by learning in relatively limited scenarios, machine is able to generalize to more unseen generic scenarios. We present a case study in Appendix F to better illustrate how Moral BBot perform better than BBot.

General Dialogue Ability The result shows that after moral training, the sensibleness and the specificity almost have no change, which suggests the moral training has little impact on the general dialogue ability. We claim that this is benefit from the mixed general corpus in the multi-task training.

## 5.5 Moral Foundation Analysis

As introduced in the moral system (Haidt, 2012) and annotated in MIC dataset (Ziems et al., 2022), there are 6 moral foundations: care, liberty, loyalty, fairness, sanctity, and authority. We analyze the moral foundations of Moral BBot trained under our framework, which could provide a clearer presentation of the internal morality of the model. We pick up those controversial questions in test set. There are 1,659 questions and 3,553 original answers/RoTs in total and each question has at least two answers with different moral foundations. For each question, we also generate an answer and an RoT (by ME flow) using Moral BBot. For each moral foundation, we calculate the ratio of the number of Moral BBot’s generated answers based on the foundation to the number of original answers based on the foundation. Refer to Appendix C.1 for the calculation implementation in detail. The ratio reflects the moral foundation tendency of Moral BBot. As shown in Figure 3, it suggests that Moral BBot is more likely to form its answer and explanation from the moral perspective “care” such as “It is wrong to bully others” and “You should not break into someone’s house”. We speculate that the foundation tendency is sourced from the data distribution in our constructed moral discussion (Appendix C.2), which indicates another approach to shape the internal moral foundation of the trained model.

## 6 Related Work

Morality in Languages Morality in artificial intelligence draws great attention since many years ago (Moor, 2006; Savulescu and Maslen, 2015; Hendrycks et al., 2020). Language is one of the primary ways to express and embody morality (Hare and Hare, 1991). In NLP communities, to analyze morality in language, Forbes et al. (2020) propose and collect a well annotated Rules of Thumb corpora, which provides conceptual units to model morality for the follow-up studies such as MIC (Ziems et al., 2022). As another line of work, over the development of large-scale language models, some researchers find that language models contain inner morality (Schramowski et al., 2021) and is promising to judge morality in a specific situation (Jiang et al., 2021). Meanwhile, previous works discover some safety defects about morality in large language models (Brown et al., 2020; Perez et al., 2022), which leads us to further study morality modeling in languages.

![](images/36792037207ffbcbf21e3c3c5590b5e3a02d1f94713f2432091d65a88ecfe022.jpg)  
Figure 3: Moral foundation tendency of Moral BBot.

Multifacetedness of Morality Morality is multifaceted. The judgment of an action may change when the situation changes (Forbes et al., 2020). Beside situation, morality may also vary across cultures, parties (Ziems et al., 2022; Bang et al., 2022), history time (Joyce, 2007), and even individuals. Based on that, Talat et al. (2021) criticize that Delphi (Jiang et al., 2021) neglects the diversity of human values. For the multifacetedness of morality, the concurrent work Bang et al. (2022) studies how to answer ethical quandary questions. In our framework, We pay particular attention to the multifaceted nature of morality and design the moral conflict sub-module. Moreover, we specially distinguish between universal and dynamic RoTs when evaluating moral answer generation.

Dialogue Safety and Morality With the great improvement of the open-domain dialogue system these years (Roller et al., 2020; Adiwardana et al., 2020; Rae et al., 2021), the safety bottleneck of dialogue system emerges gradually, hinders the deployment in real world. Numerous works study safety detection and safe generation in dialogue system (Xu et al., 2020; Dinan et al., 2021, 2019). Also, researchers discover morality is a core requirement in dialogue safety (Henderson et al.,

2018; Sun et al., 2021; Bommasani et al., 2021). However, few works directly train a moral dialogue system for lack of relevant moral expression framework and corresponding evaluation methods. The concurrent work ProsocialDialog (Kim et al., 2022) applies RoTs into dialogue response generation to better detect and counter the unsafe context. Differently, we explore the communication mechanisms of morality and train moral dialogue system by constructing discussion dataset. Our method improves the comprehensive morality of dialogue system (from the four sub-modules in our framework). Also, our method does not require any extra plugins or parameters in conversational models.

## 7 Conclusion and Future Work

We present the framework, MORALDIAL, to explore the communication mechanisms of morality. Based on the framework, we construct moral discussions to form a moral dialogue dataset, which makes dialogue systems learn morality in a very natural manner. Meanwhile, we design some metrics to measure morality performance based on our framework. We adopt a multi-task paradigm to make conversational models learn MA, ME, MR, RIL tasks simultaneously. In experiments, we analyze and prove the effectiveness of the sub-modules in our framework using both automatic and manual evaluation results. We show that adopting our proposed framework and method is quite helpful to train and evaluate a moral dialogue system. As future work, we will further use our proposed metrics to supervise moral dialogue system training (e.g. reinforcement learning). Besides, it is also important to expand current modules in our framework and collect more fine-grained moral dialogue data.

## Acknowledgment

This work was supported by the National Science Foundation for Distinguished Young Scholars (with No. 62125604). This work was also supported by the Guoqiang Institute of Tsinghua University, with Grant No. 2020GQG0005. This work was also supported by Tsinghua Precision Medicine Foundation.

## Limitations

We don’t consider the completeness of the framework and the communication mechanisms of morality may have other modules. A typical chance is that the user has an unsafe moral standpoint and may hack our moral conversational models. Though we clean these data when constructing moral discussion as described in §3.3, moral models may still perform poorly because unsafe user RoTs are out of the domain of our training data.

The pre-training (PT) step in our experiments is based on sentence-format data and may injure the overall performance of conversational models, which we have discussed in §5.3.

We adopt a trainable agreement scorer to measure the moral scores. The scorer may carry potential bias or error limited to training data and deep learning techniques. We do some data augmentation to make it more robust. However, it may still have some impact on the final experimental results.

## Ethics Statement

This paper is to propose a framework, which is to train and evaluate moral dialogue systems. We do not claim the completeness of our framework. Instead, we summarize some important communication mechanisms of morality and expect future work could explore more modules to enhance the overall moral performance of dialogue systems.

In this paper, we use the concept “Rules of Thumb” (RoTs) and related datasets. Note that the RoTs do not reflect absolutely “right” or “wrong” morals. Instead, RoTs are written by crowdworkers and the contents are based on summaries of life experience, which varies a lot across different people. We define “Safety RoTs” as those RoTs with the highest violation severity and global consensus. If an answer by dialogue system violates the safety RoTs, it should raise more attention by moderators. However, we never claim that a user or a dialogue system should obey each piece of RoTs. We pay special attention to the minority, and we utilize the user’s RoTs to evaluate the many aspects of moral performance.

Our method: discussion construction also especially considers the multifacetedness of morality, where we never pre-set that any side is right or wrong. We expect that in the discussion, both sides could express and exchange their moral views, which promotes the diversity of moral values.

Although we construct a new discussion dataset in this paper, we do not collect dataset from the Internet or crowd-sourcing. The relevant information in the meta dataset is reported in (Ziems et al., 2022). We strictly follow the protocols of the meta datasets. We would share our dataset by sharing the complete script to process meta datasets. In human interactive experiments, we don’t collect any private information. And we inform in advance crowd-workers how their interacting data will be used. We pay them 25 USD per hour, which is higher than the average wage of the local residents.

For a real-world application, our proposed moral dialogue system is expected to respect the moral views of the users and can output its own moral views. However, we still notice that the trained dialogue system could also output something undesired. Considering the diversity and complexity of users, Utilizing safety classifier as post-processing is helpful to alleviate the problem. Besides, the moral standpoints output by our proposed dialogue system should not be seen as the golden standard for real-world applications like moral education. Some promising applications may include moral debate, auxiliary moral dialogue generation, and some scenarios requiring a stronger sense of morality. The applications should set up feasible human intervention mechanisms to avoid moral misleading.

## References

Daniel Adiwardana, Minh-Thang Luong, David R So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Gaurav Nemade, Yifeng Lu, et al. 2020. Towards a human-like open-domain chatbot. arXiv preprint arXiv:2001.09977.

Francisco J Ayala. 1987. The biological roots of morality. Biology and philosophy, 2(3):235–252.

Yejin Bang, Nayeon Lee, Tiezheng Yu, Leila Khalatbari, Yan Xu, Dan Su, Elham J Barezi, Andrea Madotto, Hayden Kee, and Pascale Fung. 2022. Aisocrates: Towards answering ethical quandary questions. arXiv preprint arXiv:2205.05989.

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understand ing. arXiv preprint arXiv:1810.04805.

Emily Dinan, Gavin Abercrombie, A Stevie Bergman, Shannon Spruit, Dirk Hovy, Y-Lan Boureau, and Verena Rieser. 2021. Anticipating safety issues in e2e conversational ai: Framework and tooling. arXiv preprint arXiv:2107.03451.

Emily Dinan, Samuel Humeau, Bharath Chintagunta, and Jason Weston. 2019. Build it break it fix it for dialogue safety: Robustness from adversarial human attack. arXiv preprint arXiv:1908.06083.

Oxford English. 1976. Oxford english dictionary. Encyclopedia of Swearing, page 334.

Maxwell Forbes, Jena D Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2020. Social chemistry 101: Learning to reason about social and moral norms. arXiv preprint arXiv:2011.00620.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. Simcse: Simple contrastive learning of sentence embeddings. arXiv preprint arXiv:2104.08821.

Jonathan Haidt. 2012. The righteous mind: Why good people are divided by politics and religion. Vintage.

Richard Mervyn Hare and Richard Mervyn Hare. 1991. The language of morals. 77. Oxford Paperbacks.

Peter Henderson, Koustuv Sinha, Nicolas Angelard-Gontier, Nan Rosemary Ke, Genevieve Fried, Ryan Lowe, and Joelle Pineau. 2018. Ethical challenges in data-driven dialogue systems. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pages 123–129.

Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. 2020. Aligning ai with shared human values. arXiv preprint arXiv:2008.02275.

Liwei Jiang, Jena D Hwang, Chandra Bhagavatula, Ronan Le Bras, Maxwell Forbes, Jon Borchardt, Jenny Liang, Oren Etzioni, Maarten Sap, and Yejin Choi. 2021. Delphi: Towards machine ethics and norms. arXiv preprint arXiv:2110.07574.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547.

Richard Joyce. 2007. The evolution of morality. MIT press.

Hyunwoo Kim, Youngjae Yu, Liwei Jiang, Ximing Lu, Daniel Khashabi, Gunhee Kim, Yejin Choi, and Maarten Sap. 2022. Prosocialdialog: A prosocial backbone for conversational agents. arXiv preprint arXiv:2205.12688.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.

Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. Dailydialog: A manually labelled multi-turn dialogue dataset. arXiv preprint arXiv:1710.03957.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

James H Moor. 2006. The nature, importance, and difficulty of machine ethics. IEEE intelligent systems, 21(4):18–21.

Executive Office of the President, Cecilia Munoz, Domestic Policy Council Director, Megan (US Chief Technology Officer Smith (Office of Science, Technology Policy)), DJ (Deputy Chief Technology Officer for Data Policy, Chief Data Scientist Patil (Office of Science, and Technology Policy)). 2016. Big data: A report on algorithmic systems, opportunity, and civil rights. Executive Office of the President.

Gonçalo Pereira, Rui Prada, and Pedro A Santos. 2016. Integrating social power into the decision-making of cognitive agents. Artificial Intelligence, 241:1–44.

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red teaming language models with language models. arXiv preprint arXiv:2202.03286.

Liang Qiu, Yizhou Zhao, Jinchao Li, Pan Lu, Baolin Peng, Jianfeng Gao, and Song-Chun Zhu. 2021. Valuenet: A new dataset for human value driven dialogue system. arXiv preprint arXiv:2112.06346.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446.

Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M Smith, et al. 2020. Recipes for building an open-domain chatbot. arXiv preprint arXiv:2004.13637.

Julian Savulescu and Hannah Maslen. 2015. Moral enhancement and artificial intelligence: moral ai? In Beyond Artificial Intelligence, pages 79–95. Springer.

Patrick Schramowski, Cigdem Turan, Nico Andersen, Constantin Rothkopf, and Kristian Kersting. 2021. Language models have a moral dimension. arXiv preprint arXiv:2103.11790.

Heung-Yeung Shum, Xiao-dong He, and Di Li. 2018. From eliza to xiaoice: challenges and opportunities with social chatbots. Frontiers ofInformation Technology & Electronic Engineering, 19(1):10–26.

Keng Siau and Weiyu Wang. 2020. Artificial intelligence (ai) ethics: ethics of ai and ethical ai. Journal of Database Management (JDM), 31(2):74–87.

Eric Michael Smith, Mary Williamson, Kurt Shuster, Jason Weston, and Y-Lan Boureau. 2020. Can you put it all together: Evaluating conversational agents’ ability to blend skills. arXiv preprint arXiv:2004.08449.

Hao Sun, Guangxuan Xu, Jiawen Deng, Jiale Cheng, Chujie Zheng, Hao Zhou, Nanyun Peng, Xiaoyan Zhu, and Minlie Huang. 2021. On the safety of conversational models: Taxonomy, dataset, and benchmark. arXiv preprint arXiv:2110.08466.

Zeerak Talat, Hagen Blix, Josef Valvoda, Maya Indira Ganesh, Ryan Cotterell, and Adina Williams. 2021. A word on machine ethics: A response to jiang et al.(2021). arXiv preprint arXiv:2111.04158.

Judith Jarvis Thomson. 1976. Killing, letting die, and the trolley problem. The monist, 59(2):204–217.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Jing Xu, Da Ju, Margaret Li, Y-Lan Boureau, Jason Weston, and Emily Dinan. 2020. Recipes for safety in open-domain chatbots. arXiv preprint arXiv:2010.07079.

Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. 2019. Dialogpt: Large-scale generative pre-training for conversational response generation. arXiv preprint arXiv:1911.00536.

Caleb Ziems, Jane A Yu, Yi-Chia Wang, Alon Halevy, and Diyi Yang. 2022. The moral integrity corpus: A benchmark for ethical dialogue systems. arXiv preprint arXiv:2204.03021.

## A Details of Moral Discussion Construction

In moral views pre-training, we finally construct 711,844 RoTs and split them into train (80%), dev (10%), and test (10%) sets. In moral discussion construction, we insert some phrases to make the whole conversation more fluent. We list the phrases in Table 5. At last, we randomly remove the situation part and exchange the order between the main and subordinate clauses to enhance diversity.

We do some filtering in MA generation and MR generation. we filter out the revised answers when the corresponding RoTs are in a low consensus degree. This process is to avoid degrading the morality performance of chatbots.

The number of RIL dialogue flows is far less because most of the RoTs correspond to only one QA-pair in MIC dataset (Ziems et al., 2022).

## B Details of Metrics

## B.1 Data of Agreement Scorer

We do some data augmentation to enhance the generalization of the dataset and make better fit in real test scenarios. (1) Irrelevant Answer: we randomly match the answer and other RoTs in the dataset and label them as “Neutral”. (2) Nonsense Explanation: RoT should not be “because they are wrong” if the answer is “they are wrong”. We don’t hope that RoT has nothing new other than the answer. To detect the situation, we back translate some sentences (thus the pair has the same meaning) and make them as the answer-RoT pair of label “Neutral”. After data augmentation, the dataset overview is shown as Table 6.

## B.2 Safety RoTs

We pick safety RoTs from large-scale RoT corpora. In MIC dataset, we choose those RoTs annotated as the highest violation severity (worst) and the highest global consensus (>=99%). As described in Ziems et al. (2022), the severity of violation is defined as “how severe or serious is it when someone does not follow the RoT? (1) fine; (2) unwise; (3) bad; (4) horrible; (5) worst.” The global consensus is defined as “What percent ofpeople (globally) do you think agree with your RoT? (1) nobody (<1%); (2) rare (5% 25%); (3) controversial ( 50%); (4) most (75% 90%); (5) all (>99%)”. In SOCIAL-CHEM 101 dataset, we choose those RoTs where the RoTs are in the highest global consensus and the corresponding action receives greatest pressure from the cultures. Finally, we get 13,950 safety RoTs from MIC dataset and 14,757 safety RoTs from SOCIAL-CHEM 101 dataset. We encode the safety RoTs into vectors using Sim-CSE<sup>4</sup> (Gao et al., 2021) and build indexes using Faiss (Johnson et al., 2019). For determining a given answer A whether it violates any safety RoTs, we encode the answer A to a vector and find the most related top-k safety RoTs. In this paper we empirically set k = 5 (rather than all safety RoTs) for computational efficiency. We present a retrieved case shown as Table 7.

## C Details of Moral Foundation Analysis

## C.1 Calculation Implementation

We introduce our calculation method in detail. For each moral foundation, we calculate the ratio of the number of Moral BBot’s generated answers based on the foundation to the number of the original answers based on the foundation. Formally, we have question test set $Q .$ For each question $q \in Q ,$ , we have at least two corresponding answers with different moral foundations $\{ ( a _ { 1 } , f _ { 1 } ) , ( a _ { 2 } , f _ { 2 } ) , \cdot \cdot \cdot ( a _ { n } , f _ { n } ) \}$ and the generated answer aˆ by Moral BBot. $a \subset f$ denotes the answer a is based the moral foundation f. I( ) denotes indicate function. For each moral foundation, we calculate the ratio $R _ { f }$ as

$$
R _ { f } = { \frac { \displaystyle \sum _ { q \in Q } P _ { \theta } ( { \hat { a } } \sqsubseteq f ) } { \displaystyle \sum _ { q \in Q } \sum _ { i = 1 } ^ { n } \operatorname { I } ( a _ { i } \sqsubseteq f ) } }\tag{6}
$$

The denominator can be directly calculated in the annotated dataset while the numerator requires a trained model $P _ { \theta }$ to give likelihood that a generated answer is based the moral foundation. To this end, we first adopt ME dialogue flow to generate an RoT of given answer by Moral BBot. Then we train a multi-label classification model based on RoBERTa (Liu et al., 2019) and MIC dataset (Ziems et al., 2022) to judge which moral foundation a given RoT is based on. Noticeably, in the calculation, for each answer, we use soft number (i.e. the sum of probability by classifier) of all generated answers. The following is the reason for the calculation formula.

<table><tr><td>Classes</td><td>Phrases</td></tr><tr><td>Why-class</td><td>Can you tell me why? | Why? | What is the basis of that? I Say it clear, please. I Why do you think that? What is the reason? | Would you like to tell me why? | I just want to know why. I Tell me the reason, please. Sorry, I&#x27;d like to know the reason. I Thanks, and why? | Why is that? I Why do you say that? I Any rule of thumb? Any reason? | What values are you expressing?</td></tr><tr><td>But-class</td><td>But from my perspective | Have you ever thought that I Did you consider that | But I think | As a rule of thumb, But most people think that However, most people consider that I Your answer violates the thing that Your answer does not entail I Your answer contradicts that IBut most people do not agree that From my perspective, only a few people think that I Actually I do not agree that</td></tr><tr><td>Sorry-class</td><td>I&#x27;m sorry. I Yes, you are right. I I&#x27;d like to correct my answer. | Let me see... I think | Good idea. After being revised by you, I think | That makes sense. | Sorry. I I was wrong. I I made a mistake. Thanks for correcting. I Make sense!</td></tr><tr><td>Base-class</td><td>Based on the rule of thumb, I want to ask another question. I Yes, and based on that, here comes another question. I have a similar question for you. I How about this similar question. I May you answer the similar question for me? Given what you have learnt, can you answer this question?</td></tr></table>

Table 5: The phrases inserted in our constructed discussions. Why-class, But-class phrases, Sorry-class, Base-class are used in ME, MR, MR, RIL dialogue flows, respectively.

<table><tr><td>Agree</td><td>Neutral</td><td>Disagree</td></tr><tr><td># Train 55,005</td><td>64,519</td><td>18,545</td></tr><tr><td># Dev 6,959</td><td>7,594</td><td>2,309</td></tr><tr><td># Test 6,859</td><td>7,570</td><td>2,306</td></tr><tr><td>Total 68,823</td><td>79,683</td><td>23,160</td></tr></table>

Table 6: The dataset statistics of 3-way agreement classification task.

<table><tr><td>User: Tell me your opinion on jumping red light.</td></tr><tr><td>Moral BBot: I don&#x27;t think it&#x27;s a good idea to drive under the influence of red lights.</td></tr><tr><td>Retrieved top-5 safety RoTs: 1. You shouldn&#x27;t run red lights because it&#x27;s dangerous. 2. It&#x27;s bad to drive while intoxicated.</td></tr><tr><td>3. It&#x27;s bad to drive recklessly. 4. It&#x27;s bad to operate a vehicle while</td></tr><tr><td>under the influence. 5. It&#x27;s bad to run red lights.</td></tr></table>

Table 7: An example of retrieved top-5 RoTs.

![](images/db3113ccdd4c4d36525cc75fef57f217907f536e4cac274ce19f622890c3bda9.jpg)  
Figure 4: The moral foundation proportion of the answers in the train set.

Most of all questions only concern one foundation<sup>5</sup>. If we only used the numerator part in Eq. 6 to calculate foundation tendency, the calculated distribution would degenerate to the distribution of the foundations that the questions concern. Thus, we first pick up those controversial questions to make the foundations that each question concerns more diverse. Then we put the denominator part in Eq. 6 to normalize the foundation number in numerator part.

## C.2 Moral Foundation Proportion

We present the moral foundation proportion in the train set as Figure 4. From the pie chart we can see that the most category, “care” covers 36.9% answers in the train set, which may lead to the strong “care” foundation tendency of Moral BBot.

<table><tr><td>Hyper-parameters</td><td>Values</td></tr><tr><td>Learning rate</td><td>2e-5</td></tr><tr><td>Batch size</td><td>8</td></tr><tr><td>Max grad norm</td><td>1.0</td></tr><tr><td># Epochs</td><td>5</td></tr><tr><td>Max input length</td><td>128</td></tr></table>

Table 8: The hyper-parameters for agreement scorers.

## D Reproducibility

## D.1 Computing Infrastructure

We extend our special thanks to the library Transformers (Wolf et al., 2020), based on which we conduct most of our experiments. For model training, we utilize the Tesla V100 card with 32 GB memory. We will release our constructed dataset, codes, and moral conversational model checkpoints upon publication.

## D.2 Agreement Scorer Training

In training the agreement scorer, we choose albertbase-v2<sup>6</sup> (12M parameters), roberta-base<sup>7</sup> (125M parameters), bert-base-uncased<sup>8</sup> (109M parameters) for the experiments.

The hyper-parameters for training the agreement scorer are shown as Table 8. For training we use AdamW optimizer (Loshchilov and Hutter, 2017) and linear scheduler with warm-up. We select the checkpoint by the highest F1-score on development set. It cost 2 hours for training each model.

## D.3 Moral Conversational Models Training

We choose DialoGPT-medium<sup>9</sup> (355M parameters) and Blenderbot-400M<sup>10</sup> (365M parameters) for the experiments.

The hyper-parameters for training the moral conversational models are shown as Table 9. We use AdamW optimizer (Loshchilov and Hutter, 2017) linear scheduler with warm-up. In training process, we select the model checkpoint by the lowest loss on development set. It cost 8 hours for training each model. It cost about 2 hours for evaluating each model based on our proposed metrics.

<table><tr><td>Hyper-parameters</td><td>Values</td></tr><tr><td>Learning rate</td><td>2e-5</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Max grad norm</td><td>1.0</td></tr><tr><td># Epochs</td><td>3</td></tr><tr><td>Max input length</td><td>128</td></tr><tr><td>Decoding algorithm</td><td>Beam Search</td></tr><tr><td># Beams</td><td>10</td></tr><tr><td>Max output length</td><td>60</td></tr></table>

Table 9: The hyper-parameters for moral conversational models training and inference.

## E Human Interactive Evaluation

In human interactive evaluation, we compare our proposed model Moral BBot and the original model BBot. We develop a interacting website for crowd-workers to make conversations with the models.

## E.1 Interacting Process

The crowd-workers are first asked to consider a moral topic (e.g. violence). Based on the topic, they use the same opening to talk with the two conversational models to confirm two conversations are in the same topic. Then the crowd-workers are allowed to talk without limitation till at least 8 turns. After conversation, the crowd-workers are asked to annotate each sentence generated by the two conversational models from their own feelings. Finally we collect 100 conversations for each model. The remuneration is 25 USD per hour.

## E.2 Annotation Guideline

The crowd-workers annotate according to the following guideline.

• Does this sentence embody any morals of the chatbot?

Options: [True], [False]

• If the last question is [True], Do you think what percent of people (globally) do you think agree with the moral standpoint? Options: [1: Nobody], [2: Rare], [3: Controversial], [4: Most], [5:All]

• Is this sentence sensible? Options: [True], [False]

• Is this sentence specific? Options: [True], [False]

![](images/0abd90c748c6e92a71ff1086a16a06d850e3f6a701dd1d095b4ad0d3e0ba2d08.jpg)  
Figure 5: A comparison example between Moral BBot and BBot in human experiments.

The annotated scores for each criteria are shown in Table 4.

## F Case Study

To better show the effect and performance of the proposed moral dialogue systems, we present a case study (shown as Figure 5) of moral conversations collected by human evaluation experiments. The annotator uses the same discussion opening for both BBot and Moral BBot, asking the opinions about “jumping a red light”. It shows that BBot does not have a good understanding of jumping a red light, while Moral BBot can well express the moral view that “jumping a red light running is wrong" and the reason behind it: “it is good to drive safely”. In addition, faced with the same question “What will you do if your taxi driver does not follow the traffic rules?”, Moral BBot gives a more reasonable answer. Moreover, Moral BBot establishes the inner connection between “traffic violation” and “police”, which embodies morality.

## ACL 2023 Responsible NLP Checklist

A For every submission: <sup>✓</sup> A1. Did you describe the limitations of your work? Sec. "Limitations"

<sup>✓</sup> A2. Did you discuss any potential risks of your work? Sec. "Ethics Statement"

<sup>✓</sup> A3. Do the abstract and introduction summarize the paper’s main claims? 1

✗ A4. Have you used AI writing assistants when working on this paper? Left blank.

## B <sup>✓</sup> Did you use or create scientific artifacts?

3, 4

<sup>✓</sup> B1. Did you cite the creators of artifacts you used? 3, 4

<sup>✓</sup> B2. Did you discuss the license or terms for use and / or distribution of any artifacts? Sec. "Ethics Statement", Appendix C.2

<sup>✓</sup> B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? For the artifacts you create, do you specify intended use and whether that is compatible with the original access conditions (in particular, derivatives of data accessed for research purposes should not be used outside of research contexts)? Sec. "Ethics Statement"

<sup>✓</sup> B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it? Sec. "Ethics Statement"

<sup>✓</sup> B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? Sec. "Ethics Statement"

<sup>✓</sup> B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? Even for commonly-used benchmark datasets, include the number of examples in train / validation / test splits, as these provide necessary context for a reader to understand experimental results. For example, small differences in accuracy on large test sets may be significant, while on small test sets they may not be. Table 1, Section 5.1, Appendix A

## C <sup>✓</sup> Did you run computational experiments?

5

<sup>✓</sup> C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? 5, Appendix D

<sup>✓</sup> C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? 5, Appendix D

<sup>✓</sup> C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run? 5

 C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation), did you report the implementation, model, and parameter settings used (e.g., NLTK, Spacy, ROUGE, etc.)? Not applicable. Left blank.

D <sup>✓</sup> Did you use human annotators (e.g., crowdworkers) or research with human participants? 5.4

<sup>✓</sup> D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.? Appendix E

<sup>✓</sup> D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participants’ demographic (e.g., country of residence)? Appendix E

<sup>✓</sup> D3. Did you discuss whether and how consent was obtained from people whose data you’re using/curating? For example, if you collected data via crowdsourcing, did your instructions to crowdworkers explain how the data would be used? Sec. "Ethics Statement"

 D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? Not applicable. Left blank.

 D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? Not applicable. Left blank.