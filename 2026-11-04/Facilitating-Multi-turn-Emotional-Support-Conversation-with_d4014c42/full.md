# Facilitating Multi-turn Emotional Support Conversation with Positive Emotion Elicitation: A Reinforcement Learning Approach

Jinfeng Zhou<sup>1,2</sup>∗ † Zhuang Chen<sup>1</sup>† Bo Wang<sup>2</sup>‡ Minlie Huang<sup>1</sup>

<sup>1</sup>The CoAI Group, DCST, Institute for Artificial Intelligence, State Key Lab of Intelligent Technology and Systems,

<sup>1</sup>Beijing National Research Center for Information Science and Technology, Tsinghua University, Beijing 100084, China

<sup>2</sup>College of Intelligence and Computing, Tianjin University, Tianjin, China

jfzhou.mail@gmail.com, zhchen-nlp@mail.tsinghua.edu.cn, bo\_wang@tju.edu.cn, aihuang@tsinghua.edu.cn

## Abstract

Emotional support conversation (ESC) aims to provide emotional support (ES) to improve one’s mental state. Existing works stay at fitting grounded responses and responding strategies (e.g., question), which ignore the effect on ES and lack explicit goals to guide emotional positive transition. To this end, we introduce a new paradigm to formalize multi-turn ESC as a process of positive emotion elicitation. Addressing this task requires finely adjusting the elicitation intensity in ES as the conversation progresses while maintaining conversational goals like coherence. In this paper, we propose SUPPORTER, a mixture-of-expert-based reinforcement learning model, and well design ES and dialogue coherence rewards to guide policy’s learning for responding. Experiments verify the superiority of SUPPORTER in achieving positive emotion elicitation during responding while maintaining conversational goals including coherence.

## 1 Introduction

Emotional support (ES) aims to reassure a person to recover from emotional distress and improve one’s mental state (Burleson, 2003). It is a manifestation of emotional intelligence in social interactions (Heaney and Israel, 2008; Atoum and Al-Shoboul, 2018). Endowing ES into social dialogue systems for building helpful and trustful agents is an emerging trend (Huang et al., 2020; Rains et al., 2020).

To achieve this goal, a typical practice is modeling empathy, which aims to perceive and understand the situation and feelings of others (Keskin, 2014). Yet, the empathetic conversation (Rashkin et al., 2019) is inherently deficient in providing ES as (1) Lack of consideration of multi-turn conversation. Just making empathetic responses in each single dialogue turn leads to ignoring the user’s feedback and mental state changes in multi-turn interaction. (2) Lack of awareness of emotional elicitation. Only emanating emotional resonance fails to help users jump out of negative mental states. Although Liu et al. (2021) design emotional support conversation (ESC) task promising to remedy these deficiencies, existing works (Tu et al., 2022; Cheng et al., 2022; Peng et al., 2022) stay at fitting grounded responses and responding strategies (e.g., question) while ignoring the effects of such efforts on ES. They do not fully model the essential working mechanism of ESC and lack explicit goals to guide a user’s emotion to a positive transition in the multi-turn process. Thus, they are still insufficient to lay out an entire ESC process and cannot effectively improve one’s mental state.

![](images/dc4b87d2aec2475178a5fe20f62917796c27423ee044b52f0f4b11a0107cdf2c.jpg)  
Figure 1: A simplified multi-turn ESC example between the user (left) and agent (right). The agent progressively adjusts the intensity of empathy and elicitation to achieve the goal of improving the user’s mental state.

To this end, we introduce multi-turn ESC with positive emotion elicitation, a new paradigm aims to progressively empathize and elicit users to reach a better mental state through multi-turn conversation. Addressing this task is challenging (an example is in Figure 1): First, in a realistic multi-turn ESC, the user’s emotions often transit towards positive (e.g., the user’s emotion starts with negative and ends with positive, i.e., “My school was closed”

“Ifeel better now”) with fluctuation (e.g., the user’s negative emotions in the first two turns gradually deepen, i.e., “My school was closed” “I don’t even know”), which requires the agent to equip with the mechanism dealing with complex situations to respond satisfactorily (Shibata et al., 2014; Yoshino and Kawahara, 2015). Second, for ES, the ES response requires a delicate balance between empathy and elicitation. Only empathizing without eliciting falls into a negative emotional cycle, while the opposite setting brings a sense of distance in communication. They need to be progressively and purposefully adjusted in ongoing interactions, e.g., the agent expresses empathy of varying emotional polarity (negative negative positive) and carefully increase the intensity of elicitation (only empathy  weak elicitation  strong elicitation). Third, for language expression, the ES response purposefully elicits positive emotions but should not undermine general conversational goals like coherence. Making an eliciting response that is out of the dialogue context, e.g., replacing “I understand you. I would ... happened to me.” with “Come on! I believe ... find a solution!”, may cause users to resent and block useful feedback.

In this paper, we propose SUPPORTER<sup>1</sup> to facilitate multi-turn emotional SUPPORT conversation with positive emotion Elicitation using a mixtureof-expert(MoE) based Reinforcement learning(RL). MoE designs heuristic experts associated with specific tasks to learn diverse semantics by characterizing dialogue context, where: (1) To cope with the user’s emotional fluctuation in the ongoing conversation, experts are devised as positive and negative experts as a whole; (2) To inspire ES of responding, the emotion experts of MoE are designed to predict the user’s emotional states that are possibly transited to; (3) To inspire the expression of responding, the keyword experts of MoE are designed to predict the keywords that maintain the dialogue coherence. With experts as candidates, our RL agent learns conversational semantic encoding policy and purposefully selects experts with expert selection policy for response generation. To achieve the goal of positive emotion elicitation during responding while maintaining conversational goals like coherence, we optimize policy by carefully constructing the rewards: (1) ES rewards consider the conversation progress to dynamically adjust the elicitation intensity of positive emotion; (2) Dialogue coherence rewards involve keyword-level and sentencelevel guides to finely maintain coherence.

Our contributions are summarized as follows:

(1) We introduce a new paradigm by carefully dissecting the challenges of formalizing multi-turn ESC as a process of positive emotion elicitation.

(2) We propose SUPPORTER, an MoE-based RL model with carefully constructed ES and dialogue coherence rewards, elicits positive emotion during responding while maintaining dialogue coherence.

(3) Extensive experiments show the superiority of SUPPORTER with automatic, interactive human, and novel ES and dialogue coherence evaluations.

## 2 Related Work

Empathetic Conversation To construct a warm dialogue system, a milestone is to endow it with empathy (Rashkin et al., 2019). Considering affective empathy (Lin et al., 2019; Majumder et al., 2020; Li et al., 2020, 2022), i.e., perceiving the user’s emotion, and cognitive empathy (Zheng et al., 2021; Sabour et al., 2022; Zhou et al., 2022), i.e., understanding the user’s situation, puts the psychological theory of empathy into practice. Limited by focusing on a single-turn empathy and lack of emotional induction, it is difficult to achieve the higher goal of improving the user’s mental state due to failure to help one jump out of the negative situation.

Emotional Support Conversation To remedy above deficiencies, Liu et al. (2021) design ESC for providing ES in interactions. Our work is related to existing works on ESC but differs in task definition as we focus on enhancing the elicitation effect of positive emotion of responses instead of responding strategy prediction (e.g., question) and grounded response generation. Although fusing knowledge (Tu et al., 2022; Peng et al., 2022) and planning strategy (Cheng et al., 2022) are beneficial for wordoverlap metrics (e.g., Bleu), we argue whether the gains serve to ES is opaque and less convincing due to lacking corresponding evaluation mechanisms.

Positive Emotion Elicitation Conversation To free users from emotional distress and advance the conversation towards an optimistic state, positive emotion elicitation is an intuitive solution (Mishara et al., 2007; Jiang et al., 2021). Previous works (Hasegawa et al., 2013; Lubis et al., 2018, 2019a,b) posit the emotional elicitation process as an ideal single-turn dialogue with linear emotional changes (Wang et al., 2022). However, realistic scenarios often involve multi-turn interactions with complex emotional fluctuations. To weaken the previous strong hypothesis, we extend positive emotion elicitation to ESC by well defining challenges, and take it as a real-world application of the solution.

![](images/567bbdaa1b81ef4613ab519ab58a2e009dcef274f9422a4f00d146796e7b5eaf.jpg)  
Figure 2: The architecture of the proposed SUPPORTER model. DC is an abbreviation for Dialogue Coherence.

## 3 Preliminaries

At the t-th turn of dialogue, given dialogue context $C _ { t } = \{ x _ { 1 } , y _ { 1 } , \ldots , x _ { t - 1 } , y _ { t - 1 } , x _ { t } \}$ , our goal is to generate the response y<sub>t</sub> which serves to improve the user’s mental state. To equip this ability, the response generation process should achieve specific goals related to ES and language expression.

ES for Positive Emotion Elicitation Providing effective elicitation during multi-turn ESC suffers from two issues: First, the elicitation intensity of positive emotion needs to be adjusted progressively as the conversation progresses. Maintaining weak elicitation (e.g., “I understand you”) or strong elicitation (e.g., “Come on”) may fail to shake one’s mental state. Second, the elicitation effect of positive emotion needs to be indirectly verified by the feedback from the user’s next turn utterance. It means the elicitation intensity should consider the future fluctuation of the user’s emotional states. In this work, we construct conversation-level and turn-level ES rewards to guide the model’s learning of elicitation policy and conduct corresponding automatic and interactive human evaluations for measuring the ES performance of responding.

Language Expression for Dialogue Coherence The purpose of generative processes to enhance elicitation induces two attendant issues: First, without proper controls may lead to greedily pursuing the goals of elicitation while discarding the contextual coherence, e.g., “Come on!” with strong elicitation as a response in the context of the user continuing to express negative emotions. Second, whether the response meets the user’s expectations needs feedback from the user’s future utterance. It means maintaining coherence with future dialogue is also crucial. In this work, we construct contextual and future dialogue coherence rewards to guide the model’s learning of bi-coherent expressions and perform the automatic and interactive human evaluation of conversational goals including coherence.

## 4 Methodology

In Figure 2, our SUPPORTER takes dialogue context as input to construct state sequence, which is encoded by a dialogue encoder as the conversational semantic encoding policy. The mixture-of-expert associated with emotion and keyword prediction tasks characterize state semantics to yield action candidates of the expert selection policy, which are purposefully selected for inducing state update. We use the updated state to generate response and further optimize the policy by measuring how well the response reaches the goal of ES and dialogue coherence with the well-designed parallel rewards.

## 4.1 Multi-task Mixture-of-Expert

As a key component of SUPPORTER, we first introduce the structure of multi-task mixture-of-expert.

Dialogue Encoder Following Liu et al. (2021), the dialogue encoder is implemented with Blender-Bot (Roller et al., 2021). Given an input sequence X, we concatenate all input tokens and prepend with a [CLS] token, e.g., for the dialogue context, getting $[ C L S ] \oplus x _ { 1 } \oplus y _ { 1 } \ldots \oplus x _ { t - 1 }$ . The sequence is fed into the dialogue encoder to obtain the hidden state $H _ { X }$ . We denote the sequence representation derived from [CLS] as $h _ { X }$

Emotion Experts To track possible transitions of user’s emotional states, emotion experts are associated with contextual and future user emotion predictions. We extract M fine-grained emotional reactions for each utterance in the corpus, which are inferred from COMET (Bosselut et al., 2019) using the “xReact” relation. Since emotional reactions are often emotional words (e.g., happy, sad), we use VAD (Mohammad, 2018) to identify the emotional polarity of each word according to its valence as a positive or negative emotional category. The high-frequency categories are finally retained as supervised labels for the emotion prediction task.

We divide contextual emotion experts into positive and negative emotion experts, which are two MLP transforming $H _ { X }$ into $H _ { X , p o s }$ and $H _ { X , n e g } \mathrm { : }$

$$
\begin{array} { r } { { \pmb { H } } _ { X , p o s } = M L P _ { p o s } \left( { \pmb { H } } _ { X } \right) , } \\ { { \pmb { H } } _ { X , n e g } = M L P _ { n e g } \left( { \pmb { H } } _ { X } \right) . } \end{array}\tag{1}
$$

We project the [CLS] representations $h _ {  { { X } } , p o s }$ and $h _ { X , n e g }$ of positive and negative experts to predict positive and negative emotion, respectively:

$$
\begin{array} { r } { P _ { p o s } = \operatorname { s o f t m a x } \left( W _ { p o s } h _ { X , p o s } \right) , } \\ { P _ { n e g } = \operatorname { s o f t m a x } \left( W _ { n e g } h _ { X , n e g } \right) , } \end{array}\tag{2}
$$

which is supervised by the positive and negative emotions collected in the $e _ { p o s } ^ { * }$ and $e _ { n e g } ^ { * }$ sets of the user’s last utterance in the dialogue context using cross-entropy loss:

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { p o s } ^ { c t x - e m o } = - \frac { 1 } { \displaystyle { \left| e _ { p o s } ^ { * } \right| } } \sum _ { i = 1 } ^ { \displaystyle { \left| e _ { p o s } ^ { * } \right| } } \log P _ { p o s } \left( e _ { i } ^ { * } \right) } , }  \\ { { \displaystyle { \cal L } _ { n e g } ^ { c t x - e m o } = - \frac { 1 } { \displaystyle { \left| e _ { n e g } ^ { * } \right| } } \sum _ { i = 1 } ^ { \displaystyle { \left| e _ { n e g } ^ { * } \right| } } \log P _ { n e g } \left( e _ { i } ^ { * } \right) } . }  \end{array}\tag{3}
$$

Note that an utterance may be inferred to the emotions with different polarities due to cognitive differences (Westbrook et al., 2011; Zhou et al., 2022). For future emotion experts, we adopt the above method to get $L _ { p o s } ^ { f t r - e m o }$ and $L _ { n e g } ^ { f t r - e m o }$ losses and train them to predict the positive and negative emotions of the user’s future utterance (i.e., next turn utterance). In this way, emotion experts can learn various emotion-level features by $L _ { e m c }$ loss: $L _ { e m o } =$ $L _ { p o s } ^ { c t x - e m o } + L _ { n e g } ^ { c t x - e m o } + L _ { p o s } ^ { f t \bar { r } - e m o } + L _ { n e g } ^ { f t r - e m o }$

Keyword Experts To meet the need for dialogue coherence, keyword experts are associated with keyword predictions that act on maintaining coherence with contextual and future utterances. Here, a bidirectional emotion keyword graph $\mathcal { G }$ is constructed, which is also used in coherence rewards designing (a construction example is in Appendix $\mathbf { A } )$ . We extract the salient keywords of each utterance in the corpus as vertices using a rule-based approach (Tang et al., 2019), and employ VAD to identify the emotional polarity of each keyword. The pointwise mutual information (PMI) (Church and Hanks, 1989) is adopted to construct bidirectional edges by characterizing the association between keyword pairs, where theforward edge depicts the keyword pairs extracted from the context and response, and the backward edge depicts the ones are from the future utterance and response. We further construct positive edges to describe the keywords with positive tail vertices, and negative edges are negative ones. Finally, each head vertex selects the tail vertices with the top PMI scores for building connections. The vertices of $\mathcal { G }$ serve as supervised labels for the keyword prediction task.

Contextual keyword experts are transformed similarly to emotion experts, and their [CLS] representations $h _ { X , p o s } ^ { c t x - k w s }$ and $h _ { X , n e g } ^ { c t x - k w s }$ can be obtained from positive and negative keyword experts ${ \pmb { H } } _ { X , p o s } ^ { c t x - k w s }$ and ${ \cal H } _ { X , n e g } ^ { c t x - k w s }$ , respectively. We infer the one-hop neighbors of contextual keywords from the “forward-positive” and “forward-negative” relations respectively in $\mathcal { G }$ to enhance the perception of the target keywords in the golden response. Specifically, we use attention (Bahdanau et al., 2015) to obtain fused embeddings $e _ { p o s } ^ { c t x - k w s }$ and $e _ { n e g } ^ { c t x - k w s }$

$$
\begin{array} { r l } & { e _ { p o s } ^ { c t x - k w s } = \mathrm { A t t e n t i o n } ( \pmb { h } _ { X , p o s } ^ { c t x - k w s } , \pmb { E } _ { p o s } ^ { c t x - k w s } ) , } \\ & { e _ { n e g } ^ { c t x - k w s } = \mathrm { A t t e n t i o n } ( \pmb { h } _ { X , n e g } ^ { c t x - k w s } , \pmb { E } _ { n e g } ^ { c t x - k w s } ) , } \end{array}\tag{4}
$$

where $E _ { p o s } ^ { c t x - k w s }$ and $E _ { n e g } ^ { c t x - k w s }$ are positive and negative neighbor embedding matrices that share parameters with the dialogue encoder. We then concatenate $e _ { p o s } ^ { c t x - k w s }$ and $e _ { n e g } ^ { c t x - k w s }$ with ${ \pmb { H } } _ { X , p o s } ^ { c t x - k w s }$ and ${ \pmb { H } } _ { X , n e g } ^ { c t x - k w s }$ respectively at the token level, and use an MLP layer to fuse them to obtain keywordenhanced experts ${ \pmb { H } } _ { X , p o s - k w s } ^ { c t x - k w s }$ and $\pmb { H } _ { X , n e g - k w s } ^ { c t x - k w s } ;$

$$
\pmb { H } _ { X , p o s - k w s } ^ { c t x - k w s } [ i ] = \mathrm { M L P } ( \pmb { H } _ { X , p o s } ^ { c t x - k w s } [ i ] \oplus \pmb { e } _ { p o s } ^ { c t x - k w s } )
$$

$$
\begin{array} { r } { \pmb { H } _ { X , n e g - k w s } ^ { c t x - k w s } [ i ] = \mathrm { M L P } ( \pmb { H } _ { X , n e g } ^ { c t x - k w s } [ i ] \oplus \pmb { e } _ { n e g } ^ { c t x - 1 } } \end{array}\tag{5}
$$

Further, we take the positive and negative keywords in the golden response as supervision to optimize the $L _ { p o s } ^ { c t x - k w s }$ and $L _ { n e g } ^ { c t x - k w s }$ losses adopting cross-entropy (this process can refer to above emotion prediction task). Similarly, multi-hop reasoning on , i.e., “forward forward backward-$p o s i t i v e ^ { , \ast }$ and “forward forward backwardnegative” (clarified in Appendix $\mathbf { A } )$ , is performed to obtain keywords coherent with the future utterance. Taking the positive and negative keywords in future utterance as the prediction target, the keyword-enhanced future keyword experts can be optimized by $L _ { p o s } ^ { f t r - k w s }$ and $L _ { n e g } ^ { f t r - k \hat { w } s }$ losses. In this way, keyword experts can learn various expression-level features by $L _ { k w s }$ loss: $L _ { k w s } =$ $L _ { p o s } ^ { \dot { c } t x - k w s } + L _ { n e g } ^ { c t x - k w s } + L _ { p o s } ^ { \dot { f } \dot { t } r - \ddot { k w s } } + L _ { n e g } ^ { f t r - \ddot { k } \overrightarrow { w s } } .$

Multi-task Training To make the experts retain the primitive semantics without hindering their respective diversity, we give them a minor constraint. Specifically, we average the representations of emotion and keyword experts to get $h _ { X , e x p } .$ , and make it close to sequence representation $h _ { X }$ by optimizing the MSE loss with a minor hyperparameter α:

$$
L _ { m s e } = \frac { \alpha } { d _ { h } } \sum _ { i = 1 } ^ { d _ { h } } \left( { \pmb h } _ { X } [ i ] - { \pmb h } _ { X , e x p } [ i ] \right) ^ { 2 } ,\tag{6}
$$

where $d _ { h }$ is the dimension of $h _ { X }$ . Then, we jointly train the multi-task MoE by optimizing $L _ { e x p }$ loss:

$$
L _ { e x p } = L _ { e m o } + L _ { k w s } + L _ { m s e } .\tag{7}
$$

## 4.2 MoE-based Reinforcement Learning

We use the standard reinforcement learning framework (Sutton and Barto, 2018) as the backbone.

State We concatenate the dialogue context and the extracted keywords as the initial state $s _ { 1 } \in S$ i.e., $s _ { 1 } = \{ C , C _ { k w s } \}$ (we omit the subscript t of dialogue context $C _ { t }$ for simplicity). At each step, the prompt token sequence $\mathcal { E }$ generated by the policy determined expert (i.e., action) triggers an update of the state. We record the observed state $s _ { k } \in \mathcal S$ at k-th step, i.e., $s _ { k } = \{ C , \mathcal { E } _ { 1 } , \ldots , \mathcal { E } _ { k - 1 } \}$ , which is encoded by the dialogue encoder to get ${ \cal H } _ { S , k }$ and $h _ { S , k }$ . We concatenate sequence representations of historical states to obtain current state embedding $\pmb { s } _ { k } = \pmb { h } _ { S , 1 } \oplus . . . \oplus \pmb { h } _ { S , k }$ . If k is smaller than the set maximum iteration steps $K ,$ , we pad $\scriptstyle { \pmb { s } } _ { k }$ with zeros for fixing dimension. Note that when $k > 1$ we discard the keywords $C _ { k w s }$ because: (1) It has already acted on the first iteration; (2) The input sequence length is limited due to the constraint of the pre-trained model (i.e., BlenderBot).

Action The action space $\mathcal { A } _ { k }$ at k-th step is defined as the multi-task associated experts transformed by state $s _ { k }$ . At state $s _ { k }$ , our agent learns to choose an expert in $\mathcal { A } _ { k }$ as expert action $a _ { k }$ . We utilize a BlenderBot-based dialogue decoder to generate expert prompt $\mathcal { E } _ { k }$ of $a _ { k }$ .

Policy Besides the above dialogue encoder as the semantic encoding policy network, we design an expert selection policy network using REINFORCE with baseline (Sutton and Barto, 2018) that includes an actor network and a value network. Actor learns an expert finding policy $\pi _ { \varphi } \left( a _ { k } , s _ { k } , \mathcal { A } _ { k } \right)$ which selects the appropriate expert action $a _ { k }$ based on the current state $s _ { k }$ and action space $\mathcal { A } _ { k }$ by emitting the probability distribution of actions in $\mathcal { A } _ { k }$ . The value network measures the value $Q _ { \delta } \left( s _ { k } \right)$ of state $s _ { k }$ as the baseline in REINFORCE. Their network structures are defined as:

$$
\begin{array} { c } { { \pmb { o } _ { k } = \eta \left( \left( \eta \left( \pmb { s } _ { k } \pmb { W } _ { 1 } \right) \pmb { W } _ { 2 } \right) \right) , } } \\ { { \pi _ { \varphi } \left( a _ { k } , \pmb { s } _ { k } , \pmb { \mathcal { A } } _ { k } \right) = \phi \left( \pmb { A } _ { k } \odot \pmb { o } _ { k } \pmb { W } _ { \varphi } \right) , } } \\ { { Q _ { \delta } \left( \pmb { s } _ { k } \right) = o _ { k } \pmb { W } _ { \delta } , } } \end{array}\tag{8}
$$

where $\eta ( \cdot )$ is an ELU activation function with a dropout layer, $\odot$ is the hadamard product, $\phi ( \cdot )$ is the softmax function. $\pmb { A } _ { k }$ is a binarized vector for pruning the action space, and we set it as a full-one vector due to the small number of experts.

Rewards To guide policy learning, we reward the decision made at each step by measuring how well the response generated from updated state $s _ { k + 1 }$ provides ES and maintains dialogue coherence.

(1) Conversation-level ES Reward: aims to dynamically adjust the elicitation intensity of positive emotion as the conversation progresses defined as:

$$
\begin{array} { c } { { P E D _ { c E S } = f _ { E S } ( y ) - f _ { E S } \left( c _ { t } \right) , } } \\ { { { \displaystyle r _ { c E S } = \sum _ { t = 1 } ^ { T } \cos ( \frac { \pi } { 2 } \cdot \frac { t } { M T } ) \cdot P E D _ { c E S } . } } } \end{array}\tag{9}
$$

Here, $f _ { E S } ( \cdot )$ measures the positive emotion level of an utterance using the emotion classification model developed by Hartmann (2022). The model is trained on six datasets containing diverse text types and achieves 66% accuracy for emotion classification. Positive emotion scores are collected as positive level. We encourage the positive emotion distance $P E D _ { c E S }$ of the generated response y and the contextual user’s post $c _ { t } \mathrm { : }$ (a) is non-negative, i.e., expressing empathy (equal to 0) or elicitation (greater than 0) is the underlying requirement; (b)

synchronously increases with the dialogue turn t, i.e., the early stage of the conversation is dominated by empathy, and the latter is elicitation. MT is the maximum turn of conversation, T is current turn.

(2) Turn-level ES Reward: aims to capture the feedback of user’s next turn emotion defined as:

$$
\begin{array} { c } { { P E D _ { t E S } = | f _ { E S } ( y ) - f _ { E S } ( c _ { f } ) | , } } \\ { { r _ { t E S } = \cos ( \displaystyle \frac { \pi } { 2 } \cdot \frac { T } { M T } ) \cdot \cos ( \displaystyle \frac { \pi } { 2 } \cdot P E D _ { t E S } ) . } } \end{array}\tag{10}
$$

Here, $P E D _ { t E S }$ measures the relative positive emotion distance between the generated response y and the user’s future (i.e., next turn) utterance $c _ { f } .$ We encourage $P E D _ { t E S }$ to get smaller with the approaching of current turn $T$ to MT, i.e., supervising smooth elicitation in the latter stage and improving tolerance to emotional fluctuations.

(3) Contextual Dialogue Coherence Reward: aims to constrain generated response y to maintain coherence with context C by measuring their coherence at keyword-level and sentence-level. First, we reconstruct a dataset (Liu et al., 2021) containing coherent and incoherent context-response pairs, where the response of the incoherent pairs is an utterance randomly sampled from the dataset. Next, a BERT-based (Devlin et al., 2019) text classification model $f _ { c D C }$ is trained by feeding sentencekeyword pairs and achieves 85% accuracy. We take the coherence probability as the coherence score, the reward is defined as:

$$
r _ { c D C } = f _ { c D C } \left( C \oplus C _ { k w s } , y \oplus y _ { k w s } \right) \cdot e ^ { \frac { N _ { c , k w s } } { | y _ { k w s } | } - 1 } ,
$$

where $y _ { k w s }$ is the keyword set of $y$ and $N _ { c , k w s }$ is the number of keywords in $y _ { k w s }$ that are the forward neighbors of contextual keywords in $\mathcal { G }$

(4) Future Dialogue Coherence Reward: aims to introduce the consideration of coherence with the user’s future utterance $c _ { f } .$ . Similarly, we reconstruct a dataset (Liu et al., 2021) containing coherent and incoherent future utterance-response pairs and train another text classification model $f _ { f D C }$ which achieves 77% accuracy. The reward is defined as:

$$
r _ { f D C } = f _ { f D C } \left( c _ { f } \oplus c _ { f _ { k w s } } , y \oplus y _ { k w s } \right) \cdot e ^ { \frac { N _ { f , k w s } } { \left| y _ { k w s } \right| } - 1 } ,\tag{12}
$$

where $N _ { f , k w s }$ is the number of keywords in $y _ { k w s }$ that have a backward relation with keywords $c _ { f _ { k w s } }$ of $c _ { f }$ in ${ \mathcal { G } } .$

(5) Total reward. The total reward is $r = w _ { c E S }$ \* $r _ { c E S } + w _ { t E S } * r _ { t E S } + w _ { c D C } * r _ { c D C } + w _ { f D C } * r _ { f D C } .$

<table><tr><td rowspan="2">Corpus Info.</td><td rowspan="2">#Dialogues #Utterances Avg. length of dialogues Avg. length of utterances #Split Ratio</td><td rowspan="2">1,053</td></tr><tr><td>31,410 29.8 17.8 8:1:1</td></tr><tr><td rowspan="4">Graph  $\mathcal { G }$  Info.</td><td>#Keywords</td><td>2,433</td></tr><tr><td>Avg. forward neighbors</td><td>21.24</td></tr><tr><td>Avg. backward neighbors</td><td>21.17</td></tr><tr><td>Avg. positive neighbors Avg. negative neighbors</td><td>33.94 8.46</td></tr></table>

Table 1: Statistics of our dataset after preprocessing.

## 4.3 Optimization

We set K-step iterations, and the goal of agent learning is to maximize the expected cumulative reward: $\begin{array} { r } { J _ { \theta } = \mathbb { E } _ { \pi } \left[ \sum _ { k = 1 } ^ { K } \gamma ^ { k } r _ { k + 1 } \right] } \end{array}$ , where θ is the learned parameter and $\gamma$ is the discount coefficient. The agent is optimized by $L _ { a g e n t }$ loss and its policy gradient is defined as:

$$
\nabla _ { \theta } J _ { \theta } = \mathbb { E } _ { \pi } [ \nabla _ { \theta } \log \pi _ { \varphi } ( a _ { k } , s _ { k } , \mathcal { A } _ { k } ) ( G - Q _ { \delta } ( s _ { k } ) ) ] ,\tag{13}
$$

where G is the discounted cumulative reward from the initial state to the terminal state. Finally, we take the hidden state ${ \cal H } _ { S , K + 1 }$ of the state $s _ { K + 1 }$ to generate the response, where the decoder is optimized by $L _ { g e n }$ loss:

$$
{ \cal L } _ { g e n } = - \sum _ { m = 1 } ^ { M } \log P ( y _ { m } \mid { \cal H } _ { S , K + 1 } , y _ { < m } ) .\tag{14}
$$

Warm Start We use the pretrained small version of BenderBot for initializing our model. The initial state is used as input to fine-tune the model for warm start by optimizing $L _ { w a r m } = L _ { e x p } + L _ { g e n }$

Joint Training Our model is finally jointly trained by optimizing $L _ { j o i n t }$ loss:

$$
L _ { j o i n t } = L _ { a g e n t } + L _ { g e n } + \frac { 1 } { K + 1 } \sum _ { k = 1 } ^ { K + 1 } L _ { e x p , k }\tag{15}
$$

## 5 Experiments

## 5.1 Experimental Setup

Dataset Our experiments are conducted on the widely used ESConv (Liu et al., 2021), a multi-turn conversation dataset for ES. In a conversation, the user confides personal negative situation, and the supporter provides comfort and support to improve the user’s mental state. The statistics of ESConv and graph $\mathcal { G }$ after preprocessing are in Table 1.

<table><tr><td>Models</td><td>PPL↓</td><td>B-1↑</td><td>B-2↑</td><td>B-3↑</td><td>D-1↑</td><td>D-2↑</td><td>D-3↑</td><td>cES↑</td><td>tES↑</td><td>cDC↑</td><td>fDC↑</td><td>Len</td></tr><tr><td>MoEL</td><td>112.34</td><td>18.14</td><td>6.77</td><td>3.22</td><td>2.43</td><td>17.03</td><td>38.08</td><td>0.658</td><td>0.390</td><td>0.391</td><td>0.384</td><td>20.36</td></tr><tr><td>MIME</td><td>68.49</td><td>15.89</td><td>6.58</td><td>3.27</td><td>2.02</td><td>10.51</td><td>22.60</td><td>0.598</td><td>0.370</td><td>0.450</td><td>0.412</td><td>19.44</td></tr><tr><td>BlenderBot-Joint</td><td>14.78</td><td>17.97</td><td>7.17</td><td>3.31</td><td>4.56</td><td>24.65</td><td>49.71</td><td>0.611</td><td>0.398</td><td>0.710</td><td>0.459</td><td>17.69</td></tr><tr><td>MISC</td><td>16.16</td><td></td><td>7.31</td><td></td><td>4.41</td><td>19.71</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GLHG</td><td>15.67</td><td>19.66</td><td>7.57</td><td>3.74</td><td>3.50</td><td>21.61</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bart-Joint</td><td>16.05</td><td>19.99</td><td>7.92</td><td>3.93</td><td>4.24</td><td>21.98</td><td>43.33</td><td>0.635</td><td>0.402</td><td>0.723</td><td>0.475</td><td>18.85</td></tr><tr><td>SUPPORTER</td><td>15.37</td><td>19.50</td><td>7.49</td><td>3.58</td><td>4.93</td><td>27.73</td><td>53.78</td><td>0.743</td><td>0.409</td><td>0.681</td><td>0.472</td><td>18.37</td></tr><tr><td>w/o EmoExperts</td><td>15.35</td><td>18.32</td><td>7.12</td><td>3.38</td><td>4.79</td><td>27.20</td><td>53.01</td><td>0.711</td><td>0.392</td><td>0.679</td><td>0.460</td><td>18.14</td></tr><tr><td>w/o KwsExperts</td><td>15.54</td><td>17.76</td><td>6.74</td><td>3.19</td><td>4.69</td><td>26.16</td><td>50.92</td><td>0.728</td><td>0.394</td><td>0.636</td><td>0.443</td><td>17.72</td></tr><tr><td>w/o Multi-Task</td><td>15.49</td><td>16.79</td><td>6.54</td><td>3.18</td><td>4.78</td><td>27.17</td><td>53.45</td><td>0.651</td><td>0.399</td><td>0.651</td><td>0.450</td><td>16.48</td></tr><tr><td>w/o ESRewards</td><td>15.46</td><td>18.49</td><td>7.10</td><td>3.36</td><td>4.69</td><td>26.92</td><td>52.49</td><td>0.664</td><td>0.391</td><td>0.660</td><td>0.457</td><td>¯18.41</td></tr><tr><td>w/o DCRewards</td><td>15.43</td><td>17.28</td><td>6.80</td><td>3.25</td><td>4.80</td><td>27.45</td><td>53.04</td><td>0.707</td><td>0.401</td><td>0.652</td><td>0.448</td><td>17.12</td></tr><tr><td>w/o ExpertPolicy</td><td>15.54</td><td>18.30</td><td>7.23</td><td>3.54</td><td>4.75</td><td>27.23</td><td>52.85</td><td>0.683</td><td>0.395</td><td>0.657</td><td>0.454</td><td>18.54</td></tr><tr><td>Warm-Start Only</td><td>15.03</td><td>17.42</td><td>6.74</td><td>3.21</td><td>4.67</td><td>26.24</td><td>51.82</td><td>0.629</td><td>0.402</td><td>0.644</td><td>0.444</td><td>17.35</td></tr><tr><td>w/o Warm-Start</td><td>15.01</td><td>17.98</td><td>6.86</td><td>3.18</td><td>4.55</td><td>26.06</td><td>51.62</td><td>0.673</td><td>0.403</td><td>0.638</td><td>0.453</td><td>18.26</td></tr></table>

Table 2: Automatic evaluation results. “Len” indicates the average length of the generated responses.

Baselines (1) MoEL (Lin et al., 2019): An empathetic conversation model that uses multiple decoders to capture possible user emotions for generating. (2) MIME (Majumder et al., 2020): An empathetic conversation model that mimics user’s emotions during responding. (3) BlenderBot-Joint (Liu et al., 2021): An ESC model that prepends a predicted strategy token on the backbone of Blender-Bot. (4) MISC (Tu et al., 2022): An ESC model that fuses commonsense. (5) GLHG (Peng et al., 2022): A commonsense-based ESC model that designs a global-to-local graph. (6) We design Bart-Joint by replacing the backbone of BlenderBot-Joint with Bart (Lewis et al., 2020). It achieves comparable performance to MultiESC (Cheng et al., 2022) as its replacement since MultiESC’s code is unavailable.

Implementation Details We implement all models with Pytorch, and all pretrained models (i.e., BlenderBot, Bart) use small versions. We set the number of steps K = 2 and reward weights $w _ { c E S } = w _ { c D C } = 0 . 1 , w _ { t E S } = w _ { f D C } = 1 . 0$ (selected using a grid-search approach with two values {0.1, 1.0} for each hyperparameter). We extract M = 10 emotional reactions for each utterance. The maximum number of conversation turn MT is set to 10. The discount factor γ is 0.99, the hyperparameter α is 1e-5, and the batch size is 16. We use Adam optimizer (Kingma and Ba, 2015) with an initial learning rate of 2e-5 and a linear warmup of 120 steps for training on a GPU-V100 machine. The warm start stage is trained for 5 epochs, and the joint training stage is set to 3 epochs. The decoding settings are consistent with Liu et al. (2021). For a fair comparison, all baselines with available codes are reproduced under the same setting.

## 5.2 Automatic Evaluation

We adopt Perplexity (PPL), Bleu (B-n) and Distinct (D-n) to evaluate the general generation quality and diversity of the models. To measure how well the generated responses achieve goals, we define (1) ES scores containing conversation-level (cES) and turn-level (tES), i.e., r<sub>cES</sub> and r<sub>tES</sub>, measure the elicitation intensity of positive emotion involving conversation progress and the perceived intensity to the user’s next turn emotion; (2) Dialogue coherence scores containing contextual (cDC) and future (fDC), i.e., r<sub>cDC</sub> and $r _ { f D C }$ , measure the coherence with the context and the user’s future utterance.

Overall Performance In Table 2, compared with all baselines, our SUPPORTER achieves the most diverse expressions and highest ES (12.9% outperforms the second best MoEL on cES) while maintaining competitive dialogue quality (PPL, Bleu) and coherence (cDC, fDC). Supportive responses generated by MoEL are often accompanied by low diversity and low coherence due to the retelling of generic responses (e.g., “I am glad I could help you” with high positive emotion) that are found from its outputs. Bart-based models benefit from robust sequence modeling (Lewis et al., 2020) with inherent advantages in coherence and Bleu but perform poorly in ES and diversity. The contextual coherence (cDC) of our SUPPORTER is inferior to BlenderBot-Joint, which is acceptable as ES for positive emotion elicitation needs to sacrifice a little coherence to jump out of negative topics.

Ablation Study In Table 2: First, we remove the emotion experts (w/o EmoExperts), keyword experts (w/o KwsExperts), and the multi-task associated with the experts (w/o Multi-Task), respectively. Emotion experts mainly act on ES, including cES and tES. Keyword experts contribute significantly to dialogue coherence, including cDC and fDC. Multi-task training endows experts with specific abilities and thus has an impressive impact on overall performance. Second, we remove the ES rewards (w/o ESRewards) and dialogue coherence rewards (w/o DCRewards), respectively. The former improves positive support, and the latter maintains grounded expression. Therefore, besides achieving their own goals, they also benefit dialogue diversity and quality, respectively. Moreover, we replace the expert selection policy network with random sampling (w/o ExpertPolicy). Random experts lead to uncertainty in decision-making and thus damage overall performance, especially on ES and coherence. Third, we test using only warm start and without joint training (Warm-Start Only) as well as without warm start and only joint training (w/o Warm-Start). The former reaches comparable or even worse results than the baselines, and the latter greedily achieves the goal of maximizing the rewards resulting in low dialogue quality.

<table><tr><td rowspan="2">SUPPORTER vs.</td><td colspan="3">BlenderBot-Joint</td><td colspan="3">Bart-Joint</td><td colspan="3">w/o EmoExperts</td><td colspan="3">w/o ExpertPolicy</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Fluency</td><td>67.5</td><td>23.7</td><td>8.8</td><td>66.5</td><td>26.5</td><td>7.0</td><td>44.5†</td><td>40.0</td><td>15.5</td><td>42.9†</td><td>37.5</td><td>19.6</td></tr><tr><td>Informativeness</td><td>55.2‡</td><td>40.7</td><td>4.1</td><td>56.7‡</td><td>38.8</td><td>4.5</td><td>48.6‡</td><td>36.8</td><td>14.6</td><td>38.5</td><td>35.9</td><td>25.6</td></tr><tr><td>Coherence</td><td>53.8‡</td><td>31.8</td><td>14.4</td><td>45.4</td><td>43.8</td><td>10.8</td><td>53.7‡</td><td>35.7</td><td>10.6</td><td>55.1‡</td><td>32.4</td><td>12.5</td></tr><tr><td>Supportiveness</td><td>59.2‡</td><td>34.1</td><td>6.7</td><td>51.4</td><td>37.6</td><td>11.0</td><td>54.5</td><td>33.4</td><td>12.1</td><td>51.4</td><td>34.3</td><td>14.3</td></tr><tr><td>Overall</td><td>56.5</td><td>30.4</td><td>13.1</td><td>48.6</td><td>37.1</td><td>14.3</td><td>50.0</td><td>34.3</td><td>15.7</td><td>49.6</td><td>32.1</td><td>18.3</td></tr></table>

Table 3: Results of interactive human evaluation (%). / denote p-value < 0.1/0.05 (statistical significance test).

## 5.3 Interactive Human Evaluation

We recruited three crowdsourcing workers and exposed them to 100 negative situations randomly sampled from the test set. They were asked to engage in multi-turn conversation with the models to simulate the process of seeking ES and to choose the better one (Win) from a model pair by considering five aspects, respectively: (1) Fluency: which bot’s response is more fluent and understandable? (2) Informativeness: which bot’s response is more diverse and specific, and contains more information? (3) Coherence: which bot’s response is more coherent with context in a multi-turn conversation? (4) Supportiveness: which bot provides more effective ES, i.e., is more likely to elicit users to change their emotions from negative to positive? (5) Overall: generally, which bot is more preferred?

![](images/4520d3c1699af67dfc889069e7a61f7446835a5d65f0470694f1fc8fac7cc726.jpg)  
Figure 3: Latent space visualization of experts. Separate clusters show MoE has diverse and specific semantics.

As in Table 3, from the comparison with baselines, we found that a single incoherent response (cDC in Table 2) has less impact on the coherence of the overall multi-turn conversation. Comparisons with variants of SUPPORTER demonstrate that key components of our model, i.e., emotion experts and expert selection policy, lead to significant advantages in the overall performance.

## 5.4 Qualitative Analysis

Specificity of Experts To analyze the quality of the experts, we show the specificity of the experts learned by SUPPORTER. As shown in Figure 3, we visualize the latent space of experts using t-SNE on 200 conversation samples. The latent space distributions of multi-task-associated experts are clearly separated and clustered in specific regions. Some overlap is also intuitive due to the similarity between experts with the same polarity, e.g., contextual and future positive emotion experts. This verifies our MoE has diverse and specific semantics and the superiority of multi-task learning.

Adjustability of Elicitation To further explore the adjustability of elicitation intensity of positive emotion in multi-turn conversation, we analyze the trend of positive emotion distance with the dialogue turns, i.e., $\begin{array} { r } { P E D = f _ { E S } ( y ) - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } f _ { E S } \left( c _ { t } \right) } \end{array}$ As shown in Figure 4, the PED score of all models tends to rise first and then fall. In the early stage of the conversation (turn<6), SUPPORTER keeps the same trend as the empathy model (i.e., MoEL, MIME) and gradually increases the intensity of elicitation. This is attributed to our encouragement that it should progressively transform the conversation from empathy-dominated to elicitation-dominated. In the later stage of the conversation (turn>6), SUP-PORTER still maintains a higher level of elicitation than baselines and shows robust adjustment ability.

![](images/6e8e42008b962244f484a2e45d0590761aac711475446582314caf65f185d9b8.jpg)

Figure 4: SUPPORTER progressively enhances the elicitation intensity and exhibits robust adjustment ability in the later stage of the conversation.
<table><tr><td>Models</td><td>D-1</td><td>B-2</td><td> $\overline { { c E S } }$ </td><td>tES</td><td> $\overline { { c D C } }$ </td><td>fDC</td></tr><tr><td>SUPPORTERK=1</td><td>4.40</td><td>7.55</td><td>0.801</td><td>0.382</td><td>0.668</td><td>0.466</td></tr><tr><td>SUPPORTERK=2</td><td>4.93</td><td>7.49</td><td>0.743</td><td>0.409</td><td>0.681</td><td>0.472</td></tr><tr><td>SUPPORTERK=3</td><td>5.22</td><td>6.71</td><td>0.699</td><td>0.405</td><td>0.657</td><td>0.459</td></tr><tr><td>SUPPORTERK=4</td><td>5.05</td><td>6.10</td><td>0.673</td><td>0.413</td><td>0.594</td><td>0.431</td></tr></table>

Table 4: Parameter analysis for iteration steps K. SUP-PORTER outperforms the best baselines in most settings.

## 5.5 Parameter Analysis

We further analyze the impact of the number of iteration steps K. In Table 4, with the increase of steps, diversity and tES show an upward trend, while other metrics show a downward one. This happens possibly because the informativeness of the generated responses increases with selected experts, making it possible to lose focus and thus lead to poor dialogue quality. Furthermore, SUP-PORTER outperforms the best baselines in most cases, confirming its effectiveness.

## 6 Conclusions

In this paper, we introduce a new paradigm to formalize multi-turn ESC as a process of positive emotion elicitation and propose an MoE-based reinforcement learning model SUPPORTER with welldesigned ES and dialogue coherence rewards. Extensive experiments verify the superiority of our model in providing effective ES for positive emotion elicitation while maintaining conversational goals including coherence. Our work will facilitate future work to develop ESC with positive emotion elicitation for improving the users’ mental state.

## Limitations

We discuss three limitations of this work as follows.

The first one is the instability of reinforcement learning. Reward-driven policy learning is an essential advantage of this work because it is better equipped with the positive emotion-driven process of ESC than existing works and can model flexible ESC expression beyond the training data. However, this flexibility also suffers from instability, which calls for additional knowledge or strategies to refine the learning process.

The second one is the need for further reference to psychological theory. An advantage of our work is to learn posterior ESC patterns integrating the dialogue context and future feedback in the form of rewards. However, there is still other valuable prior knowledge to be referred from psychology studies, e.g., the CBT (cognitive-behavioral therapy) methods. This kind of prior knowledge can be used as additional knowledge to refine the learning process as mentioned in the first limitation.

The third one is that the reward design can be further optimized. The ideal case is to construct a high-quality dataset with human-feedback labels for training reward model (e.g., the constructed example of ChatGPT). At the same time, the larger parameter of the reward model, the more conducive it is to learn a robust policy and avoid it overfitting to the reward function. However, such optimizations need a trade-off with cost.

## Ethical Considerations

In this paper, the ESConv dataset used in our experiments is a publicly-available benchmark for emotional support conversation, which does not contain sensitive and personal information as well as unethical language. Our work builds on this dataset to study positive emotion elicitation to improve the user’s mental state. Therefore, we focus on constructing a dialogue system to provide emotional support from families and friends in the daily scenarios limited by this dataset rather than professional psychological counseling or psychological treatment. For risky non-daily scenarios such as self-harm or suicide-related conversations, we do not claim that the dialogue system we built has a treatment or improvement effect on them. Additionally, we also ensure the anonymity of our interactive human evaluation. We believe our work meets ACL’s Code of Ethics.

## Acknowledgements

This work was supported by the National Science Foundation for Distinguished Young Scholars (with No. 62125604). This work was also supported by the Guoqiang Institute of Tsinghua University, with Grant No. 2020GQG0005. This work was also supported by Tsinghua Precision Medicine Foundation.

This work was also supported by the National Natural Science Foundation of China (with No. 62272340, 61876128, 62276187).

## References

Adnan Yousef Atoum and Rasha Ahmed Al-Shoboul. 2018. Emotional support and its relationship to emotional intelligence. Advances in social sciences research journal, 5(1).

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chaitanya Malaviya, Asli Celikyilmaz, and Yejin Choi. 2019. COMET: commonsense transformers for automatic knowledge graph construction. In Proceedings ofthe 57th Conference ofthe Associationfor Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4762–4779. Association for Computational Linguistics.

Brant R Burleson. 2003. Emotional support skills. In Handbook of communication and social interaction skills, pages 569–612. Routledge.

Yi Cheng, Wenge Liu, Wenjie Li, Jiashuo Wang, Ruihui Zhao, Bang Liu, Xiaodan Liang, and Yefeng Zheng. 2022. Improving multi-turn emotional support dialogue generation with lookahead strategy planning. CoRR, abs/2210.04242.

Kenneth Ward Church and Patrick Hanks. 1989. Word association norms, mutual information and lexicography. In 27th Annual Meeting ofthe Associationfor

Computational Linguistics, 26-29 June 1989, University of British Columbia, Vancouver, BC, Canada, Proceedings, pages 76–83. ACL.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics.

Jochen Hartmann. 2022. Emotion english distilrobertabase. https://huggingface.co/j-hartmann/ emotion-english-distilroberta-base/.

Takayuki Hasegawa, Nobuhiro Kaji, Naoki Yoshinaga, and Masashi Toyoda. 2013. Predicting and eliciting addressee’s emotion in online dialogue. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics, ACL 2013, 4-9 August 2013, Sofia, Bulgaria, Volume 1: Long Papers, pages 964–972. The Association for Computer Linguistics.

Catherine A Heaney and Barbara A Israel. 2008. Social networks and social support. Health behavior and health education: Theory, research, and practice, 4:189–210.

Minlie Huang, Xiaoyan Zhu, and Jianfeng Gao. 2020. Challenges in building intelligent open-domain dialog systems. ACM Trans. Inf. Syst., 38(3):21:1– 21:32.

Hao Jiang, Yutao Zhu, Xinyu Zhang, Zhicheng Dou, Pan Du, Te Pi, and Yantao Jia. 2021. Emotion eliciting machine: Emotion eliciting conversation generation based on dual generator. CoRR, abs/2105.08251.

Sevgi Co¸skun Keskin. 2014. From what isn’t empathy to empathic learning process. Procedia-Social and Behavioral Sciences, 116:4932–4938.

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 7871–7880. Association for Computational Linguistics.

Qintong Li, Hongshen Chen, Zhaochun Ren, Pengjie Ren, Zhaopeng Tu, and Zhumin Chen. 2020. Empdg: Multi-resolution interactive empathetic dialogue generation. In Proceedings of the 28th International

Conference on Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020, pages 4454–4466. International Committee on Computational Linguistics.

Qintong Li, Piji Li, Zhaochun Ren, Pengjie Ren, and Zhumin Chen. 2022. Knowledge bridging for empathetic dialogue generation. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pages 10993–11001. AAAI Press.

Zhaojiang Lin, Andrea Madotto, Jamin Shin, Peng Xu, and Pascale Fung. 2019. Moel: Mixture of empathetic listeners. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 121–132. Association for Computational Linguistics.

Siyang Liu, Chujie Zheng, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang, and Minlie Huang. 2021. Towards emotional support dialog systems. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 3469–3483. Association for Computational Linguistics.

Nurul Lubis, Sakriani Sakti, Koichiro Yoshino, and Satoshi Nakamura. 2018. Eliciting positive emotion through affect-sensitive dialogue response generation: A neural network approach. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications ofArtificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 5293–5300. AAAI Press.

Nurul Lubis, Sakriani Sakti, Koichiro Yoshino, and Satoshi Nakamura. 2019a. Dialogue model and response generation for emotion improvement elicitation. In Proc. 33rd Conf. Neural Inf. Process. Syst.(NIPS), pages 1–11.

Nurul Lubis, Sakriani Sakti, Koichiro Yoshino, and Satoshi Nakamura. 2019b. Positive emotion elicitation in chat-based dialogue systems. IEEE ACM Trans. Audio Speech Lang. Process., 27(4):866–877.

Navonil Majumder, Pengfei Hong, Shanshan Peng, Jiankun Lu, Deepanway Ghosal, Alexander F. Gelbukh, Rada Mihalcea, and Soujanya Poria. 2020. MIME: mimicking emotions for empathetic response generation. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020,

pages 8968–8979. Association for Computational Linguistics.

Brian L Mishara, François Chagnon, Marc Daigle, Bogdan Balan, Sylvaine Raymond, Isabelle Marcoux, Cécile Bardon, Julie K Campbell, and Alan Berman. 2007. Which helper behaviors and intervention styles are related to better short-term outcomes in telephone crisis intervention? results from a silent monitoring study of calls to the us 1-800-suicide network. Suicide and Life-Threatening Behavior, 37(3):308–321.

Saif M. Mohammad. 2018. Obtaining reliable human ratings of valence, arousal, and dominance for 20, 000 english words. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pages 174–184. Association for Computational Linguistics.

Wei Peng, Yue Hu, Luxi Xing, Yuqiang Xie, Yajing Sun, and Yunpeng Li. 2022. Control globally, understand locally: A global-to-local hierarchical graph network for emotional support conversation. In Proceedings ofthe Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 4324–4330. ijcai.org.

Stephen A Rains, Corey A Pavlich, Bethany Lutovsky, Eric Tsetsi, and Anjali Ashtaputre. 2020. Support seeker expectations, support message quality, and supportive interaction processes and outcomes: The case of the comforting computer program revisited. Journal ofSocial and Personal Relationships, 37(2):647–666.

Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. 2019. Towards empathetic opendomain conversation models: A new benchmark and dataset. In Proceedings of the 57th Conference of the Associationfor Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 5370–5381. Association for Computational Linguistics.

Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Eric Michael Smith, Y-Lan Boureau, and Jason Weston. 2021. Recipes for building an open-domain chatbot. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, pages 300–325. Association for Computational Linguistics.

Sahand Sabour, Chujie Zheng, and Minlie Huang. 2022. CEM: commonsense-aware empathetic response generation. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pages 11229–11237. AAAI Press.

Tomohide Shibata, Yusuke Egashira, and Sadao Kurohashi. 2014. Chat-like conversational system based on selection of reply generating module with reinforcement learning. In Situated Dialog in Speech-Based Human-Computer Interaction, 5th International Workshop on Spoken Dialogue Systems, IWSDS 2014, Napa, CA, USA, January 18-20, 2014, Signals and Communication Technology, pages 63– 69. Springer.

Richard S Sutton and Andrew G Barto. 2018. Reinforce ment learning: An introduction. MIT press.

Jianheng Tang, Tiancheng Zhao, Chenyan Xiong, Xiaodan Liang, Eric P. Xing, and Zhiting Hu. 2019. Target-guided open-domain conversation. In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 5624–5634. Association for Computational Linguistics.

Quan Tu, Yanran Li, Jianwei Cui, Bin Wang, Ji-Rong Wen, and Rui Yan. 2022. MISC: A mixed strategyaware model integrating COMET for emotional support conversation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 308–319. Association for Computational Linguistics.

Shihang Wang, Xinchao Xu, Wenquan Wu, Zheng-Yu Niu, Hua Wu, and Haifeng Wang. 2022. Towards multi-turn empathetic dialogs with positive emotion elicitation. CoRR, abs/2204.10509.

David Westbrook, Helen Kennerley, and Joan Kirk. 2011. An introduction to cognitive behaviour therapy: Skills and applications. Sage.

Koichiro Yoshino and Tatsuya Kawahara. 2015. Conversational system for information navigation based on POMDP with user focus tracking. Comput. Speech Lang., 34(1):275–291.

Chujie Zheng, Yong Liu, Wei Chen, Yongcai Leng, and Minlie Huang. 2021. Comae: A multi-factor hierarchical framework for empathetic response generation. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pages 813–824. Association for Computational Linguistics.

Jinfeng Zhou, Chujie Zheng, Bo Wang, Zheng Zhang, and Minlie Huang. 2022. CASE: aligning coarse-tofine cognition and affection for empathetic response generation. CoRR, abs/2208.08845.

![](images/f398fe9d631f904f55af510a25c9a557d8d6bf3e66154c3cf13f3842b981e32d.jpg)  
Figure 5: A construction example of the bidirectional emotion keyword graph . The valence (i.e., the number in the figure) is used to identify positive and negative keywords.

## A Bidirectional Emotion Keyword Graph

A construction example of the bidirectional emotion keyword graph is in Figure 5.

One-hop Reasoning on Graph  For the contextual keyword “close”, its one-hop neighbor reasoned by the “forward-positive” relation is “understand”, and the one reasoned by the “forwardnegative” relation is “frustrated”. Further, the one-hop neighbors reasoned by the “forward” relation are the union of the one-hop neighbors of the above two relations, i.e., “understand” and “frustrated”. For the keyword “frustrated” of the response, it cannot reason the one-hop neighbor using the “backward-positive” relation. Therefore, its one-hop neighbors reasoned by the “backward” relation are the same as the one-hop neighbors reasoned by the “backward-negative” relation, i.e., “close”, “warning”, and “pandemic”.

Multi-hop Reasoning on Graph Taking the “forward forward backward-positive” multihop reasoning as an example, using the “forward” relationship for the contextual keywords to perform one-hop reasoning can obtain the set of neighbors that contain the keywords of the response, which we regard as the extended keyword set of the response determined by the context. Using the keywords in this set as a starting point to perform the second-hop reasoning by the “forward” relation can result in the expanded keyword set of the user’s future utterance (i.e., the user’s next turn utterance) determined by the response. Further, similarly, the third-hop reasoning is performed using the “backward-positive” relation to determine the extended positive keywords set of the response coherent to the future utterance.

## B Case Study

In Table 5, two cases in multi-turn conversation generated by five models are selected. We found that the advantage of SUPPORTER is that it is more likely to elicit users’ emotions to transit positively by adaptively adjusting the intensity of empathy and elicitation. For example, low turns have a tone of strong empathy (e.g., “that is so awful”, “you are struggling financially and that you have not been able to afford your kids presents”). As the conversation deepens, the elicitation intensity of positive emotion progressively increases (e.g., “do you know anyone who has gone through similar situations?”, “it is okay to be proud ofyourselffor being able to make them”), and finally the expression with strong elicitation will help users jump out of negative situations (e.g., “I am sure you will be successful in the new job market”, “you can really enjoy seeing them happy”). During these multi-turn conversations, SUPPORTER effectively provides ES with diverse expressions while maintaining the dialogue coherence, being consistent with the observations from automatic and interactive human evaluations.

<table><tr><td rowspan="2">Situation#1</td><td>I have been laid out for not performing. But the actual situation is the customer was satisfied my service. But the expectation of the management was not to do that.</td></tr><tr><td>User: Hi!</td></tr><tr><td>Context</td><td>Bot: Hi can I help you today? I am here to listen. Do you have anything happen recently that has you feeling badly?</td></tr><tr><td>(1st-2nd turn)</td><td>User: I am laid off.</td></tr><tr><td></td><td>I am sorry to hear that. What is going on with your thoughts?</td></tr><tr><td>MoEL</td><td></td></tr><tr><td>MIME</td><td>I am sorry to hear that. I am sorry to hear that. I am so sorry to hear that.</td></tr><tr><td>BlenderBot-Joint</td><td>I am sorry to hear that. Is there something you would like to talk about?</td></tr><tr><td>Bart-Joint SUPPORTER</td><td>I have been laid off before. I was laid off too. Oh no, that is so awful. I am so sorry to hear that. I know that feeling well, especially when you have this situation.</td></tr><tr><td>Ground-Truth</td><td>Oh I am really sorry to hear that, did you have the same job for a long time? That sounds very difficult to deal with. User: I was attending a customer. He was having a grievance which I sorted out. But the management did not like that. For the past</td></tr><tr><td>Context  $/ \ast \_ \ast /$ </td><td>10 years I have been with this job.</td></tr><tr><td>(3rd_5th turn)</td><td></td></tr><tr><td></td><td>Bot: That is really unfair and hard to deal with are you close to any family? User: I am not in any close to any family related to job.</td></tr><tr><td>MoEL</td><td>I think that it is great you have a great job!</td></tr><tr><td>MIME</td><td>It sounds like you are trying to get your job that you are not alone.</td></tr><tr><td>BlenderBot-Joint Bart-Joint</td><td>I can tell you are really feeling badly about this.</td></tr><tr><td>SUPPORTER</td><td>I know that is hard to deal with. I am glad that you are reaching out to me to help out!</td></tr><tr><td>Ground-Truth</td><td>That is a really difficult position to be in. do you know anyone who has gone through similar situations?</td></tr><tr><td></td><td>Do you have any close friends to talk to about any new job prospects?</td></tr><tr><td>Context</td><td>User: I have few friends, I have been talking to them. They also tell I was not any wrong. For the experience and qualification, I will be getting a better job.</td></tr><tr><td>(6th_9th turn)</td><td>/*.…*/ Bot: Oh, wow then you do have a lot of contacts and some support or at least understanding. Music is a great way to relax and that</td></tr><tr><td></td><td>is very positive in your life and current situation.</td></tr><tr><td>MoEL</td><td>User: I am so confident I will be having a better job within the next few months.</td></tr><tr><td>MIME</td><td>I have found that online too. I do not know if I have heard of my work. I found out online that I could talk to you. I would suggest looking for a new job</td></tr><tr><td>BlenderBot-Joint</td><td>I am glad to hear that! I bet you could try to get some breathing techniques.</td></tr><tr><td>reflect on your progress.</td><td>It is very difficult to think about it right now but I would definitely encourage you to take some deep breaths and take some time to</td></tr><tr><td>Bart-Joint</td><td>I am so happy that you are feeling better and have found some new ways to relax.</td></tr><tr><td>SUPPORTER</td><td>I am glad that I could be of some help to you. It sounds like you are a very strong and intelligent person. I am sure you will be</td></tr><tr><td>Ground-Truth</td><td>successful in the new job market.</td></tr><tr><td>Situation#2</td><td>That is a great outlook. Thank you for sharing today</td></tr><tr><td>User: Hi!</td><td>I don&#x27;t have enough money to buy my kids Christmas gifts.</td></tr><tr><td>Context</td><td></td></tr><tr><td>(1st-3rd turn)</td><td>Bot: How are things going? User: Not good. I am depressed.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Bot: Is there anything in particular that has been bothering you recently?</td></tr><tr><td>MoEL</td><td>User: I do not have enough money to buy my 5 kids X-mas gifts.</td></tr><tr><td>MIME I understand how vou are feeling. I know how hard it must be.</td><td>I am sorry to hear that. I am sorry vou are having this right now.</td></tr><tr><td>BlenderBot-Joint</td><td></td></tr><tr><td>Bart-Joint</td><td>So you are feeling depressed because of covid and not enough money to buy gifts?</td></tr><tr><td>SUPPORTER</td><td>I am sorry to hear that. I have had to give up my job to help my kids during the holidays. It seems to me that you are struggling financially and that you have not been able to afford your kids presents.</td></tr><tr><td>Ground-Truth</td><td>I am really sorry to hear that, but I am sure your love for your kids can still shine through.</td></tr><tr><td>Context</td><td>User: I do love my kids, but they are expecting some toys and stuff from Santa.</td></tr><tr><td>(4th_5th turn)</td><td>Bot: Would it be possible to look at some more home made stuff for this vear that will still be fun and meaningful?</td></tr><tr><td></td><td></td></tr><tr><td>MoEL</td><td>User: I can not craft. If I made them something it would be so bad.</td></tr><tr><td>MIME</td><td>I am so sorry to hear that. I am sorry you are having to go through this.</td></tr><tr><td>BlenderBot-Joint</td><td>I understand how you feel. I have been there too. I know how hard it is to the next year. I am sure there are some options out there. Have you considered doing crafts with your kids?</td></tr><tr><td></td><td>Have you considered getting some more craft supplies? Maybe something that you can use to make a list of things you would like</td></tr><tr><td>Bart-Joint</td><td>to get your kids for Christmas?</td></tr><tr><td>SUPPORTER</td><td>I know how hard it can be to make things. I know it is a terrible feeling to feel bad about yourself, but it is okay to be proud of</td></tr><tr><td>Ground-Truth</td><td>yourself for being able to make them. I know it can be really hard to feel like you can not make anything worthwhile, but I am sure there&#x27;s something you could come up</td></tr><tr><td></td><td>with that could still put a smile on your kids’ faces. User: Do you have any craft ideas?</td></tr><tr><td>Context (6th_7th turn)</td><td>Bot: Maybe you could look at involving your kids in the process. The feeling of doing something together as a family can make it</td></tr><tr><td></td><td>way more fun. It could be a shared experience your family can look back on fondly later on. User: That is a good idea. I would like to have it done before Christmas. How about some wooden toys or a cardboard box fort?</td></tr><tr><td>MoEL MIME</td><td>I am so glad to hear that. I hope that you can do something fun!</td></tr><tr><td></td><td>I think vou can do it! I have been through this, and vou are not alone? That sounds like a great idea! Maybe even something for them to look forward to after Christmas. I know I love the idea of making</td></tr><tr><td>BlenderBot-Joint</td><td>toys and toys!</td></tr><tr><td>Bart-Joint</td><td>Yes, I think you should try to get it done before Christmas, but maybe even after that. You can see how hard it is to find a good time to</td></tr><tr><td>do something. SUPPORTER</td><td>I think that would be a great idea! That way you can feel like your kids are being cared for and you can really enjoy seeing</td></tr></table>

Table 5: Cases generated from baselines and SUPPORTER. $/ ! \cdots !$ indicates that some turns of dialogue are omitted.

## ACL 2023 Responsible NLP Checklist

A For every submission: <sup>✓</sup> A1. Did you describe the limitations of your work? Sec. Limitations

<sup>✓</sup> A2. Did you discuss any potential risks of your work? Sec. Ethical Considerations

<sup>✓</sup> A3. Do the abstract and introduction summarize the paper’s main claims? Sec. Abstract and Sec. 1

✗ A4. Have you used AI writing assistants when working on this paper? Left blank.

## B <sup>✓</sup> Did you use or create scientific artifacts?

Left blank.

<sup>✓</sup> B1. Did you cite the creators of artifacts you used? Sec. 4, Sec. 5

 B2. Did you discuss the license or terms for use and / or distribution of any artifacts? Not applicable. Left blank.

<sup>✓</sup> B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? For the artifacts you create, do you specify intended use and whether that is compatible with the original access conditions (in particular, derivatives of data accessed for research purposes should not be used outside of research contexts)? Sec. 4, Sec. 5

<sup>✓</sup> B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it? Sec. Ethical Considerations

<sup>✓</sup> B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? Sec. 4, Sec. 5

<sup>✓</sup> B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? Even for commonly-used benchmark datasets, include the number of examples in train / validation / test splits, as these provide necessary context for a reader to understand experimental results. For example, small differences in accuracy on large test sets may be significant, while on small test sets they may not be. Sec. 5

## C <sup>✓</sup> Did you run computational experiments?

Left blank.

<sup>✓</sup> C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? Sec. Appendix B

<sup>✓</sup> C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? Sec. Appendix B

<sup>✓</sup> C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run? Sec. 5

 C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation), did you report the implementation, model, and parameter settings used (e.g., NLTK, Spacy, ROUGE, etc.)? Not applicable. Left blank.

D <sup>✓</sup> Did you use human annotators (e.g., crowdworkers) or research with human participants? Left blank.

<sup>✓</sup> D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.? Sec. 5

✗ D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participants’ demographic (e.g., country of residence)? Limited by the space. Crowdsourcing workers arefrom Amazon Mechanical Turk.

 D3. Did you discuss whether and how consent was obtained from people whose data you’re using/curating? For example, if you collected data via crowdsourcing, did your instructions to crowdworkers explain how the data would be used? Not applicable. Left blank.

 D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? Not applicable. Left blank.

 D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? Not applicable. Left blank.