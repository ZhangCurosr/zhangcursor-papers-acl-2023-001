# Improving Continual Relation Extraction by Distinguishing Analogous Semantics

Wenzheng Zhao† Yuanning Cui† Wei Hu†<sup>,</sup> ‡<sup>,</sup> ∗

† State Key Laboratory for Novel Software Technology, Nanjing University, China ‡ National Institute of Healthcare Data Science, Nanjing University, China wzzhao.nju.cs@gmail.com, yncui.nju@gmail.com, whu@nju.edu.cn

## Abstract

Continual relation extraction (RE) aims to learn constantly emerging relations while avoiding forgetting the learned relations. Existing works store a small number of typical samples to re-train the model for alleviating forgetting. However, repeatedly replaying these samples may cause the overfitting problem. We conduct an empirical study on existing works and observe that their performance is severely affected by analogous relations. To address this issue, we propose a novel continual extraction model for analogous relations. Specifically, we design memory-insensitive relation prototypes and memory augmentation to overcome the overfitting problem. We also introduce integrated training and focal knowledge distillation to enhance the performance on analogous relations. Experimental results show the superiority of our model and demonstrate its effectiveness in distinguishing analogous relations and overcoming overfitting.

## 1 Introduction

Relation extraction (RE) aims to detect the relation between two given entities in texts. For instance, given a sentence “Remixes oftracksfrom Persona 5 were supervised by Kozuka and original composer Shoji Meguro” and an entity pair (Persona 5, Shoji Meguro), the “composer” relation is expected to be identified by an RE model. Conventional RE task assumes all relations are observed at once, ignoring the fact that new relations continually emerge in the real world. To deal with emerging relations, some existing works (Wang et al., 2019; Han et al., 2020; Wu et al., 2021; Cui et al., 2021; Zhao et al., 2022; Zhang et al., 2022; Hu et al., 2022; Wang et al., 2022) study continual RE. In continual RE, new relations and their involved samples continually emerge, and the goal is to classify all observed relations. Therefore, a continual RE model is expected to be able to learn new relations while retaining the performance on learned relations.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Max sim.</td><td colspan="2">FewRel</td><td colspan="2">TACRED</td></tr><tr><td>Accuracy</td><td>Drop</td><td>Accuracy</td><td>Drop</td></tr><tr><td rowspan="3">CRL</td><td>[0.85, 1.00)</td><td>71.1</td><td>9.7</td><td>64.8</td><td>11.4</td></tr><tr><td>[0.70, 0.85)</td><td>78.8</td><td>5.7</td><td>76.6</td><td>5.0</td></tr><tr><td>(0.00, 0.70)</td><td>87.9</td><td>3.2</td><td>89.6</td><td>0.6</td></tr><tr><td rowspan="3">CRECL</td><td>[0.85, 1.00)</td><td>60.4</td><td>18.9</td><td>60.7</td><td>13.9</td></tr><tr><td>[0.70, 0.85)</td><td>78.4</td><td>6.8</td><td>70.0</td><td>8.4</td></tr><tr><td>(0.00, 0.70)</td><td>83.0</td><td>5.1</td><td>79.9</td><td>4.3</td></tr></table>

Table 1: Results of our empirical study. We divide all relations into three groups according to their maximum similarity to other relations. “Accuracy” indicates the average accuracy (%) of relations after the model finishes learning. “Drop” indicates the average accuracy drop (%) from learning the relation for the first time to the learning process finished.

Existing works primarily focus on storing and replaying samples to avoid catastrophic forgetting (Lange et al., 2022) of the learned relations. On one hand, considering the limited storage and computational resources, it is impractical to store all training samples and re-train the whole model when new relations emerge. On the other hand, replaying a small number of samples every time new relations emerge would make the model prone to overfit the stored samples (Verwimp et al., 2021; Lange et al., 2022). Moreover, existing works simply attribute catastrophic forgetting to the decay of previous knowledge as new relations come but seldom delve deeper into the real causation. We conduct an empirical study and find that the severe decay of knowledge among analogous relations is a key factor of catastrophic forgetting.

Table 1 shows the accuracy and accuracy drop of two existing models on the FewRel (Han et al., 2018) and TACRED (Zhang et al., 2017) datasets. CRL (Zhao et al., 2022) and CRECL (Hu et al., 2022) are both state-of-the-art models for continual RE. All relations in the datasets are divided into three groups according to the maximum cosine similarity of their prototypes to other relation prototypes. A relation prototype is the overall representation of the relation. We can observe that the performance on relations with higher similarity is poorer, which is reflected in less accuracy and greater accuracy drop. Given that a relation pair with high similarity is often analogous to each other, the performance on a relation tends to suffer a significant decline, i.e., catastrophic forgetting, when its analogous relations appear. For example, the accuracy of the previously learned relation “location” drops from 0.98 to 0.6 after learning a new relation “country of origin”. Therefore, it is important to maintain knowledge among analogous relations for alleviating catastrophic forgetting. See Appendix A for more details of our empirical study.

To address the above issues, we propose a novel continual extraction model for analogous relations. Specifically, we introduce memory-insensitive relation prototypes and memory augmentation to reduce overfitting. The memory-insensitive relation prototypes are generated by combining static and dynamic representations, where the static representation is the average of all training samples after first learning a relation, and the dynamic representation is the average of stored samples. The memory augmentation replaces entities and concatenates sentences to generate more training samples for replay. Furthermore, we propose integrated training and focal knowledge distillation to alleviate knowledge forgetting of analogous relations. The integrated training combines the advantages of two widely-used training methods, which contribute to a more robust feature space and better distinguish analogous relations. One method uses contrastive learning for training and generates prototypes for relation classification, while the other trains a linear classifier. The focal knowledge distillation assigns high weights to analogous relations, making the model more focus on maintaining their knowledge.

Our main contributions are summarized below:

• We explicitly consider the overfitting problem in continual RE, which is often ignored by previous works. We propose memory-insensitive relation prototypes and memory augmentation to alleviate overfitting.

• We conduct an empirical study and find that analogous relations are hard to distinguish and their involved knowledge is more easily to be forgotten. We propose integrated training and focal knowledge distillation to better distinguish analogous relations.

• The experimental results on two benchmark datasets demonstrate that our model achieves state-of-the-art accuracy compared with existing works, and better distinguishes analogous relations and overcomes overfitting for continual RE. Our source code is available at https://github.com/nju-websoft/CEAR.

## 2 Related Work

Continual learning studies the problem of learning from a continuous stream of data (Lange et al., 2022). The main challenge of continual learning is avoiding catastrophic forgetting of learned knowledge while learning new tasks. Existing continual learning models can be divided into three categories: regularization-based, dynamic architecture, and memory-based. The regularization-based models (Li and Hoiem, 2016; Kirkpatrick et al., 2016) impose constraints on the update of parameters important to previous tasks. The dynamic architecture models (Mallya and Lazebnik, 2018; Qin et al., 2021) dynamically extend the model architecture to learn new tasks and prevent forgetting previous tasks. The memory-based models (Lopez-Paz and Ranzato, 2017; Rebuffi et al., 2017; Chaudhry et al., 2019) store a limited subset of samples in previous tasks and replay them when learning new tasks.

In continual RE, the memory-based models (Wang et al., 2019; Han et al., 2020; Wu et al., 2021; Cui et al., 2021; Zhao et al., 2022; Zhang et al., 2022; Hu et al., 2022) are the mainstream choice as they have shown better performance for continual RE than others. To alleviate catastrophic forgetting, previous works make full use of relation prototypes, contrastive learning, multi-head attention, knowledge distillation, etc. EA-EMR (Wang et al., 2019) introduces memory replay and the embedding aligned mechanism to mitigate the embedding distortion when training new tasks. CML (Wu et al., 2021) combines curriculum learning and meta-learning to tackle the order sensitivity in continual RE. RP-CRE (Cui et al., 2021) and KIP-Framework (Zhang et al., 2022) leverage relation prototypes to refine sample representations through multi-head attention-based memory networks. Additionally, KIP-Framework uses external knowledge to enhance the model through a knowledge-infused prompt to guide relation prototype generation. EMAR (Han et al., 2020), CRL (Zhao et al., 2022), and CRECL (Hu et al., 2022) leverage contrastive learning for model training. Besides, knowledge distillation is employed by CRL to maintain previously learned knowledge. ACA (Wang et al., 2022) is the only work that considers the knowledge forgetting of analogous relations ignored by the above works and proposes an adversarial class augmentation strategy to enhance other continual RE models. All these models do not explicitly consider the overfitting problem (Lange et al., 2022; Verwimp et al., 2021), which widely exists in the memory-based models. As far as we know, a few works (Wang et al., 2021) in other continual learning fields have tried to reduce the overfitting problem and achieve good results. We address both the problems of distinguishing analogous relations and overfitting to stored samples, and propose an end-to-end model.

![](images/a58e8196e244a93da0ec096710b03ea3af69ca3692ce126bf0d6612b32c06744.jpg)  
Figure 1: Framework of the proposed model for task $T _ { k }$

## 3 Task Definition

A continual RE task consists of a sequence of tasks $\mathcal { T } = \{ T _ { 1 } , T _ { 2 } , \dots , T _ { K } \}$ . Each individual task is a conventional RE task. Given a sentence, the RE task aims to find the relation between two entities in this sentence. The dataset and relation set of $T _ { k } \in$ $\tau$ are denoted by $D _ { k }$ and $R _ { k } .$ , respectively. $D _ { k }$ contains separated training, validation and test sets, denoted by $D _ { k } ^ { \mathrm { t r a i n } } , D _ { k } ^ { \mathrm { v a l i d } }$ and $D _ { k } ^ { \mathrm { t e s t } }$ , respectively. $R _ { k }$ contains at least one relation. The relation sets of different tasks are disjoint.

Continual RE aims to train a classification model that performs well on both current task $T _ { k }$ and previously accumulated tasks $\begin{array} { r } { \tilde { T } _ { k - 1 } = \bigcup _ { i = 1 } ^ { k - 1 } T _ { i } } \end{array}$ . In other words, a continual RE model is expected to be capable of identifying all seen relations $\tilde { R } _ { k }$ $\textstyle \bigcup _ { i = 1 } ^ { k } R _ { i }$ and would be evaluated on all the test sets of seen tasks $\begin{array} { r } { \tilde { D } _ { k } ^ { \mathrm { t e s t } } = \bigcup _ { i = 1 } ^ { k } D _ { i } ^ { \mathrm { t e s t } } } \end{array}$

## 4 Methodology

## 4.1 Overall Framework

The overall framework is shown in Figure 1. For a new task $T _ { k }$ , we first train the continual RE model on $D _ { k }$ to learn this new task. Then, we select and store a few typical samples for each relation $r \in R _ { k }$ . Next, we calculate the prototype p<sub>r</sub> of each relation $r \in \tilde { R } _ { k }$ according to the static and dynamic representations of samples. We also conduct memory augmentation to provide more training data for memory replay. Note that the augmented data are not used for prototype generation. Finally, we perform memory replay consisting of integrated training and focal knowledge distillation to alleviate catastrophic forgetting. The parameters are updated in the first and last steps. After learning $T _ { k }$ the model continually learns the next task $T _ { k + 1 }$

## 4.2 New Task Training

When the new task $T _ { k }$ emerges, we first train the model on $D _ { k } ^ { \mathrm { t r a i n } }$ . We follow the works (Cui et al., 2021; Zhao et al., 2022; Zhang et al., 2022; Hu et al., 2022) to use the pre-trained language model BERT (Devlin et al., 2019) as the encoder.

Given a sentence x as input, we first tokenize it and insert special tokens $[ E _ { 1 1 } ] / [ E _ { 1 2 } ]$ and $[ E _ { 2 1 } ] / [ E _ { 2 2 } ]$ to mark the start/end positions of head and tail entities, respectively. We use the hidden representations of $\left[ E _ { 1 1 } \right]$ and $[ E _ { 2 1 } ]$ as the representations of head and tail entities. The representation of x is defined as

$$
\mathbf { h } _ { x } = \mathrm { L a y e r N o r m } \big ( \mathbf { W } _ { 1 } [ \mathbf { h } _ { x } ^ { 1 1 } ; \mathbf { h } _ { x } ^ { 2 1 } ] + \mathbf { b } \big ) ,\tag{1}
$$

where $\mathbf { h } _ { x } ^ { 1 1 } , \mathbf { h } _ { x } ^ { 2 1 } \ \in \ \mathbb { R } ^ { d }$ are the hidden representations of head and tail entities, respectively. d is the dimension of the hidden layer in BERT. $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { d \times 2 d }$ and $\textbf { b } \in \mathbb { R } ^ { d }$ are two trainable parameters.

Then, we use a linear softmax classifier to calculate the classification probability of x according to the representation $\mathbf { h } _ { x }$ :

$$
P ( x ; \theta _ { k } ) = \mathrm { s o f t m a x } ( \mathbf { W } _ { 2 } \mathbf { h } _ { x } ) ,\tag{2}
$$

where $\theta _ { k }$ denotes the model when learning $T _ { k }$ $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { | \tilde { R } _ { k } | \times d }$ is the trainable parameter of the linear classifier.

Finally, the classification loss of new task training is calculated as follows:

$$
\mathcal { L } _ { \mathrm { { n e w } } } = - \frac { 1 } { | D _ { k } ^ { \mathrm { { t r a i n } } } | } \sum _ { \substack { \boldsymbol { x } _ { i } \in D _ { k } ^ { \mathrm { t r a i n } } } } \sum _ { r _ { j } \in R _ { k } } \delta _ { \boldsymbol { y } _ { i } , \boldsymbol { r } _ { j } } \log P ( r _ { j } | \boldsymbol { x } _ { i } ; \boldsymbol { \theta } _ { k } ) ,\tag{3}
$$

where $P ( r _ { j } | x _ { i } ; \theta _ { k } )$ is the probability of input $x _ { i }$ classified as relation $r _ { j }$ by the current model $\theta _ { k } . \ y _ { i }$ is the label of $x _ { i }$ such that if $y _ { i } = r _ { j } , \delta _ { y _ { i } , r _ { j } } = 1$ and 0 otherwise.

## 4.3 Memory Sample Selection

To preserve the learned knowledge from previous tasks, we select and store a few typical samples for memory replay. Inspired by the works (Han et al., 2020; Cui et al., 2021; Zhao et al., 2022; Zhang et al., 2022; Hu et al., 2022), we adopt the k-means algorithm to cluster the samples of each relation $r \in R _ { k }$ . The number of clusters is defined as the memory size $m$ . For each cluster, we select the sample whose representation is closest to the medoid and store it in the memory space $M ^ { r }$ . The accumulated memory space is $\begin{array} { r } { \tilde { M } _ { k } = \bigcup _ { r \in \tilde { R } _ { k } } M ^ { r } } \end{array}$

## 4.4 Memory-Insensitive Relation Prototype

A relation prototype is the overall representation of the relation. Several previous works (Han et al., 2020; Zhao et al., 2022; Hu et al., 2022) directly use relation prototypes for classification and simply calculate the prototype of r using the average of the representations of its typical samples. But, such a relation prototype is sensitive to the typical samples, which may cause the overfitting problem. To reduce the sensitivity to typical samples, Zhang et al. (2022) propose a knowledge-infused relation prototype generation, which employs a knowledge-infused prompt to guide prototype generation. However, it relies on external knowledge and thus brings additional computation overhead.

To alleviate the overfitting problem, we first calculate and store the average representation of all training samples after first learning a relation. This representation contains more comprehensive knowledge about the relation. However, as we cannot store all training samples, it is static and cannot be updated to adapt to the new feature space in the subsequent learning. In this paper, the dynamic representation of typical samples is used to finetune the static representation for adapting the new feature space. The memory-insensitive relation prototype of relation r is calculated as follows:

$$
\mathbf { p } _ { r } = \left( 1 - \beta \right) \mathbf { p } _ { r } ^ { \mathrm { s t a t i c } } + \frac { \beta } { \vert M ^ { r } \vert } \sum _ { x _ { i } \in M ^ { r } } \mathbf { h } _ { x _ { i } } ,\tag{4}
$$

where $\mathbf { p } _ { r } ^ { \mathrm { s t a t i c } }$ is the average representation of all training samples after learning relation $r$ for the first time, and $\beta$ is a hyperparameter.

## 4.5 Memory Augmentation

The memory-based models (Wang et al., 2019; Han et al., 2020; Cui et al., 2021; Zhao et al., 2022; Zhang et al., 2022; Hu et al., 2022) select and store a small number of typical samples and replay them in the subsequent learning. Due to the limited memory space, these samples may be replayed many times during continual learning, resulting in overfitting. To address this issue, we propose a memory augmentation strategy to provide more training samples for memory replay.

For a sample $\boldsymbol { x } _ { i } ^ { r }$ of relation r in $M ^ { r }$ , we randomly select another sample $x _ { j } ^ { r } \neq x _ { i } ^ { r }$ from $M ^ { r }$ Then, the head and tail entities of $\boldsymbol { x } _ { i } ^ { r }$ are replaced by the corresponding entities of $\boldsymbol { x } _ { j } ^ { r }$ and the new sample, denoted by $x _ { i j } ^ { r }$ , can be seen as an additional sample of relation r. Also, we use sentence concatenation to generate training samples. Specifically, we randomly select another two samples $x _ { m }$ and $x _ { n }$ from $\tilde { M } _ { k } \backslash M ^ { r }$ and append them to the end of $\boldsymbol { x } _ { i } ^ { r }$ and $x _ { i j } ^ { r }$ , respectively. Note that $x _ { m }$ and $x _ { n }$ are not the typical samples of relation r. Then, we obtain two new samples of relation r, denoted by $x _ { i - m } ^ { r }$ and $x _ { i j - n } ^ { r } .$ . The model is expected to still identify the relation r though there is an irrelevant sentence contained in the whole input. We conduct this augmentation strategy on all typical samples in $\tilde { M _ { k } }$ , but the augmented data are only used for training, not for prototype generation, as they are not accurate enough. Finally, the overall augmented memory space is $\hat { M _ { k } }$ , and $| \hat { M } _ { k } | = 4 | \tilde { M } _ { k } |$

## 4.6 Memory Replay

## 4.6.1 Integrated Training

There are two widely-used training methods for continual RE: Han et al. (2020); Zhao et al. (2022); Hu et al. (2022) use contrastive learning for training and make predictions via relation prototypes; Cui et al. (2021); Zhang et al. (2022) leverage the cross entropy loss to train the encoder and linear classifier. We call these two methods the contrastive method and the linear method, respectively.

The contrastive method contributes to a better feature space because it pulls the representations of samples from the same relation and pushes away those from different relations, which improves the alignment and uniformity (Wang and Isola, 2020). However, its prediction process is sensitive to the relation prototypes, especially those of analogous relations that are highly similar to each other. The linear classifier decouples the representation and classification processes, which ensures a more taskspecific decision boundary. We adopt both contrastive and linear methods to combine their merits:

$$
\mathcal { L } _ { \mathrm { c l s } } = \mathcal { L } _ { \mathrm { c \_ c l s } } + \mathcal { L } _ { \mathrm { l \_ c l s } } ,\tag{5}
$$

where $\mathcal { L } _ { \mathrm { c \_ c l s } }$ and $\mathcal { L } _ { \mathrm { l \_ c l s } }$ denote the losses of the contrastive and linear methods, respectively.

In the contrastive method, we first leverage twolayer MLP to reduce dimension:

$$
\mathbf { z } _ { x } = \mathrm { N o r m } \big ( \mathrm { M L P } ( \mathbf { h } _ { x } ) \big ) .\tag{6}
$$

Then, we use the InfoNCE loss (van den Oord et al., 2018) and the triplet loss (Schroff et al., 2015) in contrastive learning:

$$
\begin{array} { r } { \mathcal { L } _ { \mathtt { c } _ { - } \mathtt { c } \mathtt { l s } } = - \frac { 1 } { | \hat { M } _ { k } | } \displaystyle \sum _ { x _ { i } \in \hat { M } _ { k } } \log \frac { \exp ( \mathbf { z } _ { x _ { i } } \cdot \mathbf { z } _ { y _ { i } } / \tau _ { 1 } ) } { \sum _ { r \in \hat { R } _ { k } } \exp ( \mathbf { z } _ { x _ { i } } \cdot \mathbf { z } _ { r } / \tau _ { 1 } ) } } \\ { + \frac { \mu } { | \hat { M } _ { k } | } \displaystyle \sum _ { x _ { i } \in \hat { M } _ { k } } \operatorname* { m a x } ( \omega - \mathbf { z } _ { x _ { i } } \mathbf { z } _ { y _ { i } } + \mathbf { z } _ { x _ { i } } \mathbf { z } _ { y _ { i } ^ { \prime } } , 0 ) } \end{array} ;\tag{7}
$$

where $\mathbf { z } _ { r }$ is the low-dimensional prototype of relation r. $y _ { i } ^ { \prime } = { \arg \operatorname* { m a x } } _ { y _ { i } ^ { \prime } \in \tilde { R } _ { k } \backslash \{ y _ { i } \} } \mathbf { z } _ { x _ { i } } \cdot \mathbf { z } _ { y _ { i } ^ { \prime } }$ is the most similar negative relation label of sample $x _ { i } , \tau _ { 1 }$ is the temperature parameter. $\mu$ and ω are hyperparameters.

At last, the relation probability is computed through the similarity between the representations of test sample and relation prototypes:

$$
P _ { c } ( x _ { i } ; \theta _ { k } ) = \mathrm { s o f t m a x } ( { \bf z } _ { x _ { i } } \cdot { \bf Z } _ { \tilde { R } _ { k } } ) ,\tag{8}
$$

where $\mathbf { Z } _ { \tilde { R } _ { k } }$ denotes the matrix of prototypes of all seen relations.

In the linear method, a linear classifier obtains the relation probability similar to that in the new task training step. The loss function is

$$
\mathcal { L } _ { \mathrm { l _ { - } c l s } } = - \frac { 1 } { \vert \hat { M } _ { k } \vert } \sum _ { x _ { i } \in \hat { M } _ { k } } \sum _ { r _ { j } \in \tilde { R } _ { k } } \delta _ { y _ { i } , r _ { j } } \log P ( r _ { j } \mid x _ { i } ; \theta _ { k } ) .\tag{9}
$$

## 4.6.2 Focal Knowledge Distillation

During the continual training process, some emerging relations are similar to other learned relations and are difficult to distinguish. Inspired by the focal loss (Lin et al., 2020), we propose the focal knowledge distillation, which forces the model to focus more on analogous relations.

Specifically, we assign a unique weight for each sample-relation pair, according to the classification probability of the sample and the similarity between the representations of sample and relation prototype. Difficult samples and analogous sample-relation pairs are assigned high weights. The weight $w _ { i , j }$ for sample $x _ { i }$ and relation $r _ { j }$ is

$$
s _ { x _ { i } , r _ { j } } = \frac { \exp { \left( \sin ( \mathbf { h } _ { x _ { i } } , \mathbf { p } _ { r _ { j } } ) / \tau _ { 2 } \right) } } { \sum _ { r _ { m } \in \tilde { R } _ { k - 1 } } \exp { \left( \sin ( \mathbf { h } _ { x _ { i } } , \mathbf { p } _ { r _ { m } } ) / \tau _ { 2 } \right) } } ,\tag{10}
$$

$$
w _ { x _ { i } , r _ { j } } = s _ { x _ { i } , r _ { j } } \big ( 1 - P ( y _ { i } | x _ { i } ; \theta _ { k } ) \big ) ^ { \gamma } ,\tag{11}
$$

where $\mathbf { p } _ { r _ { j } }$ is the prototype of relation $r _ { j }$ . sim $( \cdot )$ is the similarity function, e.g., cosine. $\tau _ { 2 }$ is the temperature parameter and γ is a hyperparameter.

With $w _ { x _ { i } , r _ { j } }$ , the focal knowledge distillation loss is calculated as follows:

$$
a _ { x _ { i } , r _ { j } } = w _ { x _ { i } , r _ { j } } P ( r _ { j } | x _ { i } ; \theta _ { k - 1 } ) ,\tag{12}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f k d } } = - \frac { 1 } { \vert \hat { M } _ { k } \vert } \displaystyle \sum _ { x _ { i } \in \hat { M } _ { k } } \displaystyle \sum _ { r _ { j } \in \tilde { R } _ { k - 1 } } a _ { x _ { i } , r _ { j } } \log P ( r _ { j } \mid x _ { i } ; \theta _ { k } ) , } \end{array}\tag{13}
$$

where $P ( r _ { j } | x _ { i } ; \theta _ { k - 1 } )$ denotes the probability of sample $x _ { i }$ predicted to relation $r _ { j }$ by the previous model $\theta _ { k - 1 }$

The focal knowledge distillation loss is combined with the training losses of contrastive and linear methods. The overall loss is defined as

$$
\mathcal { L } _ { \mathrm { r e p l a y } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { c \_ f k d } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { l \_ f k d } } ,\tag{14}
$$

where $\mathcal { L } _ { \mathrm { c } \_ \mathrm { f k d } }$ and $\mathcal { L } _ { \mathrm { l \_ f k d } }$ are the focal knowledge distillation losses of contrastive and linear methods, respectively. $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are hyperparameters.

## 4.7 Relation Prediction

After learning task $T _ { k }$ , the contrastive and linear methods are combined to predict the relation label of the given test sample $x _ { i } ^ { * }$ :

$$
y _ { i } ^ { * } = \arg \operatorname* { m a x } _ { y _ { i } ^ { * } \in \tilde { R } _ { k } } \left( ( 1 - \alpha ) P _ { c } ( x _ { i } ^ { * } ; \theta _ { k } ) + \alpha P _ { l } ( x _ { i } ^ { * } ; \theta _ { k } ) \right) ,\tag{15}
$$

where $P _ { c } ( x _ { i } ^ { * } ; \theta _ { k } )$ and $P _ { l } ( x _ { i } ^ { * } ; \theta _ { k } )$ are the probabilities calculated by the contrastive and linear methods, respectively. α is a hyperparameter.

## 5 Experiments and Results

In this section, we report the experimental results of our model. The source code is accessible online.

## 5.1 Datasets

We conduct our experiments on two widely-used benchmark datasets:

• FewRel (Han et al., 2018) is a popular RE dataset originally built for few-shot learning. It contains 100 relations and 70,000 samples in total. To be in accord with previous works (Cui et al., 2021; Zhao et al., 2022), we use 80 relations each with 700 samples (i.e., in the training and validation sets), and split them into 10 subsets to simulate 10 disjoint tasks.

• TACRED (Zhang et al., 2017) is a large-scale RE dataset having 42 relations and 106,264 samples. Following the experiment setting of previous works, we remove “no\_relation” and divide other relations into 10 tasks.

## 5.2 Experiment Setting and Baseline Models

RP-CRE (Cui et al., 2021) proposes a completelyrandom strategy to split all relations into 10 subsets corresponding to 10 tasks, and accuracy on all observed relations is chosen as the evaluation metric, which is defined as the proportion of correctly predicted samples in the whole test set. This setting is widely followed by existing works (Zhao et al., 2022; Zhang et al., 2022; Hu et al., 2022). For a fair comparison, we employ the same setting and obtain the divided data from the open-source code of RP-CRE to guarantee exactly the same task sequence. Again, following existing works, we carry out the main experiment with a memory size of 10 and report the average result of five different task sequences. See Appendix B for the details of the hyperparameter setting.

For comparison, we consider the following baseline models: EA-EMR (Wang et al., 2019), EMAR (Han et al., 2020), CML (Wu et al., 2021), RP-CRE (Cui et al., 2021), CRL (Zhao et al., 2022), CRECL (Hu et al., 2022) and KIP-Framework (Zhang et al., 2022). See Section 2 for their details.

## 5.3 Results and Analyses

## 5.3.1 Main Results

Table 2 shows the results of all compared baselines in the main experiment. The results of EA-EMR, EMAR, CML, and RP-CRE are obtained from the RP-CRE’s original paper, and the results of other baselines are directly cited from their original papers. We additionally report the standard deviations of our model. Based on the results, the following observations can be drawn:

Our proposed model achieves an overall state-ofthe-art performance on the two different datasets for the reason that our model can reduce overfitting to typical samples and better maintain knowledge among analogous relations. Thus, we can conclude that our model effectively alleviates catastrophic forgetting in continual RE.

As new tasks continually emerge, the performance of all compared models declines, which indicates that catastrophic forgetting is still a major challenge to continual RE. EA-EMR and CML do not use BERT as the encoder, so they suffer the most performance decay. This demonstrates that BERT has strong stability for continual RE.

All models perform relatively poorer on TA-CRED and the standard deviations of our model on TACRED are also higher than those on FewRel. The primary reason is that TACRED is classimbalanced and contains fewer training samples for each relation. Therefore, it is more difficult and leads to greater randomness in the task division.

<table><tr><td>FewRel</td><td> $T _ { 1 }$ </td><td> $T _ { 2 }$ </td><td> $T _ { 3 }$ </td><td> $T _ { 4 }$ </td><td> $T _ { 5 }$ </td><td> $T _ { 6 }$ </td><td> $T _ { 7 }$ </td><td> $T _ { 8 }$ </td><td> $T _ { 9 }$ </td><td> $T _ { 1 0 }$ </td></tr><tr><td>EA-EMR</td><td>89.0</td><td>69.0</td><td>59.1</td><td>54.2</td><td>47.8</td><td>46.1</td><td>43.1</td><td>40.7</td><td>38.6</td><td>35.2</td></tr><tr><td>EMAR (BERT)</td><td>98.8</td><td>89.1</td><td>89.5</td><td>85.7</td><td>83.6</td><td>84.8</td><td>79.3</td><td>80.0</td><td>77.1</td><td>73.8</td></tr><tr><td>CML</td><td>91.2</td><td>74.8</td><td>68.2</td><td>58.2</td><td>53.7</td><td>50.4</td><td>47.8</td><td>44.4</td><td>43.1</td><td>39.7</td></tr><tr><td>RP-CRE</td><td>97.9</td><td>92.7</td><td>91.6</td><td>89.2</td><td>88.4</td><td>86.8</td><td>85.1</td><td>84.1</td><td>82.2</td><td>81.5</td></tr><tr><td>CRL</td><td>98.2</td><td>94.6</td><td>92.5</td><td>90.5</td><td>89.4</td><td>87.9</td><td>86.9</td><td>85.6</td><td>84.5</td><td>83.1</td></tr><tr><td>CRECL</td><td>97.8</td><td>94.9</td><td>92.7</td><td>90.9</td><td>89.4</td><td>87.5</td><td>85.7</td><td>84.6</td><td>83.6</td><td>82.7</td></tr><tr><td>KIP-Framework</td><td>98.4</td><td>93.5</td><td>92.0</td><td>91.2</td><td>90.0</td><td>88.2</td><td>86.9</td><td>85.6</td><td>84.1</td><td>82.5</td></tr><tr><td>Ours</td><td> $9 8 . 1 \pm 0 . 6 $ </td><td> $\mathbf { 9 5 . 8 _ { \pm 1 . 7 } }$ </td><td> ${ \bf 9 3 . 6 _ { \pm 2 . 1 } }$ </td><td> $\mathbf { 9 1 . 9 } _ { \pm 2 . 0 }$ </td><td> ${ \bf 9 1 . 1 _ { \pm 1 . 5 } }$ </td><td> ${ \bf 8 9 . 4 } _ { \pm 2 . 0 }$ </td><td> ${ \bf 8 8 . 1 _ { \pm 0 . 7 } }$ </td><td> ${ \mathbf 8 6 . 9 } _ { \pm 1 . 3 }$ </td><td> ${ \bf 8 5 . 6 _ { \pm 0 . 8 } }$ </td><td> $\mathbf { 8 4 . 2 \bot 0 . 4 }$ </td></tr><tr><td>TACRED</td><td>1 T1</td><td> $T _ { 2 }$ </td><td> $T _ { 3 }$ </td><td> $T _ { 4 }$ </td><td> $T _ { 5 }$ </td><td> $T _ { 6 }$ </td><td> $T _ { 7 }$ </td><td> $T _ { 8 }$ </td><td> $T _ { 9 }$ </td><td> $T _ { 1 0 }$ </td></tr><tr><td>EA-EMR</td><td>47.5</td><td>40.1</td><td>38.3</td><td>29.9</td><td>28.4</td><td>27.3</td><td>26.9</td><td>25.8</td><td>22.9</td><td>19.8</td></tr><tr><td>EMAR (BERT)</td><td>96.6</td><td>85.7</td><td>81.0</td><td>78.6</td><td>73.9</td><td>72.3</td><td>71.7</td><td>72.2</td><td>72.6</td><td>71.0</td></tr><tr><td>CML</td><td>57.2</td><td>51.4</td><td>41.3</td><td>39.3</td><td>35.9</td><td>28.9</td><td>27.3</td><td>26.9</td><td>24.8</td><td>23.4</td></tr><tr><td>RP-CRE</td><td>97.6</td><td>90.6</td><td>86.1</td><td>82.4</td><td>79.8</td><td>77.2</td><td>75.1</td><td>73.7</td><td>72.4</td><td>72.4</td></tr><tr><td>CRL</td><td>97.7</td><td>93.2</td><td>89.8</td><td>84.7</td><td>84.1</td><td>81.3</td><td>80.2</td><td>79.1</td><td>79.0</td><td>78.0</td></tr><tr><td>CRECL KIP-Framework</td><td>96.6</td><td>93.1</td><td>89.7</td><td>87.8</td><td>85.6</td><td>84.3</td><td>83.6</td><td>81.4</td><td>79.3</td><td>78.5</td></tr><tr><td></td><td>98.3</td><td>95.0</td><td>90.8</td><td>87.5</td><td>85.3</td><td>84.3</td><td>82.1</td><td>80.2</td><td>79.6</td><td>78.6</td></tr><tr><td>Ours</td><td> $\underline { { 9 7 . 7 } } \pm 1 . 6$ </td><td> $\underline { { 9 4 . 3 } } \pm 2 . 9$ </td><td> $\mathbf { 9 2 . 3 _ { \pm 3 . 3 } }$ </td><td> ${ \bf 8 8 . 4 _ { \pm 3 . 7 } }$ </td><td> $\mathbf { 8 6 . 6 _ { \pm 3 . 0 } }$ </td><td>84.5 ±2.1</td><td> $\underline { { 8 2 . 2 } } \pm 2 . 8$ </td><td> $8 1 . 1 _ { \pm 1 . 6 }$ </td><td> ${ \bf 8 0 . 1 _ { \pm 0 . 7 } }$ </td><td> ${ \bf 7 9 . 1 _ { \pm 1 . 1 } }$ </td></tr></table>

Table 2: Accuracy (%) on all observed relations after learning each task. The best results are marked in bold, and the second-best ones are marked with underlines. “ ” indicates the model using external knowledge.

## 5.3.2 Ablation Study

We conduct an ablation study to validate the effectiveness of individual modules in our model. Specifically, for “w/o FKD”, we remove the focal knowledge distillation loss in memory replay; for “w/o LM” or “w/o CM”, the model is only trained and evaluated with the contrastive or linear method; for “w/o MA”, we only train the model with original typical samples in memory replay; and for “w/o DP” or “w/o SP”, we directly generate relation prototypes based on the average of static or dynamic representations.

The results are shown in Table 3. It is observed that our model has a performance decline without each component, which demonstrates that all modules are necessary. Furthermore, the proposed modules obtain greater improvement on the TACRED dataset. The reason is that TACRED is more difficult than FewRel, so the proposed modules are more effective in difficult cases.

## 5.3.3 Influence of Memory Size

Memory size is defined as the number of stored typical samples for each relation. For the memorybased models in continual RE, their performance is highly influenced by memory size. We conduct an experiment with different memory sizes to compare our model with CRL and CRECL for demonstrating that our model is less sensitive to memory size. We re-run the source code of CRL and CRECL with different memory sizes and show the results in Figure 2. Note that we do not compare with KIP-

<table><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1> $T _ { 6 }$      $T _ { 7 }$      $T _ { 8 }$      $T _ { 9 }$      $T _ { 1 0 }$ </td></tr><tr><td rowspan=7 colspan=1>Fel</td><td rowspan=2 colspan=3>Intact Modelw/o FKD</td><td rowspan=1 colspan=1>89.4 88.1  86.9  85.6 84.2</td></tr><tr><td rowspan=1 colspan=1>89.3  88.0 86.8  85.5  84.0</td></tr><tr><td rowspan=1 colspan=2>w/o</td><td rowspan=1 colspan=2>LM</td><td rowspan=1 colspan=1>89.0 87.5  86.5  85.1  83.6</td></tr><tr><td rowspan=4 colspan=3>w/o CMw/o MAw/o DPw/o SP</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>89.3 87.5  86.8  85.6 84.0</td></tr><tr><td rowspan=1 colspan=1>88.4 87.4 86.4 85.4 83.7</td></tr><tr><td rowspan=1 colspan=1>89.2 87.9 86.6 85.3  83.8</td></tr><tr><td rowspan=1 colspan=1>89.3 87.8  86.6 85.2  83.5</td></tr><tr><td rowspan=7 colspan=1>TAED</td><td rowspan=1 colspan=3>Intact Model</td><td rowspan=1 colspan=1>84.5 82.2 81.1  80.1  79.1</td></tr><tr><td rowspan=1 colspan=3>w/o FKD</td><td rowspan=1 colspan=1>83.4 81.3  79.5  79.2  78.2</td></tr><tr><td rowspan=1 colspan=3>w/o LM</td><td rowspan=1 colspan=1>83.7 81.2  79.6 79.4 78.2</td></tr><tr><td rowspan=1 colspan=3>w/o CM</td><td rowspan=1 colspan=1>84.0 81.9  80.1  79.2 78.0</td></tr><tr><td rowspan=1 colspan=3>w/o MA</td><td rowspan=1 colspan=1>82.9 81.2 79.3 79.0 77.9</td></tr><tr><td rowspan=1 colspan=3>w/o DP</td><td rowspan=1 colspan=1>83.2  80.8 79.1 79.1  78.3</td></tr><tr><td rowspan=1 colspan=3>w/o SP</td><td rowspan=1 colspan=1>83.5 81.1 79.6 79.3  78.2</td></tr></table>

Table 3: Ablation study results. We remove focal knowledge distillation (FKD), linear method (LM), contrastive method (CM), memory augmentation (MA), dynamic prototypes (DP), and static prototypes (SP) in order and report the accuracy (%) on all observed relations.

Framework because it uses external knowledge to enhance performance, which is beyond our scope.

In most cases, our model achieves state-ofthe-art performance with different memory sizes, which demonstrates the strong generalization of our model. However, our model does not obtain the best performance on TACRED with memory size 15 because the overfitting problem that we consider is not serious in this case. In fact, as the memory size becomes smaller, the overfitting problem is getting worse, and analogous relations are more difficult to distinguish due to the limited training data samples. From Figures 2(a), (b), (e),

(g) Memory size 15

![](images/85f580663a9a18afdc43dd72e40490cfaf3b91015e6b4ea01f73ce16a02a322e.jpg)  
(a) Memory size 2

![](images/919428038e550f87818f2853b9c71342a4fdc2abd967a0764cd3b4ed7a2b8770.jpg)  
(b) Memory size 5

![](images/137cdfbeca18fdf726d257d7208caf8e4434d6aba8141c91649e240113822664.jpg)  
(c) Memory size 15

![](images/f2727403d56b88076ff561415eb8afe42ba9cd9ef6a75aa917c54f26271abf5c.jpg)

![](images/bc405860e29784b2cac1e58d68e83ac1a9ee65fdfca766c3332c9ad5596fdf27.jpg)

![](images/ed65de5f98b1f1b233b28fe3144ea7cbe34fd803fe0a13ca63cccf32a4969a7d.jpg)  
(d) Accuracy difference of memory size 2 and 15

(e) Memory size 2  
(f) Memory size 5  
![](images/bdd87d906e3eae92c1057db8e1871d6c153480453383be9f1e1f97b163bf4eef.jpg)

![](images/263852666e71ed435eb3214b763f027a768693323ff901ce832a73b0a7a39f51.jpg)  
(h) Accuracy difference of memory size 2 and 15  
Figure 2: Accuracy w.r.t. different memory sizes and accuracy difference between memory sizes.

and (f), our model has greater advantages when the memory size is small, which indicates that our model can better deal with the overfitting problem in continual RE.

We also observe that the performance of each model declines due to the decrease of memory size, which demonstrates that memory size is a key factor in the performance of continual RE models. From Figures 2(d) and (h), the performance difference between different memory sizes is smaller. Thus, we draw the conclusion that our model is more robust to the change of memory size.

## 5.3.4 Performance on Analogous Relations

One strength of our model is to distinguish analogous relations for continual RE. We conduct an experiment to explore this point. Specifically, we select relations in the former five tasks which have analogous ones in the latter tasks, and report the accuracy and drop on them in Table 4. We consider that two relations are analogous if the similarity between their prototypes is greater than 0.85. As aforementioned, knowledge of the relations is more likely to be forgotten when their analogous relations emerge. Thus, all compared models are challenged by these relations. However, the performance of our model is superior and drops the least, which shows that our model succeeds in alleviating knowledge forgetting among analogous relations.

## 5.3.5 Case Study

We conduct a case study to intuitively illustrate the advantages of our model. Figure 3 depicts the visualization result. It is observed that the relations analogous in semantics (e.g., “mouth of the watercourse” and “tributary”) have relatively similar relation prototypes, which reflects that our model learns a reasonable representation space. Moreover, we see that the discrimination between similar relation prototypes (e.g., “director” and “screenwriter”) is still obvious, which reveals that our model can distinguish analogous relations. Please see Appendix C for the comparison with CRECL.

<table><tr><td rowspan="2">Models</td><td colspan="2">FewRel</td><td colspan="2">TACRED</td></tr><tr><td>Accuracy</td><td>Drop</td><td>Accuracy</td><td>Drop</td></tr><tr><td>CRL</td><td>69.7</td><td>19.0</td><td>68.9</td><td>20.4</td></tr><tr><td>CRECL</td><td>66.0</td><td>23.6</td><td>62.3</td><td>25.3</td></tr><tr><td>Ours</td><td>71.1</td><td>18.7</td><td>70.4</td><td>18.3</td></tr></table>

Table 4: Accuracy (%) and accuracy drop (%) on analogous relations. We select relations in the former five tasks that have similar ones in the latter tasks. Accuracy and drop are calculated in the same way as Table 1.

## 6 Conclusion

In this paper, we study continual RE. Through an empirical study, we find that knowledge decay among analogous relations is a key reason for catastrophic forgetting in continual RE. Furthermore, the overfitting problem prevalent in memorybased models also lacks consideration. To this end, we introduce a novel memory-based model to address the above issues. Specifically, the proposed memory-insensitive relation prototypes and memory augmentation can reduce overfitting to typical samples. In memory replay, the integrated training and focal knowledge distillation help maintain the knowledge among analogous relations, so that the model can better distinguish them. The experimental results on the FewRel and TACRED datasets demonstrate that our model achieves stateof-the-art performance and effectively alleviates catastrophic forgetting and overfitting for continual RE. In future work, we plan to explore whether our model can be used in few-shot RE to help distinguish analogous relations.

(1) director (2) screenwriter (3) country of origin (4) location (5) location of formation (6) headquarters location (7) mouth of the watercourse (8) tributary (9) located in or next to body of water (10) crosses  
![](images/e7fa5a8d96d7a3b0a5fea1bca05a3ff935b1f6fd4f8d1af0f230ed2f220565f3.jpg)  
Figure 3: Visualization of cosine similarity between relation prototypes generated by our model. We select 10 relations involving three highly-similar groups, i.e., [(1), (2)], [(3), (4), (5), (6)] and [(7), (8), (9), (10)].

## 7 Limitations

Our model may have several limitations: (1) As a memory-based model, our model consumes additional space to store typical samples and static prototypes, which causes the performance to be influenced by the storage capacity. (2) Although we propose memory-insensitive relation prototypes and memory augmentation, our model still relies on the selection of typical samples. The selected samples of low quality may harm the performance of our model. (3) The recent progress in large language models may alleviate catastrophic forgetting and overfitting, which has not been explored in this paper yet.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62272219) and the Collaborative Innovation Center of Novel Software Technology & Industrialization.

## References

Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. 2019. Efficient lifelong learning with A-GEM. In ICLR.

Li Cui, Deqing Yang, Jiaxin Yu, Chengwei Hu, Jiayang Cheng, Jingjie Yi, and Yanghua Xiao. 2021. Refining sample embeddings with relation prototypes to enhance continual relation extraction. In ACL, pages 232–243.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, pages 4171–4186.

Xu Han, Yi Dai, Tianyu Gao, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2020. Continual relation learning via episodic memory activation and reconsolidation. In ACL, pages 6429–6440.

Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018. FewRel: A large-scale supervised few-shot relation classification dataset with state-of-the-art evaluation. In EMNLP, pages 4803–4809.

Chengwei Hu, Deqing Yang, Haoliang Jin, Zhen Chen, and Yanghua Xiao. 2022. Improving continual relation extraction through prototypical contrastive learning. In COLING, pages 1885–1895.

James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. 2016. Overcoming catastrophic forgetting in neural networks. CoRR, abs/1612.00796.

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Gregory G. Slabaugh, and Tinne Tuytelaars. 2022. A continual learning survey: Defying forgetting in classification tasks. IEEE Trans. Pattern Anal. Mach. Intell., 44(7):3366–3385.

Zhizhong Li and Derek Hoiem. 2016. Learning without forgetting. In ECCV, pages 614–629.

Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollár. 2020. Focal loss for dense object detection. IEEE Trans. Pattern Anal. Mach. Intell., 42(2):318–327.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. In NeurIPS, pages 6467–6476.

Arun Mallya and Svetlana Lazebnik. 2018. PackNet: Adding multiple tasks to a single network by iterative pruning. In CVPR, pages 7765–7773.

Qi Qin, Wenpeng Hu, Han Peng, Dongyan Zhao, and Bing Liu. 2021. BNS: Building network structures dynamically for continual learning. In NeurIPS, pages 20608–20620.

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. 2017. iCaRL: Incremental classifier and representation learning. In CVPR, pages 5533–5542.

Florian Schroff, Dmitry Kalenichenko, and James Philbin. 2015. FaceNet: A unified embedding for face recognition and clustering. In CVPR, pages 815– 823.

Aäron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. CoRR, abs/1807.03748.

Eli Verwimp, Matthias De Lange, and Tinne Tuytelaars. 2021. Rehearsal revealed: The limits and merits of revisiting samples in continual learning. In ICCV, pages 9365–9374.

Hong Wang, Wenhan Xiong, Mo Yu, Xiaoxiao Guo, Shiyu Chang, and William Yang Wang. 2019. Sentence embedding alignment for lifelong relation extraction. In NAACL, pages 796–806.

Peiyi Wang, Yifan Song, Tianyu Liu, Binghuai Lin, Yunbo Cao, Sujian Li, and Zhifang Sui. 2022. Learning robust representations for continual relation extraction via adversarial class augmentation. CoRR, abs/2210.04497.

Quanziang Wang, Yuexiang Li, Dong Wei, Renzhen Wang, Kai Ma, Yefeng Zheng, and Deyu Meng. 2021. Revisiting experience replay: Continual learning by adaptively tuning task-wise relationship. CoRR, abs/2112.15402.

Tongzhou Wang and Phillip Isola. 2020. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In ICML, pages 9929–9939.

Tongtong Wu, Xuekai Li, Yuan-Fang Li, Gholamreza Haffari, Guilin Qi, Yujin Zhu, and Guoqiang Xu. 2021. Curriculum-meta learning for order-robust continual relation extraction. In AAAI, pages 10363– 10369.

Han Zhang, Bin Liang, Min Yang, Hui Wang, and Ruifeng Xu. 2022. Prompt-based prototypical framework for continual relation extraction. IEEE ACM Trans. Audio Speech Lang. Process., 30:2801–2813.

Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D. Manning. 2017. Position-aware attention and supervised data improve slot filling. In EMNLP, pages 35–45.

Kang Zhao, Hua Xu, Jiangong Yang, and Kai Gao. 2022. Consistent representation learning for continual relation extraction. In Findings of ACL, pages 3402– 3411.

## A More Results of Empirical Study

As mentioned in Section 1, we conduct an empirical study to explore the causation of catastrophic forgetting and find that the knowledge among analogous relations is more likely to be forgotten. As a supplement, we further report more results of our empirical study. Table 5 shows the average change of maximum similarity when the accuracy on relations suffers a sudden drop. Note that the number of relations greater than a 40% drop of CRECL on the TACRED dataset is quite small, thus the result may not be representative. It is observed that, if the maximum similarity of a relation to others obviously increases, its accuracy suddenly drops severely, which indicates that there tends to be a newly emerging relation analogous to it. In short, we can conclude that a relation may suffer catastrophic forgetting when its analogous relations appear. This also emphasizes the importance of maintaining knowledge among analogous relations.

<table><tr><td rowspan="2">Models</td><td rowspan="2">Sudden drop</td><td colspan="2">Maximum similarity change</td></tr><tr><td>FewRel</td><td>TACRED</td></tr><tr><td rowspan="2">CRL</td><td>(0.0, 20.0)</td><td>0.715 → 0.715</td><td>0.780 → 0.773</td></tr><tr><td>[20.0, 40.0) [40.0, 100.0)</td><td>0.700 → 0.888 0.784 → 0.944</td><td>0.798 → 0.899 0.860 → 0.924</td></tr><tr><td>CRECL</td><td>(0.0, 20.0) [20.0, 40.0) [40.0, 100.0)</td><td>0.596 → 0.601 0.665 → 0.889 0.556 → 0.904</td><td>0.649 → 0.642 0.650 → 0.827 0.649 → 0.820</td></tr></table>

Table 5: More results of our empirical study. We report the average change of maximum similarity when the accuracy of relations suffers varying degrees of a sudden drop. “Sudden drop” denotes the accuracy drop between two adjacent tasks.

## B Implementation Details

We carry out all experiments on a single NVIDIA RTX A6000 GPU with 48GB memory. Our implementation is based on Python 3.9.7 and the version of PyTorch is 1.11.0.

We find the best hyperparameter values through grid search with a step of 0.1 except 0.05 for ω and 0.25 for $\gamma .$ The search spaces for various hyperparameters are $\alpha \in [ 0 . 2 , 0 . 8 ] , \beta \in [ 0 . 1 , 0 . 5 ] , \mu \in$ $[ 0 . 1 , 1 . 0 ] , \omega \in [ 0 . 0 5 , 0 . 2 5 ] , \gamma \in [ 1 . 0 , 2 . 0 ]$ and $\lambda _ { 1 }$ $\lambda _ { 2 } \ \in \ [ 0 . 5 , 1 . 5 ]$ . Besides, we fix $\tau _ { 1 }$ and $\tau _ { 2 }$ to 0.1 and 0.5, respectively. The used hyperparameter values are listed below:

• For FewRel, $\alpha = 0 . 5 , \beta = 0 . 5 , \tau _ { 1 } = 0 . 1 \mathrm { { , } }$ $\mu = 0 . 5 , \ : \omega = 0 . 1 , \ : \tau _ { 2 } = 0 . 5 , \ : \gamma = 1 . 2 5 ,$ $\lambda _ { 1 } = 0 . 5 , \lambda _ { 2 } = 1 . 1 .$

• For TACRED, $\alpha = 0 . 6 , \beta = 0 . 2 , \tau _ { 1 } = 0 . 1 $ $\mu = 0 . 8 , \ : \omega = 0 . 1 5 , \ : \tau _ { 2 } = 0 . 5 , \ : \gamma = 2 . 0 ,$ $\lambda _ { 1 } = 0 . 5 , \lambda _ { 2 } = 0 . 7 .$

## C Case Study of Our Model and CRECL

To intuitively illustrate that our model can better distinguish analogous relations, we conduct a comparison to CRECL based on the case study in Section 5.3.5. As depicted in Figure 4, it is true for both our model and CRECL that if the relations are dissimilar in semantics, the similarity between their prototypes is low. However, we can observe that our model learns relatively dissimilar prototypes among analogous relations (e.g., lighter color between “director” and “screenwriter”), which demonstrates that our model can better distinguish analogous relations.

## D Comparison with ACA

As aforementioned in Section 2, Wang et al. (2022) propose an adversarial class augmentation (ACA) strategy, aiming to learn robust representations to overcome the influence of analogous relations. Specifically, ACA utilizes two class augmentation methods, namely hybrid-class augmentation and reversed-class augmentation, to build hard negative classes for new tasks. When new tasks arrive, the model is jointly trained on new relations and adversarial augmented classes to learn robust initial representations for new relations. As a data augmentation strategy, ACA can be combined with other continual RE models. Therefore, we conduct an experiment to explore the performance of our model with ACA.

We re-run the source code of ACA and report the results of RP-CRE + ACA, EMAR + ACA, and our model + ACA in Table 6. Compared with the original models, both EMAR and RP-CRE gain improvement, which demonstrates the effectiveness of ACA in learning robust representations for analogous relations. However, as we also explicitly consider the knowledge forgetting of analogous relations, there exist overlaps between ACA and our model. Thus, the performance of our model declines when combined with ACA. We leave the combination of our model and other augmentation methods in future work.

<table><tr><td>FewRel</td><td> $T _ { 1 }$ </td><td> $T _ { 2 }$ </td><td> $T _ { 3 }$ </td><td> $T _ { 4 }$ </td><td> $T _ { 5 }$ </td><td> $T _ { 6 }$ </td><td> $T _ { 7 }$ </td><td> $T _ { 8 }$ </td><td> $T _ { 9 }$ </td><td> $T _ { 1 0 }$ </td></tr><tr><td>RP-CRE + ACA  $\mathrm { E M A R + A C A }$ </td><td>97.7 98.3</td><td>95.2</td><td>92.8</td><td>91.0</td><td>90.1</td><td>88.7</td><td>86.9</td><td>86.4</td><td>85.3</td><td>83.8</td></tr><tr><td>Ours</td><td>98.1</td><td>94.6 95.8</td><td>92.6 93.6</td><td>90.6 91.9</td><td>90.4 91.1</td><td>88.8 89.4</td><td>87.7 88.1</td><td>86.7 86.9</td><td>85.6 85.6</td><td>84.1 84.2</td></tr><tr><td>Ours + ACA</td><td>98.4</td><td>94.8</td><td>92.8</td><td>91.4</td><td>90.4</td><td>88.9</td><td>87.8</td><td>86.8</td><td>86.0</td><td>83.9</td></tr><tr><td>TACRED RP-CRE + ACA</td><td>1  $T _ { 1 }$  97.1</td><td> $T _ { 2 }$ </td><td> $T _ { 3 }$ </td><td> $T _ { 4 }$ </td><td> $T _ { 5 }$ </td><td> $T _ { 6 }$ </td><td> $T _ { 7 }$ </td><td> $T _ { 8 }$ </td><td> $T _ { 9 }$ </td><td> $T _ { 1 0 }$  76.5</td></tr><tr><td> $\mathrm { E M A R + A C A }$ </td><td>97.6</td><td>93.5 92.4</td><td>89.4 90.5</td><td>84.5 86.7</td><td>83.7 84.3</td><td>81.0 82.2</td><td>79.3 80.6</td><td>78.0 78.6</td><td>77.5 78.3</td><td>78.4</td></tr><tr><td>Ours  $\mathrm { O u r s } + \mathrm { A C A }$ </td><td>97.7</td><td>94.3</td><td>92.3</td><td>88.4</td><td>86.6</td><td>84.5</td><td>82.2</td><td>81.1</td><td>80.1</td><td>79.1</td></tr></table>

Table 6: Accuracy (%) on all observed relations after learning each task.

![](images/d12fc7e9ff3806195ddd45208d63f13df036dd42e10725b7a07431fc420e3669.jpg)  
(1) director (2) screenwriter (3) country of origin (4) location (5) location of formation (6) headquarters location (7) mouth of the watercourse (8) tributary (9) located in or next to body of water (10) crosses

(a) Visualization of our model.  
![](images/a35a77c528d00ee4aed8c5fd8cfe0b4a31e053b495d950788b0d8059a04c8b0c.jpg)  
(b) Visualization of CRECL.  
Figure 4: Visualization of cosine similarity between relation prototypes generated by our model and CRECL.

## E Performance on Dissimilar Relations

We further conduct an experiment to explore the performance on dissimilar relations. We consider that relations with the highest similarity to other relations lower than 0.7 are dissimilar relations. As shown in Table 7, our model achieves the best accuracy on dissimilar relations. We attribute this to the better representations it learns through integrated training. However, our model does not always obtain the smallest drop as it focuses on alleviating the forgetting of analogous relations. Overall, from the results in Tables 4 and 7, we can conclude that our model achieves the best accuracy on both analogous and dissimilar relations as well as the least drop on analogous relations.

<table><tr><td rowspan="2">Models</td><td colspan="2">FewRel</td><td colspan="2">TACRED</td></tr><tr><td>Accuracy</td><td>Drop</td><td>Accuracy</td><td>Drop</td></tr><tr><td>CRL</td><td>90.2</td><td>5.9</td><td>92.1</td><td>1.4</td></tr><tr><td>CRECL</td><td>90.6</td><td>5.3</td><td>91.2</td><td>3.8</td></tr><tr><td>Ours</td><td>92.4</td><td>4.1</td><td>93.7</td><td>2.3</td></tr></table>

Table 7: Accuracy (%) and accuracy drop (%) on dissimilar relations. Relations with the highest similarity to other relations lower than 0.7 are considered as dissimilar relations. Accuracy and drop are calculated in the same way as Table 1.

## A For every submission:

<sup>✓</sup> A1. Did you describe the limitations of your work? Section 7.

✗ A2. Did you discuss any potential risks of your work? No, our paper is a foundational research.

<sup>✓</sup> A3. Do the abstract and introduction summarize the paper’s main claims? Section 1.

✗ A4. Have you used AI writing assistants when working on this paper? Left blank.

## B <sup>✓</sup> Did you use or create scientific artifacts?

Sections 4 and 5.

<sup>✓</sup> B1. Did you cite the creators of artifacts you used? Sections 4 and 5.

✗ B2. Did you discuss the license or terms for use and / or distribution of any artifacts? The artifacts that we use are all public.

<sup>✓</sup> B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? For the artifacts you create, do you specify intended use and whether that is compatible with the original access conditions (in particular, derivatives of data accessed for research purposes should not be used outside of research contexts)? Section 5.

✗ B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it? The datasets that we use are all public

✗ B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? The artifacts that we use are all public.

<sup>✓</sup> B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? Even for commonly-used benchmark datasets, include the number of examples in train / validation / test splits, as these provide necessary context for a reader to understand experimental results. For example, small differences in accuracy on large test sets may be significant, while on small test sets they may not be. Section 5.

## C <sup>✓</sup> Did you run computational experiments?

Section 5 and Appendix B.

<sup>✓</sup> C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? Appendix B.

<sup>✓</sup> C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? Section 5 and Appendix B.

<sup>✓</sup> C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run? Section 5.

<sup>✓</sup> C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation), did you report the implementation, model, and parameter settings used (e.g., NLTK, Spacy, ROUGE, etc.)? Appendix B.

D ✗ Did you use human annotators (e.g., crowdworkers) or research with human participants? Left blank.

 D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.? Not applicable. Left blank.

 D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participants’ demographic (e.g., country of residence)? Not applicable. Left blank.

 D3. Did you discuss whether and how consent was obtained from people whose data you’re using/curating? For example, if you collected data via crowdsourcing, did your instructions to crowdworkers explain how the data would be used? Not applicable. Left blank.

 D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? Not applicable. Left blank.

 D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? Not applicable. Left blank.