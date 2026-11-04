# Revealing Single Frame Bias for Video-and-Language Learning

Jie Lei Tamara L. Berg Mohit Bansal

Department of Computer Science University of North Carolina at Chapel Hill {jielei, tlberg, mbansal}@cs.unc.edu

## Abstract

Training an effective video-and-language model intuitively requires multiple frames as model inputs. However, it is unclear whether using multiple frames is beneficial to downstream tasks, and if yes, whether the performance gain is worth the drastically-increased computation and memory costs resulting from using more frames. In this work, we explore single-frame models for video-and-language learning. On a diverse set of video-andlanguage tasks (including text-to-video retrieval and video question answering), we show the surprising result that, with large-scale pretraining and a proper frame ensemble strategy at inference time, a single-frame trained model that does not consider temporal information can achieve better performance than existing methods that use multiple frames for training. This result reveals the existence of a strong “static appearance bias” in popular video-andlanguage datasets. Therefore, to allow for a more comprehensive evaluation of videoand-language models, we propose two new retrieval tasks based on existing fine-grained action recognition datasets that encourage temporal modeling. Our code is available at https: //github.com/jayleicn/singularity.

## 1 Introduction

Video and language are the two primary signals that constitute much of the world we perceive every day – we observe our surrounding environment with our eyes in the form of continuous visual input (video), and communicate with others via language. Intuitively, this leads one to assume that training an effective video-and-language model should require multiple video frames as input. Standard methods (Zhu and Yang, 2020; Xu et al., 2021; Li et al., 2020a; Luo et al., 2021) in this area typically use multiple densely sampled frames for training. Recent work (Lei et al., 2021) proposes sparse sampling for video-and-language understanding, where it claims that a few sparsely sampled clips are sufficient for learning due to the high redundancy in videos. This technique has shown (Lei et al., 2021; Zellers et al., 2021) to be successful in various video-language benchmarks (Jang et al., 2017; Xu et al., 2016; Anne Hendricks et al., 2017; Krishna et al., 2017a; Xu et al., 2017; Yu et al., 2018; Lei et al., 2018). However, as demonstrated by Bain et al. (2021); Luo et al. (2021); Lei et al. (2021), training with fewer frames (e.g., a single frame) leads to significantly worse performance compared to their multi-frame counterparts. In contrast, in this work, we show that with proper modeling, single-frame models could achieve competitive performance, hence revealing “static appearance bias” in popular video-and-language datasets.

We start by building a standard image-language model, with a vision encoder and a language encoder for image and text encoding, followed by a multi-modal encoder with cross-attention for crossmodal fusion. We pre-train the model on largescale image-text and video-text datasets (Chen et al., 2015; Krishna et al., 2017b; Ordonez et al., 2011; Sharma et al., 2018; Changpinyo et al., 2021; Bain et al., 2021). For fine-tuning, we randomly sample a single frame for training, and ensemble multiple uniformly sampled frames per video for making a video-level prediction at inference.

Single-frame predictions are often noisy and inaccurate, as they are made from incomplete information from single-frames without any context (see examples in Figure 3). Due to this issue, single-frame training typically performs significantly worse than multi-frame training (Lei et al., 2021; Bain et al., 2021; Luo et al., 2021). Previous work (Hendrycks et al., 2019) suggests that pretraining improves model robustness in the face of label corruption for image recognition. Inspired by this, we hypothesize that large-scale pre-training helps mitigate noise from single-frame training. Our analyses in Section 5 agree with our hypothesis, showing that as we increase pre-training data size, the performance of our single-frame model improves drastically and its gap with a similarly trained multi-frame model is largely eliminated. Besides training, these noisy single-frame predictions also render simple late fusion (e.g., meanpooling in ClipBERT (Lei et al., 2021)) less effective at inference time. To deal with this issue, we propose an early fusion strategy, which takes all frames as model inputs for directly making a videolevel prediction. Our analyses show that this early fusion ensemble method outperforms late fusion strategies and also delivers consistently improved performance when more frames are used.

We compare our approach with existing methods on six datasets across two video-language tasks, including text-to-video retrieval (MSRVTT (Xu et al., 2016), DiDeMo (Anne Hendricks et al., 2017), and ActivityNet Captions (Krishna et al., 2017a)) and video question answering (MSRVTT-QA (Xu et al., 2017), ActivityNet-QA (Yu et al., 2019), and MSRVTT-MC (Yu et al., 2018)). Results show that our approach achieves competitive (mostly better) performance than existing methods that use more training frames and pre-training data, setting new state-of-the-art for multiple tasks. This conclusion holds for both short 15-second MSRVTT videos and 180-second ActivityNet videos, showing the effectiveness of our approach in various scenarios.

More importantly, this strong single-frame performance reveals that the current evaluation is biased towards still objects, scenes, etc., while the temporal dynamics seem negligible, which in fact should be important for “true” video-language understanding. To address this issue, we next propose two new tasks that are designed to test models’ true temporal modeling ability. Based on the find-grained action recognition dataset Something-Something v2 (SSv2) (Goyal et al., 2017a), we create two text-to-video retrieval tasks, one that use SSv2’s action template as text queries, e.g., “Throwing [something] in the air and catching it”, and another that uses its annotated label as text queries, e.g., “Throwing keys in the air and catching it”. See examples in Figure 2. This template task removes the objects and only keeps the actions, enabling an evaluation that focuses almost solely on temporal modeling. The label task, on the other hand, contains both actions and objects, requiring an understanding of both still objects and their motion. Lastly, we present several baselines on these new tasks and show that temporal modeling is essential in achieving high scores.

In summary, our contributions are three-fold: (i) We explore single-frame training for videolanguage tasks. While simple, our approach can achieve state-of-the-art performance on a range of datasets, including both text-to-video retrieval and video question answering. Importantly, this result reveals the surprising static appearance bias in existing datasets. (ii) We conduct careful analyses, which show that large-scale pre-training and a proper multi-frame ensemble strategy are the core for single-frame trained models to be successful. (iii) We propose two new tasks specifically designed for testing models’ temporal modeling ability. These two new tasks complement existing benchmarks for a more comprehensive evaluation.

## 2 Related Work

Vision and Language. Vision and language learning considers the problem of learning from both visual and textual signals. Depending on their visual input type, methods in this area can be roughly categorized into two types, one with image (Anderson et al., 2018; Tan and Bansal, 2019; Lu et al., 2019; Chen et al., 2020; Li et al., 2019, 2020b, 2021b; Radford et al., 2021) and another with video (Anne Hendricks et al., 2017; Sun et al., 2019; Xu et al., 2021; Li et al., 2020a; Zellers et al., 2021; Bain et al., 2021; Lin et al., 2021). Standard video-language methods (Zhu and Yang, 2020; Xu et al., 2021; Li et al., 2020a; Lei et al., 2021; Luo et al., 2021) are typically trained with multiple video frames. This multi-frame training strategy has been the norm and is shown to work well across various datasets (Xu et al., 2016; Anne Hendricks et al., 2017; Krishna et al., 2017a; Jang et al., 2017; Xu et al., 2017; Lei et al., 2018). Unlike previous work that uses multiple frames for training, we explore single-frame training (i.e., similar to training an image-text model) and show it achieves strong performance on existing video-text benchmarks. Concurrent work (Buch et al., 2022) proposes a new module, atemporal probe, for selecting the best single-frame as inputs to a trained image-text model during inference; whereas we utilize multiple uniformly sampled frames and study effective ways of ensembling these frames.

Dataset Bias. Biases are prevalent in datasets (Goyal et al., 2017b; Li et al., 2018; Escorcia et al., 2019; Lei et al., 2020), e.g., Zhang et al. (2016) pointed out that blindly answering “yes” to yes/no questions in VQA (Antol et al., 2015) without looking at images results in an accuracy of 87%; Li et al. (2018) discovered that many video action recognition datasets, such as Kinetics (Kay et al., 2017) and UCF-101 (Soomro et al., 2012), have a strong static bias, where a linear classifier trained on static appearance (e.g., object, scene, and people) representations achieves much higher performance than chance. In this work, we find similar static bias exists in popular video-language datasets (Xu et al., 2016; Anne Hendricks et al., 2017; Krishna et al., 2017a; Xu et al., 2017; Yu et al., 2018, 2019), in which our models trained with single frames could achieve surprisingly good performance, even compared to models that perform explicit temporal modeling. When datasets are biased, they provide incorrect indications of the models’ ability. To allow for a more comprehensive evaluation, we propose two new tasks based on an existing action recognition dataset SSv2 (Goyal et al., 2017a) to test models’ true temporal modeling ability.

## 3 Methods

Model Architecture. Figure 1 shows an overview of our model (dubbed SINGULARITY). It consists of 3 main components, a vision encoder $\mathcal { F } _ { v }$ , a language encoder $\mathcal { F } _ { l }$ , and a multi-modal encoder . The vision encoder is an image-level visual backbone model, such as ViT (Dosovitskiy et al., 2020). The language encoder is a language model such as BERT (Devlin et al., 2019). For the multi-modal encoder, we use a transformer encoder (Vaswani et al., 2017), in which each layer contains a selfattention, a cross-attention, and a feed-forward network (FFN). The cross-attention layer is used to gather information from encoded visual inputs using the text as key, similar to recent work (Jaegle et al., 2021, 2022; Li et al., 2021b, 2022).

We denote a video V contains T frames as $V { = } [ f _ { 1 } , f _ { 2 } , . . . , f _ { T } ]$ , its paired text as S. During training, we randomly sample a single frame $f _ { t }$ from V as model input , where $t \in \{ 1 , . . . , T \}$ . Its encoded representation can be written as $\mathcal { F } _ { v } ( f _ { t } ) \in \mathbb { R } ^ { L _ { v } \times D }$ For text, the encoded representation is $\mathcal { F } _ { l } ( S ) ~ \in$ $\mathbb { R } ^ { L _ { l } \times D } . ~ L _ { v }$ and $L _ { l }$ are encoded sequence lengths, D is hidden size. We next make a prediction p as:

$$
\begin{array} { c } { { \boldsymbol { p } = \mathcal { H } ( \mathcal { F } _ { l } ( S ) , \mathcal { F } _ { v } ( f _ { t } ) ) , } } \\ { { \mathrm { } _ { \mathrm { Q } , \mathsf { K } , \mathsf { V } \mathrm { f o r } \mathsf { s e l f } - \mathsf { a t t } ; \atop \mathrm { Q } \mathrm { f o r } \mathsf { c r o s s } - \mathsf { a t t } } { \mathrm { \Pi } } _ { } } } \end{array}\tag{1}
$$

where Q, K, V denote the query, key, and value matrices of self- and cross-attention (Vaswani et al.,

2017). We calculate loss based on this prediction. During inference, we uniformly sample $T _ { t e s t }$ frames $\{ f _ { \tau _ { i } } \} _ { i = 1 } ^ { T _ { t e s t } }$ Each frame is encoded separately, and their encoded representations are concatenated as inputs to the multi-modal encoder to get a video-level prediction score:

$$
p = \mathcal { H } ( \mathcal { F } _ { l } ( S ) , [ \mathcal { F } _ { v } ( f _ { \tau _ { 1 } } ) ; . . . ; \mathcal { F } _ { v } ( f _ { \tau _ { T _ { t e s t } } } ) ] ) ,\tag{2}
$$

where [; ] denotes concatenation, and $[ \mathcal { F } _ { v } ( f _ { \tau _ { 1 } } ) ; . . . ; \mathcal { F } _ { v } ( f _ { \tau _ { T _ { t e s t } } } ) ] \qquad \in \qquad \mathbb { R } ^ { ( T _ { t e s t } \times L _ { v } ) \times D }$ This early fusion design allows our model to make an informed prediction given full context. In ClipBERT (Lei et al., 2021), an alternative late fusion design is used: scores are computed for each frame separately, and video-level score is obtained via a manually designed aggregation function $\mathcal { G }$ (e.g., mean-pooling):

$$
\begin{array} { r l } & { p = \mathcal { G } ( { p _ { \tau _ { 1 } } } , { p _ { \tau _ { 2 } } } , { p _ { \tau _ { T _ { t e s t } } } } ) ; } \\ & { p _ { \tau _ { i } } = \mathcal { H } ( \mathcal { F } _ { l } ( S ) , \mathcal { F } _ { v } ( f _ { \tau _ { i } } ) ) . } \end{array}\tag{3}
$$

Since the predictions in late fusion are made with incomplete information from individual frames, they can be quite noisy. In Section 5, we provide a detailed comparison w.r.t. these different frame ensemble methods and show that early fusion consistently outperforms late fusion.

Pre-Training Objectives. The model is trained with 3 losses: (i) Vision-Text Contrastive: a contrastive loss that aligns the pooled vision and text representations from the vision and language encoders. (ii) Masked Language Modeling (MLM) (Devlin et al., 2019): predicting masked tokens from their text and visual context, with multimodal encoder. (iii) Vision-Text Matching: predicting the matching score of a vision-text pair with multi-modal encoder. These losses have shown to be effective in learning multi-modal representations (Tan and Bansal, 2019; Chen et al., 2020; Li et al., 2021a,b; Lei et al., 2021; Radford et al., 2021). More details are in Appendix.

Implementation Details. We use both image-text and video-text data for pre-training. For imagetext data, we use a combination of COCO (Chen et al., 2015), Visual Genome (VG) (Krishna et al., 2017b), SBU Captions (Ordonez et al., 2011), CC3M (Sharma et al., 2018), and CC12M (Changpinyo et al., 2021). For video-text data, we use WebVid (Bain et al., 2021). Note that, for videotext data, we only sample a single frame from the video for training. We pre-train the model on two different subsets of the datasets: (i) 5M corpus that contains 5.44M images and videos from CC3M+WebVid, and (ii) 17M corpus that contains 17.28M images and videos from all the datasets.

![](images/49abce7d000037c7cb5ba56b9bca1430e1b4251dd497a65c4711a314681d9d19.jpg)  
Figure 1: SINGULARITY model overview. During training, we randomly sample a single frame as input, and make a video level prediction from it along with its paired text. During inference, we uniformly sample multiple frames, and early fuse their encoded image-level representations as input to the multi-modal encoder. Details in Section 3.

Our model is implemented in PyTorch (Paszke et al., 2019). The vision encoder is initialized using BEiT<sub>BASE</sub> (Bao et al., 2021) model trained on ImageNet-21K (Deng et al., 2009). The text encoder is initialized from the first 9 layers of BERT (Devlin et al., 2019). The multi-modal encoder is initialized from the last 3 layers of the same BERT model, and its cross-attention layers are randomly initialized. We optimize the model for 10 epochs using AdamW (Loshchilov and Hutter, 2019) with an initial learning rate of 1e-4. We warm up the learning rate in the first epoch followed by cosine decay (Loshchilov and Hutter, 2017) to 1e-6. Mixed precision is used for faster training. The batch size is set to 128/GPU, we train the model on 3 NVIDIA A100 GPUs with image size 224 224. We perform basic augmentations: random resize, crop, and flip to the images during training. This pre-training takes around 1 day on the 5M corpus, and 4 days on the 17M corpus. Our pre-training is quite efficient compared to other similar work, e.g., 10 epochs’ pre-training in Align-Prompt (Li et al., 2021a) takes 3 days on the same 5M corpus using 16 A100 GPUs, this amounts to 16 computation cost of our pre-training.

## 4 Experiments

## 4.1 Downstream Task Setup

Text-to-Video Retrieval. Given a text query, the goal of this task is to retrieve relevant videos from a large set of videos. We evaluate our model on the following datasets: (i) MSRVTT (Xu et al.,

2016) contains 10K YouTube videos, each paired with 20 captions. We follow (Yu et al., 2018; Lei et al., 2021) to use the 7K train+val videos for training, and report results on the 1K test set. (ii) DiDeMo (Anne Hendricks et al., 2017) contains 10K Flickr videos with 41K captions. We use standard train/val/test splits. (iii) ActivityNet Captions (Krishna et al., 2017a) contains 20K YouTube videos with 100K captions. We use the train split with 10K videos for training, and we report results on the widely used val1 split, with 4.9K videos. For MSRVTT, we evaluate standard text-to-video retrieval. For DiDeMo and ActivityNet Captions, we evaluate paragraph-to-video retrieval (Liu et al., 2020; Lei et al., 2021; Luo et al., 2021), where the text captions in the same video are concatenated as a single paragraph-level text for retrieval. We report performance using recall at K (R@K).

Video Question Answering. Given a video (often with a text question), this task requires generating an answer to the question or selecting the most suitable answer from a set of candidates. (i) MSRVTT-QA (Xu et al., 2017) contains 244K open-ended questions on 10K MSRVTT videos. (ii) ActivityNet-QA (Yu et al., 2019) contains 58K open-ended questions on 5.8K sampled ActivityNet (Caba Heilbron et al., 2015) videos. (iii) MSRVTT-MC (Yu et al., 2018) is a multiplechoice task that requires selecting the matched caption from 5 candidate captions for each video (3K MSRVTT videos). We use standard train/val/test splits for the three tasks, and report accuracy.

## 4.2 Comparison on Existing Datasets

Text-to-Video Retrieval Results. In Table 1, we compare SINGULARITY with existing methods on text-to-video retrieval. Across all the datasets, SIN-GULARITY (5M) achieves better performance compared to methods trained on similar amounts of data, while using only single frames for training. On DiDeMo and ActivityNet Captions, it outperforms all previous work, including many that pretrain on significantly larger amounts of data, e.g., 400M image-text pairs in CLIP4Clip, or 136M video-text pairs in VideoCLIP compared to 5M image-text and video-text pairs in SINGULARITY. We also note that our model is trained with single frames, while previous work uses many more frames, e.g., 64 frames in CLIP4Clip or 8 frames in AlignPrompt. When trained with a larger amount of data (17M), we notice a further performance boost for our model, demonstrating that SINGU-LARITY benefits from large-scale pre-training.

Table 1: Comparison to existing methods on text-to-video retrieval. #PT denotes the number of images and or videos used in cross-modal pre-training. #Train Frame denotes the number of frames used at each training step during fine-tuning. For models that use different number of frames for different datasets, we list them together with a separator “/”. We gray out methods that use significantly more pre-training data for a fair comparison. The 136M corpus is from HowTo100M (Miech et al., 2019), 0.2M refers to COCO+VG data, 138M is the combination of HowTo100M and WebVid, 400M is the private image-text data used in CLIP (Radford et al., 2021).
<table><tr><td rowspan="2">Method</td><td rowspan="2">#PT</td><td rowspan="2">#Train</td><td colspan="3">MSRVTT</td><td colspan="3">DiDeMo</td><td colspan="3">ActivityNet Cap</td></tr><tr><td>Frame</td><td>R1 R5</td><td></td><td>R10 R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td></tr><tr><td>HERO (Li et al., 2020a)</td><td>136M</td><td>310</td><td>20.5 47.6 60.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ClipBERT (Lei et al., 2021)</td><td>0.2M</td><td>16/16/8</td><td></td><td></td><td></td><td>22.0 46.8 59.9 20.4 48.0 60.8 21.3 49.063.5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ViđeoCLIP (Xu et al., 2021)</td><td>136M</td><td>960</td><td>30.9 55.4 66.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Frozen (Bain et al., 2021)</td><td>5M</td><td>4</td><td></td><td></td><td></td><td>31.0 59.5 70.5 31.0 59.8 72.4</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AlignPrompt (Li et al., 2021a)</td><td>5M</td><td>8</td><td></td><td></td><td></td><td>33.9 60.7 73.2 35.9 67.5 78.8</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>All-in-one (Wang et al., 2022)</td><td>138M</td><td>9</td><td>34.4 65.4 75.8 32.7 61.4 73.5 22.4 53.7 67.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP4Clip (Luo et al., 2021)</td><td>400M</td><td>12/64/64 42.0 68.6 78.7 42.8 68.5 79.2 40.5 72.4 98.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ECLIPSE (Lin et al., 2022)</td><td>400M</td><td>-/32/32</td><td></td><td></td><td></td><td>44.2</td><td></td><td></td><td></td><td>45.3 75.7</td><td>86.2</td></tr><tr><td>SINGULARITY</td><td>5M</td><td>1</td><td></td><td></td><td></td><td>36.865.975.5 47.4 75.284.043.0 70.6</td><td></td><td></td><td></td><td></td><td>81.3</td></tr><tr><td>SINGULARITY</td><td>17M</td><td>1</td><td></td><td></td><td></td><td>41.5 68.7 77.0 53.9 79.4 86.947.1 75.5</td><td></td><td></td><td></td><td></td><td>85.5</td></tr></table>

Table 2: Comparison to existing methods on video question answering. 69M corpus is the 69M video questions in (Yang et al., 2021), 180M refers to the 180M clip-text pairs in YT-Temporal-180M (Zellers et al., 2021).
<table><tr><td colspan="5">Method #PT #Train Frame MSRVTT-QA ActivityNet-QA MSRVTT-MC</td></tr><tr><td>ClipBERT (Lei et al., 2021)</td><td>0.2M</td><td>16</td><td>37.4</td><td>88.2</td></tr><tr><td>AlignPrompt (Li et al., 2021a)</td><td>5M</td><td>16 42.1</td><td></td><td></td></tr><tr><td>JustAsk (Yang et al., 2021)</td><td>69M 640</td><td>41.5</td><td>38.9</td><td></td></tr><tr><td>MERLOT (Zellers et al., 2021) 180M</td><td>5</td><td>43.1</td><td>41.4</td><td>90.9</td></tr><tr><td>VideoCLIP (Xu et al., 2021)</td><td>136M 960</td><td></td><td></td><td>92.1</td></tr><tr><td>All-in-one (Wang et al., 2022)</td><td>138M 9</td><td>44.3</td><td>=</td><td>92.0</td></tr><tr><td>SINGULARITY</td><td>5M 1</td><td>42.7</td><td>41.8</td><td>92.0</td></tr><tr><td>SINGULARITY</td><td>17M</td><td>1 43.5</td><td>43.1</td><td>92.1</td></tr></table>

Video QA Results. Table 2 compares SINGULAR-ITY with existing methods on video question answering. We notice SINGULARITY (5M) achieves competitive performance with previous work even when using two orders of magnitude smaller pretraining data, e.g., 180M video-text pairs in MER-

LOT vs. 5M image-text and video-text pairs. Our method also surpasses the strong video QA model JustAsk, which is specifically designed for video QA and is pre-trained on 69M video QA pairs. When pre-trained with more data, our model performance further improves. These comparisons show the effectiveness of our single-frame approach.

In Appendix, we also provide additional results: (i) SINGULARITY-temporal for retrieval and QA; (ii) zero-shot retrieval; (iii) image-text retrieval; (iv) image QA, etc.

## 4.3 New Temporal Tasks

In the previous section, we revealed the interesting observation that popular video-language datasets have strong static appearance biases – enabling our model that uses only a single frame per video at each training step to achieve competitive performance compared to state-of-the-art models that digest multiple temporally-ordered frames. The biased evaluation on these datasets favors models that are strong in recognizing static concepts, and does not provide a good indicator of whether these models are capable of recognizing fine-grained temporal relationships between neighboring frames.

![](images/ee042555c70cea8088a59eaaccc9165c752cb8bb8b26f43bfbc8428c87a83e22.jpg)  
template: Throwing [something] in the air and letting it fall. label: Throwing keys in the air and letting it fall.  
Figure 2: SSv2 examples. For each video, we show 3 temporally-ordered frames with their template and label annotations. Based on these annotations, we propose two retrieval tasks, using “template” and “label” as text queries, respectively.

Hence, to address this issue, we propose two new datasets that complement existing datasets for a more comprehensive evaluation of video-andlanguage methods. We draw inspiration from the video action recognition community, and transform the temporally-heavy action recognition dataset Something-Something v2 (SSv2) (Goyal et al., 2017a) into video-and-language datasets. In Figure 2, we show SSv2 examples. A unique property of the SSv2 dataset is that the videos often require fine-grained temporal modeling to correctly predict their action classes. For example, to match the videos and their action classes (template) in Figure 2, one has to look at multiple temporally ordered frames. Based on SSv2 videos and annotations, we define two text-to-video retrieval tasks:

• SSv2-Template Retrieval: We use the 174 templates (e.g., “Throwing [something] in the air and catching it”) in SSv2 as the text queries to retrieve videos. We use 168,913 SSv2 training videos for training. As ground-truth annotations for test videos are not available, we use validation videos: we sample 12 videos for each template, with a total of 2,088 videos for testing.

• SSv2-Label Retrieval: We use annotated labels (e.g., “Throwing keys in the air and catching it”) in SSv2 as queries to retrieve videos. We follow the same split in the template retrieval task, with 168,913 training videos, and 2,088 test videos.

Since no objects are in the queries of the template retrieval task, it requires a deeper temporal action understanding than label retrieval, while label retrieval provides a more comprehensive evaluation of both static and temporal understanding.

Experiments. We use Frozen (Bain et al., 2021) and CLIP4Clip (seqTransf version) (Luo et al., 2021) as baselines. Frozen uses a space-time transformer, CLIP4Clip is an extension based on the CLIP (Radford et al., 2021) with an extra 4-layer temporal transformer encoder. We report performance using standard text-to-video retrieval metrics R@K. For our model, in addition to the singleframe version, we build a multi-frame variant, SIN-GULARITY-temporal. Specifically, we add a twolayer temporal transformer encoder following the vision encoder, and use its outputs as inputs to the multi-modal encoder (see details in Appendix). From a single-frame pre-trained checkpoint (5M or 17M), we perform a 2nd stage video pre-training with 4 frames using WebVid videos for SINGU-LARITY-temporal. We use an initial learning rate of 5e-5, and train the model for 5 epochs.

The results are shown in Table 3. Compared to Frozen and CLIP4Clip, while SINGULAR-ITY shows competitive performance on existing benchmarks (see Table 1), it underperforms these methods on the two temporally-heavy tasks by a large margin. For example, SINGULARITY (5M) underperforms the 4-frame Frozen model by 10.9 for SSv2-template retrieval R1, though it shows a 16.4 improvement for DiDeMo R1, and 5.8 for MSRVTT R1. This is a good sign as it shows that the new tasks cannot be solved by models exploiting static appearance biases. On the other hand, after adding the 2-layer temporal encoder, the 4- frame SINGULARITY-temporal model gets a significant performance boost from the single-frame model, surpassing the baseline methods. When using more pre-training data (5M 17M), we notice a good performance gain for SSv2-label, while the performance on SSv2-template stays similar. These observations indicate that the SSv2-label task requires both static and temporal modeling, and enhancing either will improve the task performance. For SSv2-template, as no objects exist in its text queries, it requires mostly temporal modeling.

## 5 Analysis

Frames Ensemble Strategy. Our model is trained with a single-frame regime, while using multiple frames covering the full video at inference. As shown in Figure 5a (concat), encoded video frames are concatenated as input to the multi-modal encoder’s cross-attention for making a video-level prediction. A naive alternative is to compute the prediction score for each frame separately (Figure 5b), and then aggregate these frame-level scores together to get a video-level score using an aggregation function, such as LogSumExp (lse), max- and mean-pooling. This simple late fusion strategy has shown to be successful for video-and-language (Lei et al., 2021) and video action recognition methods (Bertasius et al., 2021; Wang et al., 2016).

Table 3: Comparison to existing methods on SSv2 tasks. \* The training of Frozen on the SSv2-label retrieval task fails to converge despite our best efforts in tuning the model.
<table><tr><td rowspan="2">Method</td><td rowspan="2">#PT</td><td rowspan="2">#Train Frame</td><td colspan="3">SSv2-label</td><td colspan="3">SSv2-template</td></tr><tr><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td></tr><tr><td>Frozen (Bain et al., 2021)*</td><td>5M</td><td>4</td><td></td><td>1</td><td>I</td><td>52.9</td><td>94.8</td><td>99.4</td></tr><tr><td>CLIP4Clip (Luo et al., 2021)</td><td>400M</td><td>12</td><td>43.1</td><td>71.4</td><td>80.7</td><td>77.0</td><td>96.6</td><td>98.3</td></tr><tr><td>SINGULARITY</td><td>5M</td><td>1</td><td>36.4</td><td>64.9</td><td>75.4</td><td>42.0</td><td>86.2</td><td>94.3</td></tr><tr><td>SINGULARITY-temporal</td><td>5M</td><td>4</td><td>44.1</td><td>73.5</td><td>82.2</td><td>77.0</td><td>98.9</td><td>99.4</td></tr><tr><td>SINGULARITY-temporal</td><td>17M</td><td>4</td><td>47.4</td><td>75.9</td><td>84.0</td><td>77.6</td><td>96.0</td><td>98.9</td></tr></table>

![](images/aefc7dd1f0787e2d7a24bea177724b65204761e1a7bf5897a01c14079275953c.jpg)

Figure 3: Prediction score distribution for a MSRVTT-MC example. We show frame-level score distribution for each frame, and video-level score distribution for late fusion (we use mean as an example) and our early fusion (concat). The highest score for each is indicated by ✓, the correct answer is highlighted in green. Single-frame predictions are often inaccurate, unstable and they fluctuate across the frames. Late fusion can be biased by inaccurate but high confidence frame predictions, e.g., the late fusion prediction is biased towards the 4th frame prediction.  
![](images/eac25dfebd1f265948e4af678afab14f7caad1f78ef9b7c31f89389e8211c9a3.jpg)

![](images/43b36dedce04821ab7e9640eca91a8fe9da4751f1ccf7b9d736e65ca2da96215.jpg)  
Figure 4: Impact of frame ensemble strategy. avg recall is the average of R@{1,5,10}.

In Figure 4, we compare these different frame ensemble strategies, with varying number of frames at inference. From the comparison, we can draw the following conclusions: (i) Our early fusion strategy (concat) shows a significant gain over the three late fusion strategies (lse, max, mean) for both MSRVTT retrieval and ActivityNet-QA, demonstrating the importance of considering the whole video when making the predictions. (ii) In general, for all ensemble strategies, using more frames at inference improves model performance. However, for the late fusion strategies, sometimes using more frames hurts performance, e.g., for ActivityNet-QA, inference with over 4 frames underperforms that with 4 frames for max-pooling. This observation agrees with the MSRVTT-QA results in Clip-BERT (Lei et al., 2021). In contrast, early fusion delivers consistently improved performance when more frames are used. Overall, we hypothesize that the low and unstable performance of late fusion is because its video-level prediction is obtained via aggregating frame-level predictions, while these frame-level predictions can be inaccurate and unstable (see example in Figure 3) – as they are separately predicted using incomplete information within each frame, ignoring their context. Besides better accuracy, in Appendix, we show early fusion also runs faster.

![](images/72b4ad1e27559577a2a66ec4c06b5a0d1a2228550d30e1e3b4a56ef0aec1b6c0.jpg)  
Figure 5: Comparison of frame ensemble strategies. concat is our early fusion strategy, lse, max, mean are the late fusion strategies in ClipBERT (Lei et al., 2021).

![](images/9f80b57ccb78fe17b58d34e02d1d07bd0a621d0c8973f53117d66f66ebfed267.jpg)  
#PT images/videos (M)

![](images/777772fba4fb8b83e7253186dbdaece81858b124b009aea908b112be1dfe34a0.jpg)  
#PT images/videos (M)

![](images/1aeddb7230347223f0140b6bb268e95c4213ed8b87df5ebd9e5467ffdbda74dd.jpg)

![](images/8d578ea6c47f2cccc70c8d536a996296805ab7bf771dbbc1ee9f191b5b690b77.jpg)  
Figure 6: Model performance as a function of pretraining data size, for SINGULARITY (1-frame) and SINGULARITY-temporal (4-frame). Text in red denotes performance gap. In general, as pre-training data size increases, the gap between the two models decreases.

Pre-Training Data Size. In Figure 6, we study the effect of cross-modal pre-training data size for both the single-frame and the multi-frame model. We show downstream fine-tuning performance under 4 different pre-training data setups: no cross-modal pre-training (0M), pre-train on WebVid (2.49M videos), on 5M corpus (5.44M images+videos), or on 17M corpus (17.28M images+videos).

We obsereve that both 1- and 4-frame model greatly benefit from large-scale pre-training. When comparing the two models, an interesting observation is that, as the pre-training data size increases, the performance gap between the 1-frame and the 4- frame model decreases almost monotonically. This phenomenon suggests that, when pre-trained on a sufficient amount of data, the performance of models trained with single frames might be very close to models trained with multiple frames. Though there can be exceptions for tasks that require fine-grained temporal modeling, such as SSv2-label retrieval, where multi-frame modeling is necessary.

One possible explanation is that single-frame training is noisier than multi-frame training – due to incomplete context and random sampling, single-frame predictions are often inaccurate and less stable than multi-frame predictions, and pretraining is helpful (Hendrycks et al., 2019) in this case. Meanwhile, single-frame training requires the model to extract all information from a single frame while a multi-frame model could rely on rich sources from multiple frames. Therefore, for downstream tasks, it is essential for the single-frame model to initialize from a strong pre-trained model. Training Efficiency. A core advantage of singleframe training is its training efficiency. In Section 3, we discussed our pre-training cost is only 1/16 of a recent video-language model (Li et al., 2021a). In Figure 7 we compare the training time and task performance of various models. We note our model (1-frame, SINGULARITY, 17M) trains much faster than the baselines (2.8 for 4-frame Frozen, 8.5 for 64-frame CLIP4Clip) while showing notably better performance. Besides, it is also more memory efficient, i.e., its maximum allowed batch size on a single GPU is 190 while only 50 for Frozen. Experiments conducted on a single RTX A6000 GPU with 48GB memory, training time is averaged over 8,394 DiDeMo training examples. In Appendix, we show additional comparisons of various retrieval methods in terms of inference GFLOPs and model size.

![](images/bdeaee6dde49a370128c46509e2467c72d806bd14d181dd2f4ccd68eaafae77c.jpg)  
Figure 7: Comparison of training time and downstream task performance. The maximum allowed batch size is labeled besides each model as a reference.

## 6 Conclusion

In this work, we explore single-frame training for video-and-language learning. We find that, with sufficient pre-training data and a proper frame ensemble strategy at inference, our model trained with a single frame achieves surprisingly good performance on various video-text tasks, including text-to-video retrieval and video question answering. While these results show the potential of using single-frame training for various video-text tasks, it also reveals that current benchmarks are biased towards static objects and scenes, etc. To address this issue, we propose two new tasks designed to test models’ true temporal modeling ability and build several baseline methods for these new tasks. We hope these new tasks can complement existing benchmarks for a more comprehensive video-andlanguage understanding.

## Limitations

While the proposed single-frame training approach shows strong performance on various videolanguage datasets, it does not work well on true temporal tasks like the new SSv2 tasks. Compared to multi-frame models, our single-frame model also has a higher demand for pre-training data.

## Ethics Statement

Similar to many data-driven methods, the predictions from our system reflect the distribution of data on which it is trained on, and these predictions can be inaccurate and biased by the data. Furthermore, the model is trained with a single frame strategy, which may naturally not work well on tasks that require more understanding, thus its predictions on such tasks may not be reliable. Therefore, users should not completely rely on the system for making real-world decisions.

## Acknowledgements

We thank the reviewers and area chair for the useful feedback. This work is supported by ARO Award W911NF2110220, DARPA KAIROS Grant #FA8750-19-2-1004, DARPA MCS Grant N66001- 19-2-4031, and NSF-AI Engage Institute DRL-211263. The views in this article are those of the authors and not of the funding agency.

## References

Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. 2018. Bottom-up and top-down attention for image captioning and visual question answering. In CVPR.

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. 2017. Localizing moments in video with natural language. In ICCV.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In ICCV.

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. 2021. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV.

Hangbo Bao, Li Dong, and Furu Wei. 2021. Beit: Bert pre-training of image transformers. In ICLR.

Gedas Bertasius, Heng Wang, and Lorenzo Torresani. 2021. Is space-time attention all you need for video understanding. In ICML.

Shyamal Buch, Cristobal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. 2022. Revisiting the “video” in video-language understanding. arXiv preprint arXiv:2206.01720.

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. 2015. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. 2021. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In CVPR.

Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. 2015. Microsoft coco captions: Data collection and evaluation server. arXiv.

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. Uniter: Learning universal imagetext representations. In ECCV.

Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. 2021. Unifying vision-and-language tasks via text generation. arXiv.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In CVPR.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR.

Victor Escorcia, Mattia Soldan, Josef Sivic, Bernard Ghanem, and Bryan Russell. 2019. Temporal localization of moments in video collections with natural language. arXiv.

Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. 2020. Multi-modal transformer for video retrieval. In ECCV.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. 2017a. The" something something" video database for learning and evaluating visual common sense. In ICCV.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017b. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR.

Dan Hendrycks, Kimin Lee, and Mantas Mazeika. 2019. Using pre-training can improve model robustness and uncertainty. In ICML.

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. 2022. Perceiver io: A general architecture for structured inputs & outputs. In ICLR.

Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. 2021. Perceiver: General perception with iterative attention. In ICML.

Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. 2017. Tgif-qa: Toward spatiotemporal reasoning in visual question answering. In CVPR.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. arXiv.

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. 2017. The kinetics human action video dataset. arXiv.

Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. Vilt: Vision-and-language transformer without convolution or region supervision. In ICML.

Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. 2017a. Dense-captioning events in videos. In ICCV.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. 2017b. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV.

Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. 2021. Less is more: Clipbert for video-and-language learning via sparse sampling. In CVPR.

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. 2018. Tvqa: Localized, compositional video question answering. In EMNLP.

Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. 2020. What is more likely to happen next? videoand-language future event prediction. In EMNLP.

Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, and Steven CH Hoi. 2021a. Align and prompt: Video-and-language pre-training with entity prompts. arXiv.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. arXiv preprint arXiv:2201.12086.

Junnan Li, Ramprasaath R Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven Hoi. 2021b. Align before fuse: Vision and language representation learning with momentum distillation. In NeurIPS.

Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. 2020a. Hero: Hierarchical encoder for video+ language omni-representation pretraining. In EMNLP.

Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. 2019. Visualbert: A simple and performant baseline for vision and language. arXiv.

Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, and Haifeng Wang. 2021c. Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning. In ACL.

Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. 2020b. Oscar: Objectsemantics aligned pre-training for vision-language tasks. In ECCV.

Yingwei Li, Yi Li, and Nuno Vasconcelos. 2018. Resound: Towards action recognition without representation bias. In ECCV.

Xudong Lin, Gedas Bertasius, Jue Wang, Shih-Fu Chang, Devi Parikh, and Lorenzo Torresani. 2021. Vx2text: End-to-end learning of video-based text generation from multimodal inputs. In CVPR.

Yan-Bo Lin, Jie Lei, Mohit Bansal, and Gedas Bertasius. 2022. Eclipse: Efficient long-range video retrieval using sight and sound. In ECCV.

Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. 2020. Use what you have: Video retrieval using representations from collaborative experts. In BMVC.

Ilya Loshchilov and Frank Hutter. 2017. Sgdr: Stochastic gradient descent with warm restarts. In ICLR.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In ICLR.

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In NeurIPS.

Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. 2021. Clip4clip: An empirical study of clip for end to end video clip retrieval. arXiv preprint arXiv:2104.08860.

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. 2019. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In ICCV.

Vicente Ordonez, Girish Kulkarni, and Tamara Berg. 2011. Im2text: Describing images using 1 million captioned photographs. NeurIPS.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. arXiv.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. 2012. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv.

Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. 2019. Videobert: A joint model for video and language representation learning. In ICCV.

Hao Tan and Mohit Bansal. 2019. Lxmert: Learning cross-modality encoder representations from transformers. In EMNLP.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jé- gou. 2021. Training data-efficient image transformers & distillation through attention. In ICML.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In NeurIPS.

Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2022. All in one: Exploring unified video-language pre-training. arXiv.

Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. 2016. Temporal segment networks: Towards good practices for deep action recognition. In ECCV.

Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. 2017. Video question answering via gradually refined attention over appearance and motion. In ACM MM.

Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. 2021. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In EMNLP.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. Msrvtt: A large video description dataset for bridging video and language. In CVPR.

Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. 2021. Just ask: Learning to answer questions from millions of narrated videos. In ICCV.

Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. TACL.

Youngjae Yu, Jongseok Kim, and Gunhee Kim. 2018. A joint sequence fusion model for video question answering and retrieval. In ECCV.

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. 2019. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI.

Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. 2021. Merlot: Multimodal neural script knowledge models. NeurIPS.

Peng Zhang, Yash Goyal, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2016. Yin and yang: Balancing and answering binary visual questions. In CVPR.

Linchao Zhu and Yi Yang. 2020. Actbert: Learning global-local video-text representations. In CVPR.

## A Appendix

In Section A.1, we show details of our open-ended QA model and SINGULARITY-temporal model, as well as pre-training objectives. In Section A.2, we show more experimental details, such as SIN-GULARITY-temporal results on existing datasets, SINGULARITY zero-shot results, impact of image size, and results on image-text tasks such as textto-image retrieval tasks Flickr30K (Young et al., 2014), COCO (Chen et al., 2015) and image question answering task VQA (Antol et al., 2015), model size and inference cost of our approach w.r.t. other recent approaches, as well as memory and time comparison of different frame ensemble strategies. In addition, we also show hyper-parameters and more experimental setups in this section. In Section A.3, we show more dataset details.

![](images/f84e9590315796d37c7ae0824429c389f40e2f3c2ca2550ff751865a2ab3d5da.jpg)  
Figure 8: SINGULARITY model variants for video question answering and temporal modeling (i.e., SINGULAR-ITY-temporal). The horizontal arrows indicate crossattention inputs, while the vertical arrows indicate selfattention inputs.

## A.1 Additional Modeling Details

Open-ended QA model. Figure 8a shows a graphic overview of the model architecture for open-ended video question answering. Following previous work (Cho et al., 2021; Li et al., 2021b), we formulate this task as text generation instead of classification. Based on the base model described in main text, we add an extra multi-modal decoder that takes in multi-modal encoder outputs as cross-attention inputs, and decodes answer text with “[CLS]” as the start token. This decoder has the exact same architecture as the multi-modal encoder. We initialize its weight using the pre-trained multi-modal encoder.

SINGULARITY-temporal. Figure 8b shows an overview of the model architecture for temporal modeling, this model is also referred to as SINGU-LARITY-temporal. Given multiple video frames $\{ f _ { \tau _ { i } } \} _ { i = 1 } ^ { T _ { t r a i n } }$ as input, the model firstly encode each frame into their visual representations $\{ \mathcal { F } _ { v } ( f _ { \tau _ { i } } ) \}$ with the vision encoder $\mathcal { F } _ { v }$ , where $\mathcal { F } _ { v } ( f _ { \tau _ { i } } ) \in$ $\mathbb { R } ^ { L _ { v } \times D }$ . Next, we add temporal position encoding to each frame to indicate their temporal order. This temporal position encoding is learned from scratch and is initialized as zeros. For brevity, we omit this encoding in the formulation. These framelevel representations are concatenated together as input to the temporal encoder , and we feed temporal encoder outputs to the multi-modal encoder’s cross-attention layer for making a prediction p:

$$
\begin{array} { r } { \begin{array} { r } { p = \mathcal { H } ( \mathcal { F } _ { l } ( S ) , \mathcal { T } ( [ \mathcal { F } _ { v } ( f _ { \tau _ { 1 } } ) ; . . . ; \mathcal { F } _ { v } ( f _ { \tau _ { T _ { t r a i n } } } ) ] ) ) , } \\ { \Big | \mathrm { Q } , \mathsf { K } , \mathsf { V } \mathsf { f o r } \mathsf { s e l f - a t t } ; \quad \Big | } \\ { \mathrm { Q } \mathsf { f o r } \mathsf { c r o s s - a t t } } \end{array} } \end{array}
$$

where [; ] denotes concatenation, and $[ \mathcal { F } _ { v } ( f _ { \tau _ { 1 } } ) ; . . . ; \mathcal { F } _ { v } ( f _ { \tau _ { T _ { t r a i n } } } ) ] \quad \in \quad \mathbb { R } ^ { ( T _ { t r a i n } \times L _ { v } ) \times D }$ During inference, when $T _ { t e s t }$ frames are used as inputs to the model and $T _ { t e s t } > T _ { t r a i n } ,$ , we interpolate the temporal position encoding to allow for extended temporal length. This is similar to spatial position encoding interpolation in (Touvron et al., 2021).

Pre-Training Objectives. During pre-training, we optimize the model with three standard visionand-language objectives, Vision-Text Contrastive (VTC), Masked Language Modeling (MLM) (Devlin et al., 2019), and Vision-Text Matching. We explain them in detail below.

(i) Vision-Text Contrastive (VTC) loss aims to aligns paired vision and language embeddings. Given the encoded vision embedding $\mathcal { F } _ { v } ( f _ { i , t } )$ , we use a projection head (with pooling) $\phi _ { v }$ to project the embedding sequence into a vector representation $\phi _ { v } ( \mathcal { F } _ { v } ( f _ { i , t } ) ) \in \mathbb { R } ^ { D }$ . Here $f _ { i , t }$ is the t-th frame in the i-th video in the training set, and t is randomly sampled from all available frames in this video. For brevity, we omit the subscript t and use $f _ { i }$ to denote a randomly sampled frame from the ith video during the rest of the discussion. Similarly, we have $\phi _ { l } ( \bar { \mathcal { F } _ { l } } ( S _ { j } ) ) \in \mathbb { R } ^ { D }$ for the j-th sentence. The similarity score $s _ { i , j }$ of the video and text pair is defined as their dot product:

$$
s _ { i , j } = \phi _ { v } ( \mathcal { F } _ { v } ( f _ { i } ) ) ^ { T } \phi _ { l } ( \mathcal { F } _ { l } ( S _ { j } ) )\tag{5}
$$

We apply a contrastive loss to encourage the alignment between paired vision-language embeddings:

$$
p _ { i } ^ { v } = \frac { \exp ( s _ { i , i } / \tau ) } { \sum _ { j } \exp ( s _ { i , j } / \tau ) } , ~ p _ { i } ^ { l } = \frac { \exp ( s _ { i , i } / \tau ) } { \sum _ { j } \exp ( s _ { j , i } / \tau ) } ,
$$

$$
\mathcal { L } _ { v t c } = - \sum _ { i = 1 } ^ { n } ( \log p _ { i } ^ { v } + \log p _ { i } ^ { l } ) ,\tag{6}
$$

where τ is a learned temperature parameter, and it is initialized as 0.07 following CLIP (Radford et al., 2021). n is the total number of examples in the training set.

(ii) Masked Language Modeling (MLM) loss, or more precisely, Vision Conditioned Masked Language Modeling loss, aims to predict masked text tokens from their (masked) textual context as well as the visual context. This loss is applied at the last layer of the multi-modal encoder, and we follow the exact formulation in BERT (Devlin et al., 2019), except that we add additional vision inputs and use a higher mask ratio of 50%.

(iii) Vision-Text Matching (VTM) loss works towards the same goal as the VTC loss – encouraging the alignment between paired vision and language inputs. It uses the [CLS] output from the multi-modal encoder for binary classification – whether the input vision and language pair match or not. To make the training more effective, we also leverage hard negative sampling (Li et al., 2021b; Chen et al., 2020) to sample more informative negatives within the batch for VTM.

## A.2 Additional Experiments

Text-to-video retrieval fine-tuning details. For text-to-video retrieval fine-tuning, we use the same architecture as pre-training, except that MLM loss is removed. We use an initial learning rate of 1e-5 with cosine decay to 1e-6. We use a batch size of 32, and train the model for 5 epochs for MSRVTT, 10 epochs for DiDeMo and ActivityNet Captions. During training, we use a single frame per video. During testing, we use 12 frames per video for MSRVTT and DiDeMo, and 32 frames for ActivityNet Captions since it has longer videos. On a single A100, this fine-tuning takes around 1.5 hours for MSRVTT, 0.5 hours for ActivityNet Captions or DiDeMo.

Video QA fine-tuning details. For open-ended QA, we add an extra multi-modal decoder (initialized from pre-trained multi-modal encoder) that takes in multi-modal encoder outputs as crossattention inputs, and decodes answer text (details in Appendix). We use an initial learning rate of 1e-5, and warm up the learning rate in the first half epoch, followed by cosine decay to 1e-6. We use a batch size of 32, and train the model for 10 epochs. On a single A100, this fine-tuning takes around 4 hours for MSRVTT-QA, and 1 hour for ActivityNet-QA. We use a single frame per video for training, 12 frames for testing. For MSRVTT-MC, we follow (Lei et al., 2021) to use the model trained on MSRVTT retrieval, and select the option with the highest score as prediction.

For all downstream tasks, we use the same input image size 224 224 and image augmentations as in pre-training. During inference, we resize the input video frames to 224 224.

Analysis Setup. For all ablation studies, we report results on validation splits for the datasets if available. For example, we use validation splits for DiDeMo retrieval and ActivityNet-QA, and we use the test split for MSRVTT retrieval, val1 split for ActivityNet Captions retrieval, and test split for SSv2-label. For retrieval tasks, we use the average recall, which is the average score of R@{1,5,10}) to more holistically compare the model performance. For QA tasks, we use accuracy.

SINGULARITY-temporal Results on Existing Datasets. In Table 4 and Table 5 we show results of SINGULARITY-temporal on existing textto-video retrieval and video question answering datasets. In general, the 4-frame model SINGULAR-ITY-temporal improves upon the 1-frame model SINGULARITY, but the performance gap is relatively small, especially considering the greatly increased memory and computation cost (discussed in main text) of using 4 frames.

Zero-Shot Results. In Table 6 we show zeroshot results of SINGULARITY for text-to-video retrieval. SINGULARITY achieves significantly better results compared to existing methods with a similar amount of pre-training data.

Performance of Multiple Runs. In Table 7 we show mean and standard deviation of 5 random runs, for text-to-video retrieval.

Comparison on Inference Cost. In Table 8, we compare the cost of various retrieval methods in terms of inference GFLOPs and the number of model parameters. Overall, SINGULARITY models have a similar amount of parameters and lower inference GFLOPs, with higher performance.

Memory and Time Cost of Frame Ensemble Strategies. In Section 5, we discussed that our simple early fusion based frame ensemble strategy (concat) achieves the best performance for both MSRVTT retrieval and ActivityNet-QA tasks across different number of inference frames. In this section, we continue to compare its memory and computation time cost w.r.t. other frame ensemble strategies. Results are shown in Figure 9. For both tasks, our early fusion strategy (concat) achieves better performance than late fusion strategies (lse, max, mean) while also runs faster. For memory cost, concat uses more memory for MSRVTT retrieval, but fewer memory for the ANet-QA. Overall, the early fusion approach is preferred in most cases due to its better accuracy and faster run time.

Ablation Study on Training Objectives. In Table 9, we study the effect of using different training objectives. We notice that using all objectives achieves the best performance. One interesting note is that, compared to (ITM+MLM), adding ITC loss (ITM+MLM+ITC) greatly improves MSRVTT retrieval performance, but not ActivityNet QA. This may because ITC is not applied on the multi-modal encoder which QA tasks may heavily rely on.

Table 4: SINGULARITY-temporal results on text-to-video retrieval.
<table><tr><td rowspan="2">Method</td><td rowspan="2">#PT</td><td rowspan="2">#Train</td><td colspan="3">MSRVTT</td><td colspan="3">DiDeMo</td><td colspan="3">ActivityNet Cap</td></tr><tr><td>Frame</td><td>R1</td><td>R5</td><td>R10 R1</td><td>R5</td><td></td><td>R10</td><td>R1 R5</td><td>R10</td></tr><tr><td>HERO (Li et al., 2020a)</td><td>136M</td><td>310</td><td>20.5</td><td>47.6</td><td>60.9</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMT (Gabeur et al., 2020)</td><td>136M</td><td>1K/-/3K</td><td>26.6</td><td>57.1</td><td>69.6</td><td></td><td></td><td></td><td></td><td></td><td>28.7 61.4 94.5</td></tr><tr><td>ClipBERT (Lei et al., 2021)</td><td>0.2M</td><td>16/16/8</td><td>22.0</td><td>46.8</td><td>59.9</td><td>20.4 48.0 60.8</td><td></td><td></td><td>21.3</td><td>49.0 63.5</td><td></td></tr><tr><td>VideoCLIP (Xu et al., 2021)</td><td>136M</td><td>960</td><td>30.9</td><td>55.4</td><td>66.8</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Frozen (Bain et al., 2021)</td><td>5M</td><td>4</td><td>31.0</td><td>59.5</td><td>70.5</td><td></td><td>31.0 59.8</td><td>72.4</td><td></td><td></td><td></td></tr><tr><td>AlignPrompt (Li et al., 2021a)</td><td>5M</td><td>8</td><td>33.9</td><td>60.7</td><td>73.2</td><td>35.9</td><td>67.5</td><td>78.8</td><td></td><td></td><td></td></tr><tr><td>CLIP4Clip (Luo et al., 2021)</td><td>400M</td><td>12/64/64</td><td>42.0</td><td>68.6</td><td>78.7</td><td>42.8</td><td>68.5</td><td>79.2</td><td></td><td></td><td>40.5 72.4 98.2</td></tr><tr><td>SINGULARITY</td><td>5M</td><td>1</td><td>36.8</td><td>65.9</td><td>75.5</td><td>47.4</td><td>75.2</td><td>84.0</td><td>43.0</td><td>70.6</td><td>81.3</td></tr><tr><td>SINGULARITY-temporal</td><td>5M</td><td>4</td><td>39.9</td><td>67.3</td><td>76.0</td><td>49.2</td><td>77.5</td><td>85.4</td><td>45.9</td><td>73.3</td><td>83.8</td></tr><tr><td>SINGULARITY</td><td>17M</td><td>1</td><td>41.5</td><td>68.7</td><td>77</td><td>53.9</td><td>79.4</td><td>86.9</td><td>47.1</td><td>75.5</td><td>85.5</td></tr><tr><td>SINGULARITY-temporal</td><td>17M</td><td>4</td><td>42.7</td><td>69.5</td><td>78.1</td><td>53.1</td><td>79.9</td><td>88.1</td><td>48.9</td><td></td><td>77.086.3</td></tr></table>

Table 5: SINGULARITY-temporal results on video question answering.
<table><tr><td>Method #PT</td><td colspan="5">#Train Frame MSRVTT-QA ActivityNet-QA MSRVTT-MC</td></tr><tr><td>ClipBERT (Lei et al., 2021)</td><td>0.2M</td><td>16</td><td>37.4</td><td>一</td><td>88.2</td></tr><tr><td>AlignPrompt (Li et al., 2021a)</td><td>5M</td><td>16</td><td>42.1</td><td></td><td></td></tr><tr><td>JustAsk (Yang et al., 2021)</td><td>69M</td><td>640</td><td>41.5</td><td>38.9</td><td></td></tr><tr><td>MERLOT (Zellers et al., 2021)</td><td>180M</td><td>5</td><td>43.1</td><td>41.4</td><td>90.9</td></tr><tr><td>VideoCLIP (Xu et al., 2021)</td><td>136M</td><td>960</td><td></td><td></td><td>92.1</td></tr><tr><td>SINGULARITY</td><td>5M</td><td>1</td><td>42.7</td><td>41.8</td><td>92.0</td></tr><tr><td>SINGULARITY-temporal</td><td>5M</td><td>4</td><td>43.3</td><td>43.4</td><td>92.0</td></tr><tr><td>SINGULARITY</td><td>17M</td><td>1</td><td>43.5</td><td>43.1</td><td>92.1</td></tr><tr><td>SINGULARITY-temporal</td><td>17M</td><td>4</td><td>43.9</td><td>44.1</td><td>93.7</td></tr></table>

Impact of Image Size. In Figure 10 we study the impact of image size for downstream tasks. In general, a larger image size helps improve model performance, but the performance saturates at a certain size, e.g., the model performance saturates at around 336 336 for the 3 tasks. Note that our model performance with larger image sizes might suffer from the low resolution of the raw videos we have. For example, we are only able to get videos of resolution 320 240 for MSRVTT.

Comparison on Image-Text tasks. Since our model is pre-trained with single frames, it can be directly used for image-text tasks. In Table 11 we show image-text retrieval results on Flickr30K (Young et al., 2014) and COCO (Chen et al., 2015). In Table 12 we show image question answering results on VQA (Antol et al., 2015). We observe that SINGULARITY demonstrates competitive performance on the image-text tasks. As we still see a gap with state-of-the-art image-text models such as (Li et al., 2022), one future direction is to adopt improved designs in these methods to further improve video-text task performance.

Hyper-Parameters. The hyper-parameters for our pre-training and downstream task fine-tuning are listed in Table 13 and Table 14. Note that we did not do an extensive hyper-parameter search, but mostly use the same hyper-parameters for different datasets under the same task, it is possible that better results can be achieved with more tuning.

## A.3 Additional Data Details

Statistics. We show statistics of pre-training datasets in Table 15, and downstream datasets in Table 16.

License. We show dataset licenses in Table 17.

Table 6: SINGULARITY zero-shot results on text-to-video retrieval.
<table><tr><td rowspan="2">Method</td><td rowspan="2">#PT</td><td rowspan="2">#Train</td><td colspan="3">MSRVTT</td><td colspan="3">DiDeMo</td><td colspan="3">ActivityNet Cap</td></tr><tr><td>Frame R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td></tr><tr><td>VideoCLIP (Xu et al., 2021)</td><td>137M</td><td>1K</td><td>10.4</td><td>22.2</td><td>30.0</td><td>16.6</td><td>46.9</td><td></td><td>一</td><td>1</td><td>一</td></tr><tr><td>Frozen (Bain et al., 2021)</td><td>5M</td><td>4</td><td>18.7</td><td>39.5</td><td>51.6</td><td>21.1</td><td>46.0</td><td>56.2</td><td>1</td><td>一</td><td>1</td></tr><tr><td>AlignPrompt (Li et al., 2021a)</td><td>5M</td><td>8</td><td>24.1</td><td>44.7</td><td>55.4</td><td>23.8</td><td>47.3</td><td>57.9</td><td></td><td></td><td></td></tr><tr><td>CLIP-straight</td><td>400M</td><td>1</td><td>31.2</td><td>53.7</td><td>64.2</td><td></td><td></td><td></td><td>一</td><td>=</td><td>一</td></tr><tr><td>BLIP</td><td>130M</td><td>1</td><td>43.3</td><td>65.6</td><td>74.7</td><td></td><td>一</td><td>1</td><td>一</td><td>I</td><td>一</td></tr><tr><td>SINGULARITY</td><td>5M</td><td>1</td><td>28.4</td><td>50.2</td><td>59.5</td><td>36.9</td><td>61.1</td><td>69.3</td><td>30.8</td><td>55.9</td><td>66.3</td></tr><tr><td>SINGULARITY</td><td>17M</td><td>1</td><td>34.0</td><td>56.7</td><td>66.7</td><td>37.1</td><td>61.7</td><td>69.9</td><td>30.6</td><td>55.6</td><td>66.9</td></tr></table>

Table 7: SINGULARITY results on text-to-video retrieval, with mean/std over 5 random runs. We show the results for the model pre-trained on the 17M corpus.
<table><tr><td rowspan="2">Method</td><td colspan="3">MSRVTT</td><td colspan="3">DiDeMo</td><td colspan="3">ActivityNet</td></tr><tr><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td></tr><tr><td>SINGULARITY</td><td> $4 2 . 1 \pm 0 . 5$ </td><td>69.3±0.4 78.1±0.7 53.3±1.0 78.7±1.3 86.3±1.5</td><td></td><td></td><td></td><td></td><td> $4 7 . 0 { \scriptstyle \pm 0 . 5 }$ </td><td></td><td>75.7±0.3 85.3±0.3</td></tr></table>

![](images/367cbd8147a81cf2f319f905a3eddfff57c6a378e1c29a4c78d897197e8fb518.jpg)

![](images/e124777cc995c05f94557066cf916456be532b378edffdddc9900d452829ed21.jpg)

![](images/9067631cdaf58e3b28a1acec49e7dc93957155d1b6818b2b1141284779d899f5.jpg)

![](images/69399198ff296a4a1eef1c23cc1df4fd2b902eb38b7ba1e2b7bca2cd422a28cf.jpg)

![](images/0ded7e76f09683b41e0a291b34dfe9113f1d0ab4c725d6f4daa296be8b0c4c9a.jpg)

![](images/bc6d09a12b65b2fd83ba4168562d4ecb9af8cc0338b647960ef62162d6be754d.jpg)  
Figure 9: Impact of frame ensemble strategy. Retrieval performance is shown as avg recall, i.e., average of R@{1,5,10}. Top row shows the performance, time and memory comparisons for MSRVTT retrieval task, while bottom shows the same comparisons for ActivityNet-QA (ANet-QA). We use the same fine-tuned checkpoint for each task, thus the results difference only comes from inference strategies. We measure time and memory cost by running the models on the task-specific test splits. Since the three late fusion strategies (lse, max, mean) have similar memory and time costs, we only keep lse in the figures.

Table 8: Comparison of recent retrieval methods on inference GLOPs and #params. For brevity, we show DiDeMo retrieval performance with Average Recall (AvgR) – the average of R{1,5,10}.
<table><tr><td>Method</td><td>#PT</td><td>Inference GFLOPs #params DiDeMo AvgR</td><td></td><td></td></tr><tr><td>Frozen (Bain et al., 2021)</td><td>5M</td><td>542</td><td>181M</td><td>54.4</td></tr><tr><td>AlignPrompt (Li et al., 2021a)</td><td>5M</td><td></td><td>231M</td><td>60.7</td></tr><tr><td>CLIP4Clip (Radford et al., 2021) 400M</td><td></td><td>1,121</td><td>164M</td><td>63.5</td></tr><tr><td>SINGULARITY</td><td>5M</td><td>451</td><td>202M</td><td>68.9</td></tr><tr><td>SINGULARITY-temporal</td><td>5M</td><td>485</td><td>209M</td><td>70.7</td></tr></table>

Table 9: Ablation study on training objectives. The models are pre-trained on 2.5M WebVid video-text pairs for 10 epochs and are then fine-tuned.
<table><tr><td>Objectives</td><td>MSRVTT Retrieval AvgR ActivityNet-QA</td></tr><tr><td>ITM</td><td>32.4 40.2</td></tr><tr><td>ITM + MLM</td><td>52.5 47.0</td></tr><tr><td> $\mathrm { I T M } + \mathrm { I T C }$ </td><td>54.3 44.1</td></tr><tr><td> $\mathrm { I T M } + \mathrm { M L M } + \mathrm { I T C }$ </td><td>55.7 46.4</td></tr></table>

Table 10: Impact of Image Size. We fine-tune models from the same checkpoint, pre-trained with input image size 224 224. We show average recall (average of R@{1,5,10}) for retrieval tasks, and accuracy for the QA task.
<table><tr><td>Image size</td><td>MSRVTT retrieval</td><td>DiDeMo retrieval</td><td>ActivityNet QA</td></tr><tr><td>112</td><td>58.7</td><td>65.9</td><td>46.6</td></tr><tr><td>224</td><td>62.4</td><td>73.4</td><td>49.2</td></tr><tr><td>336</td><td>65.5</td><td>73.4</td><td>49.6</td></tr><tr><td>448</td><td>64.2</td><td>72.9</td><td>49.8</td></tr></table>

Table 11: Comparison to existing methods on image-text retrieval. We show results for both text retrieval (image-totext retrieval, TR) and image retrieval (IR).
<table><tr><td rowspan="3">Method</td><td rowspan="3">#PT</td><td colspan="4">COCO (5K test)</td><td colspan="5">Flickr30K (1K test)</td></tr><tr><td colspan="2">TR</td><td colspan="2">IR</td><td colspan="2">TR</td><td colspan="2"></td><td colspan="2">IR</td></tr><tr><td></td><td>R1</td><td>R5 R10</td><td>R1 R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5 R10</td></tr><tr><td>ViLT (Kim et al., 2021)</td><td></td><td>4M 61.5</td><td>86.3</td><td>92.7 42.7</td><td>72.9 83.1</td><td>83.5</td><td>96.7</td><td>98.6</td><td>64.4</td><td>88.7</td><td></td></tr><tr><td>UNITER (Chen et al., 2020)</td><td></td><td>4M 65.7</td><td>88.6 93.8</td><td>352.9 79.9</td><td>88.0</td><td>87.3</td><td>98.0</td><td>99.2</td><td>75.6</td><td>94.1 96.8</td><td>93.8</td></tr><tr><td>OSCAR (Li et al., 2020b)</td><td></td><td></td><td></td><td>4M 70.0 91.1 95.5 54.0 80.8 88.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Frozen (Bain et al., 2021)</td><td>5M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>61.0 87.5 92.7</td><td></td></tr><tr><td>ALBEF (Li et al., 2021b)</td><td></td><td></td><td></td><td>4M 73.1 91.4 96.0 56.8 81.5 89.2</td><td></td><td>94.3</td><td>99.4</td><td>99.8</td><td></td><td>82.8</td><td>96.7 98.4</td></tr><tr><td>ALBEF (Li et al., 2021b)</td><td></td><td></td><td></td><td>14M 77.6 94.3 97.2 60.7 84.3</td><td></td><td>90.5</td><td>95.9 99.8</td><td></td><td>100.0</td><td>85.6</td><td>97.5 98.9</td></tr><tr><td>BLIP (Li et al., 2022)</td><td></td><td></td><td></td><td>14M 80.6 95.2 97.6 63.1 85.3</td><td></td><td>91.1 96.6 99.8</td><td></td><td></td><td>100.0 87.2</td><td></td><td>97.5 98.8</td></tr><tr><td>BLIP (Li et al., 2022)</td><td></td><td></td><td></td><td>129M 81.9 95.4 97.8 64.3 85.7 91.5 97.3 99.9</td><td></td><td></td><td></td><td></td><td>100.0 87.3</td><td></td><td>97.6 98.9</td></tr><tr><td>ALIGN (Jia et al., 2021)</td><td>1.2B</td><td>77.0</td><td>93.5</td><td>96.9 59.9</td><td>83.3</td><td>89.8 95.3</td><td>99.8</td><td></td><td>100.0</td><td>84.9</td><td>97.4 98.6</td></tr><tr><td>SINGULARITY</td><td></td><td>5M 71.9</td><td>90.8</td><td>95.4 54.6</td><td>80.0</td><td>87.8</td><td>93.3 99.4</td><td></td><td>99.8</td><td>81.4 95.8</td><td>97.9</td></tr><tr><td>SINGULARITY</td><td></td><td></td><td></td><td>17M 77.0 93.7 96.8 59.6 83.4 90.0 96.1 </td><td></td><td></td><td></td><td>99.8</td><td>99.9</td><td>84.7 96.8 98.3</td><td></td></tr></table>

Table 12: Comparison to existing methods on VQA.
<table><tr><td>Method</td><td>#PT</td><td>test-dev</td><td>test-std</td></tr><tr><td>ClipBERT (Lei et al., 2021) ViLT (Kim et al., 2021)</td><td>0.2M</td><td>69.08</td><td>69.43</td></tr><tr><td rowspan="4">VL-BART (Cho et al., 2021) LXMERT (Tan and Bansal, 2019) UNITER (Chen et al., 2020)</td><td>4M</td><td>70.94</td><td></td></tr><tr><td>0.2M 4M</td><td></td><td>71.30 72.54</td></tr><tr><td>4M</td><td>72.42 72.70</td><td>72.91</td></tr><tr><td>4M</td><td>73.79</td><td>74.02</td></tr><tr><td rowspan="4">OSCAR (Li et al., 2020b) ALBEF (Li et al., 2021b) ALBEF (Li et al., 2021b)</td><td>4M</td><td>73.16</td><td>73.44</td></tr><tr><td>4M</td><td>74.54</td><td>74.70</td></tr><tr><td>14M</td><td>75.84</td><td>76.04</td></tr><tr><td>14M</td><td>77.54</td><td>77.62</td></tr><tr><td>BLIP (Li et al., 2022) BLIP (Li et al., 2022)</td><td>129M</td><td>78.24</td><td>78.17</td></tr><tr><td>SINGULARITY</td><td>5M</td><td>70.30</td><td>70.53</td></tr><tr><td>SINGULARITY</td><td>17M</td><td>73.13</td><td>73.27</td></tr></table>

Table 13: SINGULARITY hyper-parameters for pre-training, video QA, image QA and text-to-image retrieval. We only list a single value if all tasks share the same value. For SINGULARITY-temporal, we train with a similar setup, except that we set #training frames to be 4. In addition, for SINGULARITY-temporal 2nd stage pre-training, we also use a smaller batch size of 32 per GPU.
<table><tr><td>config</td><td></td><td>pre-training video QA image QA</td><td></td><td>text-to-image retrieval</td></tr><tr><td>optimizer</td><td colspan="4">AdamW (Loshchilov and Hutter, 2019)  $\beta _ { 1 } , \beta _ { 2 } { = } 0 . 9 , 0 . 9 9 9$ </td></tr><tr><td>optimizer momentum</td><td colspan="4"></td></tr><tr><td>base learning rate</td><td>1e-4</td><td>1e-5</td><td>1e-5</td><td>1e-5</td></tr><tr><td>min learning rate</td><td>1e-5</td><td>1e-6</td><td>1e-6</td><td>1e-6</td></tr><tr><td>weight decay</td><td colspan="4"></td></tr><tr><td>learning rate schedule</td><td colspan="4">cosine decay (Loshchilov and Hutter, 2017)</td></tr><tr><td>image size</td><td>224</td><td>224</td><td>336</td><td>336</td></tr><tr><td>image augmentation</td><td colspan="4">random resize, crop, horizontal flip</td></tr><tr><td>#training epochs</td><td>10</td><td>10</td><td>5</td><td>10 (Flickr30K), 5 (COCO)</td></tr><tr><td>#warmup epochs</td><td>1</td><td>0.5</td><td>0.5</td><td>0</td></tr><tr><td>batch size x #GPUs</td><td>128×3</td><td>32×1</td><td>64×4</td><td>64×2</td></tr><tr><td>#training frames</td><td colspan="4"></td></tr><tr><td>#inference frames</td><td></td><td>12</td><td>1 1</td><td>1</td></tr></table>

Table 14: SINGULARITY hyper-parameters for text-to-video retrieval tasks. We only list a single value if all tasks share the same value. For SINGULARITY-temporal, we train it with a similar setup, except that we set #training frames to be 4.
<table><tr><td>config</td><td>MSRVTT</td><td>DiDeMo ActivityNet Captions SSv2-template/label</td></tr><tr><td>optimizer</td><td colspan="3">AdamW (Loshchilov and Hutter, 2019)</td></tr><tr><td>optimizer momentum</td><td></td><td>β1, β2=0.9,0.999</td><td></td></tr><tr><td>base learning rate</td><td>1e-5 1e-5</td><td>1e-5</td><td>1e-4</td></tr><tr><td>min learning rate</td><td>1e-6</td><td>1e-6</td><td>1e-6 1e-5</td></tr><tr><td>weight decay</td><td></td><td>0.02</td><td></td></tr><tr><td>learning rate schedule</td><td></td><td>cosine decay (Loshchilov and Hutter, 2017)</td><td></td></tr><tr><td>image size</td><td></td><td>224</td><td></td></tr><tr><td>image augmentation</td><td></td><td>random resize, crop, horizontal flip</td><td></td></tr><tr><td>#training epochs</td><td>5</td><td>10 10</td><td>10</td></tr><tr><td>#warmup epochs</td><td></td><td>0</td><td></td></tr><tr><td>batch size x #GPUs</td><td>32x1</td><td>32x1 32x1</td><td>32x2</td></tr><tr><td>#training frames</td><td></td><td>1</td><td></td></tr><tr><td>#inference frames</td><td>12</td><td>12 32</td><td>12</td></tr></table>

Table 15: Statistics of pre-training datasets. The average video length of WebVid is 18 seconds.

<table><tr><td>Dataset</td><td>#image/video</td><td>#text</td><td>Type</td></tr><tr><td>COCO (Chen et al., 2015)</td><td>113K</td><td>567K</td><td>image</td></tr><tr><td>VG (Krishna et al., 2017b)</td><td>100K</td><td>768K</td><td>image</td></tr><tr><td>SBU (Ordonez et al., 2011)</td><td>860K</td><td>860K</td><td>image</td></tr><tr><td>CC3M (Sharma et al., 2018)</td><td>2.95M</td><td>2.95M</td><td>image</td></tr><tr><td>CC12M (Changpinyo et al., 2021)</td><td>10.77M</td><td>10.77M</td><td>image</td></tr><tr><td>WebVid (Bain et al., 2021)</td><td>2.49M</td><td>2.49M</td><td>video</td></tr><tr><td>5M corpus = CC3M+WebVid</td><td>5.44M</td><td>5.44M</td><td>video+image</td></tr><tr><td>17M corpus = 5M+COCO+VG+SBU+CC12M</td><td>17.28M</td><td>18.41M</td><td>video+image</td></tr></table>

Table 16: Statistics of downstream datasets.
<table><tr><td rowspan="2">Dataset</td><td colspan="3">#video</td><td colspan="3">#text</td><td rowspan="2">Avg Video</td></tr><tr><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test Length (s)</td></tr><tr><td>Text-to-Video Retrieval</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ActivityNet Cap (Krishna et al., 2017a)</td><td>10,009</td><td></td><td>- 4,917</td><td>10,009</td><td></td><td>4,917</td><td>180</td></tr><tr><td>DiDeMo (Anne Hendricks et al., 2017)</td><td></td><td>8,394 1,065 1,003</td><td></td><td>8,394</td><td>1,065</td><td>1,003</td><td>29.3</td></tr><tr><td>MSRVTT (Xu et al., 2016)</td><td>7,010</td><td></td><td>1,000</td><td>140,200</td><td></td><td>1,000</td><td>15</td></tr><tr><td>SSV2-Template (Goyal et al., 2017a)</td><td>168,913</td><td></td><td>- 2,088</td><td>174</td><td></td><td>174</td><td>4</td></tr><tr><td>SSV2-Label (Goyal et al., 2017a)</td><td>168,913</td><td></td><td>- 2,088</td><td>109,968</td><td></td><td>1,989</td><td>4</td></tr><tr><td>Video Question Answering</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MSRVTT-QA (Xu et al., 2017)</td><td>6,513</td><td></td><td></td><td>4972,990158,58112,27872,821</td><td></td><td></td><td>15</td></tr><tr><td>ActivityNet-QA (Yu et al., 2019)</td><td>3,2001,800</td><td></td><td>800</td><td>32,00018,000</td><td></td><td>8,000</td><td>180</td></tr><tr><td>MSRVTT-MC (Yu et al., 2018)</td><td>7,010</td><td></td><td></td><td>- 2,990 140,200</td><td></td><td>14,950</td><td>15</td></tr></table>

Table 17: Dataset licenses.
<table><tr><td>Dataset</td><td>License</td></tr><tr><td>COCO (Chen et al., 2015)</td><td>CC BY 4.0, Flickr Terms of Use</td></tr><tr><td>VG (Krishna et al., 2017b)</td><td>CC BY 4.0</td></tr><tr><td>SBU (Ordonez et al., 2011)</td><td>Flickr Terms of Use</td></tr><tr><td>CC3M (Sharma et al., 2018)</td><td>CC3M License</td></tr><tr><td>CC12M (Changpinyo et al., 2021)</td><td>CC12M License</td></tr><tr><td>WebVid (Bain et al., 2021)</td><td>Exceptions to Copyright</td></tr><tr><td>ActivityNet Captions (Krishna et al., 2017a)</td><td>Fair Use</td></tr><tr><td>DiDeMo (Anne Hendricks et al., 2017)</td><td>BSD-2-Clause, Creative Commons</td></tr><tr><td>MSRVTT (Xu et al., 2016)</td><td>unknown</td></tr><tr><td>SSV2-Template (Goyal et al., 2017a)</td><td>SSv2 License</td></tr><tr><td>SSV2-Label (Goyal et al., 2017a)</td><td>SSv2 License</td></tr><tr><td>MSRVTT-QA (Xu et al., 2017)</td><td>MIT</td></tr><tr><td>ActivityNet-QA (Yu et al., 2019)</td><td>Apache</td></tr><tr><td>MSRVTT-MC (Yu et al., 2018)</td><td>unknown</td></tr></table>

## ACL 2023 Responsible NLP Checklist

A For every submission: <sup>✓</sup> A1. Did you describe the limitations of your work? A dedicated ‘Limitation’ section after Section 6.

<sup>✓</sup> A2. Did you discuss any potential risks of your work? A dedicated ‘Ethics Statement’ section after Section 6.

<sup>✓</sup> A3. Do the abstract and introduction summarize the paper’s main claims? Abstract and Section 1.

✗ A4. Have you used AI writing assistants when working on this paper? Left blank.

## B <sup>✓</sup> Did you use or create scientific artifacts?

Sections 3 and 4.

<sup>✓</sup> B1. Did you cite the creators of artifacts you used? Sections 3 and 4.

<sup>✓</sup> B2. Did you discuss the license or terms for use and / or distribution of any artifacts? Appendix Section A.3

<sup>✓</sup> B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? For the artifacts you create, do you specify intended use and whether that is compatible with the original access conditions (in particular, derivatives of data accessed for research purposes should not be used outside of research contexts)? Section 4.

 B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it? Not applicable. Left blank.

 B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? Not applicable. Left blank.

<sup>✓</sup> B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? Even for commonly-used benchmark datasets, include the number of examples in train / validation / test splits, as these provide necessary context for a reader to understand experimental results. For example, small differences in accuracy on large test sets may be significant, while on small test sets they may not be. Sections 3, 4 and Appendix Section A.3

## C <sup>✓</sup> Did you run computational experiments?

Sections 4 and 5.

<sup>✓</sup> C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? Section 3 and Appendix Section A.2

<sup>✓</sup> C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? Section 3, and Appendix Sections A.2 and A.3

<sup>✓</sup> C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run? Appendix Section A.2

<sup>✓</sup> C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation), did you report the implementation, model, and parameter settings used (e.g., NLTK, Spacy, ROUGE, etc.)? Section 3

D ✗ Did you use human annotators (e.g., crowdworkers) or research with human participants? Left blank.

 D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.? No response.

 D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participants’ demographic (e.g., country of residence)? No response.

 D3. Did you discuss whether and how consent was obtained from people whose data you’re using/curating? For example, if you collected data via crowdsourcing, did your instructions to crowdworkers explain how the data would be used? No response.

 D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? No response.

 D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? No response.