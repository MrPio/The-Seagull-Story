# The Seagull Story
A BERT-like transformer is used to solve an NLI task. The model is asked yes/no questions and has to answer "yes," "no," or "irrelevant" based on a given story.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8577c40e-a4be-4ced-ba8f-7601d56fa77c" width="275rem"/>
  </p>

Based on the idea and data set of [SeagullStory](https://github.com/manuu1311/SeagullStory) by [@manuu1311](https://github.com/manuu1311)

In this project, we aim to fine-tune a BERT-like transformer to solve a Natural Language Inference task. Based on a given story, the model is asked `yes`/`no` questions and has to answer `yes`, `no`, or `irrelevant`.

## The NLI task
*Natural Language Inference*, also known as *textual entailment*, is the task of determining whether a *hypothesis* is true (**entailment**), false (**contradiction**), or undetermined (**neutral**) given a *premise*.

In more precise terms, the *premise* $p$ entails the *hypothesis* $h$ ($p \implies h$) if and only if, typically, a human reading $p$ would be justified in inferring the proposition expressed by $h$ from the proposition expressed by $p$. This is a more relaxed definition than the pure *logical entailment*. The **relation is directional** because even if $p \implies h$, the reverse $h \implies p$ is much less certain.

Determining whether this relationship holds is an informal task, one which sometimes overlaps with the formal tasks of formal semantics. In this context, however, we want to tackle this task **using a BERT-like transformer**, which is now state-of-the-art for all kinds of NLP tasks.

Read more on [Textual entailment][1].

[1]:https://en.wikipedia.org/wiki/Textual_entailment

## BERT
Due to their innovative architectures, the BERT and GPT models have come to the fore when it comes to solving natural language tasks. Both are based on the transformer architecture, which is based on the attention mechanism, discovered by Google in 2017 and published in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.

### The BERT's pre-training phase
Unlike GPT, BERT is trained both to predict missing words in text (*Masked Language Modelling*), using both left and right contexts, and to recognize when two sentences in the same corpus follow each other (*Next Sentence Prediction*). 

* In the first task, each token in the input sequence is selected with a probability of $15\%$, and each selected token is replaced with the special `[MASK]` token with a probability of $80\%$, with a random token with a probability of $10\%$, and left unchanged with a probability of $10\%$. Replacement with random tokens, with a uniform probability distribution across the dictionary, is one way to fight *dataset shift*, a problem that occurs when the distribution of tokens differs greatly from training to induction. Finally, the output vector at the position of the `[MASK]` token is fed into a simple classification head, namely a feed-forward network, and then a softmax function is applied to transform the logits into a valid probability distribution over the tokens of the dictionary.
* In the second task, the transformer is fed with a vector structured as follows:
  
$$\text{\textbf{input}} = [\text{\texttt{[CLS]}} | \text{\textbf{sentence}}_1 | \texttt{[SEP]} | \text{\textbf{sentence}}_2 | \text{\texttt{[SEP]}}]$$
  
The output vector at the position of the `[CLS]` special token is processed by a binary classifier, again, a feed-forward network, to answer with a probability distribution over the classes `IsNext` and `NotNext`.

### BERT is encoder-only and bidirectional
It's interesting to note that BERT is an **encoder-only architecture**, meaning that, unlike GPT or a vanilla transformer in general, it lacks the ability to decode a vector from the dense latent space back into the vocabulary domain. Instead, BERT focuses on understanding the meaning of the input text. In the two training scenarios described above, **the classification head can be thought of as a simple one-stage decoder**, but when fine-tuning the model for a specific task, it's usually the case to remove it and add a new one based on the needs of the context.

The pre-training process implies that **BERT is a bidirectional model**, and thus is superior for tasks that require understanding the context and nuances of language. In other words, BERT is better suited to tackle our task.

### The BERT embedding strategy
Downstream of the *WordPiece* tokenizer, which uses a dictionary of $30.000$ tokens organized with a subword strategy, BERT embeds the tokens in a peculiar way.

$$E(\vec{v})=\text{LayerNorm}(\text{TokenType}(\vec{v})+\text{Position}(\vec{v})+\text{SegmentType}(\vec{v})) $$

* The $\text{TokenType}$ is the classic embedding. The one-hot encoded $(30.000\times 1)$ token is translated to a lower dimensional dense space, resulting in a smaller $(768\times 1)$ vector, where each dimension has a specific semantic meaning, but still ignores the context of the sentence.
* The $\text{Position}$ holds the information of the position of the token inside the sentence. For this purpose, $sin$ and $cos$ functions are used because they are continuous and differentiable and the relationship $sin(a+b)=f(sin(a),sin(b),cos(a),cos(b))$ allows to infer the relative position of two distinct tokens. More precisely, the value of the j-th dimension of the i-th embedded position:

$$
\mathrm{Position}(i)_j = \begin{cases}
    sin\left( \dfrac{i}{10.000^{j/768}} \right) & \text{if } i \text{ is even} \\
    cos\left( \dfrac{i}{10.000^{j/768}} \right)  & \text{if } i \text{ is odd.}
\end{cases}
$$

  This means that large dimensions encode large positional differences, while smaller dimensions encode finer positional differences.
* The $\text{SegmentType}$ is used for the *Next Sentence Prediction* training. It simply assigns a binary label based on which sentence the token belongs to. Essentially it is $\vec{0}$ if the token comes before the `[SEP]` special token, or $\vec{1}$ if it comes after it.

Read more on [Differences Between GPT and BERT][2] and [BERT Model – NLP][3].

## The choice of a BERT-like model
Over the years, several papers have been published proposing deviations from the original BERT model, both in the *pre-training process* and in the *model architecture*.

### Base models
The most downloaded models on [*HuggingFace*](https://huggingface.co/) are: 

* Facebook devised *RoBERTa*, which increased by ten times the size of the training set and introduced eight times larger mini-batches. This results often with better performance than the original BERT. *RoBERTa* removed the *Next Sentence Prediction* loss from the pre-training phase. Empirically, it was proven that this slightly improves downstream task performance.
* *ALBERT* is a light version of BERT optimized for tasks with limited computational resources. Its main aim is to reduce the number of parameters, i.e. the complexity of the model, in favour of faster fine-tuning and inference. For this purpose, *ALBERT* adds the *cross-layer parameter sharing* and *reduction*, a factorizing technique. While the former is rather intuitive, the latter reduces the embedding size to $128$ while leacing the size of the hidden layers unchanged.
* *ELECTRA* introduces *Replace Token Detection Technique* as a replacement for the BERT's *Masked Language Modelling* pre-training task. Instead of using the `[MASK]` special token, the tokens are replaced by alternative samples. This is done using a *generator-discriminator* configuration.
* Much like *ALBERT*, *DistilBERT* aims to reduce the model's complexity, but it does so with a different approach. *DistilBERT* implements *knowledge distillation* using a *teacher-student* framework. It goes without saying that *DistilBERT* sacrifices some accuracy.

Read more on  [BERT Variants and their Differences][4].

### Specialized models
All the BERT-like models discussed so far, has a lot in common with the original BERT model. In 2020, Microsoft released the paper [*DeBERTa: Decoding-enhanced BERT with Disentangled Attention*](https://arxiv.org/abs/2006.03654), introducing a BERT-like model with rather groundbreaking news.

*DeBERTa* improves *RoBERTa* with *disentangled attention* and an *enhanced mask decoder*. *Disentangled attention* differs from the classical *self-attention* in that the *content* of a token is evaluated along with its *relative position*, rather than summing its embedding into a single aggregated vector, as BERT does. This implies that *DeBERTa* uses two vectors to encode each token.

Moreover, since its 3rd version, *DeBERTa* has adopted the *ELECTRA* pre-training style.

Given the superior performance of *DeBERTa* over the other BERT-like models, we chose it to solve our *NLI* problem.

Read more on  [BERT Variants and their Differences][4].

[2]:https://www.geeksforgeeks.org/differences-between-gpt-and-bert/
[3]:https://www.geeksforgeeks.org/explanation-of-bert-model-nlp
[4]:https://360digitmg.com/blog/bert-variants-and-their-differences
