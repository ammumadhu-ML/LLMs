
# learning LLM

- [the Illustrated Transformer by Jay Alammar;](http://jalammar.github.io/illustrated-transformer/)

- [Transformers with Lucas Beyer, Google Brain, Video](https://www.youtube.com/watch?v=EixI6t5oif0) 

- [Transformers with Lucas Beyer, Google Brain, PDF](https://docs.google.com/presentation/d/1ZXFIhYczos679r70Yu8vV9uO6B1J0ztzeDxbnBxD1S0/edit#slide=id.g31364026ad_3_2)

- [a more technical blog article by Lilian Weng;](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)

- [a minimal code implementation of a generative language model for educational purposes by Andrej Karpathy;](https://github.com/karpathy/nanoGPT)

### A Catalog and family tree of all major transformers to date by Xavier Amatriain

- [Transformer models: an introduction and catalog — 2023 Edition](https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/?utm_source=substack&utm_medium=email)
- [TRANSFORMER MODELS: AN INTRODUCTION AND CATALOG](https://arxiv.org/pdf/2302.07730.pdf)
- [Transformer Catalog](https://docs.google.com/spreadsheets/d/1XI-iRulxbFQL3hB2wIrJ5xxP1XwGqiQtLQklDvA4tmo/edit#gid=0)

<br>

### Understanding the Main Architecture and Tasks
If you are new to transformers / large language models, it makes the most sense to start at the beginning.

### (1) Neural Machine Translation by Jointly Learning to Align and Translate (2014) by Bahdanau, Cho, and Bengio, https://arxiv.org/abs/1409.0473

I recommend beginning with the above paper if you have a few minutes to spare. It introduces an attention mechanism for recurrent neural networks (RNN) to improve long-range sequence modeling capabilities. This allows RNNs to translate longer sentences more accurately – the motivation behind developing the original transformer architecture later.
![image](https://user-images.githubusercontent.com/13446418/232326713-bf8e3603-661b-4790-b424-ad815ebcbba4.png)

Source: https://arxiv.org/abs/1409.0473

### 2) Attention Is All You Need (2017) by Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin, https://arxiv.org/abs/1706.03762

The paper above introduces the original transformer architecture consisting of an encoder- and decoder part that will become relevant as separate modules later. Moreover, this paper introduces concepts such as the scaled dot product attention mechanism, multi-head attention blocks, and positional input encoding that remain the foundation of modern transformers.

![image](https://user-images.githubusercontent.com/13446418/232326748-5be93e46-5b5f-4e28-bc20-76aaa5b8010b.png)

Source: https://arxiv.org/abs/1706.03762

### 3) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018) by Devlin, Chang, Lee, and Toutanova, https://arxiv.org/abs/1810.04805

Following the original transformer architecture, large language model research started to bifurcate in two directions: encoder-style transformers for predictive modeling tasks such as text classification and decoder-style transformers for generative modeling tasks such as translation, summarization, and other forms of text creation.

The BERT paper above introduces the original concept of masked-language modeling, and next-sentence prediction remains an influential decoder-style architecture. If you are interested in this research branch, I recommend following up with RoBERTa, which simplified the pretraining objectives by removing the next-sentence prediction tasks.

![image](https://user-images.githubusercontent.com/13446418/232326791-46153b54-38e0-470d-8f5f-b23374474501.png)


### 4) Improving Language Understanding by Generative Pre-Training (2018) by Radford and Narasimhan,
https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035

The original GPT paper introduced the popular decoder-style architecture and pretraining via next-word prediction. Where BERT can be considered a bidirectional transformer due to its masked language model pretraining objective, GPT is a unidirectional, autoregressive model. While GPT embeddings can also be used for classification, the GPT approach is at the core of today’s most influential LLMs, such as chatGPT.

If you are interested in this research branch, I recommend following up with the GPT-2 and GPT-3 papers. These two papers illustrate that LLMs are capable of zero- and few-shot learning and highlight the emergent abilities of LLMs. GPT-3 is also still a popular baseline and base model for training current-generation LLMs such as ChatGPT – we will cover the InstructGPT approach that lead to ChatGPT later as a separate entry.

![image](https://user-images.githubusercontent.com/13446418/233917002-a744cea1-2ad4-4fd1-a6e0-804fa6bd7c96.png)

### 5) BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (2019), by Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, and Zettlemoyer, https://arxiv.org/abs/1910.13461.

As mentioned earlier, BERT-type encoder-style LLMs are usually preferred for predictive modeling tasks, whereas GPT-type decoder-style LLMs are better at generating texts. To get the best of both worlds, the BART paper above combines both the encoder and decoder parts (not unlike the original transformer – the second paper in this list).

![image](https://user-images.githubusercontent.com/13446418/233917112-a9ac4fac-8920-4135-a5fa-7818c51994d3.png)

<br>

## Scaling Laws and Improving Efficiency

If you want to learn more about the various techniques to improve the efficiency of transformers, 
I recommend the 
[2020 Efficient Transformers: A Survey paper](https://arxiv.org/abs/2009.06732)
followed by the 2023 [A Survey on Efficient Training of Transformers paper.](https://arxiv.org/abs/2302.01107)

read more from this blog

https://magazine.sebastianraschka.com/p/understanding-large-language-models?utm_source=substack&utm_medium=email
