![image](https://github.com/harirajeev/learn_LLMS/assets/13446418/6a49827a-f2e8-4e00-a0dd-9dd1db9bac32)


- There has been a push for building up the open source developer ecosystem in the AI landscape. 
  - Meta pushed LLaMa 
  - Microsoft calls everyone a developer now 
  - Google thinks open source is the real winner in the AI race. 
  - With these open source smaller models that do not require heavy computation resources, the generative AI landscape will get even more democratised. 
  - The future would be just how Yann LeCun, Meta AI chief, visioned it – multiple smaller models working together for better performance, calling it the world model. This is what Altman predicts and wishes as well. We are headed in the right direction

1. [LLaMA, Meta’s Open Source LLM](https://thenewstack.io/why-open-source-developers-are-using-llama-metas-ai-model/)
   - So far, LLaMA might be the most impactful LLM model of 2023.
   - [LLaMa](https://aman.ai/primers/ai/LLaMA/)
   - Goat model
      - A finetuned 7B LLaMA model that outperforms GPT-4 on arithmetic tasks, Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks paper.
      - 7B Goat model outperformed a ~75x larger 540B PaLM model and GPT-4 in zero-shot settings
      - The Goat model is a special-purpose finetuned LLM that has been trained to perform well on this specific task
      - task-specific finetuned model outperforms a more general-purpose chatbot like GPT-4,well-finetuned models will always maintain an edge
      - The two main ingredients for success behind Goat are
        - supervised finetuning of a good LLM (here: LLaMA) on a target task (versus general pretraining or instruction finetuning);
        - LLaMA's digit tokenization (splits each digit into an individual token).LLaMA's specific tokenization scheme has been an essential contributor to the success of Goat
        - The Goat model was finetuned using low-rank adaptation (LoRA) to make the finetuning more parameter-efficient, which allows for finetuning a 7B-parameter LLaMA model using a single 24 Gb GPU.
        - how about training a 65B (instead of 7B) parameter LLaMA model on a single GPU Related to LoRA, using a new method called QLoRA (quantized LoRA).
        - QLORA reduces the memory requirements of a 65B LLaMA model such that it fits onto a single 48 GB GPU (like an A100)
        - The resulting 65B Guanaco model, from quantized 4-bit training, maintains full 16-bit finetuning task performance, reaching 99.3% of the ChatGPT performance after only 24h of finetuning.
   -  LIMA: [Less Is More for Alignment](https://arxiv.org/pdf/2305.11206.pdf)
        - How Much Data Do We Need For Finetuning LLMs Anyways?
        - LIMA a 65B LLaMA model finetuned on only 1000 examples
        - about half of the time, LIMA outperforms the GPT-4 predecessor ChatGPT/GPT3.5 (also referred to as DaVinci003)
        - LIMA outperforms Alpaca by such a large margin. Both are LLaMA models after supervised finetuning
        - LIMA is based on a 65B LLaMA model, whereas the original Alpaca model is based on the 7B LLaMA base model
        - the authors reproduced the Alpaca training using a 65B base model, training it on 52,000 samples
        - the difference is really in the quality of the training set that the authors carefully curated for LIMA, as it beats the same 65B LLaMA base model trained on 52x more data (i.e., Alpaca).    
   -  Hold Your LLaMAs - [The False Promise of Imitating Proprietary LLMs.](https://arxiv.org/pdf/2305.15717.pdf)
      -   can open-source fine-tuned LLMs compete with proprietary models such as GPT-4, PaLM 2, etc? 
      -   The researchers initially found that these ‘imitation models’ – the likes of Alpaca, Vicuna, GPT4All, etc – performed impressively. However, upon further investigation, it revealed a significant gap in the performance on tasks not heavily supported in the imitation data.  
      -   imitation models only tend to mimic the style of the upstream LLMs on whose data they were trained on, not their factuality and problem-solving
      -   LIMA paper does not use imitation data. Instead, it uses a carefully curated dataset
      -   Imitation data refers to the outputs from a more advanced language model (GPT-4) used to train a weaker model with the aim of achieving the performance of the powerful model. 
      -   large-scale instruction tuning, i.e. RLHF based model
   -  Can We Improve LLM Base Models By Pretraining for Multiple Epochs?
      -   [To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis](https://arxiv.org/pdf/2305.13230.pdf) 
      -   The finetuned models all require a pretrained base model. So, it's natural to ask how we can create better base models as well.
      -   It's quite common to train classic machine learning models and deep neural networks, as well as the latest vision transformers for hundreds of epochs, so what happens if we do the same for LLMs, which are commonly trained for only a single epoch?
      -   high-quality text data on the internet is slower than required.if copyrighted material is removed in the future, this could even shrink the datasets further
      -   The result is that training for multiple epochs leads to overfitting.
      -   dropout can help reduce overfitting (no surprise), but other techniques, such as weight decay, can't
      -   popular large models like LLaMA, Gopher, Chinchilla, GPT-3, and PaLM did not use dropout, since it can slow down learning
      -   Repeating only high quality data(like LLaMA data) , is it helpful ? (repeating with Wikipedia data alone was causing degradation)
      -   Does data augmentation help?       -   
      -   for finetuning, is multiple epoch helpful ?
      -   
2. From a business perspective, I can see this being useful from at least two angles: 

       1. how can we be better than the competition if we use the same off-the-shelf solution others are using and 
       2. if it's open source, we have full control and are not affected by API changes and restrictions

3. https://www.semianalysis.com/p/google-we-have-no-moat-and-neither?utm_source=bensbites&utm_medium=newsletter&utm_campaign=knighty-byte-open-source-crusade
4. [PrivatGPT](https://github.com/imartinez/privateGPT)
5. [Build, customize and control your own personal LLMs - xTuring](https://github.com/stochasticai/xturing)
   - xTuring provides fast, efficient and simple fine-tuning of LLMs, such as LLaMA, GPT-J, Galactica, and more
   - [LLaMA_lora_int4 - LLaMA INT4 efficient fine-tuning tutorial](https://github.com/stochasticai/xturing/blob/main/examples/int4_finetuning/LLaMA_lora_int4.ipynb)
   - [INT4 fine-tuning of LLMs with only 6GB of memory](https://github.com/stochasticai/xturing/blob/main/examples/int4_finetuning/README.md)
   
#*****************************************************************************************************************************************************

The Timeline

1. Feb 24, 2023 - LLaMA is Launched
  Meta launches LLaMA, open sourcing the code, but not the weights. At this point, LLaMA is not instruction or conversation tuned. Like many current models, it is a    relatively small model (available at 7B, 13B, 33B, and 65B parameters) that has been trained for a relatively large amount of time, and is therefore quite capable relative to its size.

2. March 3, 2023 - The Inevitable Happens
  https://www.vice.com/en/article/xgwqgw/facebooks-powerful-large-language-model-leaks-online-4chan-llama
Within a week, LLaMA is leaked to the public. The impact on the community cannot be overstated. Existing licenses prevent it from being used for commercial purposes, but suddenly anyone is able to experiment. From this point forward, innovations come hard and fast.

3. March 12, 2023 - Language models on a Toaster
  A little over a week later, Artem Andreenko gets the model working on a Raspberry Pi. At this point the model runs too slowly to be practical because the weights must be paged in and out of memory. Nonetheless, this sets the stage for an onslaught of minification efforts.

4. March 13, 2023 - Fine Tuning on a Laptop
  The next day, Stanford releases Alpaca, which adds instruction tuning to LLaMA. More important than the actual weights, however, was Eric Wang’s alpaca-lora repo, which used low rank fine-tuning to do this training “within hours on a single RTX 4090”.

  Suddenly, anyone could fine-tune the model to do anything, kicking off a race to the bottom on low-budget fine-tuning projects. Papers proudly describe their total   spend of a few hundred dollars. What’s more, the low rank updates can be distributed easily and separately from the original weights, making them independent of the  original license from Meta. Anyone can share and apply them.

5. March 18, 2023 - Now It’s Fast
  Georgi Gerganov uses 4 bit quantization to run LLaMA on a MacBook CPU. It is the first “no GPU” solution that is fast enough to be practical.

6. March 19, 2023 - A 13B model achieves “parity” with Bard
  The next day, a cross-university collaboration releases Vicuna, and uses GPT-4-powered eval to provide qualitative comparisons of model outputs. While the  evaluation method is suspect, the model is materially better than earlier variants. Training Cost: $300.

  Notably, they were able to use data from ChatGPT while circumventing restrictions on its API - They simply sampled examples of “impressive” ChatGPT dialogue posted on sites like ShareGPT.

7. March 25, 2023 - Choose Your Own Model
  Nomic creates GPT4All, which is both a model and, more importantly, an ecosystem. For the first time, we see models (including Vicuna) being gathered together in   one place. Training Cost: $100.

8. March 28, 2023 - Open Source GPT-3
  Cerebras (not to be confused with our own Cerebra) trains the GPT-3 architecture using the optimal compute schedule implied by Chinchilla, and the optimal scaling  implied by μ-parameterization. This outperforms existing GPT-3 clones by a wide margin, and represents the first confirmed use of μ-parameterization “in the wild”.   These models are trained from scratch, meaning the community is no longer dependent on LLaMA.

9. March 28, 2023 - Multimodal Training in One Hour
  Using a novel Parameter Efficient Fine Tuning (PEFT) technique, LLaMA-Adapter introduces instruction tuning and multimodality in one hour of training. Impressively,  they do so with just 1.2M learnable parameters. The model achieves a new SOTA on multimodal ScienceQA.

10. April 3, 2023 - Real Humans Can’t Tell the Difference Between a 13B Open Model and ChatGPT
  Berkeley launches Koala, a dialogue model trained entirely using freely available data.

  They take the crucial step of measuring real human preferences between their model and ChatGPT. While ChatGPT still holds a slight edge, more than 50% of the time  users either prefer Koala or have no preference. Training Cost: $100.

11. April 15, 2023 - Open Source RLHF at ChatGPT Levels
  Open Assistant launches a model and, more importantly, a dataset for Alignment via RLHF. Their model is close (48.3% vs. 51.7%) to ChatGPT in terms of human  preference. In addition to LLaMA, they show that this dataset can be applied to Pythia-12B, giving people the option to use a fully open stack to run the model.  Moreover, because the dataset is publicly available, it takes RLHF from unachievable to cheap and easy for small experimenters.
