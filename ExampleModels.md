
# MPT-1b-RedPajama-200b-dolly 
(pre-trained on redpajama 200B , fine tuned on dolly 15k) 
[mpt-1b-redpajama-200b-dolly-COLAB](https://colab.research.google.com/drive/19YGJ-eDe2Wm17hc9hLwobckHZjFB8lo5?usp=sharing)

- MPT-1b-RedPajama-200b-dolly is a 1.3 billion parameter decoder-only transformer pre-trained on the [RedPajama dataset]
and subsequently fine-tuned on the [Databricks Dolly [databricks-dolly-15k dataset]] instruction dataset 
- The model was pre-trained for 200B tokens by sampling from the subsets of the RedPajama dataset in the same proportions as were used by the [Llama series of models]
- This model is an instruction fine-tuned version of [mpt-1b-redpajama-200b]
- The architecture is a modification of a standard decoder-only transformer.
- The transformer has 24 layers, 16 attention heads, and width 2048.
- The model has been modified from a standard transformer in the following ways:
* It uses ALiBi and does not use positional embeddings.
* It uses QK LayerNorm.
* It does not use biases.

The model was pre-trained for 200B tokens (batch size 2200, sequence length 2048). It was trained on the following data mix:
* 67% RedPajama Common Crawl
* 15% [C4](https://huggingface.co/datasets/c4)
* 4.5% RedPajama GitHub
* 4.5% RedPajama Wikipedia
* 4.5% RedPajama Books
* 2.5% RedPajama Arxiv
* 2% RedPajama StackExchange

- This is the same mix of data as was used in the Llama series of models]
- This model was pre-trained on 440 A100-40GBs for about half a day using the [MosaicML Platform]
- The model was pre-trained with sharded data parallelism using FSDP
***
