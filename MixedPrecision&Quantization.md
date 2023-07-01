# Gradient Checkpointing
- One way to use significantly less GPU memory is to enabled “Gradient Checkpointing” (also known as “activation checkpointing”)
- When enabled, a lot of memory can be freed at the cost of small decrease in the training speed due to recomputing parts of the graph during back-propagation.
- This technique was first shared in the paper: Training Deep Nets with Sublinear Memory Cost. 
- model.gradient_checkpointing_enable()
# Mixed Precision Training
- [Accelerating Large Language Models with Mixed-Precision Techniques](https://lightning.ai/pages/community/tutorial/accelerating-large-language-models-with-mixed-precision-techniques/)  
 ![image](https://github.com/harirajeev/learn_LLMS/assets/13446418/41f0694a-5178-4a64-9123-9c932d9ee6a2)
 
# Quantization

![image](https://github.com/harirajeev/learn_LLMS/assets/13446418/38a0636f-8feb-443e-be40-55631441919a)

If we want to increase the model performance during inference even more, we can also move beyond lower floating point precision and use quantization. Quantization converts the model weights from floats to low-bit integer representations, for example, 8-bit integers (and, recently, even 4-bit integers).

Quantization is a model size reduction technique that converts model weights from high-precision floating-point representation to low-precision floating-point (FP) or integer (INT) representations, such as 16-bit or 8-bit

  - [How does Quantization Work ?](https://www.youtube.com/watch?v=IxrlHAJtqKE)
    - [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/pdf/2110.02861.pdf)
    - [8-bit Methods for Efficient Deep Learning with Tim Dettmers](https://www.youtube.com/watch?v=jyOqtw4ry2w)
    - [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
    - [QLoRa: Fine-Tune a Large Language Model on Your GPU](https://towardsdatascience.com/qlora-fine-tune-a-large-language-model-on-your-gpu-27bed5a03e2b)
  - [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers 22/3/2023](https://arxiv.org/pdf/2210.17323.pdf)
    - GPT3-175B, have in the order of 175 billion parameters and require tens-to-hundreds of GPU years to train
the parameters of GPT3-175B occupy 326GB (counting in multiples of 1024) of memory when stored in a compact float16 format
    - More complex methods for low-bitwidth quantization or model pruning usually require model retraining, which is extremely expensive for billion-parameter models
    - Compress the model in one shot, without retraining, would be very appealing
    - We present a new post-training quantization method, called GPTQ, which is efficient enough to execute on models with hundreds of billions of parameters in at most a few hours, and precise enough to compress such models to 3 or 4 bits per parameter without significant loss of accuracy
    - GPTQ can quantize the largest publicly-available models, OPT-175B and BLOOM-176B, in approximately four GPU hours, with minimal increase in
perplexity, known to be a very stringent accuracy metric  
 
    - Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% 
of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU
  
    - Int8 (LLM.int8()) inference that does not degrade predictive performance of large models and reduces the memory footprint of large models by a factor or 2x
  - [LLM.int8() and Emergent Features by Tim Dettmers](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)
     - I had two pitches for my LLM.int8() paper. 
       - One pitch is about how I use advanced quantization methods to achieve no performance degradation transformer inference at scale that makes large models more accessible. 
       - The other pitch talks about emergent outliers in transformers and how they radically change what transformers learn and how they function.  (Emergent features that were discovered in language model at scale].
    - [int8 paper - LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)  
    - How quantization is done for you through the bitsandbytes library with Hugging Face integration so that you can easily run OPT-175B and BLOOM-176B on a single machine 
      - [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
      
  - [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA - 4bit-transformers-bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)  
    - The 4bit integration comes with 2 different quantization types: FP4 and NF4. The NF4 dtype stands for Normal Float 4 and is introduced in the QLoRA paper
    - [QLORA: Efficient Finetuning of Quantized LLMs 23/5/2023](https://arxiv.org/pdf/2305.14314.pdf)
      - QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.
      - QLORA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).
      - QLORA introduces a number of innovations to save memory without sacrificing performance: 
        - (a) The recent QLoRA paper explores different data types, 4-bit Float and 4-bit NormalFloat.
        - (b) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights. bnb_4bit_quant_type="nf4" 
        - (c) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and 
        - (d) Paged Optimizers to manage memory spikes.
      - [How to propely load a model in 4bit with all its variants](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing#scrollTo=VPD7QS_DR-mw)
         - how to load a model in 4bit, understand all its variants and how to run them for inference.
         - convert a model by just adding the argument load_in_4bit [model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")]
         - nn.Linear layers are replaced by bnb.nn.Linear4bit layers  Linear4bit(in_features=1024, out_features=512, bias=False)
         - The compute dtype is used to change the dtype that will be used during computation. For example, hidden states could be in float32 but computation can be set to bf16 for speedups. By default, the compute dtype is set to float32. bnb_4bit_compute_dtype=torch.bfloat16
      - [Fine tuning Google Colab notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) 
         - This notebook shows how to fine-tune a 4bit model on a downstream task using the Hugging Face ecosystem. 
           - Hugging Face ecosystem
             - transformers 
             - peft  
               - prepare_model_for_kbit_training
               - LoraConfig, get_peft_model               -
             - accelerate
           - Load the model - GPT-neo-x-20B! Note that the model itself is around 40GB in half precision
           - Apply some preprocessing to the model to prepare it for training. For that use the prepare_model_for_kbit_training method from PEFT.
           - load a common dataset, english quotes, to fine tune our model on famous quotes.
           - train
         - We show that it is possible to fine tune GPT-neo-X 20B on a Google Colab instance!
      - computation is not done in 4bit, the weights and activations are compressed to that format and the computation is still kept in the desired or native dtype
