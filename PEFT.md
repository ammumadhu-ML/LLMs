# Parameter-efficient finetuning techniques (PEFT)
![image](https://github.com/harirajeev/learn_LLMS/assets/13446418/c4a6de9c-5982-4086-be6e-cc8c83367e98)
![image](https://github.com/harirajeev/learn_LLMS/assets/13446418/4eb30948-1e4b-4f03-94b7-9cba1c71f8ce)
- as models get larger and larger, full fine-tuning becomes infeasible to train on consumer hardware.
- storing and deploying fine-tuned models independently for each downstream task becomes very expensive, because fine-tuned models are the same size as the original pretrained model.
- [Parameter-Efficient Fine-tuning (PEFT)](https://huggingface.co/blog/peft) approaches are meant to address both problems!
- Over the years, researchers developed several techniques (Lialin et al.) to finetune LLM with high modeling performance while only requiring the training of only a small number of parameters. These methods are usually referred to as parameter-efficient finetuning techniques (PEFT).
        Some of the most widely used PEFT techniques are summarized in the figure below.
        ![image](https://user-images.githubusercontent.com/13446418/234774400-d31d4c2d-7000-45ed-a384-103f00dd11a6.png)
- PEFT approaches only fine-tune a small number of (extra) model parameters while freezing most parameters of the pretrained LLMs, thereby greatly decreasing the computational and storage costs.
- It also helps in portability wherein users can tune models using PEFT methods to get tiny checkpoints worth a few MBs compared to the large checkpoints of full fine-tuning
- The small trained weights from PEFT approaches are added on top of the pretrained LLM.

```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
```
1. [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)    
2. [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
3. [Prompt Tuning And Prefix Tuning](https://magazine.sebastianraschka.com/p/understanding-parameter-efficient)
4. [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)
    - [What is Lora](https://bdtechtalks.com/2023/05/22/what-is-lora/)
