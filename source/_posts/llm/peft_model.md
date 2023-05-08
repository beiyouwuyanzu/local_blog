---
title: PEFT State-of-the-art Parameter-Efficient Fine-Tuning [PEFT] methods
date: 2023-05-08 21:04:50
mathjax: true
---

> https://github.com/huggingface/peft
> https://github.com/huggingface/peft/blob/b1059b73aab9043b118ff19b0cf96263ea86248a/src/peft/peft_model.py#L102
> [Chinese-Vicuna/finetune.py at master · Facico/Chinese-Vicuna (github.com)](https://github.com/Facico/Chinese-Vicuna/blob/master/finetune.py)

## 特点
Parameter-Efficient Fine-Tuning (PEFT) 方法可以使预训练语言模型 (PLM) 高效适应各种下游应用程序，而无需微调模型的所有参数。微调大型 PLM 的成本通常高得令人望而却步。在这方面，PEFT 方法仅微调少量（额外）模型参数，从而大大降低了计算和存储成本。最近最先进的 PEFT 技术实现了与完全微调相当的性能。

## 微调方法
1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
3. P-Tuning: [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
4. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
5. AdaLoRA: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)

对应在peft代码里的实现
```python
PEFT_TYPE_TO_MODEL_MAPPING  = {

	PeftType.LORA: LoraModel,

	PeftType.PROMPT_TUNING: PromptEmbedding,

	PeftType.P_TUNING: PromptEncoder,

	PeftType.PREFIX_TUNING: PrefixEncoder,

	PeftType.ADALORA: AdaLoraModel,

	PeftType.ADAPTION_PROMPT: AdaptionPromptModel,

}
```

## 应用示例
llama应用的, 也是peft中主流的优化方法是LoRA
```python
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, add_eos_token=True
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```

## 实现解析
在这里将base_model封装成lora结构的形式
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305061729090.png)


