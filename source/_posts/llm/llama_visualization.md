---
title: llama 模型结构
date: 2023-05-08 21:04:50
mathjax: true
---

trainable params: 4194304 || all params: 6742609920 || trainable%: 0.06220594176090199


```json
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(32000, 4096, padding_idx=31999)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): Linear8bitLt(
                in_features=4096, out_features=4096, bias=False
                (lora_dropout): Dropout(p=0.05, inplace=False)
                (lora_A): Linear(in_features=4096, out_features=8, bias=False)
                (lora_B): Linear(in_features=8, out_features=4096, bias=False)
              )
              (k_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
              (v_proj): Linear8bitLt(
                in_features=4096, out_features=4096, bias=False
                (lora_dropout): Dropout(p=0.05, inplace=False)
                (lora_A): Linear(in_features=4096, out_features=8, bias=False)
                (lora_B): Linear(in_features=8, out_features=4096, bias=False)
              )
              (o_proj): Linear8bitLt(in_features=4096, out_features=4096, bias=False)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear8bitLt(in_features=4096, out_features=11008, bias=False)
              (down_proj): Linear8bitLt(in_features=11008, out_features=4096, bias=False)
              (up_proj): Linear8bitLt(in_features=4096, out_features=11008, bias=False)
              (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
          )
        )
        (norm): LlamaRMSNorm()
      )
      (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
    )
  )
)
```