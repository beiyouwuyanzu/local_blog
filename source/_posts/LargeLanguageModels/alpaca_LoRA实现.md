---
title: alpaca-lora实现 
date: 2023-05-08 21:04:50
mathjax: true
---
 
> 羊驼模型发展历史 
> ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305061919018.png)

## lora代码实现
- 调用
```python
      >>> from peft import LoraModel, LoraConfig

      >>> config = LoraConfig(
      ...     peft_type="LORA",
      ...     task_type="SEQ_2_SEQ_LM",
      ...     r=8,
      ...     lora_alpha=32,
      ...     target_modules=["q", "v"],
      ...     lora_dropout=0.01,
      ... )

      >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
      >>> lora_model = LoraModel(config, model)
```

- 内部实现
- > https://github1s.com/huggingface/peft/blob/b1059b73aab9043b118ff19b0cf96263ea86248a/src/peft/tuners/lora.py#L107-L119

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305061744459.png)
1. 传入base_model
2. 执行add_adapter方法, 在里面把`lora_config.target_modules` 中的 层 替换为 LoraConfig 中定义的结构
3. 保留只有lora层是训练参数可变的


## 实现细节
他是根据models的key去找target层, 然后替换成lora layer, 以llama模型为例, 看一下找的过程
```
target_modules= ["q_proj", "v_proj"]
cnt = 0
for key in key_list:
  target_module_found = any(key.endswith(target_key) for target_key in target_modules)
  if target_module_found:
    cnt += 1
    print(key)

print(cnt)
```
可以看到主要改的就是这些q, v 的层
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305061812220.png)

找到了之后就进行替换
```
self._replace_module(parent, target_name, new_module, target)
```
## llama 模型结构
从整体的模型结构也可以看出来
只有q, v 的liner被改动了.
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305061851354.png)