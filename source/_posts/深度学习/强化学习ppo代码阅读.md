---
title: 强化学习ppo代码阅读：MOSS-RLHF
date: 2023-07-17 21:34:50
mathjax: true
---


# 大模型强化学习ppo:moss_rlhf源码阅读

> 代码地址: https://github.com/OpenLMLab/MOSS-RLHF/blob/main/train_ppo.py#L106C6-L106C6

## 通常PPO的主要流程

PPO（Proximal Policy Optimization）是一种常用的强化学习算法，用于优化策略函数。PPO通过限制策略更新的幅度，保证策略更新的稳定性，从而提高学习效率。下面是PPO算法的主要流程：

1. 初始化策略函数和值函数参数。
2. 重复执行以下步骤：
   a. 收集经验：使用当前策略与环境进行交互，收集一定数量的样本轨迹（trajectories）。
   b. 估计优势函数：使用GAE（Generalized Advantage Estimation）或其他方法估计样本轨迹中的优势函数估计值。
   c. 更新策略：使用样本轨迹和优势函数估计进行策略更新。
      i. 计算旧策略的动作概率：使用旧的策略参数计算样本轨迹中每个状态的动作概率。
      ii. 计算更新比率：计算新策略与旧策略之间的概率比值，用于限制策略更新的幅度。
      iii. 计算策略损失函数：构建一个损失函数，包括策略的目标函数和更新比率的约束项。常用的目标函数是带有优势函数估计的似然比目标。
      iv. 执行策略更新：通过最小化策略损失函数来更新策略参数，可以使用梯度下降或其他优化算法。
   d. 更新值函数：使用样本轨迹中的奖励信号来更新值函数参数。可以使用基于TD误差的方法，如均方误差最小化。
   e. 重复步骤a-d直到达到停止条件（如达到最大迭代次数或目标性能）。
3. 返回训练好的策略函数。

PPO的关键思想是在策略更新过程中引入一个剪切项或者KL散度约束，以限制策略更新的幅度。这样可以确保新策略在性能上不会显著远离旧策略，从而提高训练的稳定性。PPO算法有不同的变体，如PPO-Clip和PPO-Penalty，它们使用不同的方法来限制策略更新幅度，但基本的流程和思想是相似的。

请注意，上述流程只是PPO算法的一个概述，具体实现可能会有一些细微的差异，取决于具体的应用和问题设置。


## 整体架构
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307151106823.png)

## 重复惩罚
```python3
if repetition_penalty > 1.:
                penalty_tokens = decoder_input[:, init_length:]
                penalty_scores = torch.gather(score, dim=1, index=penalty_tokens)
                penalty_scores = torch.where(penalty_scores < 0., penalty_scores * repetition_penalty, penalty_scores / repetition_penalty)
                score = score.scatter_(dim=1, index=penalty_tokens, src=penalty_scores)
```

解读:
1. 第一步获取当输出的全部token
2. 第二步把重复部分的token的score提取出来
3. 第三步对分数进行比例缩放
4. 第四步按照`self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1`重新组织分数


## 奖惩模型: LlamaRewardModel
用的LlamaForCausalLM 最后的隐藏层, 加上一个linear 变换成一个分数值

## policy_model
Llama: models/sft_model

## critic_model
LlamaRewardModel: models/moss-rlhf-reward-model-7B-zh/recover

## 核心训练流程: PPOTrainer

### 优化器: optim.lr_scheduler.LambdaLR
### 训练主流程
```
self.make_experiences()
```

```python
context_vec_sampled, resp_vec_sampled, sampled_vec = self.concat_context_and_response(context_vec,responses_vec)
sampled_vec = torch.tensor(pad_sequences(sampled_vec, pad_value=self.tokenizer.pad_token_id, padding='left'), 
                           dtype=torch.long, device=self.accelerator.device)
bsz = sampled_vec.size(0)

rewards, *_ = self.reward_model_forward(sampled_vec)
```

先从concat_context_and_response抽样出可能的答复, 然后用奖励模型计算reward打分

### 抽样过程
就是调用了llama的generate方法, 只生成一个回复

### 训练流程
```python
rewards, *_ = self.reward_model_forward(sampled_vec) # 获取奖励结果
ref_logits, *_ = self.ref_model_forward(sampled_vec) # 参考模型 计算优势估计
logits, *_ = self.policy_model_forward(sampled_vec) # 策略模型生成动作
values, *_ = self.critic_model_forward(sampled_vec) # 批评模型生成惩罚


kl_penalty = (-self.kl_penalty_weight * (logprobs - ref_logprobs)).cpu() # kl 惩罚
```

最后组织的结果
```python
                sample = {
                    'context_vec': context_vec_sampled[i],
                    'context': self.tokenizer.decode(context_vec_sampled[i], skip_special_tokens=False),
                    'resp_vec': resp_vec_sampled[i],
                    'resp': self.tokenizer.decode(resp_vec_sampled[i], skip_special_tokens=False),
                    'reward': penalized_rewards[-resp_length:].tolist(),
                    'values': values[i][-resp_length:].tolist(),
                    'ref_logprobs': ref_logprobs[i][-resp_length:].tolist(),
                    'logprobs': logprobs[i][-resp_length:].tolist(),
                    'ppl_value': ppl_value[i],
                    'ppl0_value': ppl0_value[i]
                }
```
其中最终的reward来源于 
```python
                penalized_rewards = kl_penalty[i].clone()
                penalized_rewards[-1] += rewards[i]
```

即奖励模型的reward和kl_penalty(参考模型和动作模型的kl损失)的组合

### 模型总结
整个ppo过程中一共用到了4个模型
1. policy_model 生成动作和策略, 最终输出的模型
2. ref_model 参考模型, 用于计算策略模型的优势估计、价值估计或其他相关信息。参考模型可以是过去的策略模型，也可以是通过经验数据训练的模型。
3. critic_model 指价值函数模型(GAE)，用于估计状态的价值或优势
4. reward_model 给出单步的奖惩结果

### 回顾这四个模型
```python
    # load policy model
    logging.info(f"Loading policy model from: {opt.policy_model_path}...")
    policy_model = Llama.from_pretrained(opt.policy_model_path, opt, tokenizer)
    policy_model._set_gradient_checkpointing(policy_model.model, opt.gradient_checkpoint)

    # load critic model
    logging.info(f"Loading critic model from: {opt.critic_model_path}...")
    critic_model = LlamaRewardModel.from_pretrained(opt.critic_model_path, opt, tokenizer)
    critic_model._set_gradient_checkpointing(critic_model.model, opt.gradient_checkpoint)

    # load reference model
    logging.info(f"Loading reference model from: {opt.policy_model_path}...")
    ref_model = Llama.from_pretrained(opt.policy_model_path, opt, tokenizer)

    # load reward model
    logging.info(f"Loading reward model from: {opt.critic_model_path}...")
    reward_model = LlamaRewardModel.from_pretrained(opt.critic_model_path, opt, tokenizer)
```

可以看到ref_model 和 policy_model用的是一个model, 这是因为让强化学习的过程中, 和原来的模型不要偏差太远.用两者的kl散度损失来约束

### 优势估计计算方法
```python
    def get_advantages_and_returns(self, rewards: List[float], values: List[float]):
        '''
        Copied from TRLX: https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        '''
        response_length = len(values)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(response_length)):
            nextvalues = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + self.gamma * nextvalues - values[t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            
        advantages = advantages_reversed[::-1]
        returns = [a + v for a, v in zip(advantages, values)]
        assert len(returns) == len(advantages) == len(values)
        return advantages, returns
```

可以看出, 优势估计就是根据reward 和 values计算出来的
`advantages, returns = self.get_advantages_and_returns(sample['reward'], sample['values'])`
其中values来自批评模型`values, *_ = self.critic_model_forward(sampled_vec)`

### GAE原理
GAE（Generalized Advantage Estimation）是一种用于强化学习的方法，旨在估计状态值函数或动作值函数的优势估计。

在强化学习中，优势函数衡量了一个状态或动作相对于平均预期回报的好坏程度。GAE通过对优势函数的估计，可以为策略更新提供更准确的信号。

GAE的计算原理基于一个重要的概念——马尔可夫性质。根据马尔可夫性质，当前状态的未来回报与过去状态无关，只与当前状态相关。这使得我们可以通过观察经验轨迹中的状态转换来估计优势函数。

具体来说，GAE使用一个参数λ（lambda）来平衡未来回报的折扣和引导性。对于每个状态转换，GAE计算出一个优势估计Advantage Estimate，该估计结合了一阶和二阶的TD（Temporal Difference）误差。

GAE 的计算公式为：$$\widehat{A}_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \cdots + (\gamma \lambda)^{T-t+1}\delta_{T-1} + (\gamma \lambda)^{T-t}V(s_T) - V(s_t)$$ 其中，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是时序差分的误差项，$\gamma$ 是折扣因子，$\lambda$ 是 GAE 中的超参数。


GAE的计算步骤如下：

1. 遍历经验轨迹中的每个时间步 t。
2. 计算t时刻的TD误差δ，即当前状态的奖励加上折扣因子γ乘以下一状态值的估计减去当前状态值的估计。
   δ = r_t + γ * V(s_{t+1}) - V(s_t)
3. 定义一个累积因子A，初始值为0。
4. 遍历从当前时间步开始的未来时间步，计算累积因子A。
   A ← A * λ * γ + δ
   这里λ是一个介于0和1之间的参数，用于平衡未来回报的折扣和引导性。λ越接近1，越关注未来回报；λ越接近0，越关注即时回报。
5. 根据累积因子A计算优势估计Advantage Estimate。
   GAE(t) = A

GAE的优势估计可以用于策略梯度方法中，如Actor-Critic算法，用于计算策略更新的目标。通过使用GAE，可以提高对优势函数的估计，从而改善策略的更新效果。







