---
title: llama2
date: 2023-09-05 23:16:39
---

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202309011447966.png)
训练方法:
1. 公开语料预训练
2.  sft
3. RLHF: 拒绝采样, PPO
说明: 与模型增强并行的迭代奖励建模数据积累很重要.

## 升级点
- 训练语料正价40%
- 上下文长度增加一倍
- 使用分组查询注意力(GQA)


## 预训练
1. 2万亿token

### 沿用llama1配置
1. 标准transformer
2. RMSNorm归一化
3. SwiGLU激活函数
4. RoPE旋转位置编码
5. tokenizer
    1. BPE字节对编码
    2. 32k个token

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202309011454003.png)


## 微调
1. sft数据
    1. 使用公开的指令微调数据
    2. `质量更重要`有限的干净的指令调优数据就可以达到高质量水平. (几万条数据就可以获得高质量结果)
2. 微调细节
    1. 余弦学习率, 学习率=$2*10^{-5}$, 权重衰减$0.1$, 批量大小$64$, 序列长度$4096$
    2. 格式: 提示 + [sep] + 答案
    3. 提示的损失归零, 只对答案token反向传播
    4. 微调$2$个epoch
## RLHF
人类标注模型的两个输出哪个更好, 奖励模型学习这种偏好模式

### 奖励建模目标
$$\mathcal{L}_{ranking} = -log(\sigma(r_{\theta}(x, y_c) - r_{\theta}(x, y_r)))$$

其中$r_{\theta}(x, y)$ 是奖励模型对于 模型权重$\theta$对于提示词$x$ 和 答案$y$的分数. $y_c$是采纳的答案, $y_r$是拒绝的结果
为了获得更好的有用性和安全性奖励, 增加了一个偏好评级离散函数$m(r)$
$$\mathcal{L}_{ranking} = -log(\sigma(r_{\theta}(x, y_c) - r_{\theta}(x, y_r) - m(r)))$$

### 数据组成
开源数据对学习偏好没负向影响. 因此保留在数据混合中

### 训练细节
训练一个epoch
- 70B 最大学习率 $5* 10^{-6}$. 其余模型$1*10^{-5}$. 余弦学习率, 下降到最大的$10%$, warm_up $#3$, 有效batch_size 512, 即batch_size = 1024


### 迭代微调
两种算法进行RLFH微调
1. PPO 近段策略优化
2. 拒绝采样微调
    从模型中采样K个输出, 使用奖励模型选择最佳候选.
    

##### 两种RL算法的区别
1. 宽度. 拒绝采样每次采样K个, PPO只有一个
2. 深度. 拒绝采样是从初始策略模型采样, PPO是从t时间步采样t-1步的模型输出.

##### 拒绝采样好处
最大值增加(更多的样本, 更多的机会生成良好的轨迹), 而中位数保持不变.同时更高的温度也能够采样更多样化的输出.

#### ppo优化目标

$$
\arg \max _\pi \mathbb{E}_{p \sim \mathcal{D}, g \sim \pi}[R(g \mid p)]
$$

从数据集$\mathcal{D}$中抽取提示$p$ 和  从策略$\pi$ 中抽取$g$来迭代改进策略. 并使用PPO算法和损失函数实现这一目标.

最终的奖励函数
$$
R(g \mid p)=\tilde{R}_c(g \mid p)-\beta D_{K L}\left(\pi_\theta(g \mid p) \| \pi_0(g \mid p)\right)
$$

包含偏离原始策略$\pi$的惩罚项. 这种约数对于训练稳定性是有用的.

#### 训练参数
- adamW优化器
- $\beta_1 = 0.9$
- $\beta_2 = 0.95$
- $eps = 10^{-5}$
- 权重衰减0.1
- 梯度裁剪1.0
- 恒定学习率$10^{-6}$
- 每个PPO迭代
    - batch_size=2, PPO裁剪阈值0.2
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202309011629658.png)