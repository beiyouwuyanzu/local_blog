---
title: GPT训练论文
date: 2023-09-05 23:15:15
---

> Training language models to follow instructions
with human feedback
https://arxiv.org/pdf/2203.02155.pdf

## 数据集
- 检查公共前缀的instruct来进行启发式去重
- 每个用户ID限制200

#### 数据集用途
1. SFT数据集: 13k个训练提示(来自API和人员编写)
2. RM数据集, 使用label排名训练RM: 33k个训练提示(API和人员标注)
3. PPO数据集, 没有人工标签, 用作RLHF微调: (31k个训练提示,仅来自API)



## 模型
- 我们从Brown et al.(2020)的GPT-3预训练语言模型开始。这些模型在广泛分布的互联网数据上进行训练，并能适应广泛的下游任务，但具有较差的行为特征。从这些模型开始，我们用三种不同的技术训练模型：监督微调(ST)。在标签器演示上使用监督学习对GPT-3进行微调。我们训练了16个poh,使用余弦学习率衰减，残差下降为0.2。我们根据验证集上的RM分数进行最终的SFT模型选择。与Wu et al.(2021)类似，我们发现SFT模型在1次迭代后过拟合验证损失；尽管有这种过拟合，但对更多epoch的训练对RM分数和人类偏好评级都有帮助。
- 奖励模型RM)。从最终的去嵌入层的SFT模型开始删除后，我们训练了一个模型来接收提示和响应，并输出标量奖励。在本文中，我们只使用6BM,因为这节省了大量的计算，并且我们发现175BRM训练可能不稳定，因此不太适合用作RL期间的值函数（有关详细信息，请参阅附录C)。
在Stiennon et al.(2020)中，RM是在对相同输入的两个模型输出进行比较的数据集上进行训练的。他们使用交叉熵损失，将比较作为标签一奖励的差异代表人类标记者更喜欢一种反应的对数概率。
- 为了加快比较收集，我们为标签者提供K=4和K=9之间的任何位置的排名响应。这将为显示给标记者的每个提示生成()比较。由于在每个标记任务中，比较是非常相关的，我们发现，如果我们简单地将比较混洗到一个数据集，对数据集的一次遍历会导致奖励模型过拟合。5相反，我们将每个提示作为单个批处理元素训练所有()比较。这在计算上更加高效，因为它只需要对每个完成进行一次M的前向传递（而不是对K完成进行(）前向传递)，并且，因为它不再过度拟合，它实现了大大提高的验证准确性和日志损失。

## 奖励模型
- 具体来说，奖励模型的损失函数为：
$$\operatorname{loss}\left(\theta\right)=-\frac{1}{\binom{K}2}E_{(x,y_w,y_l)\sim D}\left[\log\left(\sigma\left(r_\theta\left(x,y_w\right)-r_\theta\left(x,y_l\right)\right)\right)\right]$$

其中$r_\theta (x,y)$是提示x和带参数$\theta$的完成$y$的奖励模型的标量输出，$y_w$是$y_w$和$y_l$对中的首选完成，$D$是人工比较的数据集。
最后，由于$RM$损失对奖励的偏移是不变的，我们使用偏差对奖励模型进行归一化，以便在进行强化学习之前，标记器演示达到平均分数0。


## 强化学习
最大化以下目标函数
$$\begin{aligned}
\operatorname{objective}\left(\phi\right)=& E_{(x,y)\sim D_{\pi_{\phi}^{\mathrm{RL}}}}\left[r_{\theta}(x,y)-\beta\log\left(\pi_{\phi}^{\mathrm{RL}}(y\mid x)/\pi^{\mathrm{SFT}}(y\mid x)\right)\right]+  \\
&\gamma E_{x\sim D_{\mathrm{pretrain}}}\left[\log(\pi_{\phi}^{\mathrm{RL}}(x))\right]
\end{aligned}$$

其中$\pi_{\phi}^{RL}$是学习到的强化学习策略，$\pi_{SFT}$是监督训练的模型，$D_{pretrain}$是预训练分布。KL奖励系数$\beta$和预训练损失系数$\gamma$分别控制KIL惩罚和预训练梯度的强度。对于PPO模型，$\gamma$设置为0。除非另有说明，在本文中InstructGPT:指的是PPO-ptx模型。

