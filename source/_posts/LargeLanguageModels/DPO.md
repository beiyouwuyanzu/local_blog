---
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
date: 2023-09-04 23:45:42
---

> https://arxiv.org/abs/2305.18290
> https://zhuanlan.zhihu.com/p/634705904

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308311504249.png)
DPO 针对人类偏好进行优化，同时避免强化学习。现有的利用人类反馈微调语言模型的方法首先将奖励模型拟合到提示数据集和人类对响应对的偏好，然后使用强化学习找到最大化学习奖励的策略。相比之下，DPO 通过简单的分类目标直接优化最能满足偏好的策略，无需显式奖励函数或 RL

## 奖励模型BT
Bradley-Terry（BT）模型是一个常见选择（在可以获得多个排序答案的情况下，Plackett-Luce 是更一般的排序模型）。BT 模型规定人类偏好分布 
 可以表示成：
 
 ![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308311408664.png)
 
 如果可以访问一个从 $p*$中采样的静态对比数据集 $D={\{x^{(i)},y^{(i)}_w,y^{(i)}_l\}}^N_{i=1}$
 ，那么我们可以通过最大似然估计来参数化奖励模型 
。将问题建模为二分类问题，我们可以使用负对数似然损失：

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202308311432710.png)

其中 $\sigma$ 是逻辑函数。对于 LMs 来说，$r_{\phi}(y,x)$ 通常初始化自 SFT 模型 $\pi^{SFT} (y|x) $
 ，并在最后一层 transformer 层后添加一个线性层以获得奖励值的标量预测。为了确保奖励函数具有较低的方差，之前的工作会对奖励进行归一化，比如对所有 x
 有 $\mathbb{E}_{x,y \sim D}[r_{\phi}(x, y)] = 0$
 
 ## RL微调f
RL 微调：在 RL 阶段，我们使用学到的奖励函数来为语言模型提供反馈。特别地，我们定义了如下优化问题
$$\max _{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)}\left[r_\phi(x, y)\right]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_\theta(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right]$$


其中，$\beta$  是用于控制与基础参考策略 $\pi_{ref}$（即初始 SFT 模型$\pi^{SFT}$ ）偏离程度的参数。实际上，语言模型策略$\pi_{\theta}$ 也会被初始化成 $\pi^{SFT}$。加入的这项约束非常重要，因为它需要防止策略模型过于偏离奖励模型（能准确预测的）的分布，同时保持生成结果多样性并避免模式坍塌到单一的高奖励答案。由于语言生成的离散性，因此目标是不可微的并且通常使用强化学习来优化。标准的奖励函数如下，并通过 PPO 来最大化：
$$r(x, y)=r_\phi(x, y)-\beta\left(\log \pi_\theta(y \mid x)-\log \pi_{r e f}(y \mid x)\right)$$


## Direct Preference Optimization
与以往的 RLHF 方法（先学习一个奖励函数，然后通过强化学习优化）不同，DPO的方法跳过了奖励建模步骤，直接使用偏好数据优化语言模型。核心观点是利用从奖励函数到最优策略的解析映射，将对奖励函数的损失转化为对策略的损失。

DPO的最终目标

$$\mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right]$$


DPO更新的相关梯度
$$\begin{aligned}&\nabla_\theta\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})=\\&-\beta\mathbb{E}_{(x,y_w,y_l)\thicksim\mathcal{D}}\left[\underbrace{\sigma(\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w))}_{\text{higher weight when rewad estinate is wrong}}\left[\underbrace{\nabla_\theta\log\pi(y_w\mid x)}_{\text{increas likelihood of }y_\omega}-\underbrace{\nabla_\theta\log\pi(y_l\mid x)}_{\text{decreas likelihood of }y_l}\right]\right]\end{aligned}$$


其中 $\hat{r}_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$
 是由语言模型 $\pi_\theta$和参考模型 $\pi_{ref}$
 隐式定义的奖励（详见第 5 节）。直观上，损失函数的梯度会增加生成更优回答$y_w$ 的概率，降低非最优回答$y_l$的概率。重要的是，样本根据隐式奖励模型$\hat{r}_\theta$ 对非首选回答打分的高低进行加权并通过 $\beta$进行缩放，即隐式奖励模型对回答进行错误排序的程度，体现出 KL 约束的强度。我们的实验表明了这种加权的重要性，因为没有加权的朴素版本会导致语言模型的退化（Appendix Table 2）
 
 ## DPO概览
 一般的 DPO 流程如下：1）对于每个 prompt $x$ 采样回答$y1,y2 \sim \pi_{ref} (\cdot|x)$，基于人类偏好标注并构建离线的偏好数据集 $$
 以及 2）对于给定的 $\mathcal{D}={\{x^{(i)}, y^{(i)}_{w}, y^{(i)}_l\}^N_{i=1}}$ 对于给定的$\pi_{ref}$ 和$\mathcal{D}$ ，优化语言模型 $\pi_{\theta}$  以最小化 $\mathcal{L}_{DPO}$ 和期望的 $\beta$。实际上，有人会喜欢复用可获得的公开偏好数据集，而不是自行生成样本并收集人类偏好。由于偏好数据集是使用 $\pi^{SFT}$ 采样得到的，而只要 
 可以获得，我们就用 $\pi_{ref} = \pi^{SFT}$来初始化。然而，当它不在可以获得时，我们就通过最大化最优问答对$(x, y_w)$ 的似然来初始化 
，即：

$$\pi_{r e f}=\operatorname{argmax}_\pi \mathbb{E}_{x, y_w \sim \mathcal{D}}\left[\log \pi\left(y_w \mid x\right)\right]$$

这个过程有助于减小真实参考分布（不可知）和 DPO 实际使用的$\pi_{ref}$之间的分布偏移