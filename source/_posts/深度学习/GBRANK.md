---
title: GBRANK
date: 2023-05-08 21:04:50
mathjax: true
---

# GBRANK

## 概述
GBRank 是一种基于梯度提升决策树（Gradient Boosting Decision Tree，简称 GBDT）的排序模型，其主要思想是通过构造排序损失函数，将排序问题转化为回归问题，然后使用 GBDT 模型进行拟合。

GBRank 的核心思想是将排序问题视为一种回归问题，即将真实的排序序列视为目标值，使用 GBDT 模型预测排序序列与真实排序序列之间的距离（或其他度量），然后根据预测值进行排序。

具体来说，假设给定一个包含 $N$ 个样本的训练集，每个样本由一个特征向量 $\boldsymbol{x}_i$ 和一个相应的排序序列 $\boldsymbol{y}_i$ 组成，其中 $\boldsymbol{y}_i$ 表示样本 $\boldsymbol{x}_i$ 的真实排序序列。我们的目标是训练一个模型 $F(\boldsymbol{x})$，用于预测特征向量 $\boldsymbol{x}$ 的排序序列 $\boldsymbol{\hat{y}}$。

GBRank 模型的训练过程可以分为以下几个步骤：

1. 定义排序损失函数

    GBRank 使用的排序损失函数通常是基于 Pairwise Learning-to-Rank 算法中的 pairwise hinge loss 函数。其表达式如下：

    $$L(y_i, y_j, \boldsymbol{x}_i, \boldsymbol{x}_j) = \max(0, 1 - (y_i - y_j) \cdot (F(\boldsymbol{x}_i) - F(\boldsymbol{x}_j)))$$

    其中，$y_i$ 和 $y_j$ 是样本 $\boldsymbol{x}_i$ 和 $\boldsymbol{x}_j$ 的真实排序序列，$F(\boldsymbol{x}_i)$ 和 $F(\boldsymbol{x}_j)$ 分别表示模型对 $\boldsymbol{x}_i$ 和 $\boldsymbol{x}_j$ 的预测排序序列，$\cdot$ 表示点积运算。该损失函数的含义是，如果模型的排序预测结果与真实排序结果之间的差距大于等于 1，则损失为 0；否则，损失为 $1 - (y_i - y_j) \cdot (F(\boldsymbol{x}_i) - F(\boldsymbol{x}_j))$。
    

2. 计算梯度和残差

    由于损失函数是 pairwise hinge loss 函数，因此我们需要计算每个样本对应的梯度和残差。具体来说，对于每个样本 $\boldsymbol{x}_i$，其梯度 $g_i$ 和残差 $h_i$ 的计算方式如下：

    $$g_i = -\sum_{y_j > y_i} \frac{\partial L(y_i, y_j, \boldsymbol{x}_i, \boldsymbol{x}_j)}{\partial F(\boldsymbol{x}_i)}$$
$$h_i = \sum_{y_j > y_i} \frac{\partial^2 L(y_i, y_j, \boldsymbol{x}_i, \boldsymbol{x}_j)}{\partial F(\boldsymbol{x}_i)^2}$$ 其中，$\frac{\partial L(y_i, y_j, \boldsymbol{x}_i, \boldsymbol{x}_j)}{\partial F(\boldsymbol{x}_i)}$ 和 $\frac{\partial^2 L(y_i, y_j, \boldsymbol{x}_i, \boldsymbol{x}_j)}{\partial F(\boldsymbol{x}_i)^2}$ 分别表示 pairwise hinge loss 函数关于 $F(\boldsymbol{x}_i)$ 的一阶导数和二阶导数。$g_i$ 的含义是样本 $\boldsymbol{x}_i$ 对模型的梯度贡献，$h_i$ 的含义是样本 $\boldsymbol{x}_i$ 对模型的二阶导数贡献。

3. 训练 GBDT 模型
    在 GBRank 中，我们需要训练多个 GBDT 模型来逐步拟合梯度和残差。具体来说，我们首先用样本特征和初始的排序序列训练一个 GBDT 模型 $f_0(\boldsymbol{x})$，然后计算其在训练集中每个样本上的预测值 $\boldsymbol{\hat{y}}_0$。接下来，我们计算每个样本的残差 $h_i$，并将其作为新的排序序列，重新构造训练集，然后使用这个新的训练集和特征，训练一个新的 GBDT 模型 $f_1(\boldsymbol{x})$。重复这个过程，直到训练出 $T$ 个 GBDT 模型为止。最终，模型的预测排序序列为： $$F(\boldsymbol{x}) = \sum_{t=0}^T f_t(\boldsymbol{x})$$ 其中，$f_t(\boldsymbol{x})$ 表示第 $t$ 棵 GBDT 模型的预测输出。
4. 进行排序
训练好 GBDT 模型后，我们可以使用模型对新的样本进行排序。具体来说，对于一个新的样本 $\boldsymbol{x}$，我们计算它的预测排序序列 $\boldsymbol{\hat{y}}$，然后按照预测值从大到小排序，得到最终的排序结果。

## 训练过程: 全部样本依次训练当前树
在 GBRank 模型中，通常每棵树都是使用所有样本进行训练，而不是样本依次训练每棵树。这是因为 GBRank 模型的训练过程中需要计算梯度和 Hessian 矩阵，这些计算都需要对整个训练集进行操作，因此每棵树的训练都需要使用整个训练集。

在 GBRank 模型的训练过程中，首先需要对样本进行排序，然后依次对每个样本计算损失函数的一阶导数和二阶导数，从而得到该样本在当前模型下的梯度和 Hessian 矩阵。然后，使用这些梯度和 Hessian 矩阵来训练一棵新的树，以使得该树能够最大程度地降低损失函数。训练完一棵树之后，就将该树的输出加入到当前模型中，再次对样本进行排序，重新计算梯度和 Hessian 矩阵，以训练下一棵树。重复这个过程，直到达到预设的树的数量或者损失函数的收敛。

因此，每棵树的训练都需要使用整个训练集，而不是样本依次训练每棵树。这种训练方式可以更好地捕捉特征之间的交互关系，从而得到更加准确的排序结果。