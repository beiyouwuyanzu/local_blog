---
title: MMoE 模型
date: 2023-05-07 21:04:50
mathjax: true
---

# MMoE 模型
多任务学习的三种架构：1) 共享底部模型，2) 单门混合专家模型 (MoE)，以及 3) 多门混合专家模型 (MMoE)。前两个架构提供上下文并显示最终 MMoE 架构的增量步骤。
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305092028382.png)

1. ## Shared-bottom model
	共享底部模型是最简单和最常见的多任务学习架构。该模型有一个单一的基础（共享底部），所有特定于任务的子网络都从该基础开始。这意味着这种单一表示用于所有任务，与其他任务相比，单个任务无法调整它们从共享底部获得的信息。
2. ## Mixture-of-experts model (MoE)
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305092032003.png)
	专家混合架构通过创建多个专家网络并添加门控网络来对每个专家网络的输出进行加权，从而改进了共享底部模型。
	每个专家网络本质上都是一个独特的共享底层网络，每个网络都使用相同的网络架构。假设每个专家网络都能够学习数据中的不同模式并专注于不同的事物。
	门控网络然后生成一个加权方案，使得任务能够使用专家网络输出的加权平均值，以输入数据为条件。门控网络的最后一层是 $softmax$ 层 ($g(x)$)，用于生成专家网络输出 ($y$) 的线性组合。
	$$
y=\sum_{i=1}^n g(x)_i f_i(x)
$$
这种架构的主要创新在于，该模型能够在每个样本的基础上以不同方式激活网络的各个部分。由于门控网络以输入数据为条件（由于门控网络作为整体模型训练的一部分进行训练），该模型能够学习如何根据输入数据的属性对每个专家网络进行加权。

3. ## Multi-gate mixture-of-experts model (MMoE)
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305092035859.png)
MMoE 架构类似于 MoE 架构，不同之处在于它为每个任务提供了一个单独的门控网络，而不是为整个模型提供一个单独的门控网络。
这允许模型学习每个专家网络的每个任务和每个样本的权重，而不仅仅是每个样本的权重。这允许 MMoE 学习建模不同任务之间的关系。彼此之间几乎没有共同点的任务将导致每个任务的门控网络学习使用不同的专家网络。
MMoE 的作者通过在具有不同任务相关性级别的合成数据集上比较共享底部、MoE 和 MMoE 架构来验证这一结论。
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305092036448.png)
- 首先，我们看到共享底部模型在所有情况下都低于 MoE 和 MMoE 模型。
- 接下来，我们可以看到 MoE 和 MMoE 模型之间的性能差距随着任务之间相关性的降低而增加。
- 这表明 MMoE 能够更好地处理任务彼此无关的情况。任务多样性越大，MMoE 相对于共享底部或 MoE 架构的优势就越大。


## 代码实现
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMoE(nn.Module):
    def __init__(self, input_dim, num_experts, num_tasks, hidden_units):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        self.expert_networks = nn.ModuleList()
        for i in range(num_experts):
            expert_network = nn.Sequential(
                nn.Linear(input_dim, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units)
            )
            self.expert_networks.append(expert_network)
        
        self.gate_networks = nn.ModuleList()
        for i in range(num_tasks):
            gate_network = nn.Sequential(
                nn.Linear(hidden_units, num_experts),
                nn.Softmax(dim=-1)
            )
            self.gate_networks.append(gate_network)
        
    def forward(self, x):
        expert_outputs = []
        for expert_network in self.expert_networks:
            expert_output = expert_network(x)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=2)
        gate_outputs = []
        for gate_network in self.gate_networks:
            gate_output = gate_network(expert_outputs)
            gate_outputs.append(gate_output)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        weighted_expert_outputs = torch.bmm(gate_outputs, expert_outputs)
        final_output = torch.sum(weighted_expert_outputs, dim=2)
        return final_output
```
- 先创建专家层，这是由一组全连接层组成的。我们可以使用PyTorch中的`nn.Linear`模块创建这些层。专家层的数量和每个专家层中的隐藏单元数可以作为`MMoE`类构造函数的输入参数进行指定。专家层的输入维度可以从输入数据的形状中获得
- 接下来创建门层，它由一组门网络组成。每个门网络以专家输出作为输入，产生一组用于组合专家输出的权重。我们也可以使用`nn.Linear`模块来创建门网络。门网络的数量应该等于多任务学习问题中的任务数。
- 最后，我们需要使用门网络产生的权重将专家输出组合起来，以产生最终的输出。我们可以使用`torch.bmm`函数来完成批量矩阵乘法，从而将专家输出与门网络产生的权重相乘，并将结果相加以获得最终的输出。
- 在代码`torch.stack(expert_outputs, dim=2)`中，`dim=2`表示在第2个维度上进行连接。具体来说，在这个例子中，`expert_outputs`是一个形状为`(batch_size, num_experts, expert_output_size)`的3D张量