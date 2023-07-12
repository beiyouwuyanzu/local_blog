---
title: llama bloom chatglm对比
date: 2023-07-08 21:04:50
mathjax: true
---


# llama bloom chatglm对比
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307101650272.png)


## Casual Decoder vs Prefix Decoder
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307101655098.png)
蓝色：the attention between prefix tokens
绿色：the attention between prefix and target tokens
黄色：the attention betweetn target tokens and masked attention
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307101705035.png)

### 二者区别
以下是一些考虑因素： 
1. **实时性要求**：如果需要实时处理和输出部分解码结果，Casual decoder可能更适合。它能够即时处理和输出部分解码结果，而不需要等待所有输入符号。这对于实时通信和流式数据处理很有用。 
2. **唯一解码要求**：如果解码结果必须是唯一的，即给定一个输入序列只能有一种正确的解码方式，那么Prefix decoder是更合适的选择。前缀码的特性保证了唯一解码。 
3. **数据传输效率**：在某些情况下，Casual decoder可能具有更高的数据传输效率。由于它可以即时处理和输出部分解码结果，可以更早地获取解码的信息，而不需要等待所有输入符号。 
4. **错误容忍性**：如果输入符号中可能存在错误或丢失的情况，Prefix decoder可能更具容错性。由于它需要等待所有输入符号，可以在收到所有输入后进行完整的解码，并且能够检测到错误或丢失的符号。 综上所述，没有一个解码器可以被普遍视为更好，选择合适的解码器取决于具体的需求和约束条件。对于某些应用场景，Casual decoder可能更合适，而在其他情况下，Prefix decoder可能更适用。重要的是根据具体的需求和应用场景来评估和选择合适的解码器。

## RoPE位置编码

对于向量 q，使用 RoPE 添加位置编码后，用复数可以表示为：$q e^{i m \theta}$，其中 $\theta_i=10000^{-2i/d}​$ 与经典的 Transformer 绝对位置编码相同。通过将向量以复数方式推理， RoPE 巧妙地实现了以添加绝对位置编码的方式，在 attention 中计算了相对位置信息。

RoPE 命名联系到了复数乘法的几何意义：对于复数 $z=a+bi$，他可以表示为复平面上的向量。其中 x 轴为实部，y 轴为虚部。根据向量与 x 轴的夹角 $ϕ$，我们可以将向量表示为 $z=L(cosϕ+i * sinϕ)$ ，其中 L 为向量模长。因此，向量的乘法变为了：
$$
z_1 z_2=L_1 L_2\left(\cos \left(\phi_1+\phi_2\right)+\sin \left(\phi_1+\phi_2\right)\right)
$$

所以，复数乘法也可以看作在复平面上的向量旋转及转换。

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307102042004.png)

## 如何理解Sinusoidal Position Embedding里面任意两个相同距离时间步之间的向量距离也相同
> https://www.qinglite.cn/doc/12476476a18a86265

在Sinusoidal Position Embedding中，每个时间步的位置向量由两个正弦函数和余弦函数的组合构成。具体而言，对于每个时间步t和每个维度i，位置向量的计算公式如下：

$PE(t, i) = sin(t / 10000^{(2i/d_model)})$，当i为偶数 $PE(t, i) = cos(t / 10000^{(2i/d_model)})$，当i为奇数

其中，t表示时间步，i表示维度，d_model表示位置向量的维度。

由于正弦函数和余弦函数的周期性特点，当时间步之间的距离相同时，位置向量中对应维度的数值是相同的。这意味着任意两个相同距离的时间步之间的向量距离也相同，因为它们在相同的维度上具有相同的数值。

## ALiBi 位置编码
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307102108595.png)
直接走预设

### ALiBi和RoPE都是用于在Transformer模型中引入位置信息的方法，但RoPE在处理长序列时表现更好


## SwiGLU 激活函数
#### Gated Linear Unit
GLU激活函数的定义为：$GLU(x) = x ⊗ σ(g(x))$。其中，x是输入向量，⊗表示逐元素相乘，σ表示Sigmoid函数，g(x)是通过全连接层或卷积层得到的中间向量。

GLU激活函数通过门控机制实现对输入的选择性过滤，帮助网络捕捉长期依赖关系和上下文信息。这种机制在处理序列数据和NLP任务中具有重要的作用，可以提高模型的性能和效果


```
import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        gate = torch.sigmoid(self.linear(x))
        output = x * gate
        return output
```

SwiGLU是一种激活函数，是GLU的一种变体。SwiGLU的定义如下：$SwiGLU (x, W, V, b, c, β) = Swish β (x W + b) ⊗ (x V + c)$。$Swish β (x) = x * sigmoid(βx)$，其中$β$是指定常数

SwiGLU激活函数的优点是它结合了SWISH和GLU两者的特点 [[url1]](https://blog.csdn.net/qinduohao333/article/details/131085549)。Swish激活函数具备无上界有下届、平滑、非单调的特性，Swish在深层模型上效果优于ReLU [[url2]](https://zhuanlan.zhihu.com/p/630197707)。GLU（Gated Linear Unit）激活函数是一种用于神经网络的激活函数，它具有门控机制，可以帮助网络更好地捕捉序列数据中的长期依赖关系[[3]](https://zhuanlan.zhihu.com/p/621058772)。

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307111540793.png)

非单调激活函数的好处是它可以避免梯度消失问题，从而提高模型的训练效率

## GeLU
$$
\operatorname{GELU}(\mathrm{x})=0.5 \mathrm{x}\left(1+\tanh \left[\sqrt{2 / \pi}\left(\mathrm{x}+0.044715 \mathrm{x}^3\right)\right]\right)
$$
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307111601795.png)
GELU的优点是，它在处理负数时不会像ReLU一样将输入裁剪到0，这可能导致梯度消失的问题。

-   具有更光滑的导数：

-   GELU函数的导数是连续的，这使得在训练深度神经网络时可以更容易地传播梯度，避免了ReLU函数在 处的导数不连续的问题，从而减少了训练过程中出现的梯度消失问题

-   可以加速收敛：

-   GELU函数在激活函数的非线性变换中引入了类似于sigmoid函数的变换，这使得GELU函数的输出可以落在一个更广的范围内，有助于加速模型的收敛速度。

### SwiGLU和GeGLU的对比

1.  定义：
    -   SwiGLU：SwiGLU的定义为Swish(xW + b) ⊗ (xV + c)，其中Swish是带有参数β的Swish激活函数。
    -   GeGLU：GeGLU的定义为GELU(xW) ⊗ xV，其中GELU是高斯误差线性单元激活函数。

2.  激活函数比较：
    -   SwiGLU：SwiGLU是GLU的变种，它使用Swish激活函数而不是ReLU或GELU。Swish是ReLU的平滑近似，对于负值具有非零梯度。由于其平滑性、非单调性和门控机制，SwiGLU在各种任务中表现优于其他激活函数，包括Swish和GLU。
    -   GeGLU：GeGLU是GLU的另一种变种，它使用GELU激活函数。GELU是ReLU的平滑近似，在GPT-2和BERT等语言模型和Transformer模型中使用。它避免了梯度消失的问题，并且在0处具有连续的导数，有时可以加快训练速度。



## normal
#### layer normal
$\text{LayerNorm}(x_i) = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$

其中 $x_i$ 表示输入的第 $i$ 个特征，$\mu$ 和 $\sigma$ 分别表示这个特征的均值和标准差，$\epsilon$ 是一个很小的数

Layer normalization 的计算是在每个样本的特征维度上进行的

通过标准化、缩放和平移操作，Layer normalization 可以使每个样本的特征在不同样本之间更加一致，有助于减少不同样本之间的差异，从而提高模型的泛化能力。此外，Layer normalization 还可以帮助解决神经网络中的梯度消失和梯度爆炸问题，促进模型的收敛和训练效果

#### rms normal

RMSNorm（Root Mean Square Layer Normalization）是一种深度学习中的归一化方法，它与 Layer Normalization 的主要区别在于去掉了减去均值的部分，计算公式为：$$\text{RMSNorm}(a_i) = \frac{a_i}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2}}$$ 其中 $a_i$ 表示输入的第 $i$ 个特征，$n$ 表示特征的数量。

当应用RMS Normalization时，信号的均值并不影响信号的能量，**因为均值只是信号的直流分量，而能量主要集中在信号的变化部分**。因此，RMS Normalization仅使用信号的均方根值进行缩放，而不考虑均值。


而且节约时间和效率


## deep norm
Deep Norm 是一种深度学习中的归一化方法，它是对 Layer Normalization 的改进，它在每个子层之间添加了一个归一化层，以便在不同的子层之间进行归一化。Deep Norm 的计算公式如下：

$$
x_{l+1}=L N\left(\alpha x+G_l\left(x_l, \theta_l\right)\right)
$$

其中， $\alpha$ 是一个常数（ $\alpha$>1 ），  $G_l\left(x_l, \theta_l\right)$是参数为 $\theta_l$的第 $l$个Transformer子层（即注意力或前馈网络）的函数。DeepNet还将残差内部的权重$\theta_l$ 扩展了常数参数 $\beta$ 。

作者通过实验证实了Deep Norm在训练深层transformer模型的时候具备近乎恒定的更新规模，成功训练了1000层transformer的模型，认为Deep Norm在具备 _ **Post-LN 的良好性能_ 的同时又有 _Pre-LN 的稳定训练**_
![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202307112034774.png)
从上图可看出，相比于Post-LN结构梯度分布的不稳定，Pre-LN在各层之间梯度范数几乎保持不变，这种结构明显更利于优化器进行优化。而在进行一定轮数的 warm-up后，Post-LN的梯度范数也基本保持不变，并且其量级非常小(上图中绿色)，这也验证了Post-LN在warm-up阶段的训练不稳定性问题。
通过以上实验可发现，当使用Pre-LN结构时，warm-up阶段已不再是必需，并且Pre-LN结构可以大幅提升Transformer的收敛速度。对于机器翻译任务（IWSLT/WMT)，不需要warm-up的Pre-LN结构可以比Post-LN收敛快1倍左右，而在BERT上，Pre-LN在下游任务上达到和Post-LN相同的性能也只需要后者迭代轮数的1/3左右，并且最终的效果也更好。
