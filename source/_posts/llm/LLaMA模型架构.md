---
title: LLaMA的模型架构——RMSNorm/SwiGLU/RoPE/Transformer
date: 2023-05-08 21:04:50
mathjax: true
---

## 1. RMSNorm(均方根)（Root Mean Square）：对每个Transformer子层的输入进行归一化
RMSNorm计算公式: $y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2+\epsilon}}$

其中，$x$是输入向量，$y$是输出向量，$n$是向量的维度，$ϵ$是一个很小的常数，用于避免分母为0的情况。
  #### 一般的LN:
   $$\bar{a}_i=\frac{a_i-\mu}{\sigma} g_i$$
    其中
    $$\mu=\frac{1}{n} \sum_{i=1}^n a_i$$
    
   $$\sigma=\sqrt{\frac{1}{n} \sum_{i=1}^n\left(a_i-\mu\right)^2}$$
    
#### RMS Norm：
    
   $$\bar{a}_i=\frac{a_i}{R M S(a)} g_i$$
    
   其中
    
   $$R M S(a)=\sqrt{\frac{1}{n} \sum_{i=1}^n a_i^2}$$

### 特点
1. RMS Norm是一般LayerNorm的一种变体，可以在梯度下降时令损失更加平滑  
与layerNorm相比，RMS Norm的主要区别在于去掉了减去均值的部分(re-centering)，只保留方差部分(re-scaling)
2. RMS Norm是一种神经网络的归一化方法，可以提高模型的泛化性能。具体而言，RMS Norm是对每个神经元的输出做归一化，使得每个神经元的输出的均值为0，方差为1。这种归一化方法可以使得每个神经元的输出分布更加稳定，降低了不同神经元之间的依赖关系，提高了模型的鲁棒性和泛化性能。
3. RMS Norm的优点在于，它可以有效地降低神经元之间的依赖关系，提高了模型的鲁棒性和泛化性能。另外，RMS Norm与其他归一化方法不同，它不需要对每个mini-batch进行统计，因此可以在训练集较小的情况下获得更好的效果。


## 2. SwiGLU
> 激活函数

1. relu:
𝑅𝑒𝐿𝑈(𝑥)=𝑚𝑎𝑥(0,𝑥)

2. GeLU:
$G e L U=x \Phi(x)=x \int_{-\infty}^x \frac{1}{\sqrt{2 \pi}} e^{-\frac{t^2}{2}} d t=x \cdot \frac{1}{2}\left[1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$

	其中erf为误差函数
3. Swish激活函数：
Swish $=x \cdot \operatorname{sigmoid}(\beta x)$
激活函数就是对x乘以一些数，以对某些值进行约束
4. GLU（Gated Linear Unit），其一般形式为：
$G L U(x)=\sigma(W x+b) \otimes(V x+c)$
- 这里的𝜎可以是𝑠𝑖𝑔𝑚𝑜𝑖𝑑函数，也可以是其它的一些激活函数，其相关变体如下：
$\begin{aligned} & G T U(x, W, V, b, c)=\tanh (x W+b) \otimes \sigma(x V+c) \\ & \operatorname{Bilinear}(x, W, V, b, c)=(x W+b) \otimes(x V+c) \\ & \operatorname{Re} G L U(x, W, V, b, c)=\operatorname{Re} L U(x W+b) \otimes(x V+c) \\ & G E G L U(x, W, V, b, c)=G E L U(x W+b) \otimes(x V+c)\end{aligned}$
$\operatorname{SwiGLU}(x, W, V, b, c, \beta)=\operatorname{Swish}_\beta(x W+b) \otimes(x V+c)$

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/202305081930944.png)

                      左：GeLU，右：Swish
___




> https://www.ai-contentlab.com/2023/03/swishglu-activation-function.html


- Swish(x) = x * sigmoid(beta * x)

```Swish 已被证明在许多应用程序中表现优于 ReLU，尤其是在深度网络中。 Swish 的主要优点是它比 ReLU 更平滑，可以带来更好的优化和更快的收敛。```
- GLU(x) = x * sigmoid(Wx + b)
```GLU 与 Swish 类似，它结合了线性函数和非线性函数。然而，在 GLU 中，线性函数由 sigmoid 激活函数门控。```

- SwiGLU(x) = x * sigmoid(beta * x) + (1 - sigmoid(beta * x)) * (Wx + b)
```在SwiGLU中，Swish函数用于对GLU的线性函数进行门控。这使得 SwiGLU 可以兼顾 Swish 和 GLU 的优势，同时克服各自的劣势。```



### 特点

1. Swish 的主要优点是它比 ReLU 更平滑，可以带来更好的优化和更快的收敛
2. GLU 与 Swish 类似，它结合了线性函数和非线性函数。然而，在 GLU 中，线性函数由 sigmoid 激活函数门控。
3. 在SwiGLU中，Swish函数用于对GLU的线性函数进行门控。这使得 SwiGLU 可以兼顾 Swish 和 GLU 的优势，同时克服各自的劣势。

### 与其他激活函数相比，SwiGLU 的主要优点是：

1. 平滑度：SwiGLU 比 ReLU 更平滑，可以带来更好的优化和更快的收敛。
2. 非单调性：SwiGLU 是非单调的，这使其能够捕获输入和输出之间复杂的非线性关系。
3. 门控：SwiGLU 使用门控机制，允许它根据接收到的输入有选择地激活神经元。这有助于减少过度拟合并提高泛化能力。
4. 性能：SwiGLU 已被证明在各种任务中优于其他激活函数，包括 Swish 和 GLU。

### 代码实现

```
class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        """
 SwiGLU Activation Layer
 """
        super(SwiGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.keras.activations.swish(gate)
        x = tf.multiply(out, gate)
        return x
```

## 3. RoPE

1. RoPE 使用旋转矩阵对绝对位置进行编码，同时将显式相对位置依赖性纳入自注意公式中。
2. RoPE 具有多个优点：序列长度的灵活性、随着相对距离的增加而衰减的令牌间依赖性，以及为线性自注意力配备相对位置编码的能力。

在位置编码上，删除了绝对位置嵌入，而在网络的每一层增加了苏剑林等人(2021)提出的旋转位置嵌入(RoPE)，其思想是采用绝对位置编码的形式，实现相对位置编码


RoPE主要借助了复数的思想，为了引入复数，首先假设了在加入位置信息之前，原有的编码向量是二维行向量![q_m](https://latex.codecogs.com/gif.latex?q_m)和![k_n](https://latex.codecogs.com/gif.latex?k_n)，其中![m](https://latex.codecogs.com/gif.latex?m)和![n](https://latex.codecogs.com/gif.latex?n)是绝对位置，现在需要构造一个变换，将![m](https://latex.codecogs.com/gif.latex?m)和![n](https://latex.codecogs.com/gif.latex?n)引入到![q_m](https://latex.codecogs.com/gif.latex?q_m)和![k_n](https://latex.codecogs.com/gif.latex?k_n)中，即寻找变换：
$$
\tilde{q_m}=f(q, m), \tilde{k_n}=f(k, n)
$$

考虑到Attention的核心计算是内积：
$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

所以，寻求的这个![f(*)](https://latex.codecogs.com/gif.latex?f%28*%29)变换，应该具有特性：
$$
\langle f(q, m), f(k, n)\rangle=g(q, k, m-n)
$$

这里直接说结论，寻求的变换就是![q_me^{im\theta}](https://latex.codecogs.com/gif.latex?q_me%5E%7Bim%5Ctheta%7D)
也就是给![q_m](https://latex.codecogs.com/gif.latex?q_m)乘以![e^{im\theta}](https://latex.codecogs.com/gif.latex?e%5E%7Bim%5Ctheta%7D)，相应地，![k_n](https://latex.codecogs.com/gif.latex?k_n)乘以![e^{in\theta}](https://latex.codecogs.com/gif.latex?e%5E%7Bin%5Ctheta%7D)

做了这样一个变换之后，根据复数的特性，有：
$$
\left\langle q_m, k_n\right\rangle=\operatorname{Re}\left[q_m k_n^*\right]
$$

也就是，如果把二维向量看做复数，那么它们的内积，等于一个复数乘以另一个复数的共轭，得到的结果再取实部，代入上面的变换，也就有：
$$
\left\langle q_m e^{i m \theta}, k_n e^{i n \theta}\right\rangle=\operatorname{Re}\left[\left(q_m e^{i m \theta}\right)\left(k_n e^{i n \theta}\right)^*\right]=\operatorname{Re}\left[q_m k_n^* e^{i(m-n) \theta}\right]
$$


这样一来，内积的结果就只依赖于![(m-n)](https://latex.codecogs.com/gif.latex?%28m-n%29)，也就是相对位置了

换言之，经过这样一番操作，通过给Embedding添加绝对位置信息，可以使得两个token的编码，经过内积变换（self-attn）之后，得到结果是受它们位置的差值，即相对位置影响的

于是对于任意的位置为![m](https://latex.codecogs.com/gif.latex?m)的二维向量![[x, y]](https://latex.codecogs.com/gif.latex?%5Bx%2C%20y%5D)，把它看做复数，乘以![e^{im\theta}](https://latex.codecogs.com/gif.latex?e%5E%7Bim%5Ctheta%7D)，而根据欧拉公式，有：
$$
e^{i m \theta}=\cos m \theta+i \sin m \theta
$$

于是上述的相乘变换也就变成了：
$$
(x+i y) e^{i m \theta}=(x \cos m \theta-y \sin m \theta)+i(x \sin m \theta+y \cos m \theta)
$$

把上述式子写成矩阵形式：
$$
f\left(\left(q_0, q_1\right), m\right)=\left[\begin{array}{cc}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{array}\right]\left[\begin{array}{l}
q_0 \\
q_1
\end{array}\right]
$$

而这个变换的几何意义，就是在二维坐标系下，对向量![(q_0, q_1)](https://latex.codecogs.com/gif.latex?%28q_0%2C%20q_1%29)进行了旋转，因而这种位置编码方法，被称为旋转位置编码

根据刚才的结论，结合内积的线性叠加性，可以将结论推广到高维的情形。可以理解为，每两个维度一组，进行了上述的“旋转”操作，然后再拼接在一起：
$$
\left[\begin{array}{ccccccc}
\cos m \theta_0 & -\sin m \theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_0 & \cos m \theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_1 & -\sin m \theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_1 & \cos m \theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d / 2-1} & -\sin m \theta_{d / 2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d / 2-1} & \cos m \theta_{d / 2-1}
\end{array}\right]\left[\begin{array}{c}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right]
$$

由于矩阵的稀疏性，会造成计算上的浪费，所以在计算时采用逐位相乘再相加的方式进行：

$$
\left[\begin{array}{c}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right] \otimes\left[\begin{array}{c}
\cos m \theta_0 \\
\cos m \theta_0 \\
\cos m \theta_1 \\
\cos m \theta_1 \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_{d / 2-1}
\end{array}\right]+\left[\begin{array}{c}
-q_1 \\
q_0 \\
-q_3 \\
q_2 \\
\vdots \\
-q_{d-1} \\
q_{d-2}
\end{array}\right] \otimes\left[\begin{array}{c}
\sin m \theta_0 \\
\sin m \theta_0 \\
\sin m \theta_1 \\
\sin m \theta_1 \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_{d / 2-1}
\end{array}\right]
$$

----
另一份解释:
> https://nn.labml.ai/transformers/rope/index.html

旋转编码通过在 2D 平面中旋转来变换特征对。也就是说，它将 d 特征组织为 d/2​ 对。每对都可
以被认为是 2D 平面中的一个坐标，编码将根据令牌的位置将其旋转一个角度。
设 $x_m^{(1)}$和  $x_m^{(2)}$ 是 m 位置处任意头的键或查询的两个特征。或者为简单起见，假设 x 只有两个特征。那么转变就是，
$$
\begin{aligned}
\operatorname{RoPE}\left(x_m^{(1)}, x_m^{(2)}, m\right) & =\left(\begin{array}{cc}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{array}\right)\left(\begin{array}{l}
x_m^{(1)} \\
x_m^{(2)}
\end{array}\right) \\
& =\left(\begin{array}{c}
x_m^{(1)} \cos m \theta-x_m^{(2)} \sin m \theta \\
x_m^{(2)} \cos m \theta+x_m^{(1)} \sin m \theta
\end{array}\right)
\end{aligned}
$$

这些特征被分组成对并按上述方式处理。他们对每对使用不同的 θ 。
该论文建议对 d/2​ 特征对使用
$$
\Theta=\theta_i=10000^{\frac{2(i-1)}{d}}, i \in\left[1,2, \ldots, \frac{d}{2}\right]
$$

我们将特征 i 与特征 i+d/2​ 配对。所以对于位置 m ，我们转换
$$
\left(\begin{array}{c}
x_m^{(i)} \\
x_m^{\left(i+\frac{d}{2}\right)}
\end{array}\right)
$$
to
$$
\left(\begin{array}{l}
x_m^{(i)} \cos m \theta_i-x_m^{\left(i+\frac{d}{2}\right)} \sin m \theta_i \\
x_m^{\left(i+\frac{d}{2}\right)} \cos m \theta_i+x_m^{(i)} \sin m \theta_i
\end{array}\right)
$$

### 补充
> [https://blog.eleuther.ai/rotary-embeddings/](https://blog.eleuther.ai/rotary-embeddings/)
> 
简单来说，两个向量之间的点积是单个向量的大小和它们之间的角度的函数。考虑到这一点，RoPE 背后的直觉是我们可以将令牌嵌入表示为复数，并将它们的位置表示为我们应用于它们的纯旋转。如果我们将查询和键移动相同的量，改变绝对位置而不是相对位置，这将导致两个表示以相同的方式额外旋转——正如我们将在推导中看到的——因此角度它们之间将保持不变，因此点积也将保持不变。通过利用旋转的性质，self-attention 中使用的点积将具有我们正在寻找的属性，保留相对位置信息，同时丢弃绝对位置。

![](https://raw.githubusercontent.com/dijiatrustlight/Chart_bed/master/img/3D217A91-3E8D-4E6A-96B6-2245A4A39426.png)
