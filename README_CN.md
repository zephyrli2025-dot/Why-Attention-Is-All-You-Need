# Attention Is All You Need：学习笔记与简单复现

[English](README.md)

这个仓库是我学习论文 *Attention Is All You Need* 及 Transformer 架构时整理的学习笔记，详实记录了我循序渐进地理解论文及Transformer架构的过程，希望能有幸为和我一样刚接触这一主题的同学提供一个清晰、易懂的起点。

## 仓库会包含什么

- 用尽量平实的语言逐步拆解论文的内容
- 说明明确、可在 CPU 上运行的小型实验
- 方便阅读、修改和动手验证的代码与数据
- 持续记录学习过程中的问题、观察与实现取舍

## 更新计划

- v0：一个小型的4个词的词序还原任务。以CPU完成。
- v1：
- v2：依旧是词还原任务，但将训练范围扩大到小学词汇与更多句式。将尝试引入CUDA加速和显卡计算。
- v3：将尝试发展为关于“开学时间”FAQ的一个小问答模型，模型将从排序升级为排序+答案匹配，不可避免使用显卡完成计算。

## 目录

- [1. Attention 是什么：一个排序实验](#1-attention-是什么一个排序实验)
  - [1.1 我们要解决什么问题](#11-我们要解决什么问题)
  - [1.2 第一步：每个词变成三维向量](#12-第一步每个词变成三维向量)
  - [1.3 给四个输出位置各一个 Query](#13-给四个输出位置各一个-query)
  - [1.4 输入词怎样产生 Key 和 Value](#14-输入词怎样产生-key-和-value)
  - [1.5 Attention 真正出现了：计算匹配分数](#15-attention-真正出现了计算匹配分数)
  - [1.6 Softmax 把分数变成关注比例](#16-softmax-把分数变成关注比例)
  - [1.7 按照 Attention 权重混合 Value](#17-按照-attention-权重混合-value)
  - [1.8 三维向量怎样变成一个词](#18-三维向量怎样变成一个词)
  - [1.9 训练究竟是什么：从预测到 Loss](#19-训练究竟是什么从预测到-loss)
  - [1.10 反向传播怎样修改所有参数](#110-反向传播怎样修改所有参数)
  - [1.11 从第一次更新到学会排序](#111-从第一次更新到学会排序)
  - [1.12 训练后，Embedding 发生了什么](#112-训练后embedding-发生了什么)
  - [1.13 把整个训练过程串起来](#113-把整个训练过程串起来)
  - [1.14 这个模型和真正的 Transformer 有什么区别](#114-这个模型和真正的-transformer-有什么区别)

---

## 1. Attention 是什么：一个排序实验

这一部分将是一个小型的词序还原任务，模型将以乱序接收到四个固定单词，并学习生成正确的词序。值得注意的是：

由于token数量有限（只有4个词+2个首位提示，6个token），因此维度控制在3维（原论文能够达到512维），以便可视化训练过程。
Multi-Head Attention, Positional Encoding, Feed Forward Network, Residual connection, LayerNorm, Encoder stack, autoregressive Decoder, masked self-attention等一系列概念均暂时跳过。
仅用最简化的例子来快速入门Attention干了什么。本阶段将只使用CPU完成训练。

这个项目记录我逐步理解 Transformer 的过程。v0 从一个极简任务开始：输入四个打乱顺序的词，输出固定句子 `I enjoy eating banana`。我们保留 attention 的核心计算，把每一步的中间矩阵、梯度和参数变化都记录下来。

v0 是带可学习输出查询的单头 cross-attention 实验，不是原论文完整 Transformer，也不是自然语言语法学习的验证。本文数字来自随附代码的本次运行；之前讨论中的未核验训练数字已替换。初始 embedding 和 Query 沿用讨论中的三位小数，其余矩阵重新初始化。

### 1.1 我们要解决什么问题

词汇表只有四个词：

$$
\mathcal V=\{\text{banana},\ \text{enjoy},\ \text{eating},\ \text{I}\}.
$$

规定正确句子永远是：

$$
\boxed{\text{I enjoy eating banana}}
$$

训练时，把这四个词打乱。例如：

```text
banana enjoy eating I  → I enjoy eating banana
eating banana I enjoy  → I enjoy eating banana
enjoy I banana eating  → I enjoy eating banana
I eating enjoy banana  → I enjoy eating banana
```

四个不同元素一共有

$$
4!=24
$$

种排列，所以这个小实验可以直接把 24 种排列全部放进训练集。

我们想看的不是模型能不能记住这四个词，而是：**一组随机数字怎样经过 Attention、Loss 和反向传播，逐渐变成一个能完成固定任务的模型。**

### 1.2 第一步：每个词变成三维向量

刚开始，模型完全不知道 `banana` 是什么。每个词只对应 Embedding 矩阵中的一行数字。

这次运行随机初始化出的 Embedding 是：

$$
\begin{aligned}
\text{banana}&=(-0.147,\ 0.786,\ 0.947),\\
\text{enjoy}&=(-1.114,\ 1.691,\ -0.895),\\
\text{eating}&=(-0.356,\ 1.232,\ 0.138),\\
\text{I}&=(-1.682,\ 0.318,\ 0.133).
\end{aligned}
$$

于是，输入

```text
banana enjoy eating I
```

会变成矩阵

$$
X=
\begin{bmatrix}
-0.147 & 0.786 & 0.947\\
-1.114 & 1.691 & -0.895\\
-0.356 & 1.232 & 0.138\\
-1.682 & 0.318 & 0.133
\end{bmatrix}
\in\mathbb R^{4\times3}.
$$

这些数字一开始没有任何语言意义。训练的目的之一，就是修改这些数字。

### 1.3 给四个输出位置各一个 Query

最终要输出四个位置：

```text
位置 1：？
位置 2：？
位置 3：？
位置 4：？
```

模型一开始当然不知道它们应该分别是：

```text
I
enjoy
eating
banana
```

所以，我们给每个输出位置一个三维 Query。这次的初始值是：

$$
Q=
\begin{bmatrix}
0.027 & 0.048 & 0.279\\
0.269 & 0.488 & 0.041\\
0.490 & 0.405 & 0.356\\
-0.184 & -0.092 & -0.145
\end{bmatrix}.
$$

第一个 Query 可以理解成：

> “输出第一个词的时候，我在寻找什么信息？”

不过此时它也是随机的，还没有学会寻找任何东西。

### 1.4 输入词怎样产生 Key 和 Value

接着设置两个需要训练的 $3\times3$ 矩阵：

$$
W_K,\qquad W_V.
$$

输入矩阵 $X$ 通过它们产生 Key 和 Value：

$$
K=XW_K,
\qquad
V=XW_V.
$$

这次初始化得到的 Key 是：

$$
K=
\begin{bmatrix}
1.9395 & 2.1935 & 0.9463\\
-0.0395 & 1.1790 & -1.6189\\
1.1252 & 1.8462 & -0.0008\\
0.3021 & 0.3696 & -1.1813
\end{bmatrix},
$$

Value 是：

$$
V=
\begin{bmatrix}
0.3291 & 0.4707 & 1.9405\\
2.1514 & -1.5105 & 1.2405\\
1.0510 & -0.4843 & 1.5937\\
1.0801 & 0.3861 & 1.1850
\end{bmatrix}.
$$

四行仍然依次对应 `banana`、`enjoy`、`eating`、`I`。

可以先用一个不严格但直观的说法来区分它们：

- Key 用来回答“这个输入和 Query 匹不匹配”；
- Value 保存真正会被取出并混合的信息。

### 1.5 Attention 真正出现了：计算匹配分数

现在计算论文里最核心的式子：

$$
S=\frac{QK^T}{\sqrt{3}}.
$$

$Q$ 是 $4\times3$，$K^T$ 是 $3\times4$，所以 $S$ 是 $4\times4$。第 $i$ 行、第 $j$ 列表示：第 $i$ 个输出位置与第 $j$ 个输入词有多匹配。

先看左上角这一项。第一个 Query 是：

$$
q_1=(0.027,\ 0.048,\ 0.279),
$$

`banana` 的 Key 是：

$$
k_{\mathrm{banana}}=(1.9395,\ 2.1935,\ 0.9463).
$$

把它们做点积，再除以 $\sqrt3$：

$$
\begin{aligned}
S_{11}
&=\frac{q_1\cdot k_{\mathrm{banana}}}{\sqrt3}\\
&=\frac{(0.027)(1.9395)+(0.048)(2.1935)+(0.279)(0.9463)}{\sqrt3}\\
&\approx0.2435.
\end{aligned}
$$

把所有组合都算完，就得到：

$$
S=
\begin{bmatrix}
0.2435 & -0.2287 & 0.0686 & -0.1753\\
0.9416 & 0.2877 & 0.6949 & 0.1231\\
1.2561 & -0.0682 & 0.7498 & -0.0709\\
-0.4018 & 0.0771 & -0.2175 & 0.0472
\end{bmatrix}.
$$

除以 $\sqrt3$ 是为了控制点积的尺度。向量维度变大时，未经缩放的点积容易把 Softmax 推到过于尖锐的区域；缩放能让训练更稳定。

### 1.6 Softmax 把分数变成关注比例

分数还不是比例。对 $S$ 的每一行做 Softmax：

$$
A_{ij}=\frac{e^{S_{ij}}}{\sum_{k=1}^{4}e^{S_{ik}}}.
$$

得到：

$$
A=
\begin{bmatrix}
0.3204 & 0.1998 & 0.2690 & 0.2108\\
0.3646 & 0.1896 & 0.2849 & 0.1608\\
0.4686 & 0.1246 & 0.2824 & 0.1243\\
0.1858 & 0.2999 & 0.2233 & 0.2910
\end{bmatrix}.
$$

例如，第一行可以这样读：

| 第一个输出位置关注的输入 | 权重 |
|---|---:|
| `banana` | 32.04% |
| `enjoy` | 19.98% |
| `eating` | 26.90% |
| `I` | 21.08% |

模型现在什么都没学会。它差不多把注意力分给了四个词，这很符合随机初始化时的状态。

还要注意：$A$ 不是一组被优化器直接修改的参数。它是当前的 $Q$ 和 $K$ 计算出来的结果；参数变化以后，$A$ 才会跟着变化。

### 1.7 按照 Attention 权重混合 Value

接下来，把这些关注比例作用到 Value 上：

$$
\mathrm{Attention}(Q,K,V)=AV.
$$

维度变化是：

$$
(4\times4)(4\times3)\rightarrow4\times3.
$$

这次得到：

$$
C=
\begin{bmatrix}
1.0457 & -0.1999 & 1.5481\\
1.0011 & -0.1907 & 1.5874\\
0.8535 & -0.0565 & 1.6614\\
1.2554 & -0.3613 & 1.4333
\end{bmatrix}.
$$

以第一个输出位置为例：

$$
c_1=
0.3204v_{\mathrm{banana}}
+0.1998v_{\mathrm{enjoy}}
+0.2690v_{\mathrm{eating}}
+0.2108v_I.
$$

这一步可以理解成：

> “根据我对四个词的关注程度，把它们的信息混合起来。”

每个输出位置现在都有了一个新的三维向量，也就是 Context。

### 1.8 三维向量怎样变成一个词

Context 仍然只是三维向量。要在四个词中做选择，还需要一个输出矩阵：

$$
W_{\mathrm{out}}:\mathbb R^3\rightarrow\mathbb R^4.
$$

之所以输出四维，是因为候选词只有四个：

```text
banana
enjoy
eating
I
```

计算过程是：

$$
Z=CW_{\mathrm{out}},
\qquad
P=\mathrm{softmax}(Z).
$$

$Z$ 的每一行有四个 logits，顺序对应 `[banana, enjoy, eating, I]`。第一个输出位置的初始概率是：

$$
P_{1,:}=
\begin{bmatrix}
0.313806 & 0.171390 & 0.042492 & 0.472312
\end{bmatrix}.
$$

这里最大的概率属于 `I`，所以第一个位置暂时预测 `I`。四个位置合起来，初始输出是：

```text
输入：
banana enjoy eating I

模型预测：
I I I I

正确答案：
I enjoy eating banana
```

这个输出看起来很笨，但并不意外：模型还没有训练。

### 1.9 训练究竟是什么：从预测到 Loss

第一位置的正确答案是 `I`，模型给它的概率为：

$$
P(I)=0.472312.
$$

所以这一位置的交叉熵损失是：

$$
L_1=-\ln(0.472312)\approx0.750115.
$$

其他三个位置也按同样的方法计算，最后再对四个位置和 24 个训练样本取平均。这次初始化的整体 loss 是：

$$
\boxed{L=1.72783733}.
$$

这里有一个值得比较的基准。如果四个词完全等概率，正确词的概率就是 $1/4$：

$$
-\ln\frac14=\ln4\approx1.3863.
$$

随机初始化不保证概率恰好均匀，因此最初的 loss 可能比这个基准更高。现在的关键不是它离基准差多少，而是我们终于有了一个数字，用来回答：**模型这次错了多少。**

### 1.10 反向传播怎样修改所有参数

有了 loss，PyTorch 就可以沿着刚才的计算路线反向求导：

$$
\frac{\partial L}{\partial Q},\qquad
\frac{\partial L}{\partial W_K},\qquad
\frac{\partial L}{\partial W_V},\qquad
\frac{\partial L}{\partial E},\qquad
\frac{\partial L}{\partial W_{\mathrm{out}}}.
$$

这里的 $E$ 就是四个词的 Embedding。也就是说，一次反向传播会算出：每个可训练数字朝哪个方向移动，能让 loss 下降。

如果先用最简单的梯度下降来表示一次更新，它写成：

$$
\theta_{\mathrm{new}}
=
\theta_{\mathrm{old}}
-\eta\nabla_\theta L.
$$

其中 $\theta$ 包括：

- 四个词的 Embedding；
- 四个输出位置的 Query；
- $W_K$；
- $W_V$；
- $W_{\mathrm{out}}$。

训练不是程序员告诉模型“`I` 应该关注什么”。程序员只告诉它目标答案，并通过 loss 表达：你的答案错了，而且错了这么多。梯度下降再据此调整所有参数。

### 1.11 从第一次更新到学会排序

第一次更新后，loss 从 `1.72783733` 降到 `1.56460382`。模型还不会排序，但已经朝着让答案更接近目标的方向走了一步。

继续训练，结果如下：

| Epoch | Loss | 单词准确率 | 整句准确率 |
|---:|---:|---:|---:|
| 0 | 1.72783733 | 25% | 0% |
| 1 | 1.56460382 | 50% | 0% |
| 2 | 1.46122802 | 25% | 0% |
| 5 | 1.29390596 | 25% | 0% |
| 10 | 0.96085996 | 50% | 0% |
| 20 | 0.63846311 | 50% | 0% |
| 50 | 0.00924514 | **100%** | **100%** |
| 100 | 0.00017987 | **100%** | **100%** |
| 200 | 0.00009054 | **100%** | **100%** |

![训练损失与训练集准确率](assets/training.png)

到第 50 轮时，24 种输入排列都能得到同一个正确输出。例如：

```text
banana enjoy eating I
        ↓
I enjoy eating banana

eating I enjoy banana
        ↓
I enjoy eating banana

I banana eating enjoy
        ↓
I enjoy eating banana
```

预测变正确的同时，Attention 的分布也发生了变化。下图把训练前后的权重放在一起；横轴是输入 `banana enjoy eating I`，纵轴是四个输出位置。

![训练前后的 attention 热力图](assets/attention.png)

这里有一个很重要的现象：训练后，第一个输出位置最终预测的是 `I`，但它对 `banana` 的 Attention 权重约为 99.95%。这并不矛盾。

模型真正执行的是：

$$
\text{输入}
\rightarrow \text{Value 混合}
\rightarrow W_{\mathrm{out}}
\rightarrow \text{预测}.
$$

Value 和输出矩阵可以共同学出一套内部编码，使来自 `banana` 的 Value 帮助第一个位置输出 `I`。所以，Attention 权重描述信息怎样参与 Context，不等于词语的一一对齐，也不能直接当作因果解释。

还需要谨慎理解这里的“100%”。24 个排列包含的是同一组词，而且这个模型没有位置编码：只要输入的 Key 和 Value 同步换行，最后的 Context 就不变。因此，这个结果证明模型拟合了这个极简任务，并不证明它已经学会一般意义上的语法或排序规则。

### 1.12 训练后，Embedding 发生了什么

训练 200 次更新后，四个词的三维向量变成：

$$
\begin{aligned}
\text{banana}&=(0.766,\ 0.335,\ 1.742),\\
\text{enjoy}&=(-2.194,\ 0.790,\ -3.013),\\
\text{eating}&=(-1.778,\ 1.060,\ 0.747),\\
\text{I}&=(-3.459,\ 1.198,\ 1.441).
\end{aligned}
$$

把起点和终点放在一起，可以直接看到四个词的位置已经发生变化：

| Token | 初始坐标 | 200 次更新后 |
|---|---|---|
| `banana` | (-0.147, 0.786, 0.947) | (0.766, 0.335, 1.742) |
| `enjoy` | (-1.114, 1.691, -0.895) | (-2.194, 0.790, -3.013) |
| `eating` | (-0.356, 1.232, 0.138) | (-1.778, 1.060, 0.747) |
| `I` | (-1.682, 0.318, 0.133) | (-3.459, 1.198, 1.441) |

![四个词训练前后的三维坐标](assets/embeddings_3d.png)

上图只显示起点和终点。把每次更新后的坐标连起来，还能看到它们怎样一步步移动：

![Embedding 的三维训练轨迹](assets/embedding_paths.png)

这里最值得建立的概念是：

> Embedding 不是我们把 `banana` 的语义人工编码成三个数字。它是一个参数。

例如：

$$
e_{\mathrm{banana}}\in\mathbb R^3.
$$

如果把 `banana` 往某个方向移动会让最终 loss 降低，优化器就会不断沿着对任务有用的方向调整它。

这些三维坐标只服务于当前任务，坐标轴没有预先规定的语言含义。换一组随机初始值，模型完全可能停在另一组位置，却仍然得到相同的正确输出。

### 1.13 把整个训练过程串起来

对于输入：

```text
banana enjoy eating I
```

整个前向计算可以压缩成：

```text
Token
  ↓
Embedding
  ↓
X ∈ ℝ⁴ˣ³
  ↓
Q，K，V
  ↓
QKᵀ / √3
  ↓
Softmax
  ↓
A ∈ ℝ⁴ˣ⁴
  ↓
AV
  ↓
C ∈ ℝ⁴ˣ³
  ↓
输出矩阵
  ↓
4 × 4 logits
  ↓
Softmax
  ↓
I enjoy eating banana
```

再把预测与正确答案比较：

```text
预测结果与正确答案
  ↓
Cross Entropy
  ↓
Loss
  ↓
Backpropagation
  ↓
Embedding、Q、W_K、W_V、W_out 全部稍微修改
  ↓
重新计算下一轮
```

Attention 并不是孤立的一张权重表。它处在整条训练链中：前面接收 Embedding、Query、Key 和 Value，后面通过输出概率和 loss 获得修改方向。

### 1.14 这个模型和真正的 Transformer 有什么区别

这个例子还不是原论文中的完整 Transformer，这是有意为之。更准确地说，它是一个：

$$
\boxed{\text{Minimal Attention Sorting Model}}
$$

它暂时去掉了：

- Multi-Head Attention；
- Positional Encoding；
- Feed Forward Network；
- Residual Connection；
- LayerNorm；
- Encoder Stack；
- Autoregressive Decoder；
- Masked Self-Attention。

还有两个容易混淆的区别：

1. 这里的 $Q$ 是四个输出位置各自拥有的可训练向量，不是由输入经过 $XW_Q$ 得到的。因此，它更接近一个带固定输出 Query 的单头 cross-attention 实验，而不是完整的 self-attention。
2. 这里的 $W_{\mathrm{out}}$ 是把 Context 映射到四个词的分类矩阵，不是原论文 Multi-Head Attention 中的输出投影 $W^O$。

这个版本先把最核心的一环单独拿出来：

$$
\boxed{
QK^T
\rightarrow \mathrm{softmax}
\rightarrow V
\rightarrow \text{loss}
\rightarrow \text{gradient}
}
$$

先看清这条链，后续再逐步加入位置编码、多头机制、前馈网络和完整的 Encoder–Decoder 结构。
