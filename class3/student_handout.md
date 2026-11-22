# 第三课讲义：向量怎样吸收上下文

本课先复习前两课，再用一组能手算的小矩阵计算单头因果Attention。矩阵行对应token位置，代码位置编号从0开始：第三行是位置2。

## 1．恢复整条数据流

```text
文本 → Tokenizer → token IDs
→ Embedding → 多层Transformer Block
→ 最终Norm → LM Head → 词表分数
→ 选择下一token，追加到已有序列
```

ID通常是[B,T]，向量是[B,T,D]，LM Head输出是[B,T,V_vocab]。B是序列数量，T是token数，D是模型宽度。
讲义用$V_{\mathrm{vocab}}$表示词表大小，用粗体$\mathbf V$表示Attention中被汇总的内容矩阵。

两次Softmax面对的对象不同：

| 位置 | 最后一维的对象 | 结果用来做什么 |
|---|---|---|
| Attention内部 | 各输入位置 | 混合这些位置提供的V |
| LM Head之后 | 词表token | 形成下一个token的预测分布 |

第一课人工例子中，读取权重约为[0.248,0.248,0.503]，汇总向量约为[0.752,0.752]；
酒/牛奶的候选概率约为[0.680,0.320]。

## 2．BPE训练与编码

语料“我喜欢喝奶茶”×5、“你喜欢喝奶茶”×3、“我喜欢喝咖啡”×2，最初按字符切分。
“喜／欢”在三句话各出现一次，所以计数为$5+3+2=10$。

```python
counts[pair] += corpus[text]
pair = max(counts, key=counts.get)
```

第一行累加当前相邻组合的加权次数；第二行返回计数最多的组合。

合并以后：

```text
我 / 喜 / 欢 / 喝 / 奶 / 茶
→ 我 / 喜欢 / 喝 / 奶 / 茶
```

原先的“我＋喜”“喜＋欢”“欢＋喝”消失，出现“我＋喜欢”“喜欢＋喝”，所以按新切分重新统计。

编码执行已经冻结的规则：

| rank | 合并 |
|---|---|
| 0 | 喜＋欢 → 喜欢 |
| 1 | 喜欢＋喝 → 喜欢喝 |
| 2 | 喜欢喝＋奶 → 喜欢喝奶 |
| 3 | 喜欢喝奶＋茶 → 喜欢喝奶茶 |

“你喜欢喝奶茶”最终为：

```text
你 / 喜欢喝奶茶
```

共两个token。编码每轮从当前相邻、且存在于合并表中的候选中，选最小rank继续合并。
字符教学例与真实ByteLevel配置使用不同的片段和ID。

## 3．Embedding与位置

Embedding是一张参数表：

$$
\mathbf E\in\mathbb R^{V_{\mathrm{vocab}}\times D},
\qquad \mathbf x_p=\mathbf E[I_p].
$$

相同ID读取同一参数行。训练从预测loss反向计算梯度，再由优化器更新参数；
同一ID出现多次时，各位置贡献的梯度累加到同一行。`loss.backward()`计算梯度，`optimizer.step()`更新参数。

相加式位置编码：

$$
\mathbf x_p=\mathbf E[I_p]+\mathbf P_p.
$$

$\mathbf P_p$可以由可训练位置表查询，也可以由多频率sin/cos生成。

RoPE在Q/K投影之后、点积之前旋转每对分量：

$$
\theta_{p,i}=p\omega_i,\qquad
\omega_i=10000^{-2i/d_k}.
$$

$$
\begin{aligned}
a'&=a\cos\theta-b\sin\theta,\\
b'&=a\sin\theta+b\cos\theta.
\end{aligned}
$$

图解设未旋转q=k=[1,0]，每格转30°：位置1/3对应30°/90°，位置3/5对应90°/150°，
两次夹角均60°，单位向量点积均为0.5。一般情况下内容也影响夹角和分数。
实际RoPE使用弧度和各维度对的频率，本课接口复用第二课实现。

## 4．同一输入经过三个可训练投影

主例直接给定三行输入，方便手算；RoPE在Notebook最后单独接回。

$$
\mathbf X=
\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},
\quad
\mathbf W_Q=\begin{bmatrix}1&0\\0&2\end{bmatrix},
\quad
\mathbf W_K=\begin{bmatrix}1&1\\0&1\end{bmatrix},
\quad
\mathbf W_V=\begin{bmatrix}1&-1\\2&0\end{bmatrix}.
$$

$$
\mathbf Q=\mathbf X\mathbf W_Q,\qquad
\mathbf K=\mathbf X\mathbf W_K,\qquad
\mathbf V=\mathbf X\mathbf W_V.
$$

$$
\mathbf Q=\begin{bmatrix}1&0\\0&2\\1&2\end{bmatrix},\quad
\mathbf K=\begin{bmatrix}1&1\\0&1\\1&2\end{bmatrix},\quad
\mathbf V=\begin{bmatrix}1&-1\\2&0\\3&-1\end{bmatrix}.
$$

Q用于发起匹配，K用于接受匹配，V提供实际被读取的内容。三张W是参数，
当前Q/K/V和注意力权重是根据输入计算出的中间结果。

```python
q = x @ w_q
k = x @ w_k
v = x @ w_v
```

手算“喝”的V：

$$
[1,1]\mathbf W_V=[1\times1+1\times2,\;1\times(-1)+1\times0]=[3,-1].
$$

这里采用行向量右乘。`nn.Linear.weight`存为[out,in]，对应计算是`x @ weight.T + bias`。

## 5．一行匹配与完整矩阵

最后位置的Q是$\mathbf q_2=[1,2]$：

$$
\mathbf q_2\mathbf K^\mathsf T=[3,2,5].
$$

全体查询一起计算：

$$
\mathbf S_{\mathrm{raw}}=\mathbf Q\mathbf K^\mathsf T
=\begin{bmatrix}1&0&1\\2&2&4\\3&2&5\end{bmatrix}.
$$

| | K：我 | K：喜欢 | K：喝 |
|---|---:|---:|---:|
| Q：我 | 1 | 0 | 1 |
| Q：喜欢 | 2 | 2 | 4 |
| Q：喝 | 3 | 2 | 5 |

行是查询位置，列是被读取的位置。[T,d_k]乘[d_k,T]得到[T,T]。

## 6．缩放点积的尺度

$$
\mathbf S=\frac{\mathbf Q\mathbf K^\mathsf T}{\sqrt{d_k}}.
$$

点积累加$d_k$个乘积。假设q/k分量独立、零均值、单位方差，并且不同维乘积项互不相关，
则点积方差为$d_k$。除以$\sqrt{d_k}$后，该分析下的方差为1。

数值差过大时Softmax容易过度集中。本例第三行不缩放时约[0.114,0.042,0.844]，
除以$\sqrt2$后约[0.178,0.088,0.734]，最大项仍是第三个，分布变得缓和。

```python
scores = q @ k.T / math.sqrt(q.shape[-1])
```

## 7．因果mask与逐行Softmax

位置i允许读取$j\le i$：

$$
M_{ij}=\begin{cases}0,&j\le i,\\-\infty,&j>i.\end{cases}
\qquad
\mathbf M=
\begin{bmatrix}0&-\infty&-\infty\\0&0&-\infty\\0&0&0\end{bmatrix}.
$$

$$
A_{ij}=
\frac{\exp(S_{ij}+M_{ij})}{\sum_r\exp(S_{ir}+M_{ir})}.
$$

由于$\exp(-\infty)=0$，未来位置权重为0。普通分数0的指数是1，因此无法实现屏蔽。
当前位置的输入用于预测下一token，所以对角线保留。每行独立分配总计1的读取比例。

```python
future = torch.ones(3, 3, dtype=torch.bool).triu(1)
masked = scores.masked_fill(future, -torch.inf)
weights = masked.softmax(dim=-1)
```

$$
\mathbf A\approx
\begin{bmatrix}1&0&0\\0.5&0.5&0\\0.178&0.088&0.734\end{bmatrix}.
$$

第一行只能读自己；第二行两个可见分数相同，所以各一半；第三行读取全部位置。

## 8．加权V得到上下文向量

$$
\mathbf Z=\mathbf A\mathbf V,\qquad
\mathbf z_i=\sum_j A_{ij}\mathbf v_j.
$$

第二行（位置1）：

$$
\mathbf z_1=0.5[1,-1]+0.5[2,0]=[1.5,-0.5].
$$

$$
\mathbf Z\approx
\begin{bmatrix}1&-1\\1.5&-0.5\\2.555&-0.912\end{bmatrix}.
$$

```python
output = weights @ v
```

形状[T,T]乘[T,d_v]得到[T,d_v]，每个输入位置仍有一个输出向量。
输出分量可以是负数，行和无需为1。各行已经汇总允许读取的上下文。

## 9．通过干预解释分工

| 实验 | 固定 | 修改 | 观察 |
|---|---|---|---|
| 只改V | Q/K | 第0行V加[2,0] | 权重不变，输出变化 |
| 只改K | Q/V | 第0行K加[1,0] | 第三行权重与输出改变 |
| 改未来输入 | 参数与位置 | 最后X加[3,-2]并重新投影 | mask下前两行输出不变 |

只改V时：

$$
\Delta\mathbf z_i=A_{i,0}[2,0].
$$

三行变化分别为[2,0]、[1,0]、约[0.357,0]。

只改K时，原始分数第0列改变。第一行只可读自己，第二行Q=[0,2]与改变量[1,0]正交，
所以本例前两行权重保持；第三行第0列权重由约0.178变成0.306。

改未来输入时，较早位置的Q没有变，允许读取的K/V没有变；未来K/V被mask屏蔽。
去掉mask对照，本例前两行输出都会变化。Notebook实验09在固定位置的实际RoPE接口上再次验证。

## 10．从单头走向完整Block

每头学习自己的投影并计算注意力：

$$
\operatorname{head}_h=\operatorname{Attention}
(\mathbf X\mathbf W_Q^{(h)},\mathbf X\mathbf W_K^{(h)},\mathbf X\mathbf W_V^{(h)}).
$$

$$
\operatorname{MultiHead}(\mathbf X)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)\mathbf W_O.
$$

第20页使用独立结构配置[B,T,D]=[1,3,4]、H=2，每头宽度2，
拆头为[1,2,3,2]，权重为[1,2,3,3]，拼回[1,3,4]后执行输出投影。
第四课再接残差、Norm与FFN。

## 参考资料

- [Attention Is All You Need `3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)：缩放、多头与因果mask。
- [RoFormer](https://arxiv.org/abs/2104.09864)：RoPE二维旋转与高维推广。
- [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)：权重存储方向。
