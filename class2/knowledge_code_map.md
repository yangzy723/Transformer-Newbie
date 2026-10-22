# 第二课：知识、公式与代码对照

每个知识点按“问题 → 公式 → 关键语句 → 运行结果”阅读。公式中的矩阵用粗体，代码用相同含义的变量名；batch 维默认可以广播。

| 符号 | 含义 | 代码 / 形状 |
| --- | --- | --- |
| $V,D$ | 词表大小、向量维度 | `num_embeddings`、`embedding_dim` |
| $I,\mathbf{E},\mathbf{X}$ | ID、参数表、查表结果 | `ids`、`embedding.weight`、`vectors` |
| $\mathbf{e},\mathbf{W}_{\rm out},\mathbf{z}$ | 单位置向量、输出权重、logits | `embedding_vector`、`output_weight`、`logits` |
| $t,\omega_i,\theta_{t,i}$ | 位置、频率、旋转角 | `positions`、`frequencies`、`angles` |
| $\Pi$ | 行置换矩阵 | `x[permutation]`，不显式建矩阵 |
| $\mathbf{A},\mathbf{M}$ | 注意力权重、加性 mask | `weights`、`masked_fill(..., -inf)` |

单位置交叉熵使用行向量约定；RoPE 的几何推导把每一对分量写成列向量，张量代码直接按分量公式计算，两者表达同一个旋转。正弦和 RoPE 的频率按各自传入的维度计算；多头 RoPE 使用 $d_h$。


## PPT 03｜怎样用语料频次决定下一次合并？

**定位**：Notebook 实验 1；[code/bpe_core.py:train_bpe](code/bpe_core.py)。

$$
S(a,b)=\sum_s f_s\,\operatorname{count}_s(a,b)
$$

f_s：文本频次；count：当前切分中的相邻次数。

```python
counts[pair] += corpus[text]
pair = max(counts, key=counts.get)
```

**观察**：四轮合并频次：10 → 10 → 8 → 8。


## PPT 04｜规则学好以后，编码时先执行哪条？

**定位**：Notebook 实验 2；[code/bpe_core.py:encode_bpe](code/bpe_core.py)。

$$
(a,b)^*=\underset{(a,b)\in\mathcal{C}}{\arg\min}\;\operatorname{rank}(a,b)
$$

C：当前相邻、且在 merges 中的候选 pair。

```python
rank, pair = min(candidates, key=lambda item: item[0])
parts = merge_pair(parts, pair)
```

**观察**：abc → a / bc；不是最长前缀 ab / c。


## PPT 05｜字符级算法怎样落到真实的字节级分词器？

**定位**：Notebook 实验 3；[code/train_tokenizer.py:train](code/train_tokenizer.py)。

**观察**：从本地语料训练，完整配置保存在 JSON 中。


## PPT 06｜为什么空格、标点和字节会影响切分？

**定位**：Notebook 实验 4；[code/tokenization.py:demo](code/tokenization.py)。

**观察**：Hello world!：320 项词表得 9 项，384 项得 5 项。


## PPT 07｜为什么分词器必须和参数表配套？

**定位**：Notebook 实验 4；[code/runtime.py:load_tokenizer](code/runtime.py)。

**观察**：默认配置下：我喜欢喝牛奶。→ [300,304,262]。


## PPT 08｜为什么 ID 要变成向量，查表对应什么运算？

**定位**：Notebook 实验 5；[code/embeddings.py:demo](code/embeddings.py)。

$$
\mathbf{E}\in\mathbb{R}^{V\times D},\quad\mathbf{X}=\mathbf{E}[I]
$$

$$
\mathbf{X}=\operatorname{one\_hot}(I)\,\mathbf{E}
$$

I：[B,T]；E：[V,D]；X：[B,T,D]。

```python
vectors = embedding(ids)
```

**观察**：[2,6] → [2,6,8]；与 one-hot 乘法前向等价。


## PPT 09｜长短不同的句子怎样组成同一个 batch？

**定位**：Notebook 实验 5；[code/runtime.py:make_batch](code/runtime.py)。

**观察**：第一句 5 项补齐到 6；第二句 6 项无需补齐。


## PPT 10｜同一个 token 出现多次，梯度覆盖还是累加？

**定位**：Notebook 实验 6；[code/embeddings.py:demo](code/embeddings.py)。

$$
\mathbf{g}_{b,t}=\frac{\partial L}{\partial\mathbf{X}_{b,t}}
$$

$$
\frac{\partial L}{\partial\mathbf{E}_j}=\sum_{(b,t):\,I_{b,t}=j}\mathbf{g}_{b,t}
$$

对非 pad 行 j；遍历 batch 中所有读取它的位置。

```python
repeated = torch.tensor([[a, a, b, pad]])
embedding(repeated).sum().backward()
```

**观察**：每维梯度：a 行 2，b 行 1，pad 行 0。


## PPT 11｜预测错了，误差信号怎样传回输入向量？

**定位**：Notebook 实验 7；[code/gradient_chain.py:demo](code/gradient_chain.py)。

$$
\mathbf{z}=\mathbf{e}\mathbf{W}_{\rm out}^{\mathsf{T}},\quad L=-\log p_y
$$

$$
\frac{\partial L}{\partial\mathbf{e}}=(\mathbf{p}-\operatorname{one\_hot}(y))\mathbf{W}_{\rm out}
$$

p=softmax(z)；W_out：[V,D]；向量梯度：[D]。

```python
logits = embedding_vector @ output_weight.T
loss = F.cross_entropy(logits, target)
loss.backward()
```

**观察**：手算向量梯度与 autograd 逐项一致。


## PPT 12｜把梯度放进训练循环后，参数是否真的变化？

**定位**：Notebook 实验 8；[code/embedding_train.py:train](code/embedding_train.py)。

$$
\mathbf{e}=\mathbf{E}[x],\quad\mathbf{z}=\mathbf{e}\mathbf{W}_{\rm out}^{\mathsf{T}}+\mathbf{b}
$$

$$
L=-\frac{1}{N}\sum_{r=1}^{N}\log p(y_r\mid x_r)
$$

x/y：相邻 token；N=57；只用当前 token 预测下一项。

```python
logits = output(embedding(x))
loss = F.cross_entropy(logits, y)
loss.backward()
```

**观察**：120 步：loss 5.7565 → 0.3559；26 行更新。


## PPT 13｜输入表和输出分类器共用参数，会发生什么？

**定位**：课后扩展验证；[tools/check_experiments.py:main](code/../tools/check_experiments.py)。

$$
\mathbf{z}=\mathbf{h}\mathbf{E}^{\mathsf{T}}+\mathbf{b}
$$

$$
\nabla_{\mathbf{E}}L=\nabla_{\mathbf{E}}^{\rm lookup}L+\nabla_{\mathbf{E}}^{\rm output}L
$$

输入与输出共用同一个参数；主训练实验保持未绑定。

**观察**：课后验证：没作为输入的行也可能得到输出梯度。


## PPT 14｜为什么仅有 token 向量，还没有显式位置坐标？

**定位**：Notebook 实验 9；[code/position_mechanics.py:attention](code/position_mechanics.py)。

$$
\mathbf{S}'=\Pi\mathbf{S}\Pi^{\mathsf{T}}
$$

$$
\operatorname{Attn}(\Pi\mathbf{X})=\Pi\operatorname{Attn}(\mathbf{X})
$$

Π：行置换矩阵；假设无位置机制、无 mask。

```python
scores = x @ x.T / math.sqrt(x.shape[-1])
return scores.softmax(dim=-1) @ x
```

**观察**：重排后输出也重排：True；加入位置后通常 False。


## PPT 15｜怎样在不增加向量维度的前提下加入位置？

**定位**：Notebook 实验 10；[code/positions.py:sinusoidal](code/positions.py)。

$$
\mathbf{X}_{b,t}=\mathbf{E}[I_{b,t}]+\operatorname{PE}(t)
$$

t：token 序号；PE 来自学习式位置表或固定函数。

**观察**：相加后形状保持 [B,T,D]；本课代码演示固定函数。


## PPT 16｜正弦编码怎样从位置和频率生成一个向量？

**定位**：Notebook 实验 10；[code/positions.py:sinusoidal](code/positions.py)。

$$
\omega_i=10000^{-2i/D},\quad\theta_{t,i}=t\omega_i
$$

$$
\operatorname{PE}(t,2i)=\sin\theta_{t,i},\quad\operatorname{PE}(t,2i+1)=\cos\theta_{t,i}
$$

i：维度对编号；D=4 时 ω=[1,0.01]。

```python
angles = positions[:, None] * frequencies[None, :]
result[:, 0::2] = angles.sin()
result[:, 1::2] = angles.cos()
```

**观察**：位置 0：[0,1,0,1]；不同维度对有不同变化尺度。


## PPT 17｜为什么按绝对位置旋转，却产生相对位移？

**定位**：Notebook 实验 11；[code/positions.py:apply_rope](code/positions.py)。

$$
R(\varphi)=\begin{bmatrix}\cos\varphi&-\sin\varphi\\\sin\varphi&\cos\varphi\end{bmatrix}
$$

$$
\widetilde{\mathbf{q}}_m^{\mathsf{T}}\widetilde{\mathbf{k}}_n=\mathbf{q}^{\mathsf{T}}R((n-m)\omega)\mathbf{k}
$$

m/n：位置；ω：某一维度对的频率；φ=tω。

```python
cos, sin = angles.cos(), angles.sin()
result[..., 0::2] = even * cos - odd * sin
result[..., 1::2] = even * sin + odd * cos
```

**观察**：固定 q/k 内容：点积由相对位置 n−m 调节。


## PPT 18｜二维旋转怎样成为一个多头张量运算？

**定位**：Notebook 实验 11；[code/positions.py:apply_rope](code/positions.py)。

$$
x'_{2i}=x_{2i}\cos\theta_{t,i}-x_{2i+1}\sin\theta_{t,i}
$$

$$
x'_{2i+1}=x_{2i}\sin\theta_{t,i}+x_{2i+1}\cos\theta_{t,i}
$$

even/odd 是相邻分量；角度广播到 batch 和 head。

```python
even, odd = x[..., 0::2], x[..., 1::2]
angles = positions[:, None] * frequencies[None, :]
```

**观察**：范数保持；整体平移位置后分数误差约 10⁻¹⁵。


## PPT 19｜有了位置信息，为什么还需要 mask？

**定位**：Notebook 实验 12；[code/input_pipeline.py:demo](code/input_pipeline.py)。

$$
\mathbf{S}=\frac{\widetilde{\mathbf{Q}}\widetilde{\mathbf{K}}^{\mathsf{T}}}{\sqrt{d_h}}
$$

$$
\mathbf{A}=\operatorname{softmax}(\mathbf{S}+\mathbf{M})
$$

M：可见为 0，禁止为 −∞；softmax 沿 key 维。

```python
masked = scores.masked_fill(forbidden, -torch.inf)
weights = masked.softmax(dim=-1)
```

**观察**：未来 key 与 padding key 的权重均为 0。


## PPT 20｜真实输入的形状能否一路接到 Attention？

**定位**：Notebook 实验 12；[code/input_pipeline.py:demo](code/input_pipeline.py)。

**观察**：ID [2,6] → X [2,6,8] → Q/K [2,2,6,4] → 输出 [2,6,8]。
