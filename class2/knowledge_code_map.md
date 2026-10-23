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

- 演示语料：我喜欢喝奶茶×5、你喜欢喝奶茶×3、我喜欢喝咖啡×2。
- 第一轮“喜+欢”频次为5+3+2=10；“我+喜”只有5+2=7。
- 先合并喜+欢，再重新统计；四轮依次得到喜欢、喜欢喝、喜欢喝奶、喜欢喝奶茶。

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

- 固定上一页学到的四条 merges，对“你喜欢喝奶茶”逐轮执行。
- 编码时不再统计训练语料，也不更新 merges，只选择当前候选的最小 rank。
- 遇到重叠候选时，优先级比片段长度重要；abc 的反例得到 a/bc。

$$
(a,b)^*=\underset{(a,b)\in\mathcal{C}}{\arg\min}\;\operatorname{rank}(a,b)
$$

C：当前相邻、且在 merges 中的候选 pair。

```python
rank, pair = min(candidates, key=lambda item: item[0])
parts = merge_pair(parts, pair)
```

**观察**：你喜欢喝奶茶 → 你 / 喜欢喝奶茶；重叠反例 abc → a / bc。


## PPT 05｜字符级算法怎样落到真实的字节级分词器？

**定位**：Notebook 实验 3；[code/train_tokenizer.py:train](code/train_tokenizer.py)。

- 字符演示说明合并规则；真实库还要处理文本边界、基础字节和解码。
- 本课 ByteLevel 用完整256字节字母表覆盖UTF-8字节，并设置4个特殊符号。
- 保存的不只是词表，还包括merges、预切分和decoder；之后加载完整配置。

**观察**：从本地语料训练，完整配置保存在 JSON 中。


## PPT 06｜为什么分词器必须和参数表配套？

**定位**：Notebook 实验 4；[code/runtime.py:load_tokenizer](code/runtime.py)。

- 默认配置把“我喜欢喝牛奶。”编码为[300,304,262]，这些ID是参数表的行地址。
- 另一份同样大小的词表，也可能给同一片段分配不同ID，直接替换会读错参数。
- bos/eos由本课batch代码显式加入；pad=0只用于补齐，不代表真实文本。

**观察**：默认配置下：我喜欢喝牛奶。→ [300,304,262]。


## PPT 07｜为什么 ID 要变成向量，查表对应什么运算？

**定位**：Notebook 实验 5；[code/embeddings.py:demo](code/embeddings.py)。

- 查表等于用one-hot选择一行，既保留可学习参数，又避免显式分配大one-hot张量。
- 算术小表中，ID=[1,2,1]会读到第1行、第2行、再读一次第1行。
- 行向量最初可以随机初始化；训练任务逐步调整它们，每一维没有预先规定的语义。

**页内数值算例**：[worked_examples.py:lookup_example](code/worked_examples.py)。

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


## PPT 08｜长短不同的句子怎样组成同一个 batch？

**定位**：Notebook 实验 5；[code/runtime.py:make_batch](code/runtime.py)。

- 第一句3个普通token加bos/eos得5项；第二句4个加标记得6项，因此前者补1个pad。
- 补齐只统一张量形状；Attention仍应屏蔽pad key，训练loss仍应忽略pad标签。
- padding_idx抑制lookup路径的pad行梯度，不会替后续网络自动处理mask。

**观察**：第一句 5 项补齐到 6；第二句 6 项无需补齐。


## PPT 09｜同一个 token 出现多次，梯度覆盖还是累加？

**定位**：Notebook 实验 6；[code/embeddings.py:demo](code/embeddings.py)。

- 把同一行看成被多处引用的共享参数；每个位置贡献自己的向量梯度。
- 若ID1两次上游梯度为[1,0]和[0,2]，第1行收到[1,2]，不是取平均也不是覆盖。
- 求和损失只是特殊情况：上游全为1，出现两次的行每一维梯度为2。

**页内数值算例**：[worked_examples.py:gradient_example](code/worked_examples.py)。

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


## PPT 10｜预测错了，误差信号怎样传回输入向量？

**定位**：Notebook 实验 7；[code/gradient_chain.py:demo](code/gradient_chain.py)。

- 类别编号0/1/2，目标y=1，预测p=[0.1,0.7,0.2]；交叉熵为−ln(0.7)≈0.3567。
- 类别误差p−one_hot(1)=[0.1,−0.3,0.2]，右乘输出权重得到输入向量梯度。
- 小算例得到[-0.1,-0.2]，与autograd一致；真实词表只是把类别数扩到V。

**页内数值算例**：[worked_examples.py:cross_entropy_example](code/worked_examples.py)。

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


## PPT 11｜把梯度放进训练循环后，参数是否真的变化？

**定位**：Notebook 实验 8；[code/embedding_train.py:train](code/embedding_train.py)。

- 把相邻token构成(x,y)：当前ID查表，经Linear产生所有候选的logits，再与真实下一ID比较。
- 每步依次前向、清梯度、backward、optimizer.step；梯度负责指出如何调整参数。
- 本例只利用当前token；同一token可能有多个后续，所以训练loss不必降到0。

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


## PPT 12｜输入表和输出分类器共用参数，会发生什么？

**定位**：课后扩展验证；[tools/check_experiments.py:main](code/../tools/check_experiments.py)。

- 不绑定时，输入E与输出W是两个参数；绑定后输出直接使用E的转置。
- 输入查表只读取少数行，输出softmax却把全词表作为候选，因此更多行可能收到梯度。
- padding_idx只限制lookup路径；绑权重并不保证pad行完全没有梯度。

$$
\mathbf{z}=\mathbf{h}\mathbf{E}^{\mathsf{T}}+\mathbf{b}
$$

$$
\nabla_{\mathbf{E}}L=\nabla_{\mathbf{E}}^{\rm lookup}L+\nabla_{\mathbf{E}}^{\rm output}L
$$

输入与输出共用同一个参数；主训练实验保持未绑定。

**观察**：课后验证：没作为输入的行也可能得到输出梯度。


## PPT 13｜为什么仅有 token 向量，还没有显式位置坐标？

**定位**：Notebook 实验 9；[code/position_mechanics.py:attention](code/position_mechanics.py)。

- “我/喜欢/你”换成“你/喜欢/我”，仅查表得到的仍是同一组内容向量。
- 无位置、无mask时，Q/K/V随行重排，scores行列同时重排，输出也对应重排。
- 因果mask带有可读范围约束，不满足这里任意置换的假设；等变不等于输出不变。

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


## PPT 14｜怎样在不增加向量维度的前提下加入位置？

**定位**：Notebook 实验 10；[code/positions.py:sinusoidal](code/positions.py)。

- 内容向量表示“是什么”，位置向量提供“在第几个token”；相加后仍是D维。
- 学习式位置表通过训练更新，直接查询范围受表长L限制；正弦编码由固定函数生成。
- 相加式位置与RoPE是不同接口，本课RoPE分支不先叠加绝对位置。

**页内数值算例**：[worked_examples.py:rope_example](code/worked_examples.py)。

$$
\mathbf{X}_{b,t}=\mathbf{E}[I_{b,t}]+\operatorname{PE}(t)
$$

t：token 序号；PE 来自学习式位置表或固定函数。

**观察**：相加后形状保持 [B,T,D]；本课代码演示固定函数。


## PPT 15｜正弦编码怎样从位置和频率生成一个向量？

**定位**：Notebook 实验 10；[code/positions.py:sinusoidal](code/positions.py)。

- D=4时有两对维度，频率分别为1和0.01；同一位置形成两种变化尺度。
- t=1时角度为[1,0.01]，分别取sin/cos并交错排列，得到4维位置向量。
- 广播[T,1]×[1,D/2]生成全部角度；能计算更远位置不等于模型一定能泛化。

**页内数值算例**：[worked_examples.py:rope_example](code/worked_examples.py)。

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


## PPT 16｜为什么按绝对位置旋转，却产生相对位移？

**定位**：Notebook 实验 11；[code/positions.py:apply_rope](code/positions.py)。

- 一对维度用二维旋转R；转置相当于反向旋转，两次旋转的角度相加。
- q、k按mω、nω旋转后，点积里出现(n−m)ω，因此相对位移影响分数。
- 二维算例m=1,n=3,q=k=[1,0]得到cos(2)≈−0.4161；两位置同时加5结果不变。

**页内数值算例**：[worked_examples.py:rope_example](code/worked_examples.py)。

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


## PPT 17｜二维旋转怎样成为一个多头张量运算？

**定位**：Notebook 实验 11；[code/positions.py:apply_rope](code/positions.py)。

- 把相邻维度分成even/odd两组，分别实现旋转矩阵的两个分量。
- Q/K形状[B,H,T,d_h]，角度[T,d_h/2]广播到batch和head，旋转后形状保持。
- 相邻配对与split-half是不同坐标约定；本实现只旋转Q/K，不旋转V。

**页内数值算例**：[worked_examples.py:rope_example](code/worked_examples.py)。

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


## PPT 18｜有了位置信息，为什么还需要 mask？

**定位**：Notebook 实验 12；[code/input_pipeline.py:demo](code/input_pipeline.py)。

- 位置机制改变分数的数值关系，因果mask禁止读取未来，padding key mask禁止读取补齐项。
- 禁止位置设为−∞，softmax后权重为0；示意中最后一个key为pad，整列权重为0。
- pad query输出清零与loss忽略标签仍需各自处理；全屏蔽行需要避免softmax产生NaN。

**页内数值算例**：[worked_examples.py:mask_example](code/worked_examples.py)。

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


## PPT 19｜真实输入的形状能否一路接到 Attention？

**定位**：Notebook 实验 12；[code/input_pipeline.py:demo](code/input_pipeline.py)。

- tokenizer先生成离散ID，Embedding读取参数，线性层一次投影出Q/K/V。
- 把D=8拆成H=2个head，每头d_h=4；RoPE只处理Q/K，mask在scores上生效。
- softmax沿key维，再乘V并合并head；该脚本参数随机，未训练完整Block。

**观察**：ID [2,6] → X [2,6,8] → Q/K [2,2,6,4] → 输出 [2,6,8]。
