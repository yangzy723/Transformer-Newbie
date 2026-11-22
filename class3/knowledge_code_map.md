# 第三课知识与代码对照

代码使用小写变量，公式使用大写矩阵；位置从0开始，第三行是位置2。

| 页 | 时间 | 知识 | Notebook实验 | 源码 |
|---|---|---|---|---|
| 1 | 0–2 | 输入一句话，模型怎样继续写？ | 图示/检查 | [code/review.py](code/review.py) |
| 2 | 2–4 | 从文字到下一个 token | 图示/检查 | [code/review.py](code/review.py) |
| 3 | 4–7 | BPE 训练：合并之后重新计数 | 实验01 | [code/review.py](code/review.py) |
| 4 | 7–10 | BPE 编码：按已有规则继续合并 | 实验01 | [code/review.py](code/review.py) |
| 5 | 10–13 | ID 是行号，Embedding 是可训练的表 | 实验01 | [code/review.py](code/review.py) |
| 6 | 13–17 | 内容相同，位置不同 | 实验02 | [code/review.py](code/review.py) |
| 7 | 17–19 | 先只更新“喝”这个位置 | 图示/检查 | [code/review.py](code/review.py) |
| 8 | 19–21 | 三个位置各提供什么，分别读多少？ | 实验03 | [code/review.py](code/review.py) |
| 9 | 21–24 | 先乘读取比例，再把同一列相加 | 实验03 | [code/review.py](code/review.py) |
| 10 | 24–25 | 每个位置各算一次，整句仍有三个向量 | 图示/检查 | [code/review.py](code/review.py) |
| 11 | 25–27 | 查营业时间：问题、索引、正文各管什么？ | 图示/检查 | [code/attention.py](code/attention.py) |
| 12 | 27–28 | 换一个问题，同一份资料的读取比例也会变 | 图示/检查 | [code/attention.py](code/attention.py) |
| 13 | 28–30 | 回到句子：每个位置都有 Q、K、V | 实验04 | [code/attention.py](code/attention.py) |
| 14 | 30–31 | 只拿“喝”的 Q，把一次读取走完整 | 实验04 | [code/attention.py](code/attention.py) |
| 15 | 31–34 | 从 X 算出 Q、K、V | 实验04 | [code/attention.py](code/attention.py) |
| 16 | 34–36 | “喝”与每个位置分别算一个分数 | 实验05 | [code/attention.py](code/attention.py) |
| 17 | 36–38 | 每个位置，都有一行匹配分数 | 实验05 | [code/attention.py](code/attention.py) |
| 18 | 38–40 | 为什么要缩放点积分数？ | 实验05 | [code/attention.py](code/attention.py) |
| 19 | 40–42 | 算“喜欢”时，不能提前看到“喝” | 实验05 | [code/attention.py](code/attention.py) |
| 20 | 42–44 | 一行分数，怎样变成一行读取比例？ | 实验05 | [code/attention.py](code/attention.py) |
| 21 | 44–46 | 用权重把 V 合成新的向量 | 实验05 | [code/attention.py](code/attention.py) |
| 22 | 46–48 | 两次 Softmax，用在两个地方 | 实验03 | [code/review.py](code/review.py) |
| 23 | 48–50 | “我”的 V 变了，哪些结果会变？ | 实验06 | [code/experiments.py](code/experiments.py) |
| 24 | 50–52 | “我”的 K 变了，读取比例怎样变？ | 实验07 | [code/experiments.py](code/experiments.py) |
| 25 | 52–54 | 改动“喝”，前两个位置会受影响吗？ | 实验08 | [code/experiments.py](code/experiments.py) |
| 26 | 54–55 | 将刚才的计算写成程序 | 实验05 | [code/attention.py](code/attention.py) |
| 27 | 55–56 | 多个头，分别计算几组匹配 | 图示/检查 | [code/attention.py](code/attention.py) |
| 28 | 56–60 | 这张注意力矩阵说明了什么？ | 图示/检查 | [code/attention.py](code/attention.py) |

## 核心计算逐项对应


### 从 X 算出 Q、K、V

线性投影重组输入特征，得到用于匹配与汇总的表示。

```python
q = x @ w_q
k = x @ w_k
v = x @ w_v
```

### 每个位置，都有一行匹配分数

点积矩阵同时计算所有位置之间的匹配，形状为 T×T。

$$
\mathbf S_{\rm raw}=\mathbf Q\mathbf K^\mathsf{T}
$$

```python
raw_scores = q @ k.T
```

### 为什么要缩放点积分数？

缩放控制分数尺度，使 Softmax 更适合训练。

$$
\mathbf S=\frac{\mathbf Q\mathbf K^\mathsf{T}}{\sqrt{d_k}}
$$

```python
scores = raw_scores / math.sqrt(q.shape[-1])
```

### 算“喜欢”时，不能提前看到“喝”

Softmax 前把未来分数设为 −∞，其读取权重就变为零。

$$
\mathbf A=\operatorname{softmax}_j(\mathbf S+\mathbf M)
$$

```python
future = torch.ones(3, 3, dtype=torch.bool).triu(1)
masked = scores.masked_fill(future, -torch.inf)
```

### 一行分数，怎样变成一行读取比例？

除以总和约8.227，得到[0.5,0.5,0]。

$$
a_j=\frac{e^{s_j}}{\sum_{\ell}e^{s_{\ell}}}
$$

```python
weights = masked.softmax(dim=-1)
```

### 用权重把 V 合成新的向量

Attention 输出 Z=AV：每行是结合可见上下文后的新向量。

$$
\mathbf Z=\mathbf A\mathbf V
$$

```python
weights = masked.softmax(dim=-1)
output = weights @ v
```

### “我”的 V 变了，哪些结果会变？

固定 Q/K：权重不变，输出随 V 的变化而变化。

$$
\Delta\mathbf z_i=A_{i,0}\Delta\mathbf v_0
$$

```python
changed_v = v.clone()
changed_v[0] += v.new_tensor([2., 0.])
changed = attention(q, k, changed_v)
```

### “我”的 K 变了，读取比例怎样变？

固定 Q/V：修改 K 可以改变匹配分数、读取权重与输出。

```python
changed_k = k.clone()
changed_k[0] += k.new_tensor([1., 0.])
changed = attention(q, changed_k, v)
```

### 改动“喝”，前两个位置会受影响吗？

因果 Mask 使较早输出不受未来输入改动影响。

```python
changed_x = x.clone()
changed_x[-1] += x.new_tensor([3., -2.])
q2, k2, v2 = project(changed_x)
```

## 形状与接口约定

主例 $T=3,D=d_k=d_v=2$，省略batch维。Q/K/V为[3,2]，分数与权重为[3,3]，输出为[3,2]。
矩阵行是query位置，列是key位置，Softmax沿最后一维执行。

数学使用X@W。nn.Linear.weight存为[out,in]，计算用X@weight.T+bias。
多头页是独立的结构示意：B=1,T=3,D=4,H=2,d_h=2。

RoPE在投影Q/K之后、点积之前。实际接口复用第二课positions.apply_rope；V不旋转，
干预前后位置保持一致。主手算先分离位置旋转，以便逐项核算矩阵。

只改K的本例中，第一行只可读自身；第二行Q与改变量正交，所以前两行权重不变，第三行改变。
BPE字符演示与真实ByteLevel配置使用各自的片段和ID。完整精度数值见assets/lesson_report.json。
