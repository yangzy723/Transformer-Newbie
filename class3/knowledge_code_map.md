# 第三课知识与代码对照

代码使用小写变量，公式使用大写矩阵；位置从0开始，第三行是位置2。

| 页 | 时间 | 知识 | Notebook实验 | 源码 |
|---|---|---|---|---|
| 1 | 0–2 | 输入一句话，模型怎样继续写？ | 图示/检查 | [code/review.py](code/review.py) |
| 2 | 2–3 | 从文字到下一个 token | 图示/检查 | [code/review.py](code/review.py) |
| 3 | 3–4 | 展开一个 Block：两次变换，两条残差 | 图示/检查 | [code/attention.py](code/attention.py) |
| 4 | 4–6 | 用一个向量，看清两次残差相加 | 残差加法演示 | [code/review.py](code/review.py) |
| 5 | 6–8 | BPE 训练：合并之后重新计数 | 实验01 | [code/review.py](code/review.py) |
| 6 | 8–10 | BPE 编码：按已有规则继续合并 | 实验01 | [code/review.py](code/review.py) |
| 7 | 10–11 | Embedding：按 ID 取出一行向量 | 实验01 | [code/review.py](code/review.py) |
| 8 | 11–13 | 训练怎样修改 Embedding 的一行？ | 实验01 | [code/review.py](code/review.py) |
| 9 | 13–16 | 内容相同，位置不同 | 实验02 | [code/review.py](code/review.py) |
| 10 | 16–17 | 有了内容和位置，为什么还需要 Attention？ | 图示/检查 | [code/review.py](code/review.py) |
| 11 | 17–19 | 先只更新“喝”这个位置 | 图示/检查 | [code/review.py](code/review.py) |
| 12 | 19–21 | 三个位置各提供什么，分别读多少？ | 实验03 | [code/review.py](code/review.py) |
| 13 | 21–24 | 先乘读取比例，再把同一列相加 | 实验03 | [code/review.py](code/review.py) |
| 14 | 24–25 | 每个位置各算一次，整句仍有三个向量 | 图示/检查 | [code/review.py](code/review.py) |
| 15 | 25–27 | 查营业时间：问题、索引、正文各管什么？ | 图示/检查 | [code/attention.py](code/attention.py) |
| 16 | 27–28 | 换一个问题，同一份资料的读取比例也会变 | 图示/检查 | [code/attention.py](code/attention.py) |
| 17 | 28–29 | Q：谁在读取，就用谁的查询向量 | 图示/检查 | [code/attention.py](code/attention.py) |
| 18 | 29–30 | K：每个信息来源，怎样参与匹配？ | 图示/检查 | [code/attention.py](code/attention.py) |
| 19 | 30–31 | V（Value）：实际带回什么内容？ | 图示/检查 | [code/attention.py](code/attention.py) |
| 20 | 31–32 | 同一个输入，为什么生成三种表示？ | 图示/检查 | [code/attention.py](code/attention.py) |
| 21 | 32–33 | 回到句子：每个位置都有 Q、K、V | 实验04 | [code/attention.py](code/attention.py) |
| 22 | 33–34 | 只拿“喝”的 Q，把一次读取走完整 | 实验04 | [code/attention.py](code/attention.py) |
| 23 | 34–36 | 从 X 算出 Q、K、V | 实验04 | [code/attention.py](code/attention.py) |
| 24 | 36–38 | “喝”与每个位置分别算一个分数 | 实验05 | [code/attention.py](code/attention.py) |
| 25 | 38–40 | 每个位置，都有一行匹配分数 | 实验05 | [code/attention.py](code/attention.py) |
| 26 | 40–41 | 为什么要缩放点积分数？ | 实验05 | [code/attention.py](code/attention.py) |
| 27 | 41–43 | 算“喜欢”时，不能提前看到“喝” | 实验05 | [code/attention.py](code/attention.py) |
| 28 | 43–45 | 一行分数，怎样变成一行读取比例？ | 实验05 | [code/attention.py](code/attention.py) |
| 29 | 45–47 | 用权重把 V 合成新的向量 | 实验05 | [code/attention.py](code/attention.py) |
| 30 | 47–49 | 两次 Softmax，用在两个地方 | 实验03 | [code/review.py](code/review.py) |
| 31 | 49–51 | “我”的 V 变了，哪些结果会变？ | 实验06 | [code/experiments.py](code/experiments.py) |
| 32 | 51–53 | “我”的 K 变了，读取比例怎样变？ | 实验07 | [code/experiments.py](code/experiments.py) |
| 33 | 53–55 | 改动“喝”，前两个位置会受影响吗？ | 实验08 | [code/experiments.py](code/experiments.py) |
| 34 | 55–56 | 将刚才的计算写成程序 | 实验05 | [code/attention.py](code/attention.py) |
| 35 | 56–57 | 多个头，分别计算几组匹配 | 图示/检查 | [code/attention.py](code/attention.py) |
| 36 | 57–60 | 这张注意力矩阵说明了什么？ | 图示/检查 | [code/attention.py](code/attention.py) |

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
