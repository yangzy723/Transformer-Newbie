# 第三课知识与代码对照

代码使用小写变量，公式使用大写矩阵；位置从0开始，第三行是位置2。

| 页 | 时间 | 知识 | Notebook实验 | 源码 |
|---|---|---|---|---|
| 1 | 0–3 | 重新走进 Transformer | 图示/检查 | [code/review.py](code/review.py) |
| 2 | 3–5 | 先看整条数据流 | 图示/检查 | [code/review.py](code/review.py) |
| 3 | 5–9 | 回到第一课的加权汇总 | 实验01 | [code/review.py](code/review.py) |
| 4 | 9–12 | 两次 Softmax，各自解决什么 | 实验01 | [code/review.py](code/review.py) |
| 5 | 12–15 | BPE 训练：为什么要重新统计 | 实验02 | [code/review.py](code/review.py) |
| 6 | 15–18 | BPE 编码：规则要执行到结束 | 实验02 | [code/review.py](code/review.py) |
| 7 | 18–21 | Embedding：ID 是地址，向量是参数 | 实验02 | [code/review.py](code/review.py) |
| 8 | 21–25 | 位置怎样参与内容匹配 | 实验03 | [code/review.py](code/review.py) |
| 9 | 25–28 | 同一份输入，为什么分成 Q/K/V | 实验04 | [code/attention.py](code/attention.py) |
| 10 | 28–32 | 把三个投影真正算出来 | 实验04 | [code/attention.py](code/attention.py) |
| 11 | 32–35 | 先只算“喝”这一行 | 实验05 | [code/attention.py](code/attention.py) |
| 12 | 35–38 | 把一行扩展成整张分数表 | 实验05 | [code/attention.py](code/attention.py) |
| 13 | 38–41 | 为什么要缩放点积分数 | 实验05 | [code/attention.py](code/attention.py) |
| 14 | 41–44 | 因果 Mask：读取过去与自身 | 实验05 | [code/attention.py](code/attention.py) |
| 15 | 44–47 | 权重乘 V，得到上下文向量 | 实验05 | [code/attention.py](code/attention.py) |
| 16 | 47–49 | 实验一：只改 V，会发生什么 | 实验06 | [code/experiments.py](code/experiments.py) |
| 17 | 49–51 | 实验二：只改 K，会发生什么 | 实验07 | [code/experiments.py](code/experiments.py) |
| 18 | 51–53 | 实验三：修改未来，过去会变吗 | 实验08 | [code/experiments.py](code/experiments.py) |
| 19 | 53–55 | 把图、公式和代码对齐 | 实验05 | [code/attention.py](code/attention.py) |
| 20 | 55–57 | 多头：并行匹配，再合回一个向量 | 图示/检查 | [code/attention.py](code/attention.py) |
| 21 | 57–60 | 请你解释这张注意力矩阵 | 图示/检查 | [code/attention.py](code/attention.py) |

## 核心计算逐项对应


### 把三个投影真正算出来

线性投影重组输入特征，得到用于匹配与汇总的表示。

```python
q = x @ w_q
k = x @ w_k
v = x @ w_v
```

### 把一行扩展成整张分数表

点积矩阵同时计算所有位置之间的匹配，形状为 T×T。

$$
\mathbf S_{\rm raw}=\mathbf Q\mathbf K^\mathsf{T}
$$

```python
raw_scores = q @ k.T
```

### 为什么要缩放点积分数

缩放控制分数尺度，使 Softmax 更适合训练。

$$
\mathbf S=\frac{\mathbf Q\mathbf K^\mathsf{T}}{\sqrt{d_k}}
$$

```python
scores = raw_scores / math.sqrt(q.shape[-1])
```

### 因果 Mask：读取过去与自身

Softmax 前把未来分数设为 −∞，其读取权重就变为零。

$$
\mathbf A=\operatorname{softmax}_j(\mathbf S+\mathbf M)
$$

```python
future = torch.ones(3, 3, dtype=torch.bool).triu(1)
masked = scores.masked_fill(future, -torch.inf)
```

### 权重乘 V，得到上下文向量

Attention 输出 Z=AV：每行是结合可见上下文后的新向量。

$$
\mathbf Z=\mathbf A\mathbf V
$$

```python
weights = masked.softmax(dim=-1)
output = weights @ v
```

### 实验一：只改 V，会发生什么

固定 Q/K：权重不变，输出随 V 的变化而变化。

$$
\Delta\mathbf z_i=A_{i,0}\Delta\mathbf v_0
$$

```python
changed_v = v.clone()
changed_v[0] += v.new_tensor([2., 0.])
changed = attention(q, k, changed_v)
```

### 实验二：只改 K，会发生什么

固定 Q/V：修改 K 可以改变匹配分数、读取权重与输出。

```python
changed_k = k.clone()
changed_k[0] += k.new_tensor([1., 0.])
changed = attention(q, changed_k, v)
```

### 实验三：修改未来，过去会变吗

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
第20页是独立的多头结构示意：B=1,T=3,D=4,H=2,d_h=2。

RoPE在投影Q/K之后、点积之前。实际接口复用第二课positions.apply_rope；V不旋转，
干预前后位置保持一致。主手算先分离位置旋转，以便逐项核算矩阵。

只改K的本例中，第一行只可读自身；第二行Q与改变量正交，所以前两行权重不变，第三行改变。
BPE字符演示与真实ByteLevel配置使用各自的片段和ID。完整精度数值见assets/lesson_report.json。
