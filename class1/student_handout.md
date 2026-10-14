# Class 1 学生讲义

## 先记住这条数据流

```text
文本
 → 切成 token，再查词表得到 ID
 → 查 embedding 表：每个 ID 变成一个向量
 → 加入位置向量
 → Transformer Block：因果 Attention + FFN（含归一化和残差）
 → 最终归一化 + 输出线性层，得到 logits
 → 生成时：softmax → 选择下一 token → 接回前文 → 重复
```

本课是一个 **decoder-only** 小模型，只有一层、两头。原始 Transformer 是 Encoder–Decoder 架构；其 decoder 还使用 cross-attention 读取 encoder 的输出。

## 这些词分别指什么？

| 名称 | 本课的理解方式 |
| --- | --- |
| token | 文本切分后的单位；本课按空格切词，真实模型可能用子词等单位 |
| 词表、ID | 所有可用 token 和它们的编号；编号大小不代表语义 |
| 向量、矩阵 | 数字列表、多个数字列表叠起来 |
| embedding | 可学习的查找表，把 ID 变成向量 |
| 参数 | 训练会更新的数字，如 embedding 表和线性层的权重 |
| Q/K/V | Query/Key/Value；本课由同一序列的向量经三个投影得到 |
| Attention | 根据 Q/K 匹配得到权重，对 V 加权混合，输出向量 |
| 因果 mask | 当前及之前可见，未来不可见，防止预测时偷看答案 |
| 多头 | 多个投影下的 Attention，拼接后再投影；头数不同于层数 |
| FFN | 每个位置上应用同一个非线性网络，进一步加工表示 |
| 残差、LayerNorm | 保留原输入的相加路径、对表示做归一化 |
| logits | 输出层给每个候选 token 的原始分数，尚不是概率 |
| loss、梯度、优化器 | 误差度量、参数变化如何影响误差、据此更新参数的方法 |

## Attention：匹配、归一化、混合

对于一个 head：

```text
Q = XWq，K = XWk，V = XWv
S = QKᵀ / √dₖ
A = softmax(S + mask)   # 对每一行做；无 mask 时省略
O = AV
```

Q 和 K 用于决定权重，V 提供要混合的信息。`dₖ` 是 Q/K 每个向量的长度，缩放用于控制分数的尺度。本课因果 mask 对未来位置加负无穷，其余加 0。

权重矩阵**每一行是当前 query 位置**，**每一列是被读取的 key 位置**。对一行求 softmax 后，数值非负、总和为 1。当前例子的输出仍然是一排数字，经过后续层才会成为候选 token 的分数。

### 可以手算的第一行

```text
q = [1, 0]
K = [[1, 0], [0, 1], [1, 1]]
V = [[1, 0], [0, 2], [3, 1]]

不加 mask：
分数 ≈ [0.707, 0, 0.707]
权重 ≈ [0.401, 0.198, 0.401]
输出 ≈ [1.604, 0.797]

加因果 mask：
第一行分数 = [0.707, −∞, −∞]
第一行权重 = [1, 0, 0]
第一行输出 = [1, 0]
```

数字是人工设定的，方便计算；不能把它们当作训练好的中文语义表示。

## 本课的 Block

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

这是 Pre-LN 排列。多头计算后会拼接，再做输出投影。FFN 对每个位置使用相同的参数，且不在不同位置间直接混合信息。末尾还有一次 LayerNorm 和输出线性层。

## 训练和生成的区别

```text
完整：<bos> 我 喜欢 吃 苹果 。 <eos>
x：   <bos> 我 喜欢 吃 苹果 。
y：         我 喜欢 吃 苹果 。 <eos>
```

训练让每个位置预测右边的 token。虽可同时计算多个位置，但每个位置只能读取当前及之前的输入。模型给标签 token 的概率低，误差就大；反向传播计算梯度，优化器据此调整参数。

单个位置的交叉熵可以理解为 `−log p(正确 token)`。代码把**原始 logits**传给 `F.cross_entropy`，不提前 softmax。

生成取最后一个位置的 logits，转成概率，选一个 token，追加到前文。它**不更新参数**。本课用最高概率的贪心选择；采样则按概率随机选择，属于另一种策略。

## 对照本课代码的形状

| 张量 | 形状 | 每一维是什么 |
| --- | --- | --- |
| x、y | `[8, 6]` | 8 句，6 个输入/目标位置 |
| embedding 输出 | `[8, 6, 32]` | 每个位置 32 个数字 |
| 拆头后的 Q/K/V | `[8, 2, 6, 16]` | 8 句，2 头，6 位置，每头 16 维 |
| Attention 权重 | `[8, 2, 6, 6]` | 每头每个 query 对每个 key 的权重 |
| logits | `[8, 6, 12]` | 每个位置对 12 个词表 token 的分数 |

普通 token 有 `我、你、喜欢、吃、喝、苹果、香蕉、茶、牛奶、。`，加 `<bos>` 和 `<eos>`。8 句长度一致，因此不用 padding。位置表支持最多 16 个输入 token，本课数据只用到其中一部分。

## 运行与实验

在仓库根目录运行：

```bash
python3 class1/code/attention_demo.py
python3 class1/code/attention_demo.py --causal
# 安装 PyTorch 后：
python class1/code/tiny_transformer.py --steps 200
python class1/code/tiny_transformer.py --prompt "你 喜欢 喝"
```

或者依次运行 [Notebook](demo.ipynb)。每次脚本运行会重新训练，不是在上一次的参数上继续训练。

### 课后任务（20–30 分钟）

1. 保持 Q/K 不变，把 `V[2]` 改成 `[30,10]`。先写预测：权重变不变？输出变不变？再运行验证。
2. 加因果遮罩，只改未来位置的 V。观察前面的位置是否变化，并解释原因。
3. 训练后，比较 `我 喜欢`、`我 喜欢 吃`、`你 喜欢 喝` 的候选和生成结果，记录至少两步的上下文变化。
4. 不看讲义画出整条链路，注明 Attention 输出与 logits 的区别、训练和生成何时更新参数。

## 如何解读实验

loss 下降说明模型拟合这组训练数据的能力提升。这些句子存在多个合法后续，loss 不必等于 0。本课没有验证集，不能依据这个实验宣称模型掌握了语言或具备泛化能力。Attention 权重描述计算中的混合比例，也不直接等于人的思维或因果解释。

## 参考资料

- [Attention Is All You Need（原论文）](https://arxiv.org/abs/1706.03762)：原始架构、缩放点积 Attention、多头和遮罩。本课教学模型及图示为自行编写的简化版本。
- [PyTorch：scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：官方 Attention 操作及因果遮罩参考；本课显式写出矩阵计算，方便观察中间值。
- [PyTorch：CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：logits、标签和交叉熵的官方说明。
