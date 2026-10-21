# 第二课讲义：Tokenizer、Embedding 与位置机制

## 1. Tokenizer：从统计规则到真实编码

### 1.1 词表大小与序列长度的取舍

词表越大，token 往往能承载更长片段，但 embedding 表 E∈R^(V×D) 和输出投影也随 V 增大。稠密注意力分数矩阵为 T×T，logits 投影还涉及 T×D×V 的计算，不能只用 T² 描述整个模型成本。

例如 V=32768、D=4096、FP16，单份 E 需 `32768×4096×2=256 MiB`；未绑定的同尺寸输出权重另占一份。这不含梯度、优化器状态或其他层。

### 1.2 BPE 训练：统计当前 pair，再合并

对当前相邻片段 a、b 的加权频次：

```text
S(a,b) = Σ_s f_s × count_s(a,b)
```

f_s 是文本 s 的出现次数。可读实现逐轮执行：从基础字符开始；统计相邻 pair；按固定并列规则选最高频 pair；合并非重叠出现；重新统计，直到达到合并预算或没有 pair。

“我喜欢喝牛奶”×5、“你喜欢喝牛奶”×3、“我喜欢喝酒”×2，四轮合并频次为10、10、8、8。首轮并列，本代码按首次出现顺序选择，不把它当作所有实现的通用并列规则。

本代码每轮重扫当前语料，朴素训练上界约为 K×N，K为合并轮数，N为初始片段总量。它帮助读懂算法，不代表库实现的性能。BPE 学规则不是梯度下降。[BPE 原论文](https://aclanthology.org/P16-1162/)

### 1.3 编码：rank 不能换成最长前缀

merges 列表给出合并优先级 rank。推理时，从当前相邻、可合并的 pair 中选最小 rank，合并后再检查。token ID 是词表地址，不等于 rank。

```text
语料：bc×6、ab×5、abc×1
规则：rank 0：b+c；rank 1：a+b

abc → ab(rank1)、bc(rank0)
    → 先合并bc
    → a / bc
```

词表虽含 ab，最长前缀得到的 ab/c 在此规则下仍不正确。本课编码每轮扫描当前列表，最坏上界 O(T²)；真实实现可采用更高效数据结构。

### 1.4 真实 ByteLevel BPE

本课使用 Hugging Face Tokenizers 0.22.2：ByteLevel 预切分、256种基础字节、4个特殊符号、BPE训练器和ByteLevel解码器。预切分规定不能跨越的边界；不设置有损normalizer，`add_prefix_space=False`。分别训练320与384项词表。

保存的JSON包含词表、merges、预切分和解码配置。加载后无需下载模型。接口参考 [官方 Quicktour](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html) 与 [0.22.2 ByteLevel 示例](https://github.com/huggingface/tokenizers/blob/v0.22.2/bindings/python/examples/train_bytelevel_bpe.py)。

| 本次实际编码 | V=320 | V=384 |
| --- | --- | --- |
| 我喜欢喝牛奶。 | 3 | 3 |
| Hello world! | 9 | 5 |
| Hello, world! | 10 | 6 |
| 未见过的字🦄 | 19 | 19 |

更大词表不保证每句话都缩短。空格、标点、大小写会影响流水线结果。ByteLevel展示的是字节的可显示映射，单个token可能只含一个UTF-8字符的部分字节，不保证能单独解码成完整汉字；offsets也可能重叠。

普通测试文本整串decode可还原输入。但特殊符号跳过策略或有损规范化会改变条件，不能只凭字节覆盖就承诺所有配置无条件逐字可逆。

320项配置把“我喜欢喝牛奶。”编码为 `[300,304,262]`。这些地址只对这份配置有效。另一词表即使大小相同，地址和merges也可能不同；不可随意替换tokenizer却沿用原表地址。

## 2. Embedding：gather 与 scatter-add

### 2.1 前向是选行

```text
E：[V,D]，I：[B,T]
X = E[I]：[B,T,D]
e_i = one_hot(i) @ E
```

实现直接gather，不显式创建大one-hot。本课 `nn.Embedding(320,8,padding_idx=0)` 输入 `[2,6]`、输出 `[2,6,8]`，可以验证与one-hot乘法的前向等价；这不意味着padding的自定义反向行为也自动等价。[Embedding 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

ID没有天然语义大小。E给token提供共同D维、可训练的特征。对应重编号并重排行可保持前向结果不变；特征的几何取决于数据、任务与参数，不能把每一维或每一个距离解释成固定语言学概念。

### 2.2 重复读取同一行，反向必须累加

每个位置反传到X的梯度为g_p：

```text
∂L/∂E_j = Σ_{p:I_p=j} g_p
```

多个位置读取同一行，贡献必须相加，不能后一次覆盖前一次。这是gather的反向scatter-add。

诊断输入 `[a,a,b,pad]`，对全部输出元素求和，上游每次每维梯度为1；a行每维2、b行每维1、pad的lookup梯度为0。求和loss仅用于观察梯度，不是语言模型训练目标。

`padding_idx` 控制lookup反传，不会自动创建Attention mask或设置loss的ignore_index。默认padding行初始化为0；加载或手动修改参数后，不能假设该行内容永远为0。

### 2.3 从交叉熵算回输入向量

单位置、输出权重W_out为 `[V,D]`：

```text
z = e W_out^T + b
p = softmax(z)
L = −log p_y
∂L/∂z = p − one_hot(y)
∂L/∂e = (p − one_hot(y)) W_out
```

先算每次出现的向量梯度，再按ID累加到E。batch mean loss还包含平均因子。`gradient_chain.py` 把单位置公式与autograd逐项对照。

`F.cross_entropy` 接收原始logits，内部完成log-softmax，代码不先softmax。[交叉熵接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

### 2.4 用真实下一 token 损失更新表

模型为 `Embedding(320,16) → Linear(16,320)`，七句文本构成57个相邻token pair，加入起止符号。Adam训练120步，loss约 `5.7565 → 0.3559`。

这是只用当前token的bigram基线，不含Attention，不利用完整前文。同一输入可能有多个后续，loss不必归零；无验证集，只能观察训练拟合。

对于只能根据当前token预测的模型，训练集经验条件熵是最优平均交叉熵的下界：`−Σ_{x,y} n(x,y)/N × log(n(x,y)/n(x))`。本数据约0.35583，接近训练终点。这里使用未补齐的相邻pair，不含pad标签；不能据此删除有padding训练任务中的ignore_index。

本实验未绑权重、无weight decay，26个实际作为输入的表行发生更新。不能把“只有输入过的行更新”推广到全部训练设置。

### 2.5 权重绑定增加一条梯度路径

```text
输入：gather(E,I)
输出：logits = h E^T + b
```

若输出与输入共享E，E同时接收两条路径的梯度。softmax覆盖全部候选，未作为输入的行也可能接收输出梯度；padding_idx只限制lookup路径。绑定应在创建optimizer前完成，并避免重复传入同一参数。

输入embedding与上下文表示也不同：参数固定时同一ID查同一行，经过位置和Transformer后的隐藏表示可随上下文改变。

## 3. 位置机制：等变性、频率与旋转

### 3.1 无位置、无遮罩的置换等变性

P为行置换：

```text
Q′=PQ，K′=PK，V′=PV
Q′K′^T = P(QK^T)P^T
softmax(P S P^T) = P softmax(S) P^T
Attn(PX) = P Attn(X)
```

输入重排，输出对应重排，叫等变，不是输出矩阵完全不变。因果mask不满足这里的任意置换假设，不能忽略该条件。位置机制把可利用的坐标或相对位移带入运算。

### 3.2 学习式位置与多频率正弦

相加式 `X_p=E[I_p]+P_p`。学习式表 `[L,D]` 按token位置p查行，直接查询范围受L限制；p不是原文字符offset。

```text
θ_i = 10000^(−2i/D)
PE(p,2i)   = sin(p θ_i)
PE(p,2i+1) = cos(p θ_i)
```

D=4时频率为1、0.01，通过 `[T,1]×[1,D/2]` 广播生成角度，再交错写入sin/cos。多频率提供不同变化尺度。由三角加法公式，固定偏移Δ可用与p无关的一组二维线性变换表达，给相对偏移提供可学习结构。[原始 Transformer §3.5](https://arxiv.org/html/1706.03762v7)

### 3.3 RoPE 的二维关键等式

```text
R(φ) = [[cosφ, −sinφ], [sinφ, cosφ]]
q′_m = R(mθ)q，k′_n = R(nθ)k
q′_m^Tk′_n
  = q^T R(mθ)^T R(nθ) k
  = q^T R((n−m)θ) k
```

位置以相对位移进入旋转点积。多维特征按二维配对，各组用不同频率。旋转保持范数，但通常改变向量间点积。[RoFormer 推导](https://arxiv.org/html/2104.09864v5)

本课相邻维度配对，函数接受 `[...,T,D]`。多头Q/K为 `[B,H,T,d_h]`，角度 `[T,d_h/2]` 广播到B和H；另一种split-half约定需一致的坐标排列，不能随意混用。

保持未旋转q/k内容不变、同时平移位置且保持相对位移，旋转点积相同。完整模型的q/k还可能随上下文变化，所以不能推出任意平移文本后模型输出相同，也不能保证无限长度泛化。

### 3.4 位置、遮罩与 padding 接口互不替代

| 接口 | 作用 |
| --- | --- |
| 绝对位置或RoPE | 让位置参与数值计算 |
| causal mask `[T,T]` | 阻止读取未来key |
| padding key mask `[B,1,1,T]` | 屏蔽补齐key |
| embedding.padding_idx | 控制lookup梯度 |
| loss.ignore_index | 忽略指定标签损失 |

禁止位置在scores中设为负无穷，再softmax得到零权重。全屏蔽行需另作处理，避免NaN；本课有bos/eos，至少有可读key。padding query输出清零，不会自动处理训练标签损失。

## 4. 实验边界

bigram有真实训练但没有Attention；输入整合有多头Q/K、RoPE与mask，但参数随机，未实现完整Block或生成循环。两者不能拼称为训练好的Transformer。

运行见 [README](README.md)，反例与实验答案见 [练习](exercises.md)。
