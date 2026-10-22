# 第二课：Tokenizer、Embedding 与位置机制


## 01. 第二课：输入表示的算法与机制（0–3 分钟）

- Tokenizer：怎样学习规则，怎样执行编码
- Embedding：怎样选行，误差怎样更新表
- 位置机制：顺序怎样进入 Q/K 计算

> 文本 → token ID → 可训练向量 → 位置机制 → Attention

**授课提示**：沿用第一课的Q/K/V概念，新增B、T、D、V四个形状符号。本课有字符级BPE机制实验与真实ByteLevel BPE实现，两者的词表和ID不混用。课堂运行已保存的本地配置，不下载预训练模型。准备阶段安装Tokenizers与PyTorch。

## 02. 词表规模是计算取舍，不只是“怎么分词”（3–6 分钟）

- V 增大，embedding 与输出层参数随之增大
- T 增大，稠密注意力分数矩阵按 T² 增长
- 在覆盖、序列长度和参数规模之间权衡

> E 的参数量 V×D；V=32768、D=4096、FP16 时约占 256 MiB。

**授课提示**：只计算单个embedding矩阵，32768×4096×2字节=256MiB；未绑定的输出权重另占一份。注意力分数矩阵的T²描述稠密实现，不能说整个模型成本仅由T²决定；logits投影也有T×D×V成本。更大的词表不保证所有文本都更短，后续看实际对照。

## 03. BPE 训练：加权统计、合并、重新统计（6–9 分钟）

- 把每种文本的相邻次数，乘以该文本的出现频次
- 选择当前最高频 pair，合并非重叠出现
- 重算频次，直到词表预算或停止条件满足

> 本例语料频次 5、3、2；四轮合并频次为 10、10、8、8。

**授课提示**：打开bpe_core.train_bpe，解释corpus中的频次乘数。首轮喜+欢与欢+喝同为10，本例按首次出现顺序打破并列。边界由预切分决定，本字符示例把每句作为一个片段；真实库下一页展示正则预切分。讲清训练不是梯度下降，而是统计和贪心合并。

**对应实验**：Notebook 实验 1；`bpe_core.py:train_bpe`。

$$
S(a,b)=\sum_s f_s\,\operatorname{count}_s(a,b)
$$

f_s：文本频次；count：当前切分中的相邻次数。

```python
counts[pair] += corpus[text]
pair = max(counts, key=counts.get)
```

**运行观察**：四轮合并频次：10 → 10 → 8 → 8。

## 04. BPE 编码：按 merge rank，不按最长词匹配（9–12 分钟）

- 固定 merges：第 0 条 b+c，第 1 条 a+b
- abc 的当前候选：bc(rank 0)、ab(rank 1)
- 先合并 bc，得到 a / bc

> merge rank 表示合并优先级；token ID 表示词表地址，两者不能混用。

**授课提示**：用语料bc×6、ab×5、abc×1训练两轮，先bc频次7，再ab频次5。词表含ab、bc，但编码abc按rank得到a/bc，最长前缀会得到ab/c。运行bpe_core.demo看反例，再指出encode_bpe每轮枚举当前相邻候选并选最小rank。真实实现常用更高效数据结构，本课重在规则语义。

**对应实验**：Notebook 实验 2；`bpe_core.py:encode_bpe`。

$$
(a,b)^*=\underset{(a,b)\in\mathcal{C}}{\arg\min}\;\operatorname{rank}(a,b)
$$

C：当前相邻、且在 merges 中的候选 pair。

```python
rank, pair = min(candidates, key=lambda item: item[0])
parts = merge_pair(parts, pair)
```

**运行观察**：abc → a / bc；不是最长前缀 ab / c。

## 05. 真实实现：ByteLevel BPE 的完整流水线（12–16 分钟）

- 本地语料 → ByteLevel 预切分 → BPE 训练
- 保存 tokenizer.json：词表、merges 和配置
- 加载配置 → encode → tokens / ids / offsets

> 分词器训练与语言模型训练，是两套不同的学习过程。

**授课提示**：展示train_tokenizer.py中的Tokenizer(BPE)、ByteLevel、BpeTrainer和save。设置完整256字节基础字母表，四个特殊token，目标词表320或384。不设置有损normalizer，add_prefix_space=False。本地语料28行，重复10遍用于频次。已提交配置用于课堂复现，现场不必重训。不能把该小语料tokenizer称为商用模型的预训练配置。

**对应实验**：Notebook 实验 3；`train_tokenizer.py:train`。

**运行观察**：从本地语料训练，完整配置保存在 JSON 中。

## 06. 字节、空格和边界：观察真实编码结果（16–19 分钟）

- Hello world!：V=320 时 9 项，V=384 时 5 项
- 前导空格、标点与大小写会影响预切分和编码
- 未见字符可由基础字节覆盖，完整回解核对文本

> ByteLevel 的可显示 token 字符串，不保证每项都是完整汉字。

**授课提示**：运行tokenization.py，比较Hello world!与Hello, world!及前导空格。词表320/384对中文示例都得到3项，不把英文缩短推广为所有文本必然缩短。显示tokens里的特殊字符是字节映射表示；单个token可能只含UTF-8字符的一部分，不能逐ID解码后期待每项都是完整字符。offsets可能重叠，不当作token与字符一一对应。

**对应实验**：Notebook 实验 4；`tokenization.py:demo`。

**运行观察**：Hello world!：320 项词表得 9 项，384 项得 5 项。

## 07. tokenizer 必须与模型参数配套（19–21 分钟）

- 特殊符号也有 ID：pad=0、unk=1、bos=2、eos=3
- 相同词表大小，不代表 ID 与 merges 相同
- 先固定配置，再为它构造和训练参数表

> 本课默认配置：V=320；“我喜欢喝牛奶。”→ [300,304,262]。

**授课提示**：展示assets/tokenizer_320.json及384版本。ByteLevel与decoder让普通测试文本完整回解；skip_special_tokens默认会处理特殊符号，不能据普通文本测试承诺所有特殊字符串也逐字可逆。配置改变时须同步参数映射，不能直接拿320模型的embedding套384词表。

**对应实验**：Notebook 实验 4；`runtime.py:load_tokenizer`。

**运行观察**：默认配置下：我喜欢喝牛奶。→ [300,304,262]。

## 08. Embedding：查表与 one-hot 乘法等价（21–24 分钟）

- 每个 ID 选择参数表的一行，得到 D 维向量
- batch 的每个位置都执行同一次查表操作
- one-hot 乘法用于理解，实际实现直接 gather

> ID 没有语义距离；表示的可学习几何来自 E 和训练目标。

**授课提示**：运行embeddings.py的前半段，torch.allclose(lookup,one_hot@E)为True。真实示例E为[320,8]，batch为[2,6]，X为[2,6,8]。one-hot只为验证构造，不建议大词表训练时这么实现。对应重编号并重排行能保持前向结果不变。

**对应实验**：Notebook 实验 5；`embeddings.py:demo`。

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

**运行观察**：[2,6] → [2,6,8]；与 one-hot 乘法前向等价。

## 09. 真实 batch：padding 不等于真实文本（24–27 分钟）

- 两条不同长度序列，补齐到 T=6
- nn.Embedding(..., padding_idx=0) 控制 lookup 梯度
- valid mask 仍需传给后续注意力或损失

> padding_idx 不会自动遮住 Attention，也不会自动忽略 loss。

**授课提示**：make_batch手动加bos/eos；第一句长度5，第二句长度6，第一句末尾补pad。nn.Embedding默认将padding行初始化为0并在lookup反传中抑制其梯度；加载或手动改值时不要假定它永远为0。Attention的padding key mask以及交叉熵ignore_index各自仍要处理。

**对应实验**：Notebook 实验 5；`runtime.py:make_batch`。

**运行观察**：第一句 5 项补齐到 6；第二句 6 项无需补齐。

## 10. 重复 token 怎样更新同一行？（27–31 分钟）

- 先得到每个出现位置的向量梯度
- 同一参数行可能被多个位置共同读取
- 把这些位置的梯度相加，更新共享表行

> 诊断输入 [a,a,b,pad]：求和 loss 的梯度分别是 2、1、0。

**授课提示**：运行embeddings.py后半段，用.sum().backward()明确上游每维梯度是1；a出现两次所以对应行每维2，b每维1，padding lookup为0。这是诊断loss，不是语言模型交叉熵。未用作输入的行在此lookup路径梯度为0；绑定输出权重或施加weight decay时不能把它说成所有训练情形的更新规律。

**对应实验**：Notebook 实验 6；`embeddings.py:demo`。

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

**运行观察**：每维梯度：a 行 2，b 行 1，pad 行 0。

## 11. 从下一 token 交叉熵追到 embedding（31–34 分钟）

- 输入向量经过输出投影，得到各 token 的 logits
- 交叉熵衡量正确下一 token 的预测概率
- 误差信号右乘输出权重，传回输入向量

> 先得到每次出现的向量梯度，再按 token ID 累加到 E。

**授课提示**：W_out形状[V,D]，p-y是[V]，右乘得到[D]梯度。写一个正确类别的p_y低时误差大的例子，不推导softmax完整Jacobian。公式按单个位置；实际batch mean loss还包含平均因子。F.cross_entropy直接接原始logits，内部完成log_softmax，不先softmax。

**对应实验**：Notebook 实验 7；`gradient_chain.py:demo`。

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

**运行观察**：手算向量梯度与 autograd 逐项一致。

## 12. 运行一次真实的下一 token 训练（34–37 分钟）

- 真实 ByteLevel BPE，7 句训练文本、57 对相邻 token
- nn.Embedding(320,16) → Linear(16,320)
- Adam 训练 120 步，观察 loss 与表行变化

> 实测 loss：5.7565 → 0.3559；26 个输入表行发生更新。

**授课提示**：运行embedding_train.py或Notebook训练单元。模型只用当前token预测下一项，因此是bigram基线，不是Transformer；同一个输入可能有多个后续，loss不必归零。无验证集，仅观察训练拟合。在无weight decay、未绑定输出权重的本实验中，更新行数26与出现过的输入行数一致。 本数据仅依赖当前token时的经验条件熵下界约0.35583；此项供老师解释loss平台，不要求课内计算。训练pair未补齐，不含pad标签。

**对应实验**：Notebook 实验 8；`embedding_train.py:train`。

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

**运行观察**：120 步：loss 5.7565 → 0.3559；26 行更新。

## 13. 输入表、输出表与权重绑定（37–39 分钟）

- 输入 embedding 与输出投影可共用同一份 E
- 输入查表和输出分类，形成两条梯度路径
- 输入表行固定，上下文隐藏表示仍可变化

> 绑定权重后，E 会同时接收输入 lookup 与输出分类器的梯度。

**授课提示**：本课bigram代码未绑定，便于先看lookup行更新。若绑定，词表全体候选参与输出softmax，未作为输入的行也可能收到输出梯度；padding_idx只限制lookup路径。设置output.weight=embedding.weight应在创建optimizer之前完成，并用去重的参数迭代器。相似度受训练任务影响，不能把每维解释为固定语义标签。

**对应实验**：课后扩展验证；`tools/check_experiments.py:main`。

$$
\mathbf{z}=\mathbf{h}\mathbf{E}^{\mathsf{T}}+\mathbf{b}
$$

$$
\nabla_{\mathbf{E}}L=\nabla_{\mathbf{E}}^{\rm lookup}L+\nabla_{\mathbf{E}}^{\rm output}L
$$

输入与输出共用同一个参数；主训练实验保持未绑定。

**运行观察**：课后验证：没作为输入的行也可能得到输出梯度。

## 14. 无位置时：为什么是置换等变？（39–42 分钟）

- 把 token 行重排，Q/K/V 也对应重排
- 分数矩阵的 query 行和 key 列同时重排
- 输出跟随输入重排：没有引入固定位置坐标

> 这里假设没有位置机制、没有因果 mask；等变不等于输出完全相同。

**授课提示**：给出三行token的重排，说明输出随行重排，而非整个矩阵不变。运行position_mechanics.py：无位置时True，给新行位置加PE后False。不要求证明任意网络性质，只连接QK转置和softmax逐行对称性。因果mask并不随任意置换同样改变，所以不满足这里的假设。 用 Π 记行置换，与学习式位置表 P 区分。

**对应实验**：Notebook 实验 9；`position_mechanics.py:attention`。

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

**运行观察**：重排后输出也重排：True；加入位置后通常 False。

## 15. 绝对位置：学习表与固定函数（42–45 分钟）

- 学习式位置表按 token 序号查行
- 把内容向量与同维度的位置向量相加
- 正弦编码则用固定函数生成位置向量

> P 的索引是 token 位置；L 限制学习式表能直接查询的范围。

**授课提示**：不要把BPE的字符offset当成token位置。可学习位置表也参与反向传播；正弦表不需要训练其数值。相加式输入与RoPE是不同接口，下一页分别说明，代码不会先加绝对PE又声称相对旋转分数必然保持不变。

**对应实验**：Notebook 实验 10；`positions.py:sinusoidal`。

$$
\mathbf{X}_{b,t}=\mathbf{E}[I_{b,t}]+\operatorname{PE}(t)
$$

t：token 序号；PE 来自学习式位置表或固定函数。

**运行观察**：相加后形状保持 [B,T,D]；本课代码演示固定函数。

## 16. 正弦为什么使用多组频率？（45–48 分钟）

- 频率随维度对编号变化，提供多种变化尺度
- 每个位置乘以各组频率，得到旋转角度
- 相邻两维分别存放角度的 sin 与 cos

> 从位置 t，到角度 tω，再到每一对维度的 sin / cos。

**授课提示**：D=4时频率1与0.01，沿用上一版四维数值核对。多个频率提供不同变化尺度，单一周期不能代替全部设计。指出pos[:,None]*frequencies[None,:]的广播形状[T,D/2]，而非逐token循环。可计算更大p不等于模型长上下文质量有保证。 统一记号：ω_i 为频率，θ_{t,i}=tω_i 为角度；分别对应 frequencies 与 angles。

**对应实验**：Notebook 实验 10；`positions.py:sinusoidal`。

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

**运行观察**：位置 0：[0,1,0,1]；不同维度对有不同变化尺度。

## 17. RoPE 二维推导：相对位移从哪来？（48–51 分钟）

- 每一对维度用二维旋转矩阵变换
- q 按位置 m 旋转，k 按位置 n 旋转
- 旋转后的点积出现相对位移 n−m

> 转置是反向旋转，连续旋转可相加，因此最终角度为 (n−m)ω。

**授课提示**：从二维旋转矩阵的转置等于反向旋转、旋转可相加讲到相对位移，约两步，不展开所有矩阵项。保持未旋转q/k不变、同时平移位置时点积不变；完整模型的q/k可能随上下文变化，所以不推出任意平移文本模型输出不变。

**对应实验**：Notebook 实验 11；`positions.py:apply_rope`。

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

**运行观察**：固定 q/k 内容：点积由相对位置 n−m 调节。

## 18. 把 RoPE 写成张量运算（51–54 分钟）

- 把最后一维拆成相邻的 even / odd 分量
- 按分量公式旋转，广播到 batch 与 head
- 函数保持形状，整合时作用于多头 Q/K

> RoPE 保持每个向量范数；点积仍会随相对位置改变。

**授课提示**：展示positions.apply_rope。使用相邻维度配对，另一种split-half实现需有一致的坐标排列；不混用两种配对约定。函数支持[...,T,D]，D必须为正偶数。运行整体位置平移与范数检查，float64分数差在1e-15量级。RoPE作用Q/K，本实现不旋转V。

**对应实验**：Notebook 实验 11；`positions.py:apply_rope`。

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

**运行观察**：范数保持；整体平移位置后分数误差约 10⁻¹⁵。

## 19. 三种机制：位置、因果范围与 padding（54–56 分钟）

- RoPE 或绝对位置：让位置参与数值计算
- causal mask:[T,T]，限制读取未来
- padding key mask:[B,1,1,T]，屏蔽补齐位置

> padding_idx、Attention mask、loss ignore_index 互不替代。

**授课提示**：pipeline将上三角future和无效key位置做逻辑或，再把score设为-inf；不会修改token ID来实现mask。批中每句含bos/eos，至少有可见有效key；对全屏蔽行必须另作处理以免softmax NaN。padding query的输出最后清零，但训练损失仍需独立忽略padding标签。

**对应实验**：Notebook 实验 12；`input_pipeline.py:demo`。

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

**运行观察**：未来 key 与 padding key 的权重均为 0。

## 20. 真实输入链路：把 shape 全部接上（56–58 分钟）

- 真实 tokenizer → ID [2,6] → embedding [2,6,8]
- 投影并拆头 → Q/K [2,2,6,4]
- RoPE + 两种 mask → 输出 [2,6,8]

> 该脚本验证输入端与 Attention 接口，不包含完整 Block 或生成循环。

**授课提示**：运行input_pipeline.py，只跟踪形状与一行RoPE、一行mask。此分支不先加绝对PE，随机参数未训练；上面的bigram训练也没有Attention。明确两个实验各验证什么，避免把它们拼成已训练的大模型。首句实际token3个，加bos/eos为5，补到6；另一句4个token，加标记为6。

**对应实验**：Notebook 实验 12；`input_pipeline.py:demo`。

**运行观察**：ID [2,6] → X [2,6,8] → Q/K [2,2,6,4] → 输出 [2,6,8]。

## 21. 检查理解：解释一次编码、一次反传、一次旋转（58–60 分钟）

- 为什么 abc 在 rank 规则下是 a/bc 而非 ab/c？
- 同一 token 出现三次，它的表行梯度怎么计算？
- 为什么保持 n−m 后旋转点积相同，mask 却仍必要？

> 课后：改词表规模、检查梯度、验证旋转等式并记录结果。

**授课提示**：答案：重叠候选选最小rank；按各位置上游梯度求和，不是简单覆盖；旋转编码位置关系，mask决定哪些关系可读取。作业增加一个权重绑定实验，观察输出路径使更多embedding行获得梯度。完整答案及检查在exercises.md，保留数据与机制来源。
