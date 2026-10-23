# 第二课：Tokenizer、Embedding 与位置机制


## 01. 回顾：Transformer 的数据怎样流动？（0–3 分钟）

沿用上节课的文本生成主线：Decoder-only，图中采用 Pre-Norm 与 RoPE。
- 上节课把文字变数字、用Q/K算匹配并混合V，再从输出向量得到下一token候选概率。
- 把这些步骤放回完整生成架构：Tokenizer与Embedding在Block之前，Attention、FFN、Norm和残差在重复的Block中。
- 图中采用Pre-Norm与RoPE；向量在Block间保持[B,T,D]，输出投影变成[B,T,V]，生成时取最后位置。

> 文本变 ID，向量穿过多层 Block，最后得到下一 token 的概率。

**授课提示**：沿用第一课后半段的Decoder-only文本生成架构。图中选择Pre-Norm和RoPE：h=x+MHA(Norm(x))，输出=h+FFN(Norm(h))，两条残差分别保留输入x和中间h。Q/K在Attention内使用RoPE；没有同时给输入叠加正弦PE。先沿左列读生成数据流，再展开右侧一个Block。说明B是batch、T是序列长度、D是隐藏维度、V是词表大小。生成时取最后位置的词表logits来选择下一ID，追加到ID序列；初始文本只需先分词，示意省略KV cache。原始Transformer是Encoder–Decoder，本课程沿用第一课选定的Decoder-only主线。第一课和第二课脚本只实现部分接口，图不是已训练模型的运行截图。

## 02. 从上节课的计算，走到今天的输入端（3–6 分钟）

上节课算过 Q/K 匹配、加权混合 V 和候选选择；今天补充分词、向量与位置。
- 第一课用空格切分和人工二维向量，走通了Q/K匹配、V加权混合与下一token选择。
- 本课依次补全输入端：BPE怎样学并执行切分规则；Embedding怎样查表、接受预测误差并更新参数。
- 位置编码说明顺序怎样进入运算：相加式PE作用于输入，RoPE作用于Q/K；二者作为不同接口讲解。

> 本课聚焦：Tokenizer、Embedding、位置编码在整体架构中的作用。

**授课提示**：先用第一课代码的实际输出唤起记忆：注意力权重约[0.248,0.248,0.503]，混合向量约[0.752,0.752]，酒/牛奶概率约[0.680,0.320]。这些是人工参数的演示结果；显示值有四舍五入。区分第一次Softmax给输入位置分配混合权重，第二次给词表候选分配输出概率。再指向三个本课模块：第一课的text.split()替换为真实BPE，人工向量升级为Embedding参数及梯度，位置概念升级为正弦与RoPE机制。强调相加式PE和Q/K上的RoPE是不同接口，本课分别比较，不机械叠加。随后进入第3页奶茶BPE语料，先用一句话说明频繁片段可以合并，再按5/3/2手算统计。

## 03. BPE：先统计，再合并（6–10 分钟）

每轮重新统计当前相邻片段，选择最高频的一对。
- 演示语料：我喜欢喝奶茶×5、你喜欢喝奶茶×3、我喜欢喝咖啡×2。
- 第一轮“喜+欢”频次为5+3+2=10；“我+喜”只有5+2=7。
- 先合并喜+欢，再重新统计；四轮依次得到喜欢、喜欢喝、喜欢喝奶、喜欢喝奶茶。

> 频次来自语料；合并后要重新统计。

**授课提示**：打开bpe_core.train_bpe，解释corpus中的频次乘数。首轮喜+欢与欢+喝同为10，本例按首次出现顺序打破并列。边界由预切分决定，本字符示例把每句作为一个片段；真实库下一页展示正则预切分。讲清训练不是梯度下降，而是统计和贪心合并。 演示语料改为奶茶与咖啡，四轮频次仍为10、10、8、8。

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

## 04. BPE 编码：谁的 rank 更小？（10–14 分钟）

规则已经冻结；每次在当前候选中选择优先级最高的一对。
- 固定上一页学到的四条 merges，对“你喜欢喝奶茶”逐轮执行。
- 编码时不再统计训练语料，也不更新 merges，只选择当前候选的最小 rank。
- 遇到重叠候选时，优先级比片段长度重要；abc 的反例得到 a/bc。

> merge rank 决定合并顺序，token ID 决定查表地址。

**授课提示**：用语料bc×6、ab×5、abc×1训练两轮，先bc频次7，再ab频次5。词表含ab、bc，但编码abc按rank得到a/bc，最长前缀会得到ab/c。运行bpe_core.demo看反例，再指出encode_bpe每轮枚举当前相邻候选并选最小rank。真实实现常用更高效数据结构，本课重在规则语义。 先走完奶茶例子的四步编码，再用abc说明重叠候选为何不能最长匹配。

**对应实验**：Notebook 实验 2；`bpe_core.py:encode_bpe`。

$$
(a,b)^*=\underset{(a,b)\in\mathcal{C}}{\arg\min}\;\operatorname{rank}(a,b)
$$

C：当前相邻、且在 merges 中的候选 pair。

```python
rank, pair = min(candidates, key=lambda item: item[0])
parts = merge_pair(parts, pair)
```

**运行观察**：你喜欢喝奶茶 → 你 / 喜欢喝奶茶；重叠反例 abc → a / bc。

## 05. 把 BPE 接到真实文本上（14–18 分钟）

ByteLevel 处理字节与边界；BPE 处理合并规则。
- 字符演示说明合并规则；真实库还要处理文本边界、基础字节和解码。
- 本课 ByteLevel 用完整256字节字母表覆盖UTF-8字节，并设置4个特殊符号。
- 保存的不只是词表，还包括merges、预切分和decoder；之后加载完整配置。

> 先训练并保存配置，再加载同一配置进行编码。

**授课提示**：展示train_tokenizer.py中的Tokenizer(BPE)、ByteLevel、BpeTrainer和save。设置完整256字节基础字母表，四个特殊token，目标词表320或384。不设置有损normalizer，add_prefix_space=False。本地语料28行，重复10遍用于频次。已提交配置用于课堂复现，现场不必重训。不能把该小语料tokenizer称为商用模型的预训练配置。

**对应实验**：Notebook 实验 3；`train_tokenizer.py:train`。

**运行观察**：从本地语料训练，完整配置保存在 JSON 中。

## 06. ID 必须与参数表配套（18–20 分钟）

同样大小的词表，也可能把同一片段映射到不同的地址。
- 默认配置把“我喜欢喝牛奶。”编码为[300,304,262]，这些ID是参数表的行地址。
- 另一份同样大小的词表，也可能给同一片段分配不同ID，直接替换会读错参数。
- bos/eos由本课batch代码显式加入；pad=0只用于补齐，不代表真实文本。

> 更换 tokenizer 时，必须核对 ID 与模型参数的对应关系。

**授课提示**：展示assets/tokenizer_320.json及384版本。ByteLevel与decoder让普通测试文本完整回解；skip_special_tokens默认会处理特殊符号，不能据普通文本测试承诺所有特殊字符串也逐字可逆。配置改变时须同步参数映射，不能直接拿320模型的embedding套384词表。

**对应实验**：Notebook 实验 4；`runtime.py:load_tokenizer`。

**运行观察**：默认配置下：我喜欢喝牛奶。→ [300,304,262]。

## 07. Embedding：一个 ID 选择一行（20–24 分钟）

查表与 one-hot 乘法前向等价；实际实现直接选行。
- 查表等于用one-hot选择一行，既保留可学习参数，又避免显式分配大one-hot张量。
- 算术小表中，ID=[1,2,1]会读到第1行、第2行、再读一次第1行。
- 行向量最初可以随机初始化；训练任务逐步调整它们，每一维没有预先规定的语义。

> ID 是地址；可训练的表示存放在 E 的每一行。

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

## 08. 不同长度，怎样组成 batch？（24–27 分钟）

先添加起止标记，再补齐；pad 只是占位。
- 第一句3个普通token加bos/eos得5项；第二句4个加标记得6项，因此前者补1个pad。
- 补齐只统一张量形状；Attention仍应屏蔽pad key，训练loss仍应忽略pad标签。
- padding_idx抑制lookup路径的pad行梯度，不会替后续网络自动处理mask。

> padding_idx、注意力 mask 和损失忽略，需要分别处理。

**授课提示**：make_batch手动加bos/eos；第一句长度5，第二句长度6，第一句末尾补pad。nn.Embedding默认将padding行初始化为0并在lookup反传中抑制其梯度；加载或手动改值时不要假定它永远为0。Attention的padding key mask以及交叉熵ignore_index各自仍要处理。

**对应实验**：Notebook 实验 5；`runtime.py:make_batch`。

**运行观察**：第一句 5 项补齐到 6；第二句 6 项无需补齐。

## 09. 读了同一行，梯度就要相加（27–31 分钟）

共享参数接收所有读取位置的贡献；这里先看 lookup 路径。
- 把同一行看成被多处引用的共享参数；每个位置贡献自己的向量梯度。
- 若ID1两次上游梯度为[1,0]和[0,2]，第1行收到[1,2]，不是取平均也不是覆盖。
- 求和损失只是特殊情况：上游全为1，出现两次的行每一维梯度为2。

> 反向是 scatter-add：累加贡献，而不是覆盖旧值。

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

## 10. 预测误差怎样传回向量？（31–34 分钟）

单位置交叉熵：先得到类别误差，再乘回输出权重。
- 类别编号0/1/2，目标y=1，预测p=[0.1,0.7,0.2]；交叉熵为−ln(0.7)≈0.3567。
- 类别误差p−one_hot(1)=[0.1,−0.3,0.2]，右乘输出权重得到输入向量梯度。
- 小算例得到[-0.1,-0.2]，与autograd一致；真实词表只是把类别数扩到V。

> 先算每次出现的向量梯度，再按 ID 汇总到参数表。

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

## 11. 让 Embedding 真正学一次（34–37 分钟）

57 对相邻 token；Embedding → Linear → CrossEntropy。
- 把相邻token构成(x,y)：当前ID查表，经Linear产生所有候选的logits，再与真实下一ID比较。
- 每步依次前向、清梯度、backward、optimizer.step；梯度负责指出如何调整参数。
- 本例只利用当前token；同一token可能有多个后续，所以训练loss不必降到0。

> 这是只看当前 token 的 bigram；没有 Attention。

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

## 12. 一张表，两条梯度路径（37–39 分钟）

权重绑定：输入查表和输出分类器共享同一个 E。
- 不绑定时，输入E与输出W是两个参数；绑定后输出直接使用E的转置。
- 输入查表只读取少数行，输出softmax却把全词表作为候选，因此更多行可能收到梯度。
- padding_idx只限制lookup路径；绑权重并不保证pad行完全没有梯度。

> 绑定后，没作为输入的行也可能收到输出梯度。

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

## 13. 没有位置，重排会怎样？（39–42 分钟）

假设：没有位置机制，也没有因果 mask。
- “我/喜欢/你”换成“你/喜欢/我”，仅查表得到的仍是同一组内容向量。
- 无位置、无mask时，Q/K/V随行重排，scores行列同时重排，输出也对应重排。
- 因果mask带有可读范围约束，不满足这里任意置换的假设；等变不等于输出不变。

> 等变表示“输出跟着重排”，不是“输出完全不变”。

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

## 14. 把位置向量加进输入（42–45 分钟）

内容与位置使用同样的维度，相加后形状保持不变。
- 内容向量表示“是什么”，位置向量提供“在第几个token”；相加后仍是D维。
- 学习式位置表通过训练更新，直接查询范围受表长L限制；正弦编码由固定函数生成。
- 相加式位置与RoPE是不同接口，本课RoPE分支不先叠加绝对位置。

> 位置是 token 序号，不是原文的字符 offset。

**授课提示**：不要把BPE的字符offset当成token位置。可学习位置表也参与反向传播；正弦表不需要训练其数值。相加式输入与RoPE是不同接口，下一页分别说明，代码不会先加绝对PE又声称相对旋转分数必然保持不变。

**对应实验**：Notebook 实验 10；`positions.py:sinusoidal`。

$$
\mathbf{X}_{b,t}=\mathbf{E}[I_{b,t}]+\operatorname{PE}(t)
$$

t：token 序号；PE 来自学习式位置表或固定函数。

**运行观察**：相加后形状保持 [B,T,D]；本课代码演示固定函数。

## 15. 位置 × 频率，得到角度（45–48 分钟）

不同维度对使用不同频率，再交错存放 sin 与 cos。
- D=4时有两对维度，频率分别为1和0.01；同一位置形成两种变化尺度。
- t=1时角度为[1,0.01]，分别取sin/cos并交错排列，得到4维位置向量。
- 广播[T,1]×[1,D/2]生成全部角度；能计算更远位置不等于模型一定能泛化。

> positions → angles → sin / cos：每一步都有明确形状。

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

## 16. RoPE：相对位移从哪里来？（48–51 分钟）

q、k 分别按绝对位置旋转；点积里出现位置之差。
- 一对维度用二维旋转R；转置相当于反向旋转，两次旋转的角度相加。
- q、k按mω、nω旋转后，点积里出现(n−m)ω，因此相对位移影响分数。
- 二维算例m=1,n=3,q=k=[1,0]得到cos(2)≈−0.4161；两位置同时加5结果不变。

> 转置是反向旋转；角度相加后，剩下 n−m。

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

## 17. 把二维旋转写成两行代码（51–54 分钟）

相邻维度组成一对；同一公式广播到 batch 与 head。
- 把相邻维度分成even/odd两组，分别实现旋转矩阵的两个分量。
- Q/K形状[B,H,T,d_h]，角度[T,d_h/2]广播到batch和head，旋转后形状保持。
- 相邻配对与split-half是不同坐标约定；本实现只旋转Q/K，不旋转V。

> 保持向量范数；点积仍会随相对位置改变。

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

## 18. 位置关系与可读范围（54–56 分钟）

RoPE 改变分数；mask 决定哪些 key 可以被读取。
- 位置机制改变分数的数值关系，因果mask禁止读取未来，padding key mask禁止读取补齐项。
- 禁止位置设为−∞，softmax后权重为0；示意中最后一个key为pad，整列权重为0。
- pad query输出清零与loss忽略标签仍需各自处理；全屏蔽行需要避免softmax产生NaN。

> 有了位置机制，仍要屏蔽未来 key 与 padding key。

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

## 19. 沿真实输入，把形状接起来（56–58 分钟）

本地 tokenizer → Embedding → 多头 Q/K → RoPE + mask。
- tokenizer先生成离散ID，Embedding读取参数，线性层一次投影出Q/K/V。
- 把D=8拆成H=2个head，每头d_h=4；RoPE只处理Q/K，mask在scores上生效。
- softmax沿key维，再乘V并合并head；该脚本参数随机，未训练完整Block。

> 这是随机参数的接口实验，尚未训练完整 Transformer。

**授课提示**：运行input_pipeline.py，只跟踪形状与一行RoPE、一行mask。此分支不先加绝对PE，随机参数未训练；上面的bigram训练也没有Attention。明确两个实验各验证什么，避免把它们拼成已训练的大模型。首句实际token3个，加bos/eos为5，补到6；另一句4个token，加标记为6。

**对应实验**：Notebook 实验 12；`input_pipeline.py:demo`。

**运行观察**：ID [2,6] → X [2,6,8] → Q/K [2,2,6,4] → 输出 [2,6,8]。

## 20. 用三个问题，检查是否理解（58–60 分钟）

解释一次编码、一次反传、一次位置计算。
- 编码：按冻结merges逐步执行，重叠候选选择最小rank。
- 反传：同一ID在不同位置的向量梯度相加；padding lookup有专门约定。
- 位置与mask：前者描述位置关系，后者限制哪些key可读，二者互不替代。

> 课后：改词表、查梯度、验证旋转，并保留实验结果。

**授课提示**：答案：重叠候选选最小rank；按各位置上游梯度求和，不是简单覆盖；旋转编码位置关系，mask决定哪些关系可读取。作业增加一个权重绑定实验，观察输出路径使更多embedding行获得梯度。完整答案及检查在exercises.md，保留数据与机制来源。
