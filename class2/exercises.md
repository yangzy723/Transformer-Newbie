# 第二课练习与参考答案

## 1. BPE 的重叠候选

语料为bc×6、ab×5、abc×1，训练两次合并。写出首轮计数、两条规则和abc的编码。

答案：bc为7、ab为6，先合并bc；重算后ab为5，第二条合并ab。abc按rank优先得到a/bc；不能因为词表里有ab就改用最长前缀。

运行 `python class2/code/bpe_core.py` 核对，再交换语料频次，观察规则与编码能否变化。

## 2. 改词表规模，记录真实变化

用保存的320/384配置分别编码四个自选文本，记录token数、tokens、ID和整串decode结果。至少包含中文、英文、前导空格和未见emoji。

参考：Hello world!为9/5，中文“我喜欢喝牛奶。”均为3；不能总结为“词表越大每句话一定越短”。token ID只在各自配置内解释，不能直接跨配置喂原表。

## 3. 重复 token 的梯度，不一定等于出现次数

若ID序列 `[a,b,a]`，三个位置上游梯度分别是 `[1,2]`、`[3,4]`、`[5,6]`，求表行梯度。

答案：a为 `[6,8]`，b为 `[3,4]`。只有上游每次每维恰好为1时，梯度数值才等于出现次数；一般情况必须累加实际向量。

## 4. 单位置交叉熵

核对 `code/gradient_chain.py` 中e、W、logits、手算梯度的shape，解释为何无需先对logits做softmax再传给cross_entropy。

答案：e为 `[1,4]`、W为 `[6,4]`、logits为 `[1,6]`，`(p−one_hot(y))@W`为 `[1,4]`，与e.grad相同。交叉熵内部完成log-softmax。若平均多个位置，要计入平均因子。

## 5. 权重绑定：没有作输入的行也能有梯度

在安装PyTorch的环境运行：

```python
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)
embedding = nn.Embedding(6, 4, padding_idx=0)
head = nn.Linear(4, 6, bias=False)
head.weight = embedding.weight
loss = F.cross_entropy(head(embedding(torch.tensor([2]))), torch.tensor([3]))
loss.backward()
print(embedding.weight.grad.norm(dim=1))
```

答案：不仅输入行2有梯度，其他行也可收到输出分类器梯度；padding行0也可能如此。padding_idx只约束lookup路径。若训练绑定参数，先绑定，再用去重参数迭代器创建optimizer，例如 `nn.ModuleList([embedding, head]).parameters()`。

## 6. 置换等变性

将 `position_mechanics.py` 的置换改为 `[1,2,0]`。解释为什么无遮罩、无位置时仍应满足等变，而固定绝对位置后一般不满足。

答案：行重排同时重排Q、K、V和scores，输出对应重排；重新排列token后仍使用当前位置0/1/2，会改变内容和位置的配对。不能将这个假设直接应用于带因果mask的计算。

## 7. RoPE：范数、点积与相对位移

把positions中的整体偏移从5改成9，验证：每向量范数保持；整体平移后的旋转点积相同；只平移K的位置则通常改变点积。

答案：前两项来自正交旋转和 `R(mθ)^TR(nθ)=R((n−m)θ)`。只移K会改变n−m。测试固定未旋转q/k，不能推出改变完整上下文后输出也不变。

## 8. 三个 padding 接口

问：设置padding_idx=0后，是否可以删除Attention key mask与loss.ignore_index？

答案：不能。三者分别控制lookup梯度、可读key范围、参与损失的标签；RoPE也不会自动实现任何一种mask。
