# 第二课纯 Python 热身

本目录保留三个独立的数学热身，均只用标准库，不是机制版课堂主线。真实 Tokenizers 与 PyTorch 实验见 [第二课入口](../README.md)。也可打开 [experiments.ipynb](experiments.ipynb) 对照。

| 实验 | 命令 | 观察重点 |
| --- | --- | --- |
| 学习 BPE | `python3 class2/optional/train_bpe.py` | 统计相邻片段，四次合并，与主线固定规则一致 |
| 更新 embedding | `python3 class2/optional/embedding_update.py` | 四行二维表参与实际更新，分类交叉熵下降 |
| RoPE 二维验证 | `python3 class2/optional/rope_demo.py` | 同时平移 Q/K 位置，旋转点积只随相对位移变化 |

BPE 的八字符词表只覆盖例子中的字符；并列最高频 pair 按首次出现顺序选择。

embedding 更新实验使用“酒、牛奶、苹果、香蕉”四个 token 和“配喝、配吃”两个标签。输出层固定为恒等映射，只更新 4×2 表。它与机制版的320×8查表、320×16 bigram表相互独立，不是一个完整语言模型。

RoPE 脚本只旋转一对二维 Q/K，便于检验相对位置的代数性质。实际模型通常处理多对维度与多组频率，还包含其他层和遮罩。

回到 [第二课入口](../README.md)。

## 真实词表的课后对照

`python class2/code/tokenization.py` 保留320/384项配置的编码对照，供课后观察。当前课堂使用默认320项配置，不安排token数量对比页。
