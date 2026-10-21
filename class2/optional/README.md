# 第二课拓展实验

三个独立脚本均使用 Python 标准库。在仓库根目录运行，或打开 [experiments.ipynb](experiments.ipynb)。

| 实验 | 命令 | 观察重点 |
| --- | --- | --- |
| 学习 BPE | `python3 class2/optional/train_bpe.py` | 统计相邻片段，四次合并，与主线固定规则一致 |
| 更新 embedding | `python3 class2/optional/embedding_update.py` | 四行二维表参与实际更新，分类交叉熵下降 |
| RoPE 二维验证 | `python3 class2/optional/rope_demo.py` | 同时平移 Q/K 位置，旋转点积只随相对位移变化 |

BPE 的八字符词表只覆盖例子中的字符；并列最高频 pair 按首次出现顺序选择。

embedding 更新实验使用“酒、牛奶、苹果、香蕉”四个 token 和“配喝、配吃”两个标签。输出层固定为恒等映射，只更新 4×2 表。它与主线的 12×4 随机初始化表相互独立，不是一个完整语言模型。

RoPE 脚本只旋转一对二维 Q/K，便于检验相对位置的代数性质。实际模型通常处理多对维度与多组频率，还包含其他层和遮罩。

回到 [第二课入口](../README.md)。
