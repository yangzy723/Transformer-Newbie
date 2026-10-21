# Transformer Newbie

面向会一点 Python、没学过深度学习的同学，通过能逐行解释的代码理解 Transformer。

## 第一节课：三个问题，60 分钟

**文字怎么变成数字？怎样混合上下文？怎样选出下一个 token？**

全课沿用“我 喜欢 喝”这个例子。课堂代码只用 Python 标准库，无需安装包。

```bash
python3 class1/code/lesson.py
```

从 **[class1 教学入口](class1/README.md)** 开始：

- [简明版 PPT：16 页，含讲者备注](class1/class1_simple.pptx)
- [教师讲稿：按分钟、按输入和输出讲解](class1/teacher_guide.md)
- [学生讲义：三个步骤和两次 Softmax](class1/student_handout.md)
- [Notebook：六个短单元，保留实际执行输出](class1/demo.ipynb)

课堂脚本采用人工二维向量和输出参数，展示一次计算；完整模型训练保留在 [optional/](class1/optional/README.md)，供后续课程使用。

## 第二节课：Token、Embedding 与位置信息，60 分钟

文本为什么这样切分？token ID 为什么查成向量？模型怎样获得顺序信息？

从 **[class2 教学入口](class2/README.md)** 开始：

- [18 页授课 PPT](class2/class2_tokens_embeddings_positions.pptx)
- [第二课教案](class2/teacher_guide.md)
- [课堂 Notebook](class2/demo.ipynb)
- [学生讲义](class2/student_handout.md)

主线代码只用 Python 标准库，分别运行 BPE 合并、embedding 查表和正弦位置实验。另有实际更新 embedding 表、学习 BPE 规则与 RoPE 点积的短实验。
