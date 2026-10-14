# Transformer Newbie

面向会一点 Python、尚未系统学习深度学习的同学，用可运行的代码理解 Transformer。

## 第一节课（60 分钟）

从一句话到下一个 token：理解文本的数字表示、Attention 的计算，以及一个小型 Transformer 的训练与生成。

进入 **[class1 教学材料](class1/README.md)**。

- [授课 PPT（21 页，含讲者备注）](class1/class1_transformer.pptx)
- [逐分钟教师讲稿](class1/teacher_guide.md)
- [学生讲义](class1/student_handout.md)
- [可运行 Notebook（附实际执行输出）](class1/demo.ipynb)

无需安装依赖即可体验 Attention：

```bash
python3 class1/code/attention_demo.py
python3 class1/code/attention_demo.py --causal
```

训练小模型需要 PyTorch，安装步骤见 [class1/README.md](class1/README.md)。训练不使用外部模型、API 或 GPU。

课程是教学用的简化实现：本节使用人工语料、空格分词、学习式位置向量和一个 Pre-LN Block。
