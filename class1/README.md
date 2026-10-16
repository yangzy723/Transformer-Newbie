# Class 1 简明版：同一句例子，走通三个步骤

**对象**：会一点 Python，没学过深度学习。**时长**：60 分钟。

第一节只回答三个问题：

1. **文字变数字**：token → ID → 数字列表。
2. **混合上下文**：匹配分数 → 权重 → Attention 向量。
3. **预测下一项**：向量 → 候选分数 → 概率 → 下一个 token。

全课使用“我 喜欢 喝”，使用相同的二维向量，PPT、讲稿、脚本和 Notebook 一一对应。

## 上课时只打开这三样

| 文件 | 用法 |
| --- | --- |
| [class1_simple.pptx](class1_simple.pptx) | 新版 16 页 PPT，每页有讲者备注 |
| [teacher_guide.md](teacher_guide.md) | 按分钟写明该说什么、运行什么、检查什么 |
| [demo.ipynb](demo.ipynb) | 从上到下六个短单元，加两个小实验；所有代码直接可见 |

课后发给同学：[学生讲义](student_handout.md)。老师参考：[练习答案](exercise_answers.md)。

[一页课程地图](course_map.md)把每段的输入、输出、页码和代码对应起来。[PPT 文字版](slides.md)、[排版预览](assets/slides_preview.jpg)、[实际运行结果](sample_run.md)可作备份。排版预览由同一图形源绘制，非 PowerPoint 导出截图。

新版另存为 `class1_simple.pptx`。旧 `class1_transformer.pptx` 原样保留你的编辑，作为历史版本；本次授课请使用简明版。旧版里的代码路径以 [optional/ 的说明](optional/README.md)为准。

## 60 分钟：先动手，最后放回架构

| 时间 | PPT | 内容 | 验收问题 |
| --- | --- | --- | --- |
| 0–6 | 1–2 | 补词开场，展示三步地图 | 最后要得到什么？ |
| 6–16 | 3–5 | token、ID、向量、位置的作用 | 编号和向量有什么区别？ |
| 16–38 | 6–10 | 先理解加权，再算 Q/K/V、分数和权重 | Attention 输出是词还是向量？ |
| 38–42 | 11 | 因果遮罩，只看第一行 | 为什么未来权重必须为 0？ |
| 42–50 | 12–13 | 输出映射、候选概率、接回一个 token | 两次 Softmax 分别针对什么？ |
| 50–57 | 14–15 | 真实架构的其余组件，训练与生成 | 哪些数字来自训练？ |
| 57–60 | 16 | 自己画链路、解释一个改动 | 只改 V，什么变、什么不变？ |

多头、四维张量、归一化公式和反向传播实现留到下一节。第一节仍认识它们在完整架构中的位置。

## 代码怎么运行？

只需 Python 3。所有命令在仓库根目录执行，**无需 pip 安装、网络、GPU 或模型下载**。

```bash
python3 class1/code/lesson.py
python3 class1/code/mask_demo.py
```

Windows 可将 `python3` 换成 `py -3`。第一条一次打印整条主线，第二条是独立的短遮罩演示。打开 [lesson.py](code/lesson.py) 时，`# %%` 将代码分成六段，可在 VS Code 中逐段运行。

Notebook 的六个主线单元直接来自同一个脚本，没有隐藏的模型函数或路径初始化。已保留真实执行输出；在 VS Code/Jupyter 中选 Python 内核后可重新从头运行。查看缓存输出无需安装 Notebook 执行工具。

## 明确区分演示与完整模型

本课人工设置 embedding 和输出参数，Q/K/V 使用恒等投影，只算最后一个位置的单头 Attention。只在“酒、牛奶”两个候选间算输出概率，只生成一步。

完整 Transformer 还需要位置机制、FFN、残差、归一化等组件；实际 Q/K/V 通常由不同的可学习投影得到。真实输出层针对整个词表。本课的人工比例不是训练结果，不能把脚本称为完整语言模型。

[optional/](optional/README.md) 保留上一版真正训练的 Transformer、检查和高级 Notebook。学生先理解主线，再阅读该目录；老师可酌情展示结果。

## 维护教学材料（授课者按需使用）

课堂不需要阅读 `tools/`。内容源是 [slides.json](slides.json)，主线代码源是 [lesson.py](code/lesson.py)。

```bash
python -m pip install -r class1/requirements-tools.txt
python class1/tools/build_materials.py
python class1/tools/execute_notebook.py
```

第一条只为生成 PPT 和执行 Notebook 安装工具。重建会更新简明版 PPT、文字版、预览和 Notebook 模板，**不会写入旧 PPT**；随后执行 Notebook 可保存新输出。
