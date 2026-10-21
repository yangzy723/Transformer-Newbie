# 第二课：Token、Embedding 与位置信息

**时长：60 分钟。** 适合完成第一课、会运行 Python 列表和循环的同学。

本课回答三个问题：文本为什么切成这些 token？ID 为什么映射成一组向量？模型怎样获得顺序信息？

## 授课材料

| 材料 | 用途 |
| --- | --- |
| [授课 PPT](class2_tokens_embeddings_positions.pptx) | 18 页，16:9，含逐页讲者备注 |
| [教案](teacher_guide.md) | 教学目标、逐分钟安排、提问与答案、实验操作 |
| [课堂 Notebook](demo.ipynb) | 前八个单元与三个脚本一致，第九个单元连接整条输入链路，保留执行输出 |
| [学生讲义](student_handout.md) | 三个问题的解释、公式与常见混淆 |
| [课程地图](course_map.md) | 例句、切分、ID、向量和位置的对应关系 |
| [练习及答案](exercises.md) | 课堂验收与课后实验 |
| [PPT 文字版](slides.md) | 便于检索和备课 |
| [PPT 预览](assets/slides_preview.jpg) | 同一绘图源的排版总览，非 PowerPoint 导出截图 |
| [运行结果](sample_run.md) | 实际运行输出，供课堂核对 |

## 课程安排

| 时间 | PPT 页码 | 内容 | 课堂产出 |
| --- | --- | --- | --- |
| 0–5 分钟 | 1–2 | 回顾输入表示，提出三个问题 | 画出本课的输入链路 |
| 5–20 分钟 | 3–6 | 字、词、子词；BPE 训练与使用；字符和字节 | 解释一次切分及其 ID |
| 20–38 分钟 | 7–11 | 查表、稠密向量、参数学习、上下文表示 | 运行查表与更新实验 |
| 38–53 分钟 | 12–15 | 顺序、位置相加、正弦编码、RoPE 与遮罩 | 比较同一 token 在两个位置的表示 |
| 53–56 分钟 | 16 | 从文本连接到 Attention 输入 | 写出本例的形状变化 |
| 56–60 分钟 | 17–18 | 检查理解，布置实验 | 回答三个“为什么” |

## 运行代码

在仓库根目录执行，只需要 Python 3，无需安装包、联网或 GPU：

```bash
python3 class2/code/tokenization.py
python3 class2/code/embeddings.py
python3 class2/code/positions.py
```

Windows 可将 `python3` 换成 `py -3`。也可打开 [demo.ipynb](demo.ipynb)，选择 Python 内核，按顺序运行；VS Code 中的脚本用 `# %%` 分段。

三个课堂辅助实验也只用标准库：

```bash
python3 class2/optional/train_bpe.py
python3 class2/optional/embedding_update.py
python3 class2/optional/rope_demo.py
```

[拓展 Notebook](optional/experiments.ipynb) 收录同样的三个实验。BPE 统计和 embedding 更新在课堂中展示结果；RoPE 点积验证可放到课后。

## 例子的范围

课堂 BPE 用三种句子学习四条规则，只有八个基础字符，词表最终为 12 项。它不处理任意字符，也不复现某个商用模型的切分；改变为词表外字符会无法查 ID。字节覆盖、规范化、特殊符号等实际实现问题在讲义中区分说明。

主线 embedding 表为 12×4，采用固定随机种子初始化。四 token 分类实验另用 4×2 的可训练表，不把两套数值混用。位置实验采用正弦相加，RoPE 则另在 Q/K 接口介绍。位置机制提供顺序信号，不能据此保证模型始终正确理解顺序或无限外推。

## 维护材料

制作 PPT、执行 Notebook 时才需要工具依赖：

```bash
python -m pip install -r class2/requirements-tools.txt
python class2/tools/build_materials.py
python class1/tools/execute_notebook.py class2/demo.ipynb
python class1/tools/execute_notebook.py class2/optional/experiments.ipynb
```

第二课复用第一课的绘图工具和 Notebook 执行器，需要保留仓库的完整目录结构。重建会更新本课 PPT、文字版、预览和 Notebook 模板；执行 Notebook 后保存输出。
