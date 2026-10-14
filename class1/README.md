# Class 1：Transformer 从一句话到下一个 token

**对象**：会一点 Python，没学过深度学习。**时长**：60 分钟（包含互动和代码）。**目标**：能画出完整计算链路、解释 Attention 的加权混合、区分训练和生成。

## 直接拿来上课

| 材料 | 用途 |
| --- | --- |
| [class1_transformer.pptx](class1_transformer.pptx) | 21 页可编辑 PPT，含架构图、权重图和每页讲者备注 |
| [teacher_guide.md](teacher_guide.md) | 按分钟安排讲解、提问、现场操作和应急处理 |
| [student_handout.md](student_handout.md) | 同学课后复习的概念、公式、形状和练习 |
| [slides.md](slides.md) | PPT 的文字版，便于 GitHub 上阅读和检索 |
| [PPT 排版预览](assets/slides_preview.jpg) | 21 页总览图；由同一排版源绘制，非 PowerPoint 导出截图 |
| [demo.ipynb](demo.ipynb) | 分步骤运行两个演示，已保留一轮真实执行输出 |
| [sample_run.md](sample_run.md) | 本次实际运行的结果，可在课堂设备出问题时使用 |
| [exercise_answers.md](exercise_answers.md) | 给老师的练习参考答案 |

## 60 分钟安排

| 时间 | PPT 页码 | 内容 | 同学应能回答 |
| --- | --- | --- | --- |
| 0–5 分钟 | 1–2 | 补词开场、学习目标 | 这节课要解决什么问题？ |
| 5–12 分钟 | 3–4 | 下一 token、训练与生成 | 模型如何把一段回复写出来？ |
| 12–20 分钟 | 5–7 | token、ID、向量、位置 | 文字如何成为一组数字？ |
| 20–35 分钟 | 8–12 | Q/K/V、手算、运行 Attention | 权重和输出分别是什么？ |
| 35–43 分钟 | 13–15 | 因果遮罩、Block、整体架构 | 为什么不能看未来？ |
| 43–55 分钟 | 16–19 | 小模型训练、loss、逐步生成 | 哪一步会更新参数？ |
| 55–60 分钟 | 20–21 | 复述、检查理解、课后实验 | 能否串起整个计算链路？ |

安装和下载安排在课前，课堂不等安装。FFN、LayerNorm、多头先认识作用，反向传播的推导留给下一节。

## 运行方法

下面的命令均在**仓库根目录**执行。推荐 Python 3.11–3.13，CPU 即可；首次安装需要网络，之后两个演示可离线运行。

### 演示 1：零依赖 Attention

```bash
python3 class1/code/attention_demo.py
python3 class1/code/attention_demo.py --causal
python3 class1/code/attention_demo.py --causal --svg class1/outputs/attention.svg
```

第三条命令生成热力图，用浏览器打开 `class1/outputs/attention.svg` 即可。图的行是 query、列是 key，每行加起来为 1。

### 演示 2：CPU 训练小型 Transformer

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r class1/requirements.txt
python class1/code/tiny_transformer.py --steps 200
python class1/code/tiny_transformer.py --prompt "你 喜欢 喝"
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r class1/requirements.txt
.\.venv\Scripts\python.exe class1/code/tiny_transformer.py --steps 200
```

模型会打印分词、错位标签、张量形状、loss 和生成候选。输出保存在 `class1/outputs/`：`training_report.json` 和参数文件 `tiny_transformer.pt`。这些运行产物不提交到 Git。

默认 seed=42，200 步。生成的合法后续不止一个，不同 PyTorch 版本可能选择不同句子；重点观察计算链路和 loss 变化。

### 使用 Notebook

```bash
python -m pip install -r class1/requirements-tools.txt
```

在 VS Code / Jupyter 中打开 `class1/demo.ipynb`，选择项目 `.venv` 的 Python 内核，按顺序运行。Notebook 从仓库根目录或 `class1` 目录启动均可。使用普通 Jupyter 的同学还需安装 `jupyterlab`，再执行 `python -m jupyter lab`；使用 VS Code 则无需安装 JupyterLab。

## 修改和重新生成材料

PPT 的内容源是 [slides.json](slides.json)，排版工具是 [tools/build_materials.py](tools/build_materials.py)。可以直接用 PowerPoint/WPS 修改 PPT；要保留可复现的内容，则改 JSON 并重建。

```bash
python -m pip install -r class1/requirements-tools.txt
python class1/tools/build_materials.py
```

这会重建 PPT、PPT 文字版、Notebook 模板和预览图；Notebook 模板初始没有输出。执行并保留新输出：

```bash
python class1/tools/execute_notebook.py
```

核对教学代码的关键性质：

```bash
python class1/code/check_demo.py
```

检查涵盖行权重归一化、未来遮罩、改变未来 token 不影响之前的预测、张量形状、训练确实降低误差，以及生成终止。

## 实现的教学简化

本节演示二是真实参与梯度训练的 decoder-only Transformer，有双头因果自注意力、FFN、残差与 LayerNorm。只使用 8 句人工语料，1 个 Block，向量维度 32。使用空格分词、学习式位置向量、Pre-LN；没有 dropout、KV cache 或预训练权重。训练误差来自同一组训练句子，没有验证集，不能据此判断泛化能力。

原始 Transformer 的 Encoder–Decoder、decoder 中的 cross-attention，以及本课的简化架构会在讲义中明确区分。来源见 [学生讲义](student_handout.md#参考资料)。
