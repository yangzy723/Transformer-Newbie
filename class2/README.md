# 第二课：Tokenizer、Embedding 与位置机制

**60 分钟，20 页 PPT。** 本课在第一课的 Q/K/V 与加权混合基础上，进入算法、梯度和张量实现。需要能跟踪向量、矩阵及 `[B,T,D]` 的形状，不预设已经掌握反向传播推导。

## 三个机制

| 部分 | 内容 | 实验 |
| --- | --- | --- |
| Tokenizer | 加权 pair 统计、merge rank、预切分与字节覆盖 | 可读 BPE 算法 + Hugging Face Tokenizers |
| Embedding | gather 前向、scatter-add 反传、交叉熵梯度、权重绑定 | nn.Embedding + 自动求导 + 下一 token 训练 |
| 位置 | 置换等变性、正弦多频率、RoPE 相对旋转、两类 mask | PyTorch 张量计算与等式验证 |

## 授课材料

| 文件 | 用途 |
| --- | --- |
| [授课主线](teaching_story.md) | 三段内容如何承接、怎样提问和收束 |
| [知识、公式与代码对照](knowledge_code_map.md) | 每页公式、关键语句、实验编号与预期结果 |
| [授课 PPT](class2_tokens_embeddings_positions.pptx) | 20 页，含解释、计算步骤、关键代码和讲者备注 |
| [固定排版 PDF](class2_tokens_embeddings_positions.pdf) | 正文为内嵌字体的矢量文字，适合缩放阅读与投影 |
| [教案](teacher_guide.md) | 时间安排、板书顺序、实验操作、提问与答案 |
| [课堂 Notebook](demo.ipynb) | 准备单元 + 12 个实验单元，保留实际输出 |
| [讲义](student_handout.md) | 算法、梯度、位置公式与实现细节 |
| [课程地图](course_map.md) | 接口、shape 和参数来源 |
| [练习及答案](exercises.md) | 编码反例、梯度、权重绑定和旋转验证 |
| [运行记录](sample_run.md) | 真实输出和训练指标 |
| [PPT 文字版](slides.md) | 备课检索 |
| [排版预览](assets/slides_preview.jpg) | 同一绘图源的总览，非 PowerPoint 导出截图 |
| 开场回顾 | [完整架构](assets/preview_opening.png) · [本课范围](assets/preview_scope.png) |
| 单页示例 | [BPE训练](assets/preview_bpe_training.png) · [BPE编码](assets/preview_bpe_encoding.png) · [梯度累加](assets/preview_embedding.png) · [RoPE](assets/preview_rope.png) |

## 时间安排

| 时间 | PPT | 内容 |
| --- | --- | --- |
| 0–6 分钟 | 1–2 | 回顾完整生成架构与第一课计算，定位本课三个主题 |
| 6–20 分钟 | 3–6 | BPE 训练与编码、真实 ByteLevel 配置与参数地址 |
| 20–39 分钟 | 7–12 | batch 查表、梯度累加、交叉熵、实际训练和权重绑定 |
| 39–56 分钟 | 13–18 | 置换等变性、正弦编码、RoPE 推导与 mask |
| 56–58 分钟 | 19 | 真实 ID 到多头 Attention 的接口整合 |
| 58–60 分钟 | 20 | 检查理解与课后实验 |

课堂先看核心公式与关键语句，再运行对应结果。RoPE先用30°箭头图解建立直觉，再接到多频率旋转公式和张量代码。

## 环境与运行

推荐 Python 3.11–3.13，CPU 即可。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r class2/requirements.txt
python class2/code/bpe_core.py
python class2/code/worked_examples.py
python class2/code/embeddings.py
python class2/code/gradient_chain.py
python class2/code/embedding_train.py
python class2/code/positions.py
python class2/code/position_mechanics.py
python class2/code/input_pipeline.py
python class2/tools/check_experiments.py
```

已有项目虚拟环境时直接使用它。Windows 可用 `py -3 -m venv .venv`，随后用 `.venv\Scripts\python.exe` 运行脚本。Notebook 选择同一环境的内核。

首次安装依赖需要网络；之后加载本地 tokenizer JSON 可离线运行，不下载模型权重。

## 配置与数据

- [本地语料](data/corpus.txt)：28 行中英文文本。
- [320 项配置](assets/tokenizer_320.json) / [384 项配置](assets/tokenizer_384.json)：真实 ByteLevel BPE；固定 Tokenizers 0.22.2。
- [编码报告](assets/tokenizer_report.json)：tokens、IDs、offsets、长度和回解结果。
- [训练报告](assets/training_report.json)：bigram 的 loss 与表行更新记录。
- [页面算例](assets/worked_examples.json)：奶茶 BPE、查表、梯度、交叉熵、RoPE 和 mask 的实际计算结果。

需重训 tokenizer 时运行 `python class2/code/train_tokenizer.py`，会覆盖两份本地配置。改配置后须重新运行依赖其 ID 的实验并更新报告；正式授课使用已保存配置即可。

`code/tokenization.py` 的词表规模对照保留为课后扩展，课堂按默认320项配置运行。

原有三个纯 Python 热身保留在 [optional/](optional/README.md)，使用独立的小表和二维旋转，不作为本课真实接口的替代。

## 重建材料

```bash
python -m pip install -r class2/requirements-tools.txt
python class2/tools/build_materials.py
python class1/tools/execute_notebook.py class2/demo.ipynb
```

只调整 PPT 时运行 `python class2/tools/build_materials.py --slides-only`，保留 Notebook 已执行的输出。

PPT 使用 16:9 投影布局，代码统一 18 号字，主标题 34 号字；分词对照、参数表、padding 和 mask 用原生图形展示。每页正文包含机制解释和具体算例，讲者备注补充操作提示。数值算例由 `code/worked_examples.py` 计算，保存到 `assets/worked_examples.json`。页面内容布局位于 `tools/detail_slides.py`，通用样式位于 `tools/slide_design.py`，内容与公式索引位于 `tools/lesson_alignment.py`。

PPT 复用第一课绘图工具，需保留完整仓库。公式制作时排成透明高清图片，同时保留 LaTeX 源码；公式图片不能在 PPT 中逐字符修改，可改源码后重建。其他文字和图形仍可编辑。重建 Notebook 会清空输出，执行后再保存。

### 字体与排版重建

PPT采用等线常规体与粗体；预览为3200×1800，PDF正文为内嵌字体的矢量文字。公式使用320 dpi数学排版。
重建器会查找Office自带的等线或系统Noto Sans CJK字体；也可以用 `CLASS2_FONT_REGULAR`、`CLASS2_FONT_BOLD`、`CLASS2_FONT_FAMILY` 指定一套字体。
