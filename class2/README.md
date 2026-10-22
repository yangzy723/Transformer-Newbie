# 第二课：Tokenizer、Embedding 与位置机制

**60 分钟，21 页 PPT。** 本课在第一课的 Q/K/V 与加权混合基础上，进入算法、梯度和张量实现。需要能跟踪向量、矩阵及 `[B,T,D]` 的形状，不预设已经掌握反向传播推导。

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
| [授课 PPT](class2_tokens_embeddings_positions.pptx) | 21 页，含公式、代码接口和逐页讲者备注 |
| [教案](teacher_guide.md) | 时间安排、板书顺序、实验操作、提问与答案 |
| [课堂 Notebook](demo.ipynb) | 准备单元 + 12 个实验单元，保留实际输出 |
| [讲义](student_handout.md) | 算法、梯度、位置公式与实现细节 |
| [课程地图](course_map.md) | 接口、shape 和参数来源 |
| [练习及答案](exercises.md) | 编码反例、梯度、权重绑定和旋转验证 |
| [运行记录](sample_run.md) | 真实输出和训练指标 |
| [PPT 文字版](slides.md) | 备课检索 |
| [排版预览](assets/slides_preview.jpg) | 同一绘图源的总览，非 PowerPoint 导出截图 |

## 时间安排

| 时间 | PPT | 内容 |
| --- | --- | --- |
| 0–3 分钟 | 1 | 目标、符号与输入链路 |
| 3–21 分钟 | 2–7 | 计算取舍、BPE 训练与编码、真实 ByteLevel 配置 |
| 21–39 分钟 | 8–13 | batch 查表、梯度累加、交叉熵、实际训练和权重绑定 |
| 39–56 分钟 | 14–19 | 置换等变性、正弦编码、RoPE 推导与 mask |
| 56–58 分钟 | 20 | 真实 ID 到多头 Attention 的接口整合 |
| 58–60 分钟 | 21 | 检查理解与课后实验 |

课堂先看核心公式与关键语句，再运行对应结果，不逐行朗读所有辅助函数。bigram 训练不是 Transformer，随机 Attention 接口也不是训练好的模型。

## 环境与运行

推荐 Python 3.11–3.13，CPU 即可。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r class2/requirements.txt
python class2/code/bpe_core.py
python class2/code/tokenization.py
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

需重训 tokenizer 时运行 `python class2/code/train_tokenizer.py`，会覆盖两份本地配置。改配置后须重新运行依赖其 ID 的实验并更新报告；正式授课使用已保存配置即可。

原有三个纯 Python 热身保留在 [optional/](optional/README.md)，使用独立的小表和二维旋转，不作为本课真实接口的替代。

## 重建材料

```bash
python -m pip install -r class2/requirements-tools.txt
python class2/tools/build_materials.py
python class1/tools/execute_notebook.py class2/demo.ipynb
```

PPT 复用第一课绘图工具，需保留完整仓库。公式由 `tools/lesson_alignment.py` 统一维护，制作时排成透明高清图片，同时保留 LaTeX 源码；公式图片不能在 PPT 中逐字符修改，可改源码后重建。其他文字和图形仍可编辑。重建 Notebook 会清空输出，执行后再保存。
