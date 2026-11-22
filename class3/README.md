# 第三课：重新走进 Transformer，理解 Attention

**21页、60分钟，前25分钟详细复习前两课。** 从“我／喜欢／喝”出发，用同一组小矩阵追踪输入怎样变成上下文向量，随后分别改动V、K和未来输入，观察变化从哪一步发生。

## 上课打开什么

| 材料 | 用途 |
|---|---|
| [课堂PPT](class3_attention.pptx) / [阅读版PDF](class3_attention.pdf) | 具体算例、计算步骤、矩阵与关键代码；含讲者备注 |
| [逐页教案](teacher_guide.md) | 时间、操作、提问、预期回答及追问 |
| [课堂Notebook](demo.ipynb) | 10个代码单元，含准备单元与9个实验，保留实际输出 |
| [学生讲义](student_handout.md) | 课后重看公式、手算和实现约定 |
| [练习](exercises.md) / [答案](exercise_answers.md) | 独立解释、计算、修改、验证 |
| [课程地图](course_map.md) / [讲解主线](teaching_story.md) | 与前两课及后续课程衔接 |
| [知识代码对照](knowledge_code_map.md) | 页码、时间、实验与源码定位 |
| [运行记录](sample_run.md) / [计算报告](assets/lesson_report.json) | 页面中的完整精度数值与实际输出 |
| [PPT文字稿](slides.md) / [整套预览](assets/slides_preview.jpg) | 备课检索与浏览 |
| 单页预览 | [架构](assets/preview_architecture.png) · [投影](assets/preview_projection.png) · [分数矩阵](assets/preview_scores.png) · [因果实验](assets/preview_causality.png) |

## 60分钟怎样安排

| 时间 | PPT | 内容 |
|---|---|---|
| 0–5 | 1–2 | 补词任务、完整生成架构 |
| 5–12 | 3–4 | 第一课加权汇总、两次Softmax |
| 12–18 | 5–6 | BPE合并后重新计数、按rank编码 |
| 18–25 | 7–8 | Embedding参数学习、相加式位置与RoPE |
| 25–35 | 9–11 | Q/K/V用途、可学习投影、一个query |
| 35–47 | 12–15 | 完整矩阵、缩放、因果mask、每行Softmax、AV |
| 47–53 | 16–18 | 继续同一算例：改V、改K、改未来输入 |
| 53–57 | 19–20 | 核心代码、多头结构预览 |
| 57–60 | 21 | 解释矩阵、手算一行输出、判断修改的影响 |

## 代码怎样穿插讲解

前半节复习时分别运行Notebook实验01–03。第10页运行实验04得到Q/K/V，接下来先在PPT上算点积、缩放和Mask，到第15页再运行实验05查看完整过程，避免提前展示尚未解释的结果。

第16–18页接着修改刚才的V、K和最后位置输入，对应实验06–08。每次先判断哪些量会保持不变，再运行并解释结果。第19页回看同一段核心代码。实验09把真实RoPE接口接回计算，供课后复验。

## 运行

已有第二课的环境可直接使用。新环境先安装依赖，之后实验只读本地配置、CPU运行。

```bash
python -m pip install -r class3/requirements.txt
python class3/code/review.py
python class3/code/attention.py
python class3/code/experiments.py
python class3/tools/check_experiments.py
```

`code/attention.py` 展示核心计算，`code/experiments.py` 生成 `assets/lesson_report.json`。
第二课的冻结配置位于 `class2/assets/tokenizer_320.json`，本课只读取它。请保留完整仓库。

核心主例是便于手算的控制矩阵，使用三位置、两维输入；Notebook实验09再把第二课的真实RoPE接口接回Q/K。
第20页以独立配置D=4、H=2说明下一课的多头结构。

## 重建课件

```bash
python -m pip install -r class3/requirements-tools.txt
python class3/tools/build_materials.py
python class1/tools/execute_notebook.py class3/demo.ipynb
```

只改PPT时使用 `python class3/tools/build_materials.py --slides-only`，保留Notebook的执行结果。
重建器同时生成PPT、矢量文字PDF、3200×1800预览、教案及知识代码对照，并检查可见文字与公式的边界。

内容源在 `slides.json`，页面布局在 `tools/pages.py`，通用版面在 `tools/presentation.py`。
字体与PDF绘制复用第二课工具；使用等线常规体与粗体，公式单独以320 dpi排版。
