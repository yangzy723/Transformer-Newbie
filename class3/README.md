# 第三课：重新走进 Transformer，理解 Attention

**33页、60分钟，前25分钟详细复习前两课。** 从“我／喜欢／喝”出发，用同一组小矩阵追踪输入怎样变成上下文向量，随后分别改动V、K和未来输入，观察变化从哪一步发生。

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
| 单页预览 | [开场数据流](assets/preview_opening.png) · [Q](assets/preview_query.png) · [K](assets/preview_key.png) · [V](assets/preview_value.png) · [加权求和](assets/preview_weighted_sum.png) · [Q/K/V类比](assets/preview_qkv.png) · [一次读取](assets/preview_qkv_flow.png) · [架构](assets/preview_architecture.png) · [投影](assets/preview_projection.png) · [分数矩阵](assets/preview_scores.png) · [因果实验](assets/preview_causality.png) |

## 60分钟怎样安排

| 时间 | PPT | 内容 |
|---|---|---|
| 0–4 | 1–3 | 补词任务、完整模型架构、Block内部结构 |
| 4–10 | 4–5 | BPE训练与编码 |
| 10–17 | 6–7 | Embedding参数学习、位置与RoPE |
| 17–25 | 8–11 | 选定读取者、逐项加权、每个位置各有输出 |
| 25–31 | 12–16 | 问题/索引/正文类比，分别展开Q、K、V |
| 31–36 | 17–20 | 三组投影的作用、每个位置的Q/K/V、手算投影 |
| 36–47 | 21–26 | 点积与分数矩阵、缩放、Mask、Softmax、AV |
| 47–49 | 27 | 读取权重与下一token预测概率 |
| 49–55 | 28–30 | 沿用同一算例：改V、改K、改未来输入 |
| 55–57 | 31–32 | 核心代码、多头结构 |
| 57–60 | 33 | 解释一行矩阵、手算输出、判断修改的影响 |

## 代码怎样穿插讲解

Notebook顺序与课堂一致：实验01复习BPE与Embedding，实验02复习位置，实验03逐项显示第一课的加权贡献。第10页运行实验03，直接对应“先乘比例，再按列求和”。

第12–16页先用资料检索讲Q/K/V，再分别解释当前查询、来源匹配特征与传递内容。第17–19页说明独立投影和每个位置的三份向量；第20页运行实验04，算出本节的Q/K/V。第21–26页逐步计算，最后运行实验05查看完整过程。

第28–30页分别运行实验06–08，修改刚才的V、K和最后位置输入。每次先判断哪些量不变，再运行并解释结果。第31页回看同一段代码。实验09接回真实RoPE，供课后复验。


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
第32页以独立配置D=4、H=2说明下一课的多头结构。

## 重建课件

```bash
python -m pip install -r class3/requirements-tools.txt
python class3/tools/build_materials.py
python class1/tools/execute_notebook.py class3/demo.ipynb
```

只改PPT时使用 `python class3/tools/build_materials.py --slides-only`，保留Notebook的执行结果。
重建器同时生成PPT、矢量文字PDF、3200×1800预览、教案及知识代码对照，并检查可见文字与公式的边界。

内容源在 `slides.json`，页面布局分为 `tools/overview_pages.py`（总览）、`tools/intro_pages.py`（读取与类比）、`tools/role_pages.py`（Q/K/V分工）与 `tools/pages.py`（计算），通用版面在 `tools/presentation.py`。
字体和文本测量复用第二课工具，箭头与PDF导出使用本课的线宽和几何设置；使用等线常规体与粗体，公式单独以320 dpi排版。
