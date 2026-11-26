# 第四课：FFN、激活函数与采样

**28 页课件，约 60 分钟。** 本课承接第三课的 Attention，讲解三个问题：FFN 如何变换每个位置的向量，激活函数为什么不可省略，以及模型如何从词表分数中选出下一个 token。

| 材料 | 用途 |
|---|---|
| [PPT](class4_ffn_activation_sampling.pptx) / [PDF](class4_ffn_activation_sampling.pdf) | 结构图、逐步算例、激活曲线、概率图与代码；PPT 含讲者备注 |
| [整套预览](assets/slides_preview.jpg) | 浏览整套页面与内容顺序 |
| [逐页教案](teacher_guide.md) | 每页的讲解、提问与易错点 |
| [课堂 Notebook](demo.ipynb) | 18 个代码单元，含中间结果与数值核对；核心计算使用标准库 |
| [PPT 全文](slides.md) / [结构化内容](slides.json) | 检索、修改和重建 |
| [计算脚本](code/lesson.py) / [完整精度报告](assets/lesson_report.json) | 每个数值算例的可复核来源 |

## 课程内容

| 页码 | 内容 |
|---|---|
| 1–4 | 模型数据流、Pre-Norm Block、多头汇合、LayerNorm 的统计轴 |
| 5–10 | FFN 结构与权重，依次手算升维、ReLU 和下投影，再验证逐位置共享 |
| 11–15 | 无激活的代数合并，GELU/SiLU 与 SwiGLU，实际接回残差并对照代码 |
| 16–23 | LM Head、稳定 Softmax、贪心与抽样、频数、温度、top-k、top-p 和解码实现 |
| 24–28 | 自回归循环、训练与生成的区别、对照实验与综合练习 |

FFN 的升维、ReLU 和下投影分别展开计算。PPT 展示矩阵、运算与结果，Notebook 按相同顺序执行，并补充改变输入、激活和解码参数的对照实验。

## 算例与参数

1. **残差计算**：沿用第三课的 `h=[1.3,1.8]`。第 14 页使用本课的 FFN 参数，计算 `Norm(h)`、分支输出和残差结果；第三课的 `[-0.1,0.4]` 是当时为演示加法而给定的更新。
2. **FFN 计算**：给定输入 `[1,-2]` 及两个矩阵，依次得到四维中间特征、ReLU 结果和二维输出 `[1,2]`。多头汇合与 SwiGLU 各有独立的小型参数例。
3. **输出与采样**：LM Head 使用给定表示 `[1,2]`，得到词表分数 `[1,2,-1]`。解码实验则使用固定概率 `[.5,.3,.2]`，便于比较温度、top-k 和 top-p。

上述小矩阵由课程设定。输出实验使用三项词表“牛奶／茶／水”，局部 ID 为 0、1、2；自回归列表演示另列出完整词表。Notebook 在各节给出所用输入与参数。

## 运行与验证

核心计算无第三方依赖，安装 Python 即可运行：

```bash
python3 class4/code/lesson.py
python3 class4/tools/check_experiments.py
```

Notebook 的最后一个代码单元使用 PyTorch 核对权重转置与 LayerNorm。未安装 PyTorch 时，该单元会跳过核对。

## 重建课件

复用已有第二、三课的字体测量、原生图形与矢量 PDF 工具，请保留完整仓库。制作工具环境同第三课：

```bash
python -m pip install -r class4/requirements-tools.txt
python class4/tools/build_materials.py
python class4/tools/execute_notebook.py
python class4/tools/check_materials.py
```

预览指定页面可执行 `python class4/tools/build_materials.py --pages 7,8,9`，生成带 `_preview` 后缀的临时课件。完整重建会更新 Notebook 模板，之后需重新执行以保存输出。

结构、矩阵、表格、曲线和连接线使用 PPT 原生对象，数学公式使用透明图像。

完整预览与布局报告在 `outputs/`，由本目录 `.gitignore` 排除。最终 PPT、PDF 与页面预览保存在本目录和 `assets/`。
