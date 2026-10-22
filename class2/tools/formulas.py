"""将公式排成透明高清图片，PPT 和预览共用同一份图像。"""
import os
from pathlib import Path

# 无显示服务器和用户缓存目录时也可重建。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/transformer-newbie-matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image
from pptx.util import Inches


def render_formula(expression, destination, font_size=26):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"mathtext.fontset": "stix", "font.family": "STIXGeneral"}):
        if r"\begin{bmatrix}" in expression:
            # Mathtext 不支持 bmatrix；按同一数学字体独立排齐四个元素。
            prefix, body = expression.split(r"\begin{bmatrix}", 1)
            rows = [row.split("&") for row in body.removesuffix(r"\end{bmatrix}").split(r"\\")]
            if len(rows) != 2 or any(len(row) != 2 for row in rows):
                raise ValueError("矩阵排版要求 2×2 元素")
            fig = Figure(figsize=(4.4, .85), dpi=320)
            fig.patch.set_alpha(0)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.axis("off")
            ax.text(.03, .5, "$"+prefix+"$", fontsize=font_size-1, va="center", color="#132C3E")
            for row, y in zip(rows, (.72, .26)):
                for cell, x in zip(row, (.49, .82)):
                    ax.text(x, y, "$"+cell+"$", fontsize=font_size-1, va="center", ha="center", color="#132C3E")
            for x, direction in ((.33, 1), (.98, -1)):
                ax.plot([x+.022*direction, x, x, x+.022*direction], [.96, .96, .02, .02],
                        color="#132C3E", linewidth=1.3)
            fig.savefig(destination, transparent=True, bbox_inches="tight", pad_inches=.03)
            fig.clear()
        else:
            math_to_image("$"+expression+"$", destination, dpi=320,
                          prop=FontProperties(size=font_size), color="#132C3E")


def place_formula(canvas, expression, destination, x, y, width, height, font_size=26):
    render_formula(expression, destination, font_size)
    with Image.open(destination) as source:
        formula = source.convert("RGBA")
    w, h = formula.size
    scale = min(width/w, height/h, 1/320)
    placed_w, placed_h = w*scale, h*scale
    left, top = x+(width-placed_w)/2, y+(height-placed_h)/2
    picture = canvas.slide.shapes.add_picture(str(destination), Inches(left), Inches(top),
                                             width=Inches(placed_w), height=Inches(placed_h))
    picture.name = "Formula: " + expression
    picture._element.nvPicPr.cNvPr.set("descr", expression)
    preview = formula.resize((round(placed_w*120), round(placed_h*120)), Image.Resampling.LANCZOS)
    canvas.image.paste(preview, (round(left*120), round(top*120)), preview)


def write_alignment(root, slides):
    from lesson_alignment import LINKS, source_label, experiment_label
    text = ["# 第二课：知识、公式与代码对照\n",
            "每个知识点按“问题 → 公式 → 关键语句 → 运行结果”阅读。"
            "公式中的矩阵用粗体，代码用相同含义的变量名；batch 维默认可以广播。\n",
            "| 符号 | 含义 | 代码 / 形状 |\n| --- | --- | --- |",
            "| $V,D$ | 词表大小、向量维度 | `num_embeddings`、`embedding_dim` |",
            "| $I,\\mathbf{E},\\mathbf{X}$ | ID、参数表、查表结果 | `ids`、`embedding.weight`、`vectors` |",
            "| $\\mathbf{e},\\mathbf{W}_{\\rm out},\\mathbf{z}$ | 单位置向量、输出权重、logits | `embedding_vector`、`output_weight`、`logits` |",
            "| $t,\\omega_i,\\theta_{t,i}$ | 位置、频率、旋转角 | `positions`、`frequencies`、`angles` |",
            "| $\\Pi$ | 行置换矩阵 | `x[permutation]`，不显式建矩阵 |",
            "| $\\mathbf{A},\\mathbf{M}$ | 注意力权重、加性 mask | `weights`、`masked_fill(..., -inf)` |\n",
            "单位置交叉熵使用行向量约定；RoPE 的几何推导把每一对分量写成列向量，"
            "张量代码直接按分量公式计算，两者表达同一个旋转。正弦和 RoPE 的频率按各自传入的维度计算；多头 RoPE 使用 $d_h$。\n"]
    for number, item in enumerate(slides, 1):
        link = LINKS.get(item["kind"])
        if not link:
            continue
        target = "code/"+link["file"]
        text += [f"\n## PPT {number:02d}｜{link['question']}\n",
                 f"**定位**：{experiment_label(link)}；[{source_label(link)}]({target})。\n"]
        for formula in link.get("formulas", []):
            text += ["$$\n"+formula+"\n$$\n"]
        if link.get("symbols"):
            text += [link["symbols"]+"。\n"]
        if link.get("code"):
            text += ["```python\n"+"\n".join(link["code"])+"\n```\n"]
        text += ["**观察**："+link["result"]+"。\n"]
    (root / "knowledge_code_map.md").write_text("\n".join(text).rstrip()+"\n", encoding="utf-8")
