"""制作第二课 PPT、讲义文字版和 Notebook。"""
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import nbformat as nbf
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from figures import draw

ROOT = Path(__file__).resolve().parents[1]
base_tools = ROOT.parent / "class1/tools"
sys.path.append(str(base_tools))
spec = importlib.util.spec_from_file_location("course_drawing", base_tools / "build_materials.py")
drawing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawing)


def build_slides():
    drawing.FONT = drawing.find_font()
    items = json.loads((ROOT / "slides.json").read_text(encoding="utf-8"))
    ppt = Presentation()
    ppt.slide_width, ppt.slide_height = Inches(13.333333), Inches(7.5)
    ppt.core_properties.title = "第二课：Token、Embedding 与位置信息"
    ppt.core_properties.subject = "Transformer 输入表示，60 分钟"
    ppt.core_properties.author = "Transformer Newbie"
    previews, text, minute = [], ["# 第二课：Token、Embedding 与位置信息\n"], 0
    for number, item in enumerate(items, 1):
        start, end = minute, minute + item["minutes"]
        minute = end
        slide = ppt.slides.add_slide(ppt.slide_layouts[6])
        c = drawing.Canvas(slide)
        c.rect(0, 0, .13, 7.5, drawing.TEAL, False)
        c.text(.65, .35, 9.1, .35, item["section"], 13, drawing.TEAL, bold=True)
        c.text(10.1, .35, 2.58, .35, f"{start:02d}–{end:02d} min", 13, drawing.MUTED, center=True)
        c.text(.65, .92, 12.05, .82, item["title"], 29, bold=True)
        c.rect(7.02, 1.86, 5.65, 4.06, "FFFFFF")
        for i, bullet in enumerate(item["bullets"]):
            y = 2.02 + i * 1.05
            c.node(.65, y, .38, .38, str(i+1), drawing.GOLD, 13)
            c.text(1.2, y+.02, 5.45, .93, bullet, 19)
        if item.get("code"):
            c.rect(.65, 5.18, 6.03, .66, drawing.INK)
            c.text(.83, 5.27, 5.66, .5, item["code"], 12, "FFFFFF")
        draw(c, item["kind"])
        c.rect(.65, 6.12, 12.02, .69, drawing.LIGHT)
        c.text(.87, 6.28, 11.57, .45, item["callout"], 17, drawing.TEAL, bold=True)
        c.text(.65, 7.07, 9.2, .25, "TRANSFORMER NEWBIE  /  CLASS 2  /  TOKEN · EMBEDDING · POSITION", 10, drawing.MUTED)
        c.text(11.73, 7.07, .92, .25, f"{number:02d} / {len(items):02d}", 10, drawing.MUTED, center=True)
        slide.notes_slide.notes_text_frame.text = f"第 {number} 页｜{start}–{end} 分钟\n{item['notes']}"
        previews.append(c.image)
        text += [f"\n## {number:02d}. {item['title']}（{start}–{end} 分钟）\n",
                 "\n".join("- "+b for b in item["bullets"]), "\n> "+item["callout"],
                 "\n**授课提示**："+item["notes"]]
        if item.get("code"):
            text += ["\n```python\n"+item["code"]+"\n```"]
    assert minute == 60
    ppt.save(ROOT / "class2_tokens_embeddings_positions.pptx")
    (ROOT / "slides.md").write_text("\n".join(text).rstrip()+"\n", encoding="utf-8")
    (ROOT / "assets").mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1440, math.ceil(len(items)/3)*290), "#D9E0E4")
    output = ROOT / "outputs/slide_previews"
    output.mkdir(parents=True, exist_ok=True)
    for i, preview in enumerate(previews):
        preview.save(output / f"slide_{i+1:02d}.png")
        thumbnail = preview.copy()
        thumbnail.thumbnail((470, 265))
        sheet.paste(thumbnail, ((i%3)*480+5, (i//3)*290+5))
    sheet.save(ROOT / "assets/slides_preview.jpg", quality=90)


def build_notebooks():
    descriptions = [
        ("1. 比较切分单位", "六个汉字可以按字、按词或按子词表示。这里的四词切分仅用作对照。"),
        ("2. 读懂一次合并", "遇到指定的相邻片段，就合成一项；从左到右扫描，不跨句子合并。"),
        ("3. 应用固定规则，再查 ID", "规则来自 optional/train_bpe.py 的小语料训练。按顺序观察合并。改输入为我喜欢喝酒，保留规则再运行。这里只覆盖词表中的基础字符，不代表某个商用模型的切分。"),
        ("4. 字符、字节、token", "牛奶是两个汉字、六个 UTF-8 字节。六个基础字节仍可合并；它们不是六个最终 token 的保证。"),
        ("5. ID 查 embedding 表", "沿用同一份12项词表，查 ID 4、11 两行。四维向量采用固定随机种子初始化，尚未学习语义。"),
        ("6. 重复查表", "参数固定时，同一 ID 的输入 embedding 相同；位置和上下文可使后续表示发生变化。"),
        ("7. 构造位置编码", "D=4 时两组角度分别是 p 和 p/100。位置从0开始，指 token 位置，不是字符偏移。"),
        ("8. 内容相同，位置不同", "这里复制了前面‘我’的四维向量，分别加入位置0和2的编码。验证输入改变，不把这个结果说成模型已经学会正确理解顺序。"),
    ]
    cells = [nbf.v4.new_markdown_cell("# 第二课：Token、Embedding 与位置信息\n\n文本如何切分？编号为什么映射成向量？顺序怎样进入计算？按顺序运行以下实验，代码只用 Python 标准库。")]
    blocks = []
    for name in ("tokenization.py", "embeddings.py", "positions.py"):
        source = (ROOT / "code" / name).read_text(encoding="utf-8")
        blocks += [b.strip() for b in re.split(r"(?m)^# %% .*\n", source) if b.strip()]
    assert len(blocks) == len(descriptions)
    for (title, explanation), block in zip(descriptions, blocks):
        compile(block, "<lesson-cell>", "exec")
        cells += [nbf.v4.new_markdown_cell(f"## {title}\n\n{explanation}"), nbf.v4.new_code_cell(block)]
    cells += [nbf.v4.new_markdown_cell("## 9. 从切分结果连接到 Attention 输入\n\n使用前面的实际切分结果，重新查 ID、取表行，再为位置0、1等构造输入。修改第一个单元的 text，并从头运行，可观察 token 数和矩阵形状一起改变。这里只演示相加式位置机制。"),
              nbf.v4.new_code_cell('input_ids = [vocab.index(piece) for piece in parts]\ninput_vectors = [table[token_id] for token_id in input_ids]\nmodel_inputs = []\nfor p, vector in enumerate(input_vectors):\n    pe = position_vector(p)\n    model_inputs.append([vector[i] + pe[i] for i in range(4)])\nprint("实际切分与 ID：", parts, input_ids)\nprint("送入后续计算的输入：", [[round(v, 4) for v in row] for row in model_inputs])\nprint("形状：", [len(model_inputs), len(model_inputs[0])])'),
              nbf.v4.new_markdown_cell("## 复习\n\n1. 本例为什么会合并喜欢喝牛奶？\n2. ID 的数值距离为什么不能解释为语义距离？\n3. 同一 token 在两个位置的原始 embedding 与组合输入分别相同吗？\n\n[学生讲义](student_handout.md) · [拓展实验](optional/README.md)")]
    extension_cells = [nbf.v4.new_markdown_cell("# 第二课拓展实验\n\n三个独立实验：学习 BPE 规则、更新 embedding 表、验证二维旋转点积。无需安装额外依赖。")]
    for name, title in (("train_bpe.py", "统计语料，学习合并规则"), ("embedding_update.py", "只更新 embedding 的分类实验"), ("rope_demo.py", "保持相对位移，比较 RoPE 点积")):
        source = (ROOT / "optional" / name).read_text(encoding="utf-8")
        compile(source, name, "exec")
        extension_cells += [nbf.v4.new_markdown_cell("## "+title), nbf.v4.new_code_cell(source)]
    for name, content in (("demo.ipynb", cells), ("optional/experiments.ipynb", extension_cells)):
        notebook = nbf.v4.new_notebook(cells=content)
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "file_extension": ".py"}}
        nbf.write(notebook, ROOT / name)


if __name__ == "__main__":
    build_slides()
    build_notebooks()
    print("第二课：18 页 PPT、60 分钟、9 个课堂实验单元、3 个拓展单元。")
