"""制作第二课 PPT、讲义文字版和 Notebook。"""
import importlib.util
import json
import math
from pathlib import Path
import sys

from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from figures import draw
from formulas import draw_panel, write_alignment
from lesson_alignment import LINKS, experiment_label

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
    ppt.core_properties.title = "第二课：Tokenizer、Embedding 与位置机制"
    ppt.core_properties.subject = "Transformer 输入表示，60 分钟"
    ppt.core_properties.author = "Transformer Newbie"
    previews, text, minute = [], ["# 第二课：Tokenizer、Embedding 与位置机制\n"], 0
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
        link = LINKS.get(item["kind"])
        if link:
            c.text(.7, 5.25, 5.9, .26, experiment_label(link), 13, drawing.TEAL, bold=True)
            file = link["file"].replace("../tools/", "tools/")
            c.text(.7, 5.58, 5.9, .25, file+" · "+link["function"], 12, drawing.MUTED)
        if link and link.get("formulas"):
            draw_panel(c, link, ROOT, item["kind"])
        else:
            draw(c, item["kind"])
        c.rect(.65, 6.12, 12.02, .69, drawing.LIGHT)
        c.text(.87, 6.28, 11.57, .45, item["callout"], 17, drawing.TEAL, bold=True)
        c.text(.65, 7.07, 9.2, .25, "TRANSFORMER NEWBIE  /  CLASS 2  /  TOKEN · EMBEDDING · POSITION", 10, drawing.MUTED)
        c.text(11.73, 7.07, .92, .25, f"{number:02d} / {len(items):02d}", 10, drawing.MUTED, center=True)
        notes = f"第 {number} 页｜{start}–{end} 分钟\n{item['notes']}"
        if link:
            notes += f"\n对应：{experiment_label(link)}，{file}:{link['function']}\n观察：{link['result']}"
        slide.notes_slide.notes_text_frame.text = notes
        previews.append(c.image)
        text += [f"\n## {number:02d}. {item['title']}（{start}–{end} 分钟）\n",
                 "\n".join("- "+b for b in item["bullets"]), "\n> "+item["callout"],
                 "\n**授课提示**："+item["notes"]]
        if link:
            text += ["\n**对应实验**："+experiment_label(link)+"；`"+file+":"+link["function"]+"`。"]
            for formula in link.get("formulas", []):
                text += ["\n$$\n"+formula+"\n$$"]
            if link.get("symbols"):
                text += ["\n"+link["symbols"]+"。"]
            if link.get("code"):
                text += ["\n```python\n"+"\n".join(link["code"])+"\n```"]
            text += ["\n**运行观察**："+link["result"]+"。"]
    assert minute == 60
    ppt.save(ROOT / "class2_tokens_embeddings_positions.pptx")
    (ROOT / "slides.md").write_text("\n".join(text).rstrip()+"\n", encoding="utf-8")
    write_alignment(ROOT, items)
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
    from notebooks import build
    build(ROOT)


if __name__ == "__main__":
    build_slides()
    build_notebooks()
    print("第二课机制版：21 页、60 分钟、Tokenizers 与 PyTorch 实验。")
