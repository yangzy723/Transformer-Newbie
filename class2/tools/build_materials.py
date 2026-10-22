"""制作第二课 PPT、讲义文字版和 Notebook。"""
import importlib.util
import json
import math
from pathlib import Path
import sys

from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from formulas import write_alignment
from lesson_alignment import LINKS, experiment_label
from slide_design import DESIGN, canvas_class, header, reference, formula_slide, large_slide

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
    Canvas = canvas_class(drawing.Canvas, drawing.FONT)
    report = json.loads((ROOT / "assets/training_report.json").read_text(encoding="utf-8"))
    layout_report = []
    for number, item in enumerate(items, 1):
        start, end = minute, minute + item["minutes"]
        minute = end
        slide = ppt.slides.add_slide(ppt.slide_layouts[6])
        c = Canvas(slide)
        header(c,item,number,start,end,len(items))
        link = LINKS.get(item["kind"])
        if link and link.get("formulas"):
            formula_slide(c,item,link,ROOT,report)
        else:
            large_slide(c,item)
        reference(c,link)
        title, subtitle, takeaway = DESIGN[item["kind"]]
        notes = (f"第 {number} 页｜{start}–{end} 分钟\n{item['title']}\n"
                 +"\n".join(item["bullets"])+f"\n{item['callout']}\n{item['notes']}")
        if link:
            file = link["file"].replace("../tools/", "tools/")
            notes += f"\n对应：{experiment_label(link)}，{file}:{link['function']}\n观察：{link['result']}"
            if item["kind"] == "absolute":
                notes += "\n投影代码出自 position_mechanics.py:demo；位置函数来自 positions.py:sinusoidal。"
        slide.notes_slide.notes_text_frame.text = notes
        previews.append(c.image)
        layout_report.append({"slide":number,"title":title,"text":c.text_metrics})
        text += [f"\n## {number:02d}. {title}（{start}–{end} 分钟）\n",
                 subtitle, "\n".join("- "+b for b in item["bullets"]), "\n> "+takeaway,
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
    sheet = Image.new("RGB", (1920, math.ceil(len(items)/3)*382), "#D9E0E4")
    output = ROOT / "outputs/slide_previews"
    output.mkdir(parents=True, exist_ok=True)
    for i, preview in enumerate(previews):
        preview.save(output / f"slide_{i+1:02d}.png")
        thumbnail = preview.copy()
        thumbnail.thumbnail((628, 354))
        sheet.paste(thumbnail, ((i%3)*640+6, (i//3)*382+6))
    sheet.save(ROOT / "assets/slides_preview.jpg", quality=90)
    (output / "layout_report.json").write_text(json.dumps(layout_report, ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # 三张独立示例供浏览，无需先打开整个 PPT。
    for name, index in (("tokenizer",5),("embedding",9),("rope",16)):
        previews[index].save(ROOT / f"assets/preview_{name}.png")


def build_notebooks():
    from notebooks import build
    build(ROOT)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides-only", action="store_true", help="仅重建 PPT 与预览，保留 Notebook 的执行结果")
    args = parser.parse_args()
    build_slides()
    if not args.slides_only:
        build_notebooks()
    print("第二课机制版：21 页、60 分钟、Tokenizers 与 PyTorch 实验。")
