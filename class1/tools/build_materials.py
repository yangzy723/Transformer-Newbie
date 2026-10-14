"""从 slides.json 生成可编辑 PPT、文字版、预览和 Notebook 模板。

图和文字使用同一套坐标；PPT 里的文字和流程图均为原生可编辑对象。
预览依赖本机中文字体，是排版参考，不是 PowerPoint 导出的截图。
"""

import json
import inspect
import math
import os
import sys
from pathlib import Path

import nbformat as nbf
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from attention_demo import Q, K, V, attention

BG, INK, TEAL = "F7F5EF", "132C3E", "087F8C"
MUTED, GOLD, LIGHT, BLUE = "5C6F7B", "DAA35B", "E6EFEE", "DEE8F0"
WIDTH, HEIGHT, SCALE = 13.333333, 7.5, 120


def find_font():
    candidates = [os.environ.get("CLASS1_PREVIEW_FONT", ""),
                  "/System/Library/Fonts/PingFang.ttc",
                  "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                  "C:/Windows/Fonts/msyh.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("预览需要中文字体。请用 CLASS1_PREVIEW_FONT 指定字体文件路径。")


FONT = None


class Canvas:
    def __init__(self, slide):
        self.slide = slide
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string(BG)
        self.image = Image.new("RGB", (1600, 900), "#" + BG)
        self.draw = ImageDraw.Draw(self.image)

    def rect(self, x, y, w, h, color=LIGHT, rounded=True):
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(color)
        shape.line.fill.background()
        if rounded:
            shape.adjustments[0] = 0.06
        bounds = (round(x*SCALE), round(y*SCALE), round((x+w)*SCALE), round((y+h)*SCALE))
        if rounded:
            self.draw.rounded_rectangle(bounds, radius=10, fill="#"+color)
        else:
            self.draw.rectangle(bounds, fill="#"+color)

    def text(self, x, y, w, h, content, size=20, color=INK, bold=False, center=False):
        font = ImageFont.truetype(FONT, round(size*SCALE/72))
        lines = []
        for original in str(content).split("\n"):
            current = ""
            for char in original:
                if self.draw.textlength(current + char, font=font) > (w*SCALE-6):
                    lines.append(current)
                    current = char
                else:
                    current += char
            lines.append(current)
        line_px = size*SCALE/72*1.3
        if len(lines)*line_px > h*SCALE + 2:
            raise ValueError(f"文字超出预留高度：{content!r}")
        box = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.clear()
        tf.word_wrap = False  # 使用已经按字体测量好的换行。
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        from pptx.enum.text import PP_ALIGN
        for i, line in enumerate(lines):
            paragraph = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            paragraph.text = line
            paragraph.font.name = "Microsoft YaHei"
            paragraph.font.size = Pt(size)
            paragraph.font.bold = bold
            paragraph.font.color.rgb = RGBColor.from_string(color)
            paragraph.line_spacing = Pt(size*1.3)
            paragraph.space_before = paragraph.space_after = Pt(0)
            if center:
                paragraph.alignment = PP_ALIGN.CENTER
            px = x*SCALE
            if center:
                px += (w*SCALE - self.draw.textlength(line, font=font))/2
            self.draw.text((round(px), round(y*SCALE+i*line_px)), line,
                           font=font, fill="#"+color)

    def line(self, x1, y1, x2, y2, color=TEAL, arrow=False):
        line = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        line.line.color.rgb = RGBColor.from_string(color)
        line.line.width = Pt(1.7)
        self.draw.line((x1*SCALE, y1*SCALE, x2*SCALE, y2*SCALE), fill="#"+color, width=3)
        if arrow:
            angle = math.atan2(y2-y1, x2-x1)
            points = [(x2*SCALE, y2*SCALE)]
            for delta in (-0.55, 0.55):
                points.append(((x2-0.09*math.cos(angle+delta))*SCALE,
                               (y2-0.09*math.sin(angle+delta))*SCALE))
            self.draw.polygon(points, fill="#"+color)
            tri = self.slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                Inches(x2-0.035), Inches(y2-0.045), Inches(0.07), Inches(0.09))
            tri.rotation = math.degrees(angle) + 90
            tri.fill.solid()
            tri.fill.fore_color.rgb = RGBColor.from_string(color)
            tri.line.fill.background()

    def node(self, x, y, w, h, text, color=LIGHT, size=17, text_color=INK):
        self.rect(x, y, w, h, color)
        self.text(x+0.08, y+0.09, w-0.16, h-0.10, text, size, text_color, center=True)


def vertical_flow(c, labels, x=7.65, y=2.0, w=4.25, h=0.5, gap=0.17, size=17):
    for i, label in enumerate(labels):
        top = y+i*(h+gap)
        c.node(x, top, w, h, label, BLUE if i % 2 else LIGHT, size)
        if i < len(labels)-1:
            c.line(x+w/2, top+h+0.01, x+w/2, top+h+gap-0.01, arrow=True)


def heatmap(c, causal):
    _, weights, _ = attention(Q, K, V, causal)
    c.text(7.5, 2.05, 4.7, 0.3, "Key →    我        喜欢        猫", 16, MUTED)
    for i, row in enumerate(weights):
        c.text(7.25, 2.65+i*0.72, 0.65, 0.35, ["我", "喜欢", "猫"][i], 17)
        for j, weight in enumerate(row):
            ratio = weight
            rgb = tuple(round((1-ratio)*a+ratio*b) for a, b in zip((230,239,238),(8,127,140)))
            color = "".join(f"{v:02X}" for v in rgb)
            c.node(8.0+j*1.24, 2.5+i*0.72, 1.12, 0.62, f"{weight:.3f}", color, 18,
                   "FFFFFF" if weight > .75 else INK)
    c.text(7.5, 5.02, 4.85, 0.62, "Query ↓   每一行和为 1\n" +
           ("上三角权重为 0：未来不可见" if causal else "人工示例：先观察计算，再解释结构"), 16, MUTED)


def diagram(c, kind, report):
    if kind == "opening":
        c.text(7.4, 2.15, 4.9, 0.45, "从你熟悉的补词开始", 18, TEAL)
        c.text(7.4, 3.03, 4.95, 1.45, "我 喜欢 喝\n____", 36, bold=True)
        c.node(7.45, 4.67, 1.8, 0.56, "茶", BLUE, 23)
        c.node(9.6, 4.67, 2.25, 0.56, "牛奶", LIGHT, 23)
    elif kind in ("goals", "homework"):
        vertical_flow(c, ["画出数据流", "算一次 Attention", "看一次训练与生成"], y=2.5, h=0.65, gap=0.3, size=22)
    elif kind == "probability":
        c.text(7.5, 2.15, 4.8, 0.4, "下一 token 的概率（示意）", 18, TEAL)
        for i, (name, p) in enumerate((("茶",.6),("牛奶",.3),("其他",.1))):
            y = 2.95+i*.72
            c.text(7.5,y,1.05,.4,name,19)
            c.rect(8.65,y,2.85*p,.36,TEAL)
            c.text(11.6,y,.6,.4,f"{p:.2f}",18)
        c.text(7.5, 5.25, 4.8, .35, "这些概率加起来为 1", 18, MUTED)
    elif kind == "shift":
        c.text(7.5,2.16,4.8,.4,"输入 x 与目标 y：错开一位",18,TEAL)
        c.text(7.5,3.02,4.75,1.1,"x: <bos> 我 喜欢 吃 苹果 。\ny: 我 喜欢 吃 苹果 。 <eos>",18)
        c.line(9.8,4.25,9.8,4.65,arrow=True)
        c.node(7.6,4.82,4.35,.58,"“喜欢”位置的目标是“吃”",LIGHT,19)
    elif kind == "tokens":
        vertical_flow(c,["文本：我 喜欢 喝 茶",'split()：四个 token',"ID：[7, 5, 6, 10]", "查 embedding 表"], y=2.25,h=.56,gap=.25,size=19)
    elif kind == "embedding":
        c.text(7.45,2.14,4.9,.4,"每个位置 → 一个数字列表",18,TEAL)
        for i, token in enumerate(("我", "喜欢", "喝")):
            c.text(7.5,2.92+i*.65,.8,.4,token,19)
            for j, val in enumerate(((.2,-.7,1.1),(.4,.1,-.5),(-.1,.8,.3))[i]):
                c.node(8.45+j*1.08,2.85+i*.65,.96,.5,str(val),BLUE,18)
        c.text(7.5,5.15,4.8,.4,"示意 3 维；代码实际使用 32 维",17,MUTED)
    elif kind == "position":
        vertical_flow(c,["token 向量 [T, D]", "+ 位置向量 [T, D]", "带位置信息的 X [T, D]"],y=2.48,h=.6,gap=.3,size=20)
    elif kind in ("context", "manual"):
        heatmap(c,False)
    elif kind == "qkv":
        c.node(8.95,2.13,1.7,.55,"输入 X",BLUE,22)
        for j,label in enumerate(("Q = XWq", "K = XWk", "V = XWv")):
            x=7.25+j*1.72
            c.line(9.8,2.75,x+.78,3.48,arrow=True)
            c.node(x,3.58,1.56,.6,label,LIGHT,17)
        c.text(7.45,4.77,4.9,.72,"投影参数由训练得到\nself-attention：三者来自同一序列",18,MUTED)
    elif kind == "formula":
        vertical_flow(c,["QKᵀ / √dₖ   [T, T]", "每行 softmax   [T, T]", "权重 @ V   [T, dᵥ]"],y=2.38,h=.66,gap=.33,size=22)
    elif kind == "mask":
        heatmap(c,True)
    elif kind == "block":
        # 两个残差路径分别从自己的子层输入绕到加法节点。
        x,w=7.8,3.8
        ys=[2.0,2.49,2.98,3.47,3.96,4.45,4.94,5.43]
        labels=["输入 x", "LayerNorm", "双头因果 Attention", "＋ 残差", "LayerNorm", "FFN", "＋ 残差", "输出 x"]
        for i,(y,label) in enumerate(zip(ys,labels)):
            c.node(x,y,w,.36,label,BLUE if i in (2,5) else LIGHT,13)
            if i < len(ys)-1:
                c.line(x+w/2,y+.37,x+w/2,y+.48,arrow=True)
        for start,end,side in ((2.18,3.65,12.08),(3.65,5.12,12.37)):
            c.line(11.6,start,side,start,GOLD)
            c.line(side,start,side,end,GOLD)
            c.line(side,end,11.63,end,GOLD,True)
        c.text(7.8,5.83,4.6,.22,"Pre-LN；两条金色路径是残差相加",10,MUTED)
    elif kind == "architecture":
        vertical_flow(c,["Token ID", "Token + Position 向量", "因果 Transformer Block × 1", "最终 LayerNorm", "Linear → logits → softmax"],x=7.6,y=2.02,w=4.75,h=.45,gap=.21,size=16)
        c.text(7.65,5.48,4.6,.3,"本课 decoder-only，无 cross-attention",14,MUTED)
    elif kind == "shapes":
        vertical_flow(c,["ID： [8, 6]", "向量： [8, 6, 32]", "权重： [8, 2, 6, 6]", "logits： [8, 6, 12]"],y=2.08,h=.54,gap=.26,size=20)
    elif kind == "training":
        c.text(7.45,2.07,4.9,.4,"实际运行：训练集 loss",19,TEAL)
        points=report["history"]
        x0,y0,w,h=7.7,2.9,4.25,2.25
        c.line(x0,y0,x0,y0+h,MUTED)
        c.line(x0,y0+h,x0+w,y0+h,MUTED)
        ymax=max(p["loss"] for p in points)*1.08
        previous=None
        for point in points:
            position=(x0+w*point["step"]/report["steps"],y0+h*(1-point["loss"]/ymax))
            if previous:
                c.line(*previous,*position,TEAL)
            previous=position
        c.text(7.5,2.52,4.85,.35,f"{report['initial_loss']:.3f} → {report['final_loss']:.3f} · {report['steps']} steps",20)
        c.text(7.7,5.34,4.3,.35,"无验证集；只展示人工语料拟合",15,MUTED)
    elif kind == "generation":
        vertical_flow(c,["最后位置的 logits", "softmax → 下一 token 概率", "选择 token → 接回前文", "再计算一次，直到 <eos>"],y=2.17,h=.53,gap=.27,size=18)
    elif kind == "limits":
        c.text(7.45,2.22,4.85,.4,"本次实际运行",19,TEAL)
        c.text(7.45,2.93,4.83,.4,"训练前",17,MUTED)
        c.text(7.45,3.35,4.8,.72,report["before_training"],22)
        c.text(7.45,4.35,4.83,.4,"训练后",17,MUTED)
        c.text(7.45,4.77,4.8,.74,report["after_training"],22)
    elif kind == "quiz":
        vertical_flow(c,["Attention → 向量", "因果 mask → 未来权重 0", "生成 → 参数保持不变"],y=2.4,h=.66,gap=.33,size=20)
    elif kind == "code":
        c.text(7.45,2.24,4.83,.4,"现场操作：一次只改一项",19,TEAL)
        vertical_flow(c,["先写预测", "运行函数", "解释输出变化"],y=3.0,h=.55,gap=.27,size=22)
    else:
        raise ValueError(kind)


def build_slides(slides, report):
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(WIDTH), Inches(HEIGHT)
    presentation.core_properties.title = "Class 1 — Transformer：从一句话到下一个 token"
    presentation.core_properties.subject = "60 分钟入门教学，含 Attention 与 CPU 训练演示"
    presentation.core_properties.author = "Transformer Newbie"
    previews, markdown, minute = [], ["# Class 1 PPT 文字版\n", "由 `slides.json` 自动生成；备注也包含在 PPT 中。\n"], 0
    for number,item in enumerate(slides,1):
        start,end=minute,minute+item["minutes"]
        minute=end
        slide=presentation.slides.add_slide(presentation.slide_layouts[6])
        c=Canvas(slide)
        c.rect(0,0,.13,HEIGHT,TEAL,False)
        c.text(.65,.35,8,.35,item["section"],13,TEAL,bold=True)
        c.text(10.1,.35,2.58,.35,f"{start:02d}–{end:02d} min",13,MUTED,center=True)
        c.text(.65,.92,12.05,.82,item["title"],30,bold=True)
        c.rect(7.02,1.86,5.65,4.06,"FFFFFF")
        code=item.get("code")
        for i,bullet in enumerate(item["bullets"]):
            y=2.02+i*(.87 if code else 1.05)
            c.node(.65,y,.38,.38,str(i+1),GOLD,13)
            c.text(1.2,y+.02,5.45,.93,bullet,19)
        if code:
            c.rect(.65,4.95,6.03,.94,INK)
            c.text(.83,5.08,5.66,.72,code,10,"FFFFFF")
        diagram(c,item["kind"],report)
        c.rect(.65,6.12,12.02,.69,LIGHT)
        c.text(.87,6.28,11.57,.45,item["callout"],18,TEAL,bold=True)
        c.text(.65,7.07,9.2,.25,"TRANSFORMER NEWBIE  /  CLASS 1  /  先预测 · 再运行 · 再解释",10,MUTED)
        c.text(11.73,7.07,.92,.25,f"{number:02d} / {len(slides):02d}",10,MUTED,center=True)
        slide.notes_slide.notes_text_frame.text=f"第 {number} 页｜{start}–{end} 分钟\n{item['notes']}"
        previews.append(c.image)
        markdown += [f"\n## {number:02d}. {item['title']}（{start}–{end} 分钟）\n",
                     "\n".join(f"- {b}" for b in item["bullets"])+"\n"]
        if code:
            markdown.append("\n```python\n"+code+"\n```\n")
        markdown += ["\n> "+item["callout"]+"\n", "\n**讲者备注**："+item["notes"]+"\n"]
    if minute != 60:
        raise ValueError(f"课程时长应为 60 分钟，实际为 {minute}")
    presentation.save(ROOT/"class1_transformer.pptx")
    (ROOT/"slides.md").write_text("\n".join(markdown),encoding="utf-8")
    assets=ROOT/"assets"
    assets.mkdir(exist_ok=True)
    sheet=Image.new("RGB",(1440,math.ceil(len(previews)/3)*290),"#D9E0E4")
    for i,preview in enumerate(previews):
        thumbnail=preview.copy()
        thumbnail.thumbnail((470,265))
        sheet.paste(thumbnail,((i%3)*480+5,(i//3)*290+5))
    sheet.save(assets/"slides_preview.jpg",quality=90)
    # 单页预览供检查和复用，不将整组重复图片提交到仓库。
    preview_dir=ROOT/"outputs"/"slide_previews"
    preview_dir.mkdir(parents=True,exist_ok=True)
    for i,slide_image in enumerate(previews,1):
        slide_image.save(preview_dir/f"slide_{i:02d}.png")


def build_notebook():
    def md(text):
        return nbf.v4.new_markdown_cell(inspect.cleandoc(text))

    def code(text):
        source = inspect.cleandoc(text)
        compile(source, "<notebook-cell>", "exec")
        return nbf.v4.new_code_cell(source)

    cells=[
        md("""# Class 1：从 Attention 到一个能训练的 Transformer

        对象：会一点 Python，没有深度学习基础。请先预测每个实验的结果，再运行。
        本 Notebook 可从仓库根目录或 class1 目录启动。首次安装见 README，选项目 `.venv` 内核。
        本课没有外部模型、API 或 GPU 依赖。保留的输出来自一次实际执行；重新运行会刷新它们。"""),
        code("""import sys
        from pathlib import Path

        start = Path.cwd().resolve()
        candidates = [start, start / 'class1', start.parent / 'class1']
        CLASS1 = next((p for p in candidates if (p / 'code/attention_demo.py').is_file()), None)
        if CLASS1 is None:
            raise RuntimeError('请从仓库根目录或 class1 目录打开 Notebook')
        sys.path.insert(0, str(CLASS1 / 'code'))
        print('Python:', sys.version.split()[0])"""),
        md("""## 1. 文字变数字：先认识列表和查表

        一个数字列表称为向量，多个等长数字列表叠起来称为矩阵。
        本课按空格分词，真实模型可能用子词等单位。ID 只是编号，embedding 则是一组会训练的数字。
        第一个 Attention 演示直接使用人工 Q/K/V，不先训练 embedding。"""),
        code("""text = '我 喜欢 喝 茶'
        print('教学分词：', text.split())
        from attention_demo import Q, K, V, TOKENS, attention, print_matrix
        print('演示位置：', TOKENS)
        print_matrix('Q', Q)
        print_matrix('K', K)
        print_matrix('V', V)"""),
        md("""## 2. Attention：匹配 → softmax → 混合

        手算第一行：`[1,0]` 与三个 key 点积、除以 √2，约为 `[0.707,0,0.707]`。
        softmax 得到约 `[0.401,0.198,0.401]`，输出是 V 的加权混合，仍是向量。
        行是 query，列是 key；每行权重之和为 1。"""),
        code("""scores, weights, output = attention(Q, K, V)
        print_matrix('匹配分数', scores)
        print_matrix('Attention 权重', weights)
        print_matrix('输出向量', output)
        print('各行权重之和：', [round(sum(row), 6) for row in weights])"""),
        md("""### 实验 A：只改 V（运行前先猜）

        把最后一个 V 改成 `[30,10]`。Q/K 没变：权重会变吗？输出会变吗？"""),
        code("""changed_v = [list(row) for row in V]
        changed_v[2] = [30.0, 10.0]
        _, changed_weights, changed_output = attention(Q, K, changed_v)
        print('权重相同：', weights == changed_weights)
        print_matrix('修改 V 后的输出', changed_output)"""),
        md("""### 因果遮罩：不能读到未来的位置

        未来位置的分数设为负无穷，再计算 softmax。第一行只能看自己。
        当前位置预测的是下一个 token，因此对角线允许读取。"""),
        code("""causal_scores, causal_weights, causal_output = attention(Q, K, V, causal=True)
        print_matrix('有遮罩的分数', causal_scores)
        print_matrix('有遮罩的权重', causal_weights)
        print_matrix('有遮罩的输出', causal_output)
        _, _, causal_changed = attention(Q, K, changed_v, causal=True)
        print('只改第三个 V，前两个输出不变：', causal_output[:2] == causal_changed[:2])"""),
        code("""from IPython.display import SVG, display
        from attention_demo import save_heatmap
        heatmap_path = CLASS1 / 'outputs' / 'attention_causal.svg'
        save_heatmap(causal_weights, heatmap_path, causal=True)
        display(SVG(filename=str(heatmap_path)))"""),
        md("""## 3. 训练一个真实的小型 Transformer

        PyTorch 为我们计算梯度。模型：1 个 Block、2 头、D=32、Pre-LN。
        有因果 self-attention、FFN、残差、LayerNorm、可学习的 token/position embedding。
        数据只有八句人工语料；空格分词。没有验证集，观察的是训练集拟合。
        没安装 PyTorch 时，先按 README 安装，然后重启内核并按顺序运行。"""),
        code("""import torch
        from tiny_transformer import (SENTENCES, VOCAB, TOKEN_TO_ID, BOS, EOS,
                                      TinyTransformer, make_batch, decode,
                                      train_model, generate, encode)
        print('PyTorch:', torch.__version__)
        print('人工语料：', SENTENCES)
        print('词表：', TOKEN_TO_ID)
        x, y = make_batch()
        print('x[0]:', decode(x[0]))
        print('y[0]:', decode(y[0]))
        print('x/y 形状：', tuple(x.shape), tuple(y.shape))"""),
        md("""### 先只做一次 forward：看形状，不更新参数

        B=8，T=6，D=32，H=2；12 个词表 token。
        输入 `[8,6]` → 向量 `[8,6,32]` → logits `[8,6,12]`。
        注意权重是 `[8,2,6,6]`，最后两个维度是 query 位置和 key 位置。"""),
        code("""torch.manual_seed(42)
        initial_model = TinyTransformer()
        with torch.no_grad():
            embedded = (initial_model.token_embedding(x)
                        + initial_model.position_embedding(torch.arange(x.shape[1])))
            logits, head_weights = initial_model(x)
        print('向量形状：', tuple(embedded.shape))
        print('logits 形状：', tuple(logits.shape))
        print('Attention 形状：', tuple(head_weights.shape))
        print('第一句、第一头的权重：')
        print(head_weights[0, 0].round(decimals=3))"""),
        md("""### 训练：loss → backward → 更新参数

        每个位置预测右边的 token。交叉熵可理解为 `−log p(正确 token)`。
        原始 logits 传给 cross_entropy；不要先 softmax。
        `train_model` 会固定随机种子，在 CPU 上训练 200 步。"""),
        code("""model, report = train_model(steps=200, seed=42)
        print('参数量：', report['parameters'])
        print(f"loss: {report['initial_loss']:.4f} → {report['final_loss']:.4f}")
        print('训练前：', report['before_training'])
        print('训练后：', report['after_training'])
        print('本次训练耗时：', report['elapsed_seconds'], '秒')"""),
        md("""## 4. 逐步生成，再改 prompt

        生成取最后一个位置的 logits，softmax 后贪心选最高概率 token。
        选择结果追加到前文，循环到 `<eos>`；整个过程不更新参数。
        概率非常接近的两个候选，不同版本训练后可能交换排序。"""),
        code("""print('最终生成：', generate(model, prompt='我 喜欢', trace=True))"""),
        code("""prompt = '你 喜欢 喝'  # 实验 B：换为 '我 喜欢 吃'，先预测再运行
        print('最终生成：', generate(model, prompt=prompt, trace=True))"""),
        md("""### 检查一个关键性质：改变未来 token，不影响更早的预测

        保持前四个输入位置不变，只改后面位置，比较前四个位置的 logits。"""),
        code("""model.eval()
        altered = x.clone()
        altered[:, 4:] = TOKEN_TO_ID['茶']
        with torch.no_grad():
            original_logits, _ = model(x)
            changed_logits, _ = model(altered)
        difference = (original_logits[:, :4] - changed_logits[:, :4]).abs().max().item()
        print('前四个位置 logits 最大差异：', difference)
        torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4])"""),
        md("""## 离开课堂前，自己说一遍

        1. Attention 输出是什么？Q/K/V 分别影响什么？
        2. 为什么第一行因果权重是 `[1,0,0]`？
        3. 训练与生成的区别是什么？
        4. 画出 token 到下一 token 的流程，标出 logits 的形状。

        结果边界：loss 下降显示对八句人工语料拟合更好；没有验证集，不能判断泛化。
        多种合法后续使 loss 不必为 0。参考 [学生讲义](student_handout.md) 和 [原论文](https://arxiv.org/abs/1706.03762)。"""),
    ]
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata={"kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
                       "language_info":{"name":"python", "file_extension":".py"}}
    nbf.write(notebook,ROOT/"demo.ipynb")


def main():
    global FONT
    FONT=find_font()
    slides=json.loads((ROOT/"slides.json").read_text(encoding="utf-8"))
    report=json.loads((ROOT/"assets/sample_training_report.json").read_text(encoding="utf-8"))
    build_slides(slides,report)
    build_notebook()
    print(f"已生成 {len(slides)} 页 / 60 分钟的 PPT、文字版、预览与 Notebook 模板。")


if __name__ == "__main__":
    main()
