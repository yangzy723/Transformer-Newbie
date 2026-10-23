"""第二课投影版：大公式、可读代码和解释机制的原生图形。"""
from PIL import ImageFont
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.util import Inches, Pt

from formulas import place_formula

INK, MUTED, BLUE, TEAL = "14263D", "475569", "2458C6", "087F8C"
BG, WHITE, PALE, LINE = "F5F7FA", "FFFFFF", "E8F0FC", "CBD5E1"
GREEN, RED, DARK = "087568", "B6453B", "14263D"

# 标题概括问题，正文展示定义、解释和计算过程。
DESIGN = {
    "opening": ("回顾：Transformer 的数据怎样流动？", "从一句话出发，沿着分词、向量、注意力和输出，走完一次预测。", "文本变 ID，向量穿过多层 Block，最后得到下一 token 的概率。"),
    "lesson_scope": ("从上节课的计算，走到今天的输入端", "把上节课的人工算例，接到真实模型的三个输入接口。", "本课聚焦：Tokenizer、Embedding、位置编码在整体架构中的作用。"),
    "bpe_train": ("BPE：先统计，再合并", "每轮重新统计当前相邻片段，选择最高频的一对。", "频次来自语料；合并后要重新统计。"),
    "rank": ("BPE 编码：谁的 rank 更小？", "规则已经冻结；每次在当前候选中选择优先级最高的一对。", "merge rank 决定合并顺序，token ID 决定查表地址。"),
    "hf_pipeline": ("把 BPE 接到真实文本上", "ByteLevel 处理字节与边界；BPE 处理合并规则。", "先训练并保存配置，再加载同一配置进行编码。"),
    "artifact": ("ID 必须与参数表配套", "同样大小的词表，也可能把同一片段映射到不同的地址。", "更换 tokenizer 时，必须核对 ID 与模型参数的对应关系。"),
    "onehot": ("Embedding：一个 ID 选择一行", "查表与 one-hot 乘法前向等价；实际实现直接选行。", "ID 是地址；可训练的表示存放在 E 的每一行。"),
    "lookup": ("不同长度，怎样组成 batch？", "先添加起止标记，再补齐；pad 只是占位。", "padding_idx、注意力 mask 和损失忽略，需要分别处理。"),
    "grad": ("读了同一行，梯度就要相加", "共享参数接收所有读取位置的贡献；这里先看 lookup 路径。", "反向是 scatter-add：累加贡献，而不是覆盖旧值。"),
    "chain": ("预测误差怎样传回向量？", "单位置交叉熵：先得到类别误差，再乘回输出权重。", "先算每次出现的向量梯度，再按 ID 汇总到参数表。"),
    "training": ("让 Embedding 真正学一次", "57 对相邻 token；Embedding → Linear → CrossEntropy。", "训练让“当前 token 的向量”逐渐有助于预测下一项。"),
    "tie": ("一张表，两条梯度路径", "权重绑定：输入查表和输出分类器共享同一个 E。", "绑定后，没作为输入的行也可能收到输出梯度。"),
    "permutation": ("没有位置，重排会怎样？", "假设：没有位置机制，也没有因果 mask。", "只按内容匹配时，输出跟着输入一起重排。"),
    "absolute": ("把位置向量加进输入", "内容与位置使用同样的维度，相加后形状保持不变。", "位置是 token 序号，不是原文的字符 offset。"),
    "frequencies": ("位置 × 频率，得到角度", "不同维度对使用不同频率，再交错存放 sin 与 cos。", "positions → angles → sin / cos：每一步都有明确形状。"),
    "rotation": ("RoPE：位置变成箭头的转角", "先看一对二维分量：每前进一个位置，箭头就多转一点。", "两个箭头一起转，夹角不变；位置差因此进入注意力分数。"),
    "vector_rope": ("从两个箭头，到整个 Q / K 张量", "每两个分量组成一支箭头，各对维度按不同速度旋转。", "RoPE 旋转 Q 和 K；位置通过它们的点积进入 Attention。"),
    "mask": ("位置关系与可读范围", "RoPE 改变分数；mask 决定哪些 key 可以被读取。", "有了位置机制，仍要屏蔽未来 key 与 padding key。"),
    "pipeline": ("沿真实输入，把形状接起来", "本地 tokenizer → Embedding → 多头 Q/K → RoPE + mask。", "把分词、向量、旋转与 mask 接起来，完成一次注意力计算。"),
    "quiz": ("用三个问题，检查是否理解", "解释一次编码、一次反传、一次位置计算。", "课后：改词表、查梯度、验证旋转，并保留实验结果。"),
}


def course_fonts():
    """Use the same real regular/bold family in Office, preview and PDF."""
    import os
    from pathlib import Path
    candidates = [
        ("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Deng.ttf",
         "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Dengb.ttf", "DengXian"),
        ("C:/Windows/Fonts/Deng.ttf", "C:/Windows/Fonts/Dengb.ttf", "DengXian"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "Noto Sans CJK SC"),
    ]
    if os.environ.get("CLASS2_FONT_REGULAR") and os.environ.get("CLASS2_FONT_BOLD"):
        candidates.insert(0, (os.environ["CLASS2_FONT_REGULAR"], os.environ["CLASS2_FONT_BOLD"],
                              os.environ.get("CLASS2_FONT_FAMILY", "DengXian")))
    for regular, bold, name in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            return regular, bold, name
    raise RuntimeError("请安装等线字体，或设置 CLASS2_FONT_REGULAR / CLASS2_FONT_BOLD / CLASS2_FONT_FAMILY。")


def canvas_class(base, font_path=None):
    from PIL import Image, ImageDraw
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.oxml.xmlchemy import OxmlElement
    import math
    regular, heavy, family = course_fonts()

    class ProjectionCanvas(base):
        scale = 240

        def __init__(self, slide):
            self.slide = slide
            self.image = Image.new("RGB", (3200, 1800), "#"+BG)
            self.draw = ImageDraw.Draw(self.image)
            self.pdf_ops = []
            self.text_metrics = []
            self.ink_boxes = []
            self.font_path = regular
            self.rect(0, 0, 13.333333, 7.5, BG, False)

        @staticmethod
        def flat_shape(shape):
            # Override theme effectRef: Office otherwise adds shadows to every connector.
            from pptx.oxml.ns import qn
            properties=shape._element.find(qn("p:spPr"))
            properties.append(OxmlElement("a:effectLst"))
            style=shape._element.find(qn("p:style"))
            if style is not None:
                effect=style.find(qn("a:effectRef"))
                if effect is not None:
                    effect.set("idx","0")

        def rect(self, x, y, w, h, color=PALE, rounded=True):
            shape = self.slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                Inches(x), Inches(y), Inches(w), Inches(h))
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(color)
            shape.line.fill.background()
            self.flat_shape(shape)
            if rounded:
                shape.adjustments[0] = .04
            bounds = tuple(round(v*self.scale) for v in (x,y,x+w,y+h))
            if rounded:
                self.draw.rounded_rectangle(bounds, radius=8, fill="#"+color)
            else:
                self.draw.rectangle(bounds, fill="#"+color)
            self.pdf_ops.append(("rect", (x,y,w,h,color,rounded)))

        def line(self, x1,y1,x2,y2,color=TEAL,arrow=False):
            shape = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            shape.line.color.rgb = RGBColor.from_string(color)
            shape.line.width = Pt(1.3)
            self.flat_shape(shape)
            if arrow:
                end=OxmlElement("a:tailEnd")
                end.set("type","triangle")
                end.set("w","sm"); end.set("len","sm")
                shape.line._get_or_add_ln().append(end)
            self.draw.line(tuple(v*self.scale for v in (x1,y1,x2,y2)), fill="#"+color,
                           width=round(1.3*self.scale/72))
            if arrow:
                angle=math.atan2(y2-y1,x2-x1)
                points=[(x2*self.scale,y2*self.scale)]
                for delta in (-.45,.45):
                    points.append(((x2-.075*math.cos(angle+delta))*self.scale,
                                   (y2-.075*math.sin(angle+delta))*self.scale))
                self.draw.polygon(points,fill="#"+color)
            self.pdf_ops.append(("line",(x1,y1,x2,y2,color,arrow)))

        def text(self, x, y, w, h, content, size=22, color=INK, bold=False,
                 center=False, mono=False):
            # Use ASCII subscripts in prose; mathematical notation is typeset separately.
            content=str(content).translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉−ᵢ", "0123456789-i"))
            from pathlib import Path
            mono_path="/System/Library/Fonts/Menlo.ttc"
            if not Path(mono_path).is_file():
                from matplotlib.font_manager import findfont
                mono_path=findfont("DejaVu Sans Mono")
            name = ("Menlo" if mono_path.endswith("Menlo.ttc") else "DejaVu Sans Mono") if mono else family
            path = mono_path if mono else heavy if bold else regular
            font = ImageFont.truetype(path, round(size*self.scale/72))
            lines = []
            for original in str(content).split("\n"):
                current = ""
                for char in original:
                    if self.draw.textlength(current+char, font=font) > w*self.scale-6:
                        if mono:
                            raise ValueError(f"代码必须显式换行：{original}")
                        lines.append(current)
                        current = char
                    else:
                        current += char
                lines.append(current)
            step = size/72*1.22
            if len(lines) == 1:
                h = max(h, step)
            if len(lines)*step > h+.012:
                raise ValueError(f"文本超出高度（{size} pt）：{content!r}")
            box = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            tf = box.text_frame
            tf.clear()
            tf.word_wrap = False
            tf.auto_size = MSO_AUTO_SIZE.NONE
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            for i, line in enumerate(lines):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = line
                p.font.name = name
                p.font._element.set("lang", "zh-CN")
                p.font.size = Pt(size)
                p.font.bold = bold
                p.font.color.rgb = RGBColor.from_string(color)
                # Explicit CJK typeface prevents Office theme fallback.
                for tag in ("a:ea", "a:cs"):
                    element=OxmlElement(tag); element.set("typeface",name)
                    p.font._element.append(element)
                p.line_spacing = Pt(size*1.22)
                p.space_before = p.space_after = Pt(0)
                if center:
                    p.alignment = PP_ALIGN.CENTER
                left = x+(w-self.draw.textlength(line,font=font)/self.scale)/2 if center else x
                # Shared baseline for vector PDF and bitmap preview; no simulated stroke.
                baseline=y+i*step+size/72*.90
                self.draw.text((round(left*self.scale),round(baseline*self.scale)),line,
                               font=font,fill="#"+color,anchor="ls")
                self.pdf_ops.append(("text",(left,baseline,line,size,color,path)))
                a,b,r,d=font.getbbox(line,anchor="ls")
                self.ink_boxes.append({"text":line,"bounds":[left+a/self.scale,baseline+b/self.scale,
                                      left+r/self.scale,baseline+d/self.scale]})
            self.text_metrics.append({"text": str(content), "size": size, "mono": mono,
                                      "lines": len(lines), "box": [x,y,w,h]})

    return ProjectionCanvas


def validate_layout(canvases):
    """Check visible glyph / formula bounds, rather than empty text-box space."""
    problems=[]
    for page,c in enumerate(canvases,1):
        objects=c.ink_boxes+[{"text":"公式图片","bounds":[a[1],a[2],a[1]+a[3],a[2]+a[4]]}
                             for kind,a in c.pdf_ops if kind=="image"]
        for index,item in enumerate(objects):
            l,t,r,b=item["bounds"]
            if l<-.01 or t<-.01 or r>13.343 or b>7.51:
                problems.append(f"第{page}页越界：{item['text']}")
            for other in objects[index+1:]:
                ll,tt,rr,bb=other["bounds"]
                if min(r,rr)-max(l,ll)>.025 and min(b,bb)-max(t,tt)>.018:
                    problems.append(f"第{page}页重叠：{item['text']} / {other['text']}")
    if problems:
        raise ValueError("\n".join(problems))
    return {"pages":len(canvases),"text_formula_overlap":0,"out_of_bounds":0}


def save_vector_pdf(canvases, destination):
    """Editable PPT geometry replayed to PDF with embedded vector text."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
    from matplotlib import rc_context
    import math
    with rc_context({"pdf.fonttype":42,"pdf.compression":6}), PdfPages(destination) as pdf:
        for canvas in canvases:
            fig=Figure(figsize=(13.333333,7.5),dpi=240)
            ax=fig.add_axes([0,0,1,1],xlim=(0,13.333333),ylim=(7.5,0))
            ax.axis("off")
            for kind,args in canvas.pdf_ops:
                if kind=="rect":
                    x,y,w,h,color,rounded=args
                    patch=(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.035",
                                          facecolor="#"+color,edgecolor="none") if rounded else
                           Rectangle((x,y),w,h,facecolor="#"+color,edgecolor="none"))
                    ax.add_patch(patch)
                elif kind=="line":
                    x1,y1,x2,y2,color,arrow=args
                    ax.plot([x1,x2],[y1,y2],color="#"+color,lw=1.3,solid_capstyle="butt")
                    if arrow:
                        angle=math.atan2(y2-y1,x2-x1)
                        points=[(x2,y2)]+[(x2-.075*math.cos(angle+d),y2-.075*math.sin(angle+d)) for d in (-.45,.45)]
                        ax.add_patch(Polygon(points,color="#"+color,lw=0))
                elif kind=="text":
                    x,y,line,size,color,path=args
                    ax.text(x,y,line,fontsize=size,fontproperties=FontProperties(fname=path),
                            color="#"+color,va="baseline",parse_math=False)
                elif kind=="image":
                    source,x,y,w,h=args
                    ax.imshow(source,extent=(x,x+w,y+h,y),aspect="auto",interpolation="lanczos")
            # Preserve drawing order; Matplotlib otherwise sorts images, patches and text.
            for order,artist in enumerate(ax.get_children()):
                artist.set_zorder(order)
            pdf.savefig(fig)
            fig.clear()

def token(c, x, y, text, w=.8, fill=PALE, color=BLUE, size=24):
    c.rect(x,y,w,.58,fill)
    c.text(x+.06,y+.07,w-.12,.48,text,size,color,center=True)


def header(c, item, number, start, end, total):
    kind = item["kind"]
    title, subtitle, takeaway = DESIGN[kind]
    active = 0 if number <= 6 else 1 if number <= 12 else 2
    for i, text in enumerate(("01  分词", "02  向量", "03  位置")):
        x = .60+i*1.52
        c.rect(x,.27,1.34,.34,BLUE if i == active else "E6ECF4")
        c.text(x,.31,1.34,.26,text,14,WHITE if i == active else MUTED,center=True)
    c.text(10.0,.30,2.7,.28,f"{start:02d}–{end:02d} min  /  60",14,MUTED,center=True)
    c.text(.60,.84,12.10,.66,title,34,INK,bold=True)
    c.text(.62,1.54,12.0,.39,subtitle,20,MUTED)
    c.rect(.60,6.70,12.10,.47,PALE)
    c.text(.79,6.78,11.7,.36,takeaway,19,BLUE,bold=True)
    c.text(.62,7.27,8.8,.21,"TRANSFORMER NEWBIE  ·  CLASS 2",11,MUTED)
    c.text(11.5,7.25,1.2,.24,f"{number:02d} / {total:02d}",12,MUTED,center=True)


def reference(c, link):
    if not link or link.get("question")=="为什么按绝对位置旋转，却产生相对位移？":
        return
    experiment = f"实验 {link['experiment']:02d}" if link["experiment"] else "课后扩展"
    file = link["file"].replace("../tools/", "tools/")
    text=f"{experiment}   ·   {file} / {link['function']}"
    if link.get("worked_function"):
        text=f"{experiment}   ·   {file}   ·   算例 worked_examples.py/{link['worked_function']}"
    c.text(.65,6.38,12,.24,f"{experiment}   ·   {file}",13,MUTED)
