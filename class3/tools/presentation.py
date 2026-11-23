"""第三课版面：以计算对象为中心，保留原生文字、矩阵与连接线。"""
import hashlib
import re
from pathlib import Path
import sys

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'class2/tools'))
from slide_design import canvas_class, validate_layout
from pdf_export import save_vector_pdf
from formulas import place_formula
from detail_slides import arrow_pair

INK, MUTED, BLUE, TEAL = '172B42', '536273', '2458B8', '087E83'
BG, WHITE, PALE, LINE = 'FFFFFF', 'FFFFFF', 'EDF3FB', 'D8E0E8'
ORANGE, GREEN, DARK = 'A15D26', '087568', '172B42'
BaseCanvas = canvas_class(object)


class Canvas(BaseCanvas):
    def __init__(self, slide):
        super().__init__(slide)
        self.rect(0, 0, 13.333333, 7.5, WHITE, False)

    def line(self,x1,y1,x2,y2,color=LINE,arrow=False,width=None,head=None):
        """线宽用pt；箭头头部用独立原生三角形，长度按页面关系设置。"""
        import math
        from pptx.enum.shapes import MSO_CONNECTOR
        from pptx.util import Inches,Pt
        from pptx.dml.color import RGBColor
        width=width if width is not None else (2.5 if arrow else .9)
        if arrow and color==LINE:color='71839A'
        length=math.hypot(x2-x1,y2-y1)
        if length<1e-6:return
        ux,uy=(x2-x1)/length,(y2-y1)/length
        head=min(head if head is not None else .13,length*.40) if arrow else 0
        bx,by=x2-head*ux,y2-head*uy
        connector=self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(x1),Inches(y1),Inches(bx),Inches(by))
        connector.line.color.rgb=RGBColor.from_string(color);connector.line.width=Pt(width)
        connector.line._get_or_add_ln().set('cap','rnd')
        self.flat_shape(connector)
        self.draw.line(tuple(round(v*self.scale) for v in (x1,y1,bx,by)),
                       fill='#'+color,width=max(1,round(width*self.scale/72)))
        self.pdf_ops.append(('line',(x1,y1,bx,by,color,False,width)))
        if arrow:
            half=head*.46
            points=[(x2,y2),(bx-uy*half,by+ux*half),(bx+uy*half,by-ux*half)]
            builder=self.slide.shapes.build_freeform(round(points[0][0]*1000),
                round(points[0][1]*1000),scale=Inches(1)/1000)
            builder.add_line_segments([(round(x*1000),round(y*1000)) for x,y in points[1:]],close=True)
            shape=builder.convert_to_shape();shape.fill.solid()
            shape.fill.fore_color.rgb=RGBColor.from_string(color);shape.line.fill.background()
            self.flat_shape(shape);shape.name='Flow arrowhead'
            self.draw.polygon([(round(x*self.scale),round(y*self.scale)) for x,y in points],fill='#'+color)
            self.pdf_ops.append(('polygon',(points,color)))

    def flow(self,x1,y1,x2,y2,color=BLUE,width=3.2,head=.16):
        self.line(x1,y1,x2,y2,color,arrow=True,width=width,head=head)

    def path(self,points,color=BLUE,width=2.4,arrow=True,head=.13):
        for i,(a,b) in enumerate(zip(points,points[1:])):
            self.line(*a,*b,color,arrow=arrow and i==len(points)-2,width=width,head=head)

    def text(self, x, y, w, h, content, *args, **kwargs):
        content = str(content).translate({ord('ᵀ'): '^T', ord('ₖ'): '_k', ord('ᵥ'): '_v'})
        return super().text(x, y, w, h, content, *args, **kwargs)


def fmt(x, digits=3):
    if isinstance(x, str):
        return x
    if x is None:
        return '-∞'
    if abs(x) < 1e-10:
        return '0'
    if abs(x-round(x)) < 1e-9:
        return str(int(round(x)))
    return f'{x:.{digits}f}'


def vec(values, digits=3):
    return '['+', '.join(fmt(v, digits) for v in values)+']'


def label(c, x, y, text, w=6, color=MUTED, size=17):
    c.text(x, y, w, .35, text, size, color, bold=True)


def prose(c, x, y, w, texts, size=21, gap=.18, bottom=6.8, color=INK):
    font = ImageFont.truetype(c.font_path, round(size*c.scale/72))
    for text in texts:
        normalized = text.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉−ᵢ', '0123456789-i'))
        count = 0
        for original in normalized.split('\n'):
            current = ''; count += 1
            for char in original:
                if c.draw.textlength(current+char, font=font) > w*c.scale-6:
                    count += 1; current = char
                else:
                    current += char
        h = count*size/72*1.22+.018
        if y+h > bottom+.01:
            raise ValueError(f'段落越过内容区 {y+h:.2f}>{bottom}: {text}')
        c.text(x, y, w, h, text, size, color)
        y += h+gap
    return y


def equation(c, expr, x, y, w, h, size=29):
    expr = re.sub(r'\\mathbf\s+([A-Za-z])', r'\\mathbf{\1}', expr)
    name = hashlib.sha256((expr+str(size)).encode()).hexdigest()[:12]
    path = ROOT/f'assets/formulas/{name}.png'
    if not hasattr(c, 'formula_paths'):
        c.formula_paths = set()
    c.formula_paths.add(path)
    from matplotlib import rc_context
    with rc_context({'savefig.transparent': True}):
        place_formula(c, expr, path, x, y, w, h, font_size=size)


def code(c, lines, y=6.20, size=18, x=.78, w=11.80):
    """关键位置保留代码，不把代码栏作为每页装饰。"""
    c.line(x, y-.12, x+w, y-.12, LINE)
    for i, line in enumerate(lines):
        c.text(x, y+i*.34, w, .33, line, size, MUTED, mono=True)


def header(c, item, number, start, end, total):
    c.text(.75, .31, 10, .25, 'TRANSFORMER  /  第三课', 12, MUTED)
    c.text(.73, .83, 11.90, .70, item['title'], 32, INK, bold=True)
    c.text(.75, 1.57, 11.85, .40, item['subtitle'], 18.5, MUTED)
    c.line(.75, 7.11, 12.58, 7.11, LINE)
    c.text(.75, 7.22, 7, .21, 'Transformer Newbie', 10, MUTED)
    c.text(11.35, 7.20, 1.23, .24, f'{number:02d} / {total:02d}', 11, MUTED, center=True)


def reference(c, item):
    # 时间、Notebook编号与源码路径写入备注、教案和知识代码对照。
    pass


def table(c, x, y, w, rows, labels=None, columns=None, cell_h=.56, size=23,
          heat=False, highlight_row=None, formats=3, highlight_col=None):
    """开放式矩阵；仅为热力图或正在讨论的行列着色。"""
    label_w = .78 if labels else 0
    cols = len(rows[0]); cw = (w-label_w)/cols
    if columns:
        for j, text in enumerate(columns):
            c.text(x+label_w+j*cw, y, cw, .34, text, 17, MUTED, center=True)
        y += .48
    for i, row in enumerate(rows):
        yy = y+i*cell_h
        if labels:
            c.text(x, yy+.09, label_w-.04, .34, labels[i], 18, MUTED, center=True)
        for j, value in enumerate(row):
            fg = INK
            if heat:
                strength = max(0, min(1, float(value)))
                fill = ''.join(f'{round(a+(b-a)*strength):02X}' for a, b in zip((241,245,251),(36,88,184)))
                fg = WHITE if strength > .55 else INK
                c.rect(x+label_w+j*cw+.015, yy, cw-.03, cell_h-.025, fill, False)
            elif i == highlight_row or j == highlight_col:
                c.rect(x+label_w+j*cw, yy, cw, cell_h-.025, PALE, False)
                fg = BLUE
            c.text(x+label_w+j*cw+.04, yy+.075, cw-.08, .40,
                   fmt(value, formats), size, fg, center=True)
    if not heat:
        left=x+label_w+.015; right=x+w-.015; top=y+.02; bottom=y+len(rows)*cell_h-.055
        for xx, direction in ((left, 1), (right, -1)):
            c.line(xx, top, xx, bottom, 'AEBBCD',width=1.2)
            c.line(xx, top, xx+direction*.10, top, 'AEBBCD',width=1.2)
            c.line(xx, bottom, xx+direction*.10, bottom, 'AEBBCD',width=1.2)
    return y+len(rows)*cell_h


def bars(c, x, y, w, labels, values, color=BLUE, maximum=1, row_h=.61, digits=3):
    for i, (text, value) in enumerate(zip(labels, values)):
        yy=y+i*row_h
        c.text(x, yy, .88, .38, text, 22, INK)
        left=x+1.03; bw=w-2.2
        c.rect(left, yy+.12, bw, .17, 'EEF1F5', False)
        c.rect(left, yy+.12, bw*value/maximum, .17, color, False)
        c.text(left+bw+.18, yy-.015, 1, .40, f'{value:.{digits}f}', 23, color)


def node(c, x, y, w, h, text, fill=PALE, size=19, color=INK):
    c.rect(x, y, w, h, fill, False)
    c.text(x+.10, y+(h-size/72*1.22)/2, w-.20, h-.05,
           text, size, color, center=True)
