"""第三课的版面组件；复用第二课已校准的字体、矢量PDF和Office样式。"""
import hashlib
import re
import json
from pathlib import Path
import sys

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'class2/tools'))
from slide_design import canvas_class, save_vector_pdf, validate_layout
from formulas import place_formula
from detail_slides import architecture_review, arrow_pair

INK, MUTED, BLUE, TEAL = '14263D', '475569', '2458C6', '087F8C'
BG, WHITE, PALE, LINE = 'F5F7FA', 'FFFFFF', 'E8F0FC', 'CBD5E1'
ORANGE, GREEN, DARK = 'A95316', '087568', '14263D'
BaseCanvas = canvas_class(object)


class Canvas(BaseCanvas):
    def text(self,x,y,w,h,content,*args,**kwargs):
        content=str(content).translate({ord('ᵀ'):'^T',ord('ₖ'):'_k',ord('ᵥ'):'_v'})
        return super().text(x,y,w,h,content,*args,**kwargs)



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
    return '['+', '.join(fmt(v,digits) for v in values)+']'


def panel(c, x,y,w,h,title,color=BLUE):
    c.rect(x,y,w,h,WHITE)
    c.text(x+.20,y+.15,w-.40,.34,title,18,color,bold=True)


def prose(c, x,y,w,texts,size=19,gap=.12,bottom=None):
    font=ImageFont.truetype(c.font_path,round(size*c.scale/72))
    for text in texts:
        normalized=text.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉−ᵢ','0123456789-i'))
        current=''; lines=1
        for char in normalized:
            if c.draw.textlength(current+char,font=font)>w*c.scale-6:
                lines+=1; current=char
            else:
                current+=char
        h=lines*size/72*1.22+.015
        if bottom is not None and y+h>bottom+.01:
            raise ValueError(f'段落越过内容区 {y+h:.2f}>{bottom}: {text}')
        c.text(x,y,w,h,text,size,INK)
        y+=h+gap
    return y


def equation(c,expr,x,y,w,h,size=29):
    expr=re.sub(r'\\mathbf\s+([A-Za-z])',r'\\mathbf{\1}',expr)
    name=hashlib.sha256((expr+str(size)).encode()).hexdigest()[:12]
    place_formula(c,expr,ROOT/f'assets/formulas/{name}.png',x,y,w,h,font_size=size)


def code(c,lines,y=5.53,height=.73,size=18):
    c.rect(.60,y,12.10,height,DARK)
    c.text(.80,y+.15,.53,.29,'PY',14,'8BCFD2',bold=True)
    step=.30 if len(lines)<=2 else .32
    for i,line in enumerate(lines):
        c.text(1.35,y+.07+i*step,11.03,.32,line,size,WHITE,mono=True)


def header(c,item,number,start,end,total):
    active=0 if start<25 else 1 if start<47 else 2
    for i,label in enumerate(('01  复习','02  Attention','03  实验与衔接')):
        x=.60+i*1.90
        c.rect(x,.27,1.72,.34,BLUE if i==active else 'E6ECF4')
        c.text(x,.31,1.72,.25,label,13,WHITE if i==active else MUTED,center=True)
    c.text(10.05,.30,2.64,.29,f'{start:02d}–{end:02d} min / 60',14,MUTED,center=True)
    c.text(.60,.83,12.08,.64,item['title'],32,INK,bold=True)
    c.text(.62,1.51,12.02,.40,item['subtitle'],19,MUTED)
    c.rect(.60,6.76,12.10,.45,PALE)
    c.text(.80,6.83,11.68,.34,item['takeaway'],18,BLUE,bold=True)
    c.text(.62,7.29,8.7,.20,'TRANSFORMER NEWBIE  ·  CLASS 3',10,MUTED)
    c.text(11.54,7.26,1.16,.24,f'{number:02d} / {total:02d}',12,MUTED,center=True)


def reference(c,item):
    if item['kind']=='architecture':
        return
    source=item.get('source','')
    lab=item.get('experiment')
    label=f"实验 {lab:02d} · {source}" if lab else source
    if label:
        c.text(.66,6.43,12,.25,label,13,MUTED)


def table(c,x,y,w,rows,labels=None,columns=None,cell_h=.45,size=21,
          heat=False,highlight_row=None,formats=3):
    """原生可编辑矩阵；行标签是query，列标签是key。"""
    import math
    label_w=.75 if labels else 0
    cols=len(rows[0]); cw=(w-label_w)/cols
    if columns:
        for j,label in enumerate(columns):
            c.text(x+label_w+j*cw,y,cw,.33,label,17,MUTED,center=True)
        y+=.43
    for i,row in enumerate(rows):
        yy=y+i*cell_h
        if labels:
            c.text(x,yy+.09,label_w-.05,.32,labels[i],18,MUTED,center=True)
        for j,value in enumerate(row):
            fill=PALE if i==highlight_row else 'F0F4F9'
            fg=BLUE if i==highlight_row else INK
            if heat and isinstance(value,(int,float)):
                strength=max(0,min(1,float(value)))
                fill=''.join(f'{round(a+(b-a)*strength):02X}' for a,b in zip((238,243,251),(36,88,198)))
                fg=WHITE if strength>.55 else INK
            c.rect(x+label_w+j*cw+.03,yy,cw-.06,cell_h-.055,fill)
            c.text(x+label_w+j*cw+.055,yy+.075,cw-.11,.36,fmt(value,formats),size,fg,center=True)
    return y+len(rows)*cell_h


def formula_strip(c,expr):
    c.rect(.60,2.08,12.10,.70,WHITE)
    equation(c,expr,.85,2.16,11.60,.48,size=29)


def two_panels(c,ltitle,rtitle,y=2.97,h=2.35):
    panel(c,.60,y,5.85,h,ltitle)
    panel(c,6.65,y,6.05,h,rtitle)
