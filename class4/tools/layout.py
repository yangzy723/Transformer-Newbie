"""复用第三课原生绘图与矢量 PDF，新增第四课的统一版面组件。"""
import hashlib
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
COURSE_ROOT=Path(os.environ.get('TRANSFORMER_COURSE_ROOT',str(ROOT.parent)))
_original_path=list(sys.path)
sys.path.insert(0,str(COURSE_ROOT/'class3/tools'))
from presentation import Canvas, fmt, vec, table, save_vector_pdf, validate_layout
from formulas import place_formula
from PIL import ImageFont
sys.path[:]=_original_path

INK='172B42'; MUTED='536273'; BLUE='2458B8'; TEAL='087E83'
PALE='EDF3FB'; LINE='DDE4EB'; WHITE='FFFFFF'; GREEN='087568'
ORANGE='A15D26'; RED='B6453B'; SAND='FFF2E5'; MINT='E8F4EF'


def text(c,x,y,w,h,s,size=21,color=INK,bold=False,center=False,mono=False):
    c.text(x,y,w,h,str(s),size,color,bold=bold,center=center,mono=mono)


def para(c,x,y,w,s,size=20,color=INK,bold=False):
    """度量段落后布置文本，超出正文区则立即失败。"""
    font=ImageFont.truetype(c.font_path,round(size*c.scale/72))
    count=0
    for line in str(s).split('\n'):
        count+=1; cur=''
        for ch in line:
            if c.draw.textlength(cur+ch,font=font)>w*c.scale-6:
                count+=1;cur=ch
            else:cur+=ch
    h=count*size/72*1.22+.035
    if y+h>6.73:raise ValueError(f'正文越界 {y+h:.2f}: {s}')
    text(c,x,y,w,h,s,size,color,bold)
    return y+h


def label(c,x,y,w,s,color=BLUE,size=18):
    text(c,x,y,w,.37,s,size,color,True)


def note(c,x,y,w,s,size=18,color=MUTED):
    return para(c,x,y,w,s,size,color)


def card(c,x,y,w,h,title,body='',fill=PALE,color=BLUE,size=21,body_size=20):
    c.rect(x,y,w,h,fill,False)
    end=para(c,x+.20,y+.16,w-.40,title,size,color,True)
    if body:para(c,x+.20,end+.15,w-.40,body,body_size,INK)


def eq(c,s,x=.8,y=2.2,w=11.7,h=.70,size=29):
    if not hasattr(c,'formula_expressions'):c.formula_expressions=[]
    c.formula_expressions.append(s)
    if not hasattr(c,'formula_specs'):c.formula_specs=[]
    c.formula_specs.append((s,size))
    name=hashlib.sha256((s+str(size)).encode()).hexdigest()[:12]
    path=ROOT/'assets/formulas'/f'{name}.png'
    place_formula(c,s,path,x,y,w,h,font_size=size)
    return path


def node(c,x,y,w,h,title,detail='',fill=PALE,color=BLUE,size=22):
    c.rect(x,y,w,h,fill,False)
    if detail:
        text(c,x+.10,y+.13,w-.20,.44,title,size,color,True,True)
        text(c,x+.10,y+h-.42,w-.20,.32,detail,16,MUTED,center=True)
    else:text(c,x+.10,y+(h-.40)/2,w-.20,.44,title,size,color,True,True)


def arrow(c,x1,y1,x2,y2,color=BLUE):
    c.flow(x1,y1,x2,y2,color,width=2.3,head=.12)


def rowtable(c,x,y,w,headers,rows,widths=None,size=19,row_h=.60):
    widths=widths or [1/len(headers)]*len(headers)
    xs=[x]
    for frac in widths:xs.append(xs[-1]+w*frac)
    c.rect(x,y,w,.48,PALE,False)
    for j,v in enumerate(headers):text(c,xs[j]+.10,y+.075,w*widths[j]-.20,.36,v,17,BLUE,True)
    yy=y+.53
    for i,row in enumerate(rows):
        if i%2==0:c.rect(x,yy-.02,w,row_h,'F7F9FB',False)
        for j,v in enumerate(row):text(c,xs[j]+.10,yy+.07,w*widths[j]-.20,row_h-.08,v,size,INK)
        yy+=row_h
    return yy


def bars(c,x,y,w,labels,values,color=BLUE,row_h=.60,maximum=1,digits=4):
    for i,(word,p) in enumerate(zip(labels,values)):
        yy=y+i*row_h
        text(c,x,yy,1.00,.38,word,20,INK)
        c.rect(x+1.10,yy+.15,w-2.30,.15,'E7ECF2',False)
        if p>0:c.rect(x+1.10,yy+.15,(w-2.30)*p/maximum,.15,color,False)
        text(c,x+w-1.03,yy-.01,1.03,.42,f'{p:.{digits}f}',20,color,True)


def curve(c,x,y,w,h,functions,xlim=(-3,3),ylim=(-1,3),legend=True):
    """原生连接线绘制曲线，坐标与采样点可编辑。"""
    left,right=xlim;bottom,top=ylim
    xp=lambda v:x+(v-left)/(right-left)*w
    yp=lambda v:y+h-(v-bottom)/(top-bottom)*h
    for v in range(math.ceil(left),math.floor(right)+1):
        c.line(xp(v),y,xp(v),y+h,'EDF0F4',width=.6)
        text(c,xp(v)-.18,y+h+.08,.36,.28,str(v),14,MUTED,center=True)
    for v in range(math.ceil(bottom),math.floor(top)+1):
        c.line(x,yp(v),x+w,yp(v),'EDF0F4',width=.6)
        text(c,x-.55,yp(v)-.12,.42,.25,str(v),14,MUTED,center=True)
    c.line(x,yp(0),x+w,yp(0),'8292A6',width=1.1)
    c.line(xp(0),y,xp(0),y+h,'8292A6',width=1.1)
    for j,(name,fn,color) in enumerate(functions):
        points=[]
        for k in range(121):
            a=left+(right-left)*k/120;b=fn(a)
            if bottom<=b<=top:points.append((xp(a),yp(b)))
        for a,b in zip(points,points[1:]):c.line(*a,*b,color,width=2.4)
        if legend:
            lx=x+j*(w/len(functions))
            c.line(lx,y-.25,lx+.25,y-.25,color,width=2.7)
            text(c,lx+.35,y-.40,w/len(functions)-.35,.34,name,16,color,True)


def code(c,lines,x=.95,y=2.3,w=11.40,size=19,line_h=.36):
    c.rect(x-.17,y-.13,w+.34,len(lines)*line_h+.28,'F3F6F9',False)
    for i,line in enumerate(lines):text(c,x,y+i*line_h,w,line_h+.01,line,size,INK,mono=True)


def header(c,item,total):
    text(c,.75,.24,8.7,.26,'TRANSFORMER  /  第四课',11.5,MUTED)
    text(c,10.20,.24,2.38,.26,item['section'],11.5,TEAL,True)
    text(c,.73,.69,11.90,.62,item['title'],32,INK,True)
    text(c,.75,1.39,11.85,.45,item['subtitle'],18.5,MUTED)
    c.line(.75,6.89,12.58,6.89,LINE,width=.75)
    text(c,.75,7.01,10.95,.32,item['takeaway'],16.5,BLUE)
    text(c,11.86,7.03,.72,.25,f"{item['page']:02d}/{total:02d}",11,MUTED,center=True)
    c.line(.75,7.40,12.58,7.40,'EDF1F4',width=1.3)
    c.line(.75,7.40,.75+11.83*item['page']/total,7.40,TEAL,width=1.3)
