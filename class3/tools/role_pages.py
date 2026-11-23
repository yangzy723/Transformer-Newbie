"""把Query、Key、Value的职责分别展开，再解释独立投影。"""
from presentation import *


def draw_roles(c,kind,r):
    if kind=='query_focus':
        label(c,.80,2.24,'先选定当前读取者：“喝”',w=7.8,color=BLUE,size=22)
        label(c,9.35,2.24,'Q = Query · 查询',w=3.15,color=BLUE,size=22)
        for i,word in enumerate(('我','喜欢','喝')):
            x=.80+i*2.75
            node(c,x,3.03,2.10,.65,word+'的输入',fill=PALE if i==2 else 'F1F4F7',size=22,
                 color=BLUE if i==2 else INK)
        c.flow(7.35,3.87,7.35,4.43,width=3.5,head=.16)
        node(c,6.00,4.61,2.70,.66,'喝的 Q',size=27,color=BLUE)
        c.path([(7.35,5.42),(7.35,5.66),(1.85,5.66)],width=2.5,arrow=False)
        for i,word in enumerate(('我','喜欢','喝')):
            x=.80+i*2.75
            c.flow(x+1.05,5.66,x+1.05,5.94,width=2.5,head=.11)
            node(c,x,6.11,2.10,.60,word+'的 K',fill='EAF3F2',size=22,color=TEAL)
        c.line(8.98,2.25,8.98,6.76,LINE)
        prose(c,9.35,3.11,3.15,[
            'Q 由当前读取者的\n输入生成。',
            '同一个“喝”的 Q，\n逐个比较三个 K。',
            '换成“喜欢”读取，就使用“喜欢”的 Q。'],21,gap=.24)
    elif kind=='key_focus':
        label(c,.80,2.24,'当前查询',w=2.60,color=BLUE,size=22)
        label(c,4.72,2.24,'K = Key · 匹配线索',w=3.0,color=TEAL,size=21)
        label(c,9.32,2.24,'每个来源的匹配分数',w=3.18,color=TEAL,size=21)
        node(c,.80,4.15,2.60,.77,'喝的 Q',size=28,color=BLUE)
        c.path([(3.57,4.535),(4.10,4.535)],color=BLUE,width=2.6,arrow=False)
        c.path([(4.10,3.375),(4.10,5.775)],color=TEAL,width=2.4,arrow=False)
        for i,word in enumerate(('我','喜欢','喝')):
            yy=3.00+i*1.20
            c.flow(4.10,yy+.375,4.60,yy+.375,color=TEAL,width=2.4,head=.12)
            node(c,4.78,yy,2.72,.75,word+'的 K',fill='EAF3F2',size=25,color=TEAL)
            c.flow(7.70,yy+.375,9.10,yy+.375,color=TEAL,width=2.8,head=.15)
            node(c,9.32,yy,3.18,.75,word+'：一个分数',fill='F1F4F7',size=22)
        c.text(7.94,2.83,1.05,.31,'做点积',17,TEAL,center=True)
        c.text(.80,6.58,11.70,.38,'K 提供匹配线索；它与当前 Q 一起决定分数，分数再转换成读取比例。',22,INK)
    elif kind=='value_focus':
        label(c,.80,2.24,'同一个 Q：“几点打烊？”    同一个 K：“营业时间”',w=11.7,color=BLUE,size=21)
        for x,title,body in ((.80,'修改前','22:00 打烊'),(7.30,'修改后','23:00 打烊')):
            label(c,x,3.11,title,w=5.2,size=20)
            node(c,x,3.71,5.20,.67,'K · 营业时间',fill='EAF3F2',size=24,color=TEAL)
            c.rect(x,4.73,5.20,1.03,'FAF3EC',False)
            c.text(x+.20,4.99,4.80,.53,'V · '+body,30,ORANGE,bold=True,center=True)
        c.flow(6.19,5.245,7.07,5.245,color=ORANGE,width=3.4,head=.17)
        c.text(.80,6.39,11.70,.75,'Q/K 没变，读取比例保持；V 的内容变了，实际带回的信息也跟着变。',24,INK)
    elif kind=='projection_roles':
        label(c,.80,2.24,'三种角色，分别学习怎样表示',w=11.7,color=BLUE,size=22)
        node(c,.80,4.12,2.45,.85,'同一输入 x',size=26)
        c.path([(3.43,4.545),(3.90,4.545)],width=2.8,arrow=False)
        c.path([(3.90,3.215),(3.90,5.875)],width=2.5,arrow=False)
        for i,(tag,title,color,fill) in enumerate((('Q','查询方式',BLUE,PALE),
            ('K','参与匹配的方式',TEAL,'EAF3F2'),('V','传递的内容',ORANGE,'FAF3EC'))):
            yy=2.84+i*1.33
            c.flow(3.90,yy+.375,4.43,yy+.375,color=color,width=2.4,head=.12)
            c.rect(4.61,yy,2.26,.75,fill,False)
            equation(c,rf'\times\,\mathbf W_{{{tag}}}',4.76,yy+.10,1.96,.49,size=29)
            c.flow(7.05,yy+.375,7.95,yy+.375,color=color,width=2.8,head=.14)
            node(c,8.14,yy,1.07,.75,tag,fill=fill,size=30,color=color)
            c.text(9.64,yy+.14,2.86,.46,title,23,color,bold=True)
        c.text(.80,6.57,11.70,.40,'三张 W 分别在所有位置共享；各位置由自己的输入生成 Q、K、V。',22,INK)
    else:
        return False
    return True
