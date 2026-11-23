"""把读取者、逐项求和与Q/K/V分工分开讲清楚。"""
import math
from presentation import *

DOCS = [('营业时间','10:00 开门，22:00 打烊'),
        ('饮品价格','奶茶 18 元，咖啡 22 元'),
        ('店铺位置','学府路 12 号，地铁 A 口')]


def draw_intro(c, kind, r):
    old=r['review']['first_lesson']; tokens=r['base']['tokens']
    if kind=='reading_focus':
        label(c,.80,2.22,'提供信息：我、喜欢，以及“喝”自己',w=11.7,color=BLUE,size=21)
        for i,token in enumerate(tokens):
            x=.80+i*4.05
            node(c,x,2.93,3.60,.74,token+'的内容向量',fill=PALE if i==2 else 'F3F5F7',size=24,
                 color=BLUE if i==2 else INK)
            target=5.70+i*.95
            c.flow(x+1.80,3.85,target,4.83,width=2.9,head=.15)
        node(c,4.85,5.00,3.60,.76,'喝的新向量',size=27,color=BLUE)
        c.text(.80,4.99,3.55,.76,'现在先只更新“喝”。\n其他位置稍后各算一次。',21,INK)
        c.text(.80,6.17,11.70,.79,'下一步：先给三个来源分配读取比例，\n再把它们提供的向量加权相加。',23,INK)
    elif kind=='mix_prepare':
        label(c,.80,2.25,'位置',w=1.4,size=21)
        label(c,3.03,2.25,'提供的内容向量 V',w=3.1,size=21)
        label(c,6.62,2.25,'“喝”分给它的读取比例',w=5.8,color=BLUE,size=21)
        for i,(token,value,weight) in enumerate(zip(tokens,old['values'],old['weights'])):
            y=3.14+i*.83
            c.text(.80,y,1.50,.46,token,28,INK)
            c.text(3.03,y,3.10,.46,vec(value),28,TEAL)
            c.rect(6.62,y+.18,3.99,.16,'EEF1F5',False)
            c.rect(6.62,y+.18,3.99*weight,.16,BLUE,False)
            c.text(10.92,y,1.60,.46,f'{weight:.4f}',27,BLUE)
        c.text(.80,6.24,11.70,.48,'一个权重，对应一个位置；它会乘到那个位置向量的所有分量上。',24,INK)
    elif kind=='mix_steps':
        label(c,.80,2.25,'位置',w=1.2,size=20)
        label(c,2.18,2.25,'读取比例 × 原向量',w=4.4,size=21)
        c.text(7.10,2.25,2.70,.40,'第 0 列的贡献',21,BLUE,bold=True,center=True)
        c.text(9.80,2.25,2.70,.40,'第 1 列的贡献',21,BLUE,bold=True,center=True)
        contributions=[[a*x for x in value] for a,value in zip(old['weights'],old['values'])]
        for i,(token,weight,value) in enumerate(zip(tokens,old['weights'],old['values'])):
            yy=3.02+i*.67
            c.text(.80,yy,1.20,.45,token,25,INK)
            equation(c,rf'{weight:.4f}\,[{fmt(value[0])},{fmt(value[1])}]',2.18,yy-.01,3.56,.46,size=27)
            c.flow(6.00,yy+.23,6.65,yy+.23,width=2.7,head=.13)
        table(c,7.10,2.93,5.40,contributions,cell_h=.67,size=25,formats=4)
        c.line(.80,5.19,12.50,5.19,LINE)
        for i,expr in enumerate((r'0.2483+0+0.5035\approx0.752',r'0+0.2483+0.5035\approx0.752')):
            yy=5.53+i*.61
            c.text(.80,yy,1.52,.39,f'第 {i} 列相加',20,MUTED)
            equation(c,expr,2.42,yy-.01,5.18,.43,size=25)
        c.flow(7.88,5.99,8.47,5.99,width=3.1,head=.14)
        label(c,8.77,5.44,'属于“喝”的新向量',w=3.7,color=BLUE,size=20)
        c.text(8.77,6.04,3.72,.49,vec(old['context']),29,BLUE,bold=True)
    elif kind=='each_output':
        label(c,.80,2.25,'谁在读取',w=2.2,size=21)
        label(c,4.00,2.25,'这次可读取的内容',w=4.6,size=21)
        label(c,9.80,2.25,'输出属于谁',w=2.7,color=BLUE,size=21)
        for i,token in enumerate(tokens):
            yy=3.07+i*1.02
            node(c,.80,yy,2.10,.65,token,size=25)
            c.flow(3.10,yy+.325,3.80,yy+.325,width=2.7,head=.13)
            c.rect(4.00,yy,4.60,.65,'F1F4F7',False)
            span=(i+1)*1.38
            for j in range(i+1):
                left=4.00+(4.60-span)/2+j*1.38
                c.text(left,yy+.12,1.38,.41,tokens[j],23,INK,center=True)
            c.flow(8.80,yy+.325,9.60,yy+.325,width=2.7,head=.13)
            node(c,9.80,yy,2.70,.65,token+'的新向量',size=21,color=BLUE)
        c.text(.80,6.49,11.70,.43,'输入 3 个位置，输出仍有 3 个位置；每个位置各有自己的读取比例。',23,INK)
    elif kind=='retrieval_roles':
        label(c,.80,2.25,'Q：这次想知道什么？',w=3.2,color=BLUE,size=21)
        label(c,5.10,2.25,'K：匹配索引',w=2.4,color=TEAL,size=21)
        label(c,8.45,2.25,'V：实际读取的正文',w=4.05,color=ORANGE,size=21)
        c.rect(.80,3.72,2.60,1.10,PALE,False)
        c.text(1.00,3.89,2.20,.82,'奶茶店\n几点打烊？',24,BLUE,bold=True,center=True)
        c.line(3.57,4.27,4.15,4.27,BLUE,width=2.5)
        c.line(4.15,3.38,4.15,5.68,BLUE,width=2.5)
        for i,(key,value) in enumerate(DOCS):
            yy=2.99+i*1.15
            c.flow(4.15,yy+.39,4.92,yy+.39,color=BLUE,width=2.5,head=.13)
            node(c,5.10,yy,2.40,.78,key,fill='EAF3F2',size=23,color=TEAL)
            c.flow(7.67,yy+.39,8.27,yy+.39,color=ORANGE,width=2.5,head=.13)
            node(c,8.45,yy,4.05,.78,value,fill='FAF3EC',size=21,color=ORANGE)
        c.text(.80,6.47,11.70,.45,'用问题匹配索引，判断相关程度；真正带回信息的是对应正文。',23,INK)
    elif kind=='retrieval_shift':
        label(c,.80,2.25,'读取比例示意：两次检索，资料不变',w=11.7,size=19)
        label(c,4.40,3.03,'Q：几点打烊？',w=3.65,color=BLUE,size=23)
        label(c,8.58,3.03,'Q：奶茶多少钱？',w=3.92,color=TEAL,size=23)
        for i,(key,value) in enumerate(DOCS):
            yy=3.94+i*.69
            c.text(.80,yy,3.05,.43,key,24,INK)
            for x,weights,color in ((4.40,[.8,.1,.1],BLUE),(8.58,[.1,.8,.1],TEAL)):
                weight=weights[i]
                c.rect(x,yy+.15,2.52,.17,'EEF1F5',False)
                c.rect(x,yy+.15,2.52*weight,.17,color,False)
                c.text(x+2.76,yy-.01,1.12,.43,f'{weight:.0%}',24,color)
        c.text(.80,6.45,11.70,.46,'Q 换了，匹配比例会变；每条资料的 K 和 V 仍保持配对，低权重内容也参与读取。',22,INK)
    elif kind=='token_roles':
        for x,w,title,color in ((.80,2.8,'位置 / 输入',INK),(4.00,2.5,'Q · 用来查询',BLUE),
                (7.00,2.5,'K · 用来匹配',TEAL),(10.00,2.5,'V · 提供内容',ORANGE)):
            c.text(x,2.35,w,.42,title,21,color,bold=True,center=x>3)
        for i,token in enumerate(tokens):
            yy=3.15+i*1.08
            if i==2:c.rect(.80,yy-.04,11.70,.79,PALE,False)
            c.text(.80,yy+.10,1.30,.43,token,26,INK)
            equation(c,rf'\mathbf x_{{{i}}}',2.22,yy+.08,1.10,.47,size=29)
            for x,tag in ((4.0,'q'),(7.0,'k'),(10.0,'v')):
                equation(c,rf'\mathbf {tag}_{{{i}}}',x,yy+.08,2.5,.47,size=29)
        c.text(.80,6.59,11.70,.37,'每个位置都有三份表示；更新哪个位置，就使用对应的 Q 发起读取。',22,INK)
    elif kind=='qkv_one_reader':
        label(c,.80,2.25,'先算读取比例',w=11.7,color=BLUE,size=21)
        node(c,.80,3.01,2.40,.92,'喝的 Q',size=28,color=BLUE)
        c.flow(3.40,3.47,4.05,3.47,width=3.1,head=.15)
        c.rect(4.25,3.01,3.60,.92,'EAF3F2',False)
        c.text(4.40,3.15,3.30,.40,'逐个比较三个 K',24,TEAL,bold=True,center=True)
        c.text(4.40,3.62,3.30,.27,'我 / 喜欢 / 喝',17,MUTED,center=True)
        c.flow(8.05,3.47,8.90,3.47,width=3.1,head=.15)
        node(c,9.10,3.01,3.40,.92,'三个读取比例',size=26,color=BLUE)
        c.path([(10.80,4.10),(10.80,4.36),(7.20,4.36),(7.20,4.98)],width=2.5,head=.14)
        label(c,.80,4.62,'再按比例汇总内容',w=4.3,color=ORANGE,size=21)
        c.rect(.80,5.17,3.60,.95,'FAF3EC',False)
        c.text(.95,5.31,3.30,.40,'三个位置的 V',24,ORANGE,bold=True,center=True)
        c.text(.95,5.77,3.30,.27,'我 / 喜欢 / 喝',17,MUTED,center=True)
        c.flow(4.60,5.645,5.30,5.645,color=ORANGE,width=3.1,head=.15)
        c.rect(5.50,5.17,3.40,.95,PALE,False)
        c.text(5.65,5.27,3.10,.39,'按对应比例汇总',22,BLUE,bold=True,center=True)
        equation(c,r'a_0v_0+a_1v_1+a_2v_2',5.65,5.72,3.10,.32,size=22)
        c.flow(9.10,5.645,9.90,5.645,width=3.1,head=.15)
        node(c,10.10,5.17,2.40,.95,'喝的新向量',size=24,color=BLUE)
        c.text(.80,6.59,11.70,.38,'Q/K 负责产生比例；V 按这些比例进入加权求和，得到当前读取者的输出。',22,INK)
    elif kind=='softmax':
        b=r['base']; scores=b['scores'][1]
        exp=[math.exp(scores[0]),math.exp(scores[1]),0.0]
        label(c,.80,2.25,'例：“喜欢”这一行',w=6,color=BLUE,size=22)
        equation(c,r'a_j=\frac{e^{s_j}}{\sum_{\ell}e^{s_{\ell}}}',8.12,2.22,4.38,.93,size=32)
        for yy,vector,detail,color in (
            (3.10,'≈ [1.414, 1.414, −∞]','Mask 后的分数：依次对应我、喜欢、喝',INK),
            (4.21,'≈ [4.113, 4.113, 0]','取指数：未来位置的指数为 0',INK),
            (5.32,'[0.5, 0.5, 0]',f'除以这一行的指数之和 ≈ {sum(exp):.3f}',BLUE)):
            c.text(.80,yy,4.95,.49,vector,29,color,bold=color==BLUE)
            c.text(6.30,yy+.04,6.20,.41,detail,21,MUTED)
        for yy in (3.64,4.75):c.flow(3.05,yy,3.05,yy+.50,width=3.0,head=.14)
        c.text(.80,6.49,11.70,.44,'沿每一行的 key 轴归一化：每个读取者都有自己的一组比例，权重和为 1。',22,INK)
    else:
        return False
    return True
