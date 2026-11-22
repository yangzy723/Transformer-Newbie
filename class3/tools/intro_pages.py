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
            node(c,x,2.98,3.60,.74,token+'的内容向量',fill=PALE if i==2 else 'F3F5F7',size=24,
                 color=BLUE if i==2 else INK)
            c.line(x+1.80,3.75,x+1.80,4.45,LINE)
        c.line(2.60,4.45,10.70,4.45,LINE)
        c.line(10.70,4.45,10.70,4.94,BLUE,arrow=True)
        node(c,8.90,4.97,3.60,.76,'喝的新向量',size=27,color=BLUE)
        c.text(.80,4.99,7.40,.72,'现在只计算“喝”的输出。\n另外两个位置提供信息，稍后也会各算自己的输出。',21,INK)
        c.text(.80,6.17,11.70,.79,'“我喜欢喝”和“我不能喝”，都有“喝”，但前文不同。\n读取上下文，让“喝”的新表示带上这次出现的信息。',23,INK)
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
            c.text(6.05,yy,.54,.46,'→',27,MUTED,center=True)
        table(c,7.10,2.93,5.40,contributions,cell_h=.67,size=25,formats=4)
        c.line(.80,5.19,12.50,5.19,LINE)
        for i,expr in enumerate((r'0.2483+0+0.5035\approx0.752',r'0+0.2483+0.5035\approx0.752')):
            yy=5.53+i*.61
            c.text(.80,yy,1.52,.39,f'第 {i} 列相加',20,MUTED)
            equation(c,expr,2.42,yy-.01,5.18,.43,size=25)
        c.line(7.88,5.99,8.47,5.99,BLUE,arrow=True)
        label(c,8.77,5.44,'属于“喝”的新向量',w=3.7,color=BLUE,size=20)
        c.text(8.77,6.04,3.72,.49,vec(old['context']),29,BLUE,bold=True)
    elif kind=='each_output':
        label(c,.80,2.25,'谁在读取',w=2.2,size=21)
        label(c,4.00,2.25,'读取哪些位置的内容',w=5.5,size=21)
        label(c,10.13,2.25,'输出属于谁',w=2.4,color=BLUE,size=21)
        for i,token in enumerate(tokens):
            yy=3.07+i*1.02
            node(c,.80,yy,2.10,.58,token,size=25)
            c.line(2.93,yy+.29,3.66,yy+.29,LINE,arrow=True)
            for j in range(i+1):
                node(c,4.00+j*1.65,yy,1.44,.58,tokens[j],fill='F3F5F7',size=23)
            c.line(9.17,yy+.29,9.84,yy+.29,LINE,arrow=True)
            node(c,10.13,yy,2.37,.58,token+'的新向量',size=19,color=BLUE)
        c.text(.80,6.49,11.70,.43,'输入 3 个位置，输出仍有 3 个位置；每个位置各有自己的读取比例。',23,INK)
    elif kind=='retrieval_roles':
        label(c,.80,2.25,'Q：这次想知道什么？',w=3.55,color=BLUE,size=21)
        label(c,5.00,2.25,'K：匹配用的索引',w=2.80,color=TEAL,size=20)
        label(c,8.12,2.25,'V：实际读取的正文',w=4.38,color=ORANGE,size=21)
        c.rect(.80,3.72,3.40,1.10,PALE,False)
        c.text(1.00,3.89,3.00,.82,'奶茶店\n几点打烊？',24,BLUE,bold=True,center=True)
        c.line(4.22,4.27,4.60,4.27,LINE)
        c.line(4.60,3.38,4.60,5.68,LINE)
        for i,(key,value) in enumerate(DOCS):
            yy=2.99+i*1.15
            c.line(4.60,yy+.39,4.98,yy+.39,LINE,arrow=True)
            node(c,5.00,yy,2.60,.78,key,fill='EAF3F2',size=23,color=TEAL)
            c.line(7.62,yy+.39,8.09,yy+.39,LINE,arrow=True)
            node(c,8.12,yy,4.38,.78,value,fill='FAF3EC',size=21,color=ORANGE)
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
        for i,token in enumerate(tokens):
            x=.80+i*4.05
            node(c,x,2.71,3.60,.64,token+'的输入向量',size=23)
            c.line(x+1.80,3.37,x+1.80,3.58,LINE)
            c.line(x+.25,3.58,x+1.80,3.58,LINE)
            c.line(x+.25,3.58,x+.25,6.00,LINE)
            for yy,text,color,fill in ((3.81,'Q：查询可见位置',BLUE,PALE),
                                       (4.76,'K：供查询时匹配',TEAL,'EAF3F2'),
                                       (5.71,'V：供读取时汇总',ORANGE,'FAF3EC')):
                c.line(x+.25,yy+.27,x+.73,yy+.27,LINE,arrow=True)
                node(c,x+.75,yy,2.85,.56,text,fill=fill,size=18,color=color)
        c.text(.80,6.59,11.70,.37,'每个位置都有 Q、K、V，既能发起读取，也能提供可匹配的特征与内容。',21,INK)
    elif kind=='qkv_one_reader':
        label(c,3.00,2.25,'各位置的 K',w=1.98,color=TEAL,size=19)
        label(c,5.60,2.25,'由 Q/K 算出比例',w=2.15,color=BLUE,size=19)
        label(c,8.15,2.25,'对应位置的 V',w=2.10,color=ORANGE,size=19)
        node(c,.80,4.15,1.60,.65,'喝的 Q',size=23,color=BLUE)
        c.line(2.42,4.475,2.70,4.475,LINE)
        c.line(2.70,3.375,2.70,5.575,LINE)
        for i,token in enumerate(tokens):
            yy=3.05+i*1.10
            c.line(2.70,yy+.325,2.98,yy+.325,LINE,arrow=True)
            node(c,3.00,yy,1.98,.65,token+'的 K',fill='EAF3F2',size=21,color=TEAL)
            c.line(5.00,yy+.325,5.57,yy+.325,LINE,arrow=True)
            node(c,5.60,yy,2.10,.65,'给'+token+'的比例',size=19,color=BLUE)
            c.text(7.76,yy+.13,.34,.37,'×',20,MUTED,center=True)
            node(c,8.15,yy,1.97,.65,token+'的 V',fill='FAF3EC',size=21,color=ORANGE)
            c.line(10.14,yy+.325,10.55,yy+.325,LINE)
        c.line(10.55,3.375,10.55,5.575,LINE)
        node(c,10.34,4.26,.43,.43,'+',size=22,color=BLUE)
        c.line(10.79,4.475,10.94,4.475,BLUE,arrow=True)
        c.rect(10.97,3.91,1.53,1.12,PALE,False)
        c.text(11.07,4.09,1.33,.78,'喝的\n新向量',22,BLUE,bold=True,center=True)
        c.text(.80,6.47,11.70,.45,'每个比例都乘到同位置的 V 上，再把三个贡献相加，得到“喝”的输出。',23,INK)
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
        for yy in (3.71,4.82):c.line(3.05,yy,3.05,yy+.28,BLUE,arrow=True)
        c.text(.80,6.49,11.70,.44,'沿每一行的 key 轴归一化：每个读取者都有自己的一组比例，权重和为 1。',22,INK)
    else:
        return False
    return True
