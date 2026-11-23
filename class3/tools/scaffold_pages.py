"""用可追踪的数字和过渡例子，连接Block、Embedding与上下文读取。"""
from presentation import *


def draw_scaffold(c,kind,r):
    if kind=='block_numbers':
        d=r['review']['residual']
        for y,title,left,update,right,left_label,update_label,right_label in (
            (2.20,'① Attention 读上下文，再加回原输入',d['input'],d['attention_update'],d['after_attention'],
             '“喝”的输入向量','Attention 更新量','“喝”的中间表示'),
            (4.31,'② FFN 处理这个位置，再加回中间表示',d['after_attention'],d['ffn_update'],d['output'],
             '保留中间表示','FFN 更新量','“喝”的输出向量')):
            label(c,.80,y,title,w=11.7,color=BLUE,size=22)
            for x,label_text,value,color,fill in ((.80,left_label,left,INK,'F1F4F7'),
                    (4.89,update_label,update,ORANGE,'FAF3EC'),(9.00,right_label,right,TEAL,'EAF3F2')):
                c.text(x,y+.57,3.50,.35,label_text,20,color,bold=True,center=True)
                c.rect(x,y+1.05,3.50,.73,fill,False)
                equation(c,vec(value,1),x+.10,y+1.12,3.30,.57,size=34)
            c.text(4.40,y+1.14,.36,.44,'+',29,MUTED,center=True)
            c.text(8.51,y+1.14,.36,.44,'=',29,MUTED,center=True)
        c.line(.80,4.11,12.50,4.11,LINE)
        c.text(.80,6.62,11.70,.36,'两个分量分别相加，宽度 D=2 保持；其他位置也各有一行，整句仍是 [1,3,2]。',20,INK)
    elif kind=='embedding':
        real=r['review']['real_tokenizer']
        label(c,.80,2.23,'① 文本变成行号',w=5.1,color=BLUE,size=22)
        c.text(.80,3.22,4.75,.47,'“'+real['text']+'”',28,INK)
        c.text(.80,4.03,4.75,.36,'Tokenizer 输出 3 个 ID',21,MUTED)
        c.text(.80,4.64,4.75,.49,str(real['ids']),30,BLUE,bold=True)
        c.flow(5.48,4.24,6.49,4.24,width=3.2,head=.16)
        label(c,6.90,2.23,'② 分别取出对应的参数行',w=5.6,color=TEAL,size=22)
        for i,id_ in enumerate(real['ids']):
            yy=3.10+i*.91
            c.text(6.90,yy+.12,.94,.42,str(id_),25,MUTED,center=True)
            c.flow(8.03,yy+.35,8.58,yy+.35,color=TEAL,width=2.6,head=.12)
            c.rect(8.78,yy,3.72,.70,'EAF3F2',False)
            equation(c,rf'\mathbf x_{{{i}}}=\mathbf E[{id_}]',8.90,yy+.11,3.48,.49,size=27)
        c.line(.80,6.08,12.50,6.08,LINE)
        c.text(.80,6.43,11.70,.43,'E 是 [320,D]；查 3 行得到 X [3,D]，加入本例的 batch 维后是 [1,3,D]。',23,INK)
    elif kind=='embedding_update':
        e=r['review']['embedding'];before=e['weight'][1];grad=e['gradient'][1];after=e['after'][1]
        for x,title,value,fill,color in ((.80,'表里原来的参数',vec(before,1),'F1F4F7',INK),
                (3.81,'反向传播得到的梯度',vec(grad),'FAF3EC',ORANGE),
                (6.82,'学习率',fmt(e['learning_rate'],1),'F1F4F7',INK),
                (9.83,'更新后的参数',vec(after,1),'EAF3F2',TEAL)):
            c.text(x,2.30,2.67,.36,title,19,color,bold=True,center=True)
            c.rect(x,2.94,2.67,.79,fill,False)
            equation(c,value,x+.08,3.06,2.51,.54,size=29)
        c.text(.80,4.16,11.70,.46,'新参数 = 旧参数 − 学习率 × 梯度',27,INK,bold=True,center=True)
        label(c,.80,5.03,'两个分量，分别计算',w=5.6,color=BLUE,size=21)
        equation(c,r'0.2-0.1\times1=0.1',.80,5.60,5.30,.45,size=28)
        equation(c,r'-0.1-0.1\times2=-0.3',.80,6.20,5.30,.45,size=28)
        c.line(6.38,4.99,6.38,6.72,LINE)
        prose(c,6.89,5.00,5.61,[
            '梯度表示损失对参数的变化率。\n本例梯度为正，SGD 把这两个数调小。',
            'backward 算梯度；step 才更新参数。'],20,gap=.19)
    elif kind=='context_bridge':
        for x,w,title in ((.80,5.40,'课堂分词：前文不同'),(6.75,3.20,'“喝”的内容向量'),(10.34,2.16,'“喝”的位置')):
            c.text(x,2.28,w,.37,title,21,BLUE if x<1 else MUTED,bold=True,center=x>1)
        for i,middle in enumerate(('喜欢','不能')):
            yy=3.17+i*1.08
            c.rect(.80,yy,5.40,.77,'F1F4F7',False)
            for x,w,word in ((1.03,1.03,'我'),(2.47,1.74,middle),(4.62,1.18,'喝')):
                c.text(x,yy+.16,w,.46,word,28,BLUE if word=='喝' else INK,bold=word=='喝',center=True)
            c.text(6.75,yy+.19,3.20,.43,'同一行 E[ID]',23,TEAL,center=True)
            c.text(10.34,yy+.18,2.16,.43,'2',28,INK,center=True)
        c.text(.80,5.57,11.70,.46,'起点可以相同，但“喜欢喝”和“不能喝”需要结合不同的前文。',24,INK)
        c.rect(.80,6.21,11.70,.63,PALE,False)
        c.text(1.00,6.33,11.30,.41,'接下来：让“喝”读取这句话的内容，形成结合上下文的新表示。',23,BLUE,bold=True)
    else:
        return False
    return True
