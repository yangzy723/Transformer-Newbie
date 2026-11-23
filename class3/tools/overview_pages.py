"""用适合整页阅读的主流程展示任务、整体模型与Block。"""
from presentation import *


def stage(c,x,y,w,title,detail,fill=PALE,color=INK):
    c.rect(x,y,w,.91,fill,False)
    c.text(x+.12,y+.13,w-.24,.39,title,23,color,bold=True,center=True)
    c.text(x+.12,y+.56,w-.24,.29,detail,16,MUTED,center=True)


def draw_overview(c,kind,r):
    if kind=='restart':
        old=r['review']['first_lesson']
        for x,w,title in ((.80,3.15,'已有输入'),(5.05,3.15,'模型处理后的表示'),(9.42,3.08,'下一 token 的概率')):
            label(c,x,2.30,title,w=w,size=20,color=BLUE)
        for x,w,word in ((.80,.73,'我'),(1.73,1.29,'喜欢'),(3.14,.80,'喝')):
            if word=='喝':c.rect(x,3.13,w,.79,PALE,False)
            c.text(x,3.25,w,.58,word,34,BLUE if word=='喝' else INK,bold=True,center=True)
        c.flow(4.11,3.58,4.84,3.58,width=3.6,head=.18)
        stage(c,5.05,3.12,3.15,'“喝”的最终向量','来自多层 Transformer',color=BLUE)
        c.flow(8.41,3.58,9.18,3.58,width=3.6,head=.18)
        for i,(word,value) in enumerate(zip(old['candidate_labels'],old['lm_probabilities'])):
            y=3.13+i*.66
            c.text(9.42,y,.86,.40,word,23,INK)
            c.rect(10.44,y+.13,1.00,.16,'EEF1F5',False)
            c.rect(10.44,y+.13,value,.16,BLUE,False)
            c.text(11.66,y-.01,.84,.43,f'{value:.3f}',22,BLUE)
        prose(c,.80,4.63,3.15,['把已经出现的内容，\n一起交给模型。'],22)
        prose(c,5.05,4.63,3.15,['预测时，读取最后位置的最终表示。'],22)
        prose(c,9.42,4.63,3.08,['经过输出层与 Softmax\n得到候选概率。'],21)
        c.text(9.42,5.86,3.08,.30,'第一课两候选算例',16,MUTED)
        c.line(.80,6.24,12.50,6.24,LINE)
        c.text(.80,6.49,11.70,.45,'最后一个位置，怎样读到前文的信息？',27,INK,bold=True)
    elif kind=='architecture':
        for x,title,detail in ((.80,'文本','我喜欢喝'),(4.85,'Tokenizer','文本 → ID 序列 [T]'),(8.90,'Embedding','按 ID 查出 [B,T,D]')):
            stage(c,x,2.91,3.15,title,detail,fill='F1F4F7')
        c.flow(4.12,3.37,4.65,3.37,width=3.0,head=.14)
        c.flow(8.16,3.37,8.69,3.37,width=3.0,head=.14)
        c.flow(10.475,4.02,10.475,4.91,width=3.2,head=.16)
        stage(c,8.90,5.12,3.15,'Blocks × N','逐层更新 [B,T,D]',color=BLUE)
        stage(c,4.85,5.12,3.15,'Norm → LM Head','最后位置 → 词表分数',fill='F1F4F7')
        stage(c,.80,5.12,3.15,'Softmax → 采样','概率 → 下一个 token ID',fill='F1F4F7')
        c.flow(8.69,5.58,8.16,5.58,width=3.0,head=.14)
        c.flow(4.65,5.58,4.12,5.58,width=3.0,head=.14)
        c.text(.80,6.57,11.70,.39,'按图从上排走到下排。采样出的 ID 接回输入，再进行下一次预测。',22,INK)
    elif kind=='block_structure':
        stage(c,.80,3.16,1.20,'x','输入',fill='F1F4F7')
        stage(c,2.80,3.16,2.20,'Norm','归一化',fill='F1F4F7')
        stage(c,5.80,3.16,3.80,'Self-Attention','Q/K + RoPE · 因果 Mask',color=BLUE)
        stage(c,10.40,3.16,2.10,'+ x','得到 h',fill='F1F4F7')
        for x1,x2 in ((2.17,2.60),(5.17,5.60),(9.77,10.20)):
            c.flow(x1,3.615,x2,3.615,width=2.8,head=.13)
        c.path([(1.40,3.01),(1.40,2.66),(11.45,2.66),(11.45,3.00)],color='7B8B9D',width=1.7,head=.11)
        c.flow(11.45,4.25,11.45,5.09,width=3.0,head=.15)
        stage(c,10.40,5.26,2.10,'Norm','归一化',fill='F1F4F7')
        stage(c,5.80,5.26,3.80,'FFN','分别处理每个位置',fill='F1F4F7')
        stage(c,2.80,5.26,2.20,'+ h','残差相加',fill='F1F4F7')
        stage(c,.80,5.26,1.20,'y','输出',fill=PALE,color=BLUE)
        for x1,x2 in ((10.20,9.77),(5.60,5.17),(2.60,2.17)):
            c.flow(x1,5.715,x2,5.715,width=2.8,head=.13)
        c.path([(11.45,4.55),(3.90,4.55),(3.90,5.09)],color='7B8B9D',width=1.7,head=.11)
        c.text(.80,2.19,11.70,.34,'上排：读取上下文          下排：逐位置变换          灰线：保留原有信息',18,MUTED)
        c.text(.80,6.65,11.70,.34,'x、h 和 y 都是 [B,T,D]：每个位置一直保留一个向量。',21,INK)
    else:
        return False
    return True
