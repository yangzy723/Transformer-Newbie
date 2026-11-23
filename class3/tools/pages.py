"""第三课：按每页的计算关系安排版面与连接。"""
from presentation import *
from intro_pages import draw_intro
from overview_pages import draw_overview
from role_pages import draw_roles


def draw(c, item, r):
    kind=item['kind']; b=r['base']; review=r['review']
    if draw_overview(c,kind,r) or draw_roles(c,kind,r) or draw_intro(c,kind,r):
        return
    if kind=='two_softmax':
        old=review['first_lesson']
        for x, name, objects, scores, values, color in (
            (.80, 'Attention 内部', b['tokens'], old['scores'], old['weights'], BLUE),
            (7.05, 'LM Head 之后', old['candidate_labels'], old['lm_logits'], old['lm_probabilities'], TEAL)):
            label(c, x, 2.25, name, w=5.3, color=color, size=22)
            c.text(x, 2.97, 5.15, .38, '分数 ≈ '+vec(scores), 22, INK)
            bars(c, x, 3.65, 5.15, objects, values, color=color, row_h=.65)
            c.text(x, 5.96, 5.15, .78,
                   '在输入位置间分配权重，\n随后加权汇总 V。' if x<1 else '在词表候选间分配概率，\n随后选择下一个 token。', 22, color)
        c.line(6.40, 2.26, 6.40, 6.68, LINE)
    elif kind=='bpe_train':
        d=review['bpe']
        label(c, .80, 2.23, '语料及其出现次数')
        for i,(text,n) in enumerate(d['corpus'].items()):
            c.text(.80, 2.96+i*.53, 5.23, .43, f'{text}  × {n}', 25, INK)
        c.text(.80, 4.84, 5.15, .42, '“喜 + 欢”：5 + 3 + 2 = 10', 23, BLUE, bold=True)
        c.line(6.15, 2.25, 6.15, 5.36, LINE)
        label(c, 6.65, 2.23, '一次合并，改变了相邻关系', w=5.9)
        c.text(6.65, 3.00, 5.90, .43, '我 / 喜 / 欢 / 喝 / 奶 / 茶', 23, MUTED)
        c.flow(9.52, 3.49, 9.52, 4.03,width=3.0,head=.14)
        c.text(6.65, 4.23, 5.90, .43, '我 / 喜欢 / 喝 / 奶 / 茶', 23, BLUE, bold=True)
        c.text(6.65, 4.93, 5.90, .78, '旧的“欢 + 喝”消失；\n新的“喜欢 + 喝”出现，下一轮要重新数。', 21, INK)
        code(c, ['counts[pair] += corpus[text]', 'pair = max(counts, key=counts.get)'], y=6.08)
        c.text(7.05, 6.08, 5.44, .75, '上行：累计当前相邻对的出现次数\n下行：选择累计次数最多的一对', 17, MUTED)
    elif kind=='bpe_encode':
        d=review['bpe']
        label(c, .80, 2.23, '已学好的四条合并规则', w=4.3)
        for i,pair in enumerate(d['merges']):
            c.text(.80, 3.03+i*.64, 4.40, .43, f'{i}   '+ ' + '.join(pair), 23, INK)
        c.line(5.50, 2.25, 5.50, 6.70, LINE)
        label(c, 5.99, 2.23, '每次选当前可用的最小 rank', w=6.5)
        for i,row in enumerate(d['encode_trace']):
            prefix='起点' if row['rank'] is None else 'rank '+str(row['rank'])
            yy=3.01+i*.57
            c.text(5.99, yy+.02, 1.03, .34, prefix, 16, MUTED)
            c.text(7.18, yy, 5.38, .43, ' / '.join(row['tokens']), 23,
                   BLUE if i==4 else INK, bold=i==4)
        c.text(.80, 6.29, 4.35, .42, 'rank 是规则的执行顺序。', 20, MUTED)
        c.text(5.99, 6.29, 6.56, .42, '最终只有两个 token：你 / 喜欢喝奶茶', 22, BLUE, bold=True)
    elif kind=='embedding':
        e=review['embedding']; real=review['real_tokenizer']
        label(c, .80, 2.23, '查表：选哪一行', w=5.2, color=BLUE, size=22)
        c.text(.80, 3.00, 5.10, .42, '“我喜欢喝牛奶。”', 25, INK)
        c.text(.80, 3.61, 5.10, .43, 'ID = '+str(real['ids']), 25, BLUE, bold=True)
        equation(c, r'\mathbf X=\mathbf E[[300,304,262]]', .81, 4.38, 5.10, .55, size=27)
        c.text(.80, 5.40, 5.15, .81, '词表大小 320，向量宽度 D：\nE 是 [320,D]，查出的 X 是 [3,D]。', 21, INK)
        c.line(6.27, 2.25, 6.27, 6.71, LINE)
        label(c, 6.77, 2.23, '训练：这一行的数怎样改变', w=5.8, color=TEAL, size=22)
        c.text(6.77, 3.00, 5.72, .41, '预测 → loss → 梯度 → 参数更新', 23, INK)
        label(c, 6.77, 3.78, '第二课二维小表：第 1 行，学习率 0.1', w=5.8, size=17)
        c.text(6.77, 4.30, 5.70, .42, '[0.2, -0.1] − 0.1 × [1, 2]', 25, INK)
        c.text(6.77, 4.96, 5.70, .42, '= [0.1, -0.3]', 28, TEAL, bold=True)
        c.text(6.77, 5.76, 5.72, .79, 'backward 计算梯度；step 更新参数。\n同一 ID 出现多次，梯度累加到同一行。', 20, INK)
    elif kind=='positions':
        label(c, .80, 2.23, '相加式：位置进入输入向量', w=5.2, color=BLUE, size=21)
        equation(c, r'\mathbf x_p=\mathbf E[u_p]+\mathbf P_p', .80, 3.03, 5.20, .64, size=32)
        prose(c, .80, 4.12, 5.10, [
            '同一个 token 查到同一内容向量；换了位置，加上的位置向量不同。',
            'P 可以是一张可学习的表，也可以由不同频率的 sin/cos 生成。'], 22, gap=.24)
        c.line(6.25, 2.25, 6.25, 6.71, LINE)
        label(c, 6.75, 2.23, 'RoPE：位置进入 Q/K 的夹角', w=5.85, color=TEAL, size=21)
        # 同样的内容箭头，每前进一格转30度；对比共同平移。
        arrow_pair(c, 8.03, 4.25, (1,3))
        arrow_pair(c, 11.17, 4.25, (3,5))
        c.text(6.82, 5.28, 2.45, .38, '位置 1、3', 21, INK, center=True)
        c.text(9.96, 5.28, 2.45, .38, '位置 3、5', 21, INK, center=True)
        c.text(6.75, 2.91, 5.85, .34, '初始都向右，每格转 30°；蓝 Q、绿 K。', 18, MUTED, center=True)
        c.text(6.75, 6.03, 5.85, .75, '两个箭头一起多转 60°，夹角仍是 60°。\n点积反映相对位置；V 不旋转。', 21, TEAL)
    elif kind=='projections':
        c.text(.80, 2.23, 11.70, .44, 'X 的三行：我 [1,0]    喜欢 [0,1]    喝 [1,1]', 25, INK, bold=True)
        c.text(.80, 2.83, 11.70, .34, '固定输入 X，三张 2×2 的 W 在所有位置间共享；每行向量两维。', 18, MUTED)
        for i, (tag,key,color) in enumerate((('Q','q',BLUE),('K','k',TEAL),('V','v',ORANGE))):
            x=.80+i*4.06
            equation(c,rf'\mathbf W_{{{tag}}}',x,3.35,3.45,.40,size=23)
            table(c,x,3.91,3.45,b['w_'+key],cell_h=.46,size=23)
            equation(c,rf'\mathbf {tag}=\mathbf X\mathbf W_{{{tag}}}',x,4.97,3.45,.39,size=24)
            table(c,x,5.44,3.45,b[key],labels=b['tokens'],cell_h=.40,size=21,highlight_row=2)
        for xx in (4.45,8.51): c.line(xx,3.42,xx,6.75,LINE)
    elif kind=='one_query':
        c.text(.80, 2.27, 11.70, .48, '“喝”的查询向量 q₂ = [1, 2]', 29, BLUE, bold=True)
        label(c,.80,3.13,'被匹配的位置',w=2.6)
        label(c,3.27,3.13,'该位置的 K',w=3.0)
        label(c,7.12,3.13,'与 q₂ 的点积',w=5)
        for i,(token,key,score) in enumerate(zip(b['tokens'],b['k'],b['raw_scores'][2])):
            yy=3.81+i*.68
            c.text(.80,yy,1.80,.43,token,25,INK)
            c.text(3.27,yy,2.60,.43,vec(key),25,TEAL)
            c.text(7.12,yy,5.12,.43,f'1×{fmt(key[0])} + 2×{fmt(key[1])} = {fmt(score)}',25,BLUE,bold=True)
        c.text(.80,6.17,11.70,.44,'一行原始分数：[3, 2, 5]。其他位置的 Q 也这样计算。',23,INK)
        code(c,['raw_row = q[-1] @ k.T'],y=6.70,size=17)
    elif kind=='score_matrix':
        for x,w,title in ((.80,2.50,'Q  [3,2]'),(4.00,3.50,'K.T  [2,3]'),(8.25,4.25,'原始分数  [3,3]')):
            label(c,x,2.24,title,w=w,color=BLUE,size=21)
        table(c,.80,3.08,2.50,b['q'],labels=b['tokens'],cell_h=.65,size=24,highlight_row=2)
        kt=list(map(list,zip(*b['k'])))
        table(c,4.00,3.08,3.50,kt,columns=b['tokens'],cell_h=.65,size=24)
        c.text(3.46,3.94,.45,.51,'×',29,MUTED,center=True)
        c.text(7.63,3.94,.42,.51,'=',29,MUTED,center=True)
        table(c,8.25,3.08,4.25,b['raw_scores'],labels=b['tokens'],columns=b['tokens'],cell_h=.65,size=24,highlight_row=2)
        c.text(.80,5.87,11.70,.45,'行：谁在查询。列：查询哪个位置。',24,INK,bold=True)
        c.text(.80,6.48,11.70,.37,'例如第 1 行、第 2 列：“喜欢”查询“喝”，[0,2] · [1,2] = 4。',21,MUTED)
    elif kind=='scale':
        equation(c,r'\mathbf S=\frac{\mathbf Q\mathbf K^{\mathsf{T}}}{\sqrt{d_k}}',.80,2.35,5.05,.96,size=36)
        prose(c,.80,3.88,5.05,[
            '点积累加各维度的乘积。\n分量近似独立、尺度相近时，\n维度越高，典型波动越大。',
            '分数差大，Softmax 会更集中。\n按维度的平方根缩放，\n可以缓和这种集中。'],22,gap=.22)
        label(c,6.57,2.24,'“喝”这一行：读取权重的变化',w=6,color=BLUE,size=21)
        c.text(6.57,3.05,5.90,.38,'原始分数 [3,2,5]，Q/K 都是两维',21,INK)
        for i,token in enumerate(b['tokens']):
            yy=3.81+i*.78
            c.text(6.57,yy+.02,.80,.40,token,22,INK)
            for dy,values,color in ((0,r['scale']['unscaled_weights'][2],'9AA8B6'),(.27,b['weights'][2],BLUE)):
                value=values[i]
                c.rect(7.62,yy+dy+.08,3.48*value,.13,color,False)
                c.text(11.32,yy+dy-.035,1.14,.33,f'{value:.3f}',18,color)
        c.text(6.57,6.54,5.90,.31,'灰色：不缩放     蓝色：除以 √2',18,MUTED)
    elif kind=='mask':
        label(c,.80,2.23,'缩放后的分数：未来设为 −∞',w=5.4,color=BLUE,size=21)
        c.text(.80,2.68,5.08,.30,'“喜欢”的原始分数：[2,2,4]',18,MUTED)
        rows=[['-∞' if b['forbidden'][i][j] else b['scores'][i][j] for j in range(3)] for i in range(3)]
        table(c,.80,3.06,5.08,rows,labels=b['tokens'],columns=b['tokens'],cell_h=.65,size=24,highlight_row=1)
        c.flow(6.12,4.22,7.10,4.22,width=3.2,head=.16)
        c.text(5.95,4.66,1.48,.35,'Softmax',17,MUTED,center=True)
        label(c,7.60,2.23,'可见位置间的读取权重 A',w=4.9,color=BLUE,size=21)
        table(c,7.60,3.06,4.91,b['weights'],labels=b['tokens'],columns=b['tokens'],cell_h=.65,size=23,heat=True)
        c.text(.80,5.95,11.70,.45,'“喜欢”用“我、喜欢”预测“喝”：保留自身，屏蔽未来。',24,INK)
        c.text(.80,6.57,11.70,.35,'exp(−∞)=0；把未来分数设为 0 则仍有 exp(0)=1。Mask 要在 Softmax 之前加。',20,MUTED)
    elif kind=='mix':
        label(c,.80,2.23,'A：读取比例',w=4.1,color=BLUE,size=21)
        label(c,5.63,2.23,'V：被读取的内容',w=2.75,color=TEAL,size=20)
        label(c,9.09,2.23,'Z：新的向量',w=3.41,color=BLUE,size=21)
        table(c,.80,3.13,4.14,b['weights'],labels=b['tokens'],columns=b['tokens'],cell_h=.63,size=22,heat=True)
        table(c,5.63,3.61,2.75,b['v'],cell_h=.63,size=23)
        table(c,9.09,3.61,3.41,b['output'],cell_h=.63,size=23,highlight_row=1)
        c.text(5.04,4.15,.43,.48,'×',29,MUTED,center=True)
        c.text(8.48,4.15,.42,.48,'=',29,MUTED,center=True)
        equation(c,r'\mathbf z_1=0.5[1,-1]+0.5[2,0]=[1.5,-0.5]',.80,5.96,11.70,.51,size=29)
        c.text(.80,6.66,11.70,.35,'[3,3] @ [3,2] → [3,2]。每个位置仍有一个向量，现在包含它读取的上下文。',20,MUTED)
    elif kind=='change_v':
        d=r['interventions']['value']
        c.text(.80,2.24,11.70,.45,'固定 Q/K，把 V[0] 从 [1,−1] 改成 [3,−1]：增加 [2,0]。',24,INK)
        label(c,.80,3.18,'权重 A 保持不变',w=5,color=BLUE,size=21)
        table(c,.80,3.88,4.75,b['weights'],labels=b['tokens'],columns=b['tokens'],cell_h=.58,size=23,heat=True)
        label(c,6.26,3.18,'各行输出的变化 = A 的第 0 列 × [2,0]',w=6.2,color=BLUE,size=20)
        for i,token in enumerate(b['tokens']):
            yy=3.97+i*.60
            c.text(6.26,yy,.78,.42,token,22,INK)
            c.text(7.33,yy,5.10,.42,f'{format(b["weights"][i][0],".4f").rstrip("0").rstrip(".")} × [2,0] '+('≈ ' if i==2 else '= ')+vec(d['delta_output'][i]),22,BLUE)
        c.text(.80,6.26,11.70,.46,'匹配分数没有变；每个位置读到多少变化，取决于它给“我”的权重。',23,INK)
    elif kind=='change_k':
        d=r['interventions']['key']
        c.text(.80,2.24,11.70,.45,'固定 Q/V，把 K[0] 从 [1,1] 改成 [2,1]。',25,INK)
        label(c,.80,3.15,'只看“喝”这一行',w=5,color=BLUE,size=21)
        for y,name,before,after in (
            (3.90,'原始分数','[3, 2, 5]','[4, 2, 5]'),
            (4.67,'读取权重',vec(d['weights_before'][2]),vec(d['weights_after'][2])),
            (5.44,'输出 Z[2]',vec(d['output_before'][2]),vec(d['output_after'][2]))):
            c.text(.80,y,2.14,.43,name,22,MUTED)
            c.text(3.10,y,4.00,.43,before,26,INK)
            c.flow(7.27,y+.23,7.84,y+.23,width=2.6,head=.12)
            c.text(8.05,y,4.43,.43,after,26,BLUE,bold=True)
        c.text(.80,6.41,11.70,.39,'前两行恰好不变：第一行只能读自身；第二行 Q=[0,2] 与 K 的改变量 [1,0] 点积为 0。',20,MUTED)
    elif kind=='causality':
        d=r['interventions']['causality']
        c.text(.80,2.24,11.70,.45,'把最后位置 X[2] 从 [1,1] 改成 [4,−1]，用相同参数重新计算 Q/K/V。',23,INK)
        for x,title,before,after,color in (
            (.80,'使用因果 Mask',d['masked_before'],d['masked_after'],BLUE),
            (7.01,'去掉 Mask',d['unmasked_before'],d['unmasked_after'],ORANGE)):
            label(c,x,3.09,title,w=5.5,color=color,size=23)
            c.text(x+.84,3.82,2.20,.33,'修改前的输出',17,MUTED,center=True)
            c.text(x+3.22,3.82,2.25,.33,'修改后的输出',17,color,center=True)
            for i,token in enumerate(b['tokens']):
                yy=4.41+i*.57
                c.text(x,yy,.76,.39,token,20,MUTED)
                c.text(x+.84,yy,2.20,.39,vec(before[i]),19,INK,center=True)
                c.text(x+3.22,yy,2.25,.39,vec(after[i]),19,color,center=True)
        c.line(6.39,3.10,6.39,5.98,LINE)
        c.text(.80,6.40,11.70,.43,'因果 Mask 让前两行读不到改动后的 K/V，因此它们的输出不变。',22,INK)
    elif kind=='core_code':
        lines=[
            ('q, k, v = x @ w_q, x @ w_k, x @ w_v','三种投影 [3,2]'),
            ('scores = q @ k.T / math.sqrt(q.shape[-1])','分数 [3,3]'),
            ('future = torch.ones(3, 3, dtype=torch.bool).triu(1)','右上三角'),
            ('masked = scores.masked_fill(future, -torch.inf)','屏蔽未来'),
            ('weights = masked.softmax(dim=-1)','沿 key 轴归一化'),
            ('output = weights @ v','新向量 [3,2]')]
        c.rect(.79,2.40,11.73,3.74,'F5F7FA',False)
        for i,(line,note) in enumerate(lines):
            yy=2.71+i*.52
            c.text(.98,yy,.34,.35,str(i+1),16,MUTED,mono=True)
            c.text(1.47,yy,8.64,.36,line,18,INK,mono=True)
            c.text(10.43,yy,1.89,.36,note,16,BLUE)
        equation(c,r'\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}+M\right)V',.80,6.42,11.70,.50,size=28)
    elif kind=='multihead':
        label(c,.80,2.24,'结构示例：B=1，T=3，D=4；两个头，每头 2 维',w=11.7)
        node(c,.80,3.99,2.05,.71,'X [1,3,4]',size=21,color=BLUE)
        c.line(2.87,4.35,3.40,4.35,BLUE,width=2.5)
        c.line(3.40,3.34,3.40,5.34,BLUE,width=2.5)
        for i,yy in enumerate((2.95,4.95)):
            c.flow(3.40,yy+.39,3.88,yy+.39,color=BLUE if i==0 else TEAL,width=2.5,head=.12)
            node(c,4.01,yy,4.03,.79,f'Q/K/V 投影 {i+1} → Attention',fill=PALE if i==0 else 'EAF3F2',size=20,color=BLUE if i==0 else TEAL)
            c.text(4.01,yy+.96,4.03,.33,'各自输出 [1,3,2]',18,MUTED,center=True)
            c.line(8.06,yy+.39,8.57,yy+.39,BLUE if i==0 else TEAL,width=2.5)
        c.line(8.57,3.34,8.57,5.34,BLUE,width=2.5)
        c.flow(8.57,4.35,8.97,4.35,width=2.7,head=.12)
        node(c,9.15,3.93,3.35,.83,'拼接 → W_O',size=24,color=BLUE)
        c.text(9.15,5.02,3.35,.36,'输出 [1,3,4]',20,INK,center=True)
        c.text(.80,6.44,11.70,.41,'不同的投影产生各头的读取权重；拼接后，再用 W_O 合并信息。',22,INK)
    elif kind=='quiz':
        label(c,.80,2.23,'同一张权重矩阵 A',w=5.4,color=BLUE,size=22)
        table(c,.80,3.15,5.02,b['weights'],labels=b['tokens'],columns=b['tokens'],cell_h=.69,size=25,heat=True)
        c.text(.80,6.00,5.07,.82,'V[0]=[1,−1]，V[1]=[2,0]\nV[2]=[3,−1]',21,INK)
        c.line(6.30,2.25,6.30,6.80,LINE)
        prose(c,6.80,2.99,5.70,[
            '“喜欢”这一行读了谁？为什么没有读“喝”？',
            '用这一行权重和左侧 V，算出“喜欢”的新向量。',
            '只改 V[0]，A 会变吗？如果改的是 K[0] 呢？'],24,gap=.34)
    else:
        raise ValueError(kind)
