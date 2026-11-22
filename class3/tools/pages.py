"""21页课堂内容：实际矩阵、可编辑图表、计算展开与控制变量实验。"""
from presentation import *


def draw(c,item,r):
    kind=item['kind']; b=r['base']; review=r['review']
    if kind=='architecture':
        architecture_review(c)
    elif kind=='restart':
        panel(c,.60,2.08,5.85,3.90,'先回到最熟悉的补词任务')
        c.text(.88,2.75,5.20,.65,'我 喜欢 喝 ____',32,BLUE,bold=True)
        prose(c,.88,3.64,5.25,[
            '你会填“牛奶”“茶”，还是别的片段？',
            '模型根据已有内容，为词表中的候选计算分数，再选择下一个 token。',
            '今天追踪：输入的一行向量，怎样吸收前文信息？'],20,gap=.20,bottom=5.78)
        for i,(label,body) in enumerate((
            ('第一课：走完一次预测','Q/K 匹配 → 加权混合 V → 候选概率'),
            ('第二课：补全输入端','BPE 分词 → Embedding → 位置信息'),
            ('第三课：放大 Attention','可训练投影 → 匹配矩阵 → 上下文向量'))):
            y=2.08+i*1.35
            panel(c,6.65,y,6.05,1.19,label)
            c.text(6.90,y+.58,5.55,.46,body,19,INK)
        c.text(.86,6.15,11.7,.31,'课堂节奏：25 分钟复习 → 30 分钟深入 Attention → 5 分钟验收与衔接',18,MUTED)
    elif kind=='old_attention':
        old=review['first_lesson']
        panel(c,.60,2.08,5.85,3.25,'第一课原例：三个位置提供什么 V')
        table(c,.88,2.77,5.20,old['values'],labels=['我','喜欢','喝'],
              columns=['分量 0','分量 1'],cell_h=.53,highlight_row=2)
        c.text(.89,5.00,5.1,.31,'三个 token，每个向量两个分量。',18,MUTED)
        panel(c,6.65,2.08,6.05,3.25,'“喝”这一行：按比例汇总')
        c.text(6.91,2.76,5.52,.37,'权重 ≈ '+vec(old['weights']),22,BLUE,bold=True)
        c.text(6.91,3.36,5.52,.36,'第 0 分量：a₀×1 + a₁×0 + a₂×1',20,INK)
        c.text(6.91,3.88,5.52,.36,'第 1 分量：a₀×0 + a₁×1 + a₂×1',20,INK)
        c.text(6.91,4.56,5.52,.41,'结果 ≈ '+vec(old['context']),25,GREEN,bold=True)
        code(c,['context = weights @ values'])
    elif kind=='two_softmax':
        old=review['first_lesson']
        for x,title,labels,scores,weights,color in (
            (.60,'Attention：给输入位置分配读取比例',['我','喜欢','喝'],old['scores'],old['weights'],BLUE),
            (6.65,'LM Head：给候选 token 分配预测概率',old['candidate_labels'],old['lm_logits'],old['lm_probabilities'],TEAL)):
            panel(c,x,2.08,5.85 if x<1 else 6.05,3.25,title,color)
            c.text(x+.25,2.73,5.3,.34,'对象：'+' / '.join(labels),21,color,bold=True)
            c.text(x+.25,3.28,5.3,.35,'分数 ≈ '+vec(scores),21,INK)
            c.text(x+.25,3.86,5.3,.35,'Softmax ≈ '+vec(weights),21,color)
            c.text(x+.25,4.55,5.3,.50,'用于混合 V，输出向量。' if x<1 else '用于选择下一 token。',22,INK)
        code(c,['weights = scores.softmax(dim=-1)','probabilities = logits.softmax(dim=-1)'])
    elif kind=='bpe_train':
        d=review['bpe']
        panel(c,.60,2.08,5.85,3.25,'① 加权语料：每个出现位置都要计数')
        for i,(text,n) in enumerate(d['corpus'].items()):
            c.text(.88,2.74+i*.48,5.15,.38,f'{text}  × {n}',23,INK)
        c.text(.88,4.42,5.15,.38,'“喜＋欢”：5 + 3 + 2 = 10 次',21,BLUE,bold=True)
        panel(c,6.65,2.08,6.05,3.25,'② 合并后，相邻片段已经变化')
        c.text(6.91,2.74,5.53,.35,'原来：我 / 喜 / 欢 / 喝 / 奶 / 茶',21,MUTED)
        c.line(9.5,3.22,9.5,3.50,arrow=True)
        c.text(6.91,3.68,5.53,.36,'现在：我 / 喜欢 / 喝 / 奶 / 茶',21,BLUE,bold=True)
        prose(c,6.91,4.33,5.48,[
            '新增“我＋喜欢”“喜欢＋喝”。下一轮重新统计当前相邻对。'],19,bottom=5.19)
        code(c,['counts[pair] += corpus[text]','pair = max(counts, key=counts.get)'])
    elif kind=='bpe_encode':
        d=review['bpe']
        panel(c,.60,2.08,4.22,3.25,'冻结的规则：rank 小的先用')
        for i,pair in enumerate(d['merges']):
            c.text(.85,2.76+i*.48,3.74,.38,f'{i}：'+ ' + '.join(pair),20,INK)
        panel(c,5.02,2.08,7.68,3.25,'每次合并后，检查新出现的相邻候选')
        for i,row in enumerate(d['encode_trace']):
            prefix='起点' if row['rank'] is None else str(row['rank'])+' →'
            c.text(5.26,2.71+i*.43,7.13,.37,prefix+'  '+' / '.join(row['tokens']),
                   21,BLUE if i==4 else INK,bold=i==4)
        c.rect(.60,5.58,12.10,.66,PALE)
        c.text(.85,5.74,11.58,.37,'最终：你 / 喜欢喝奶茶，共两个 token。“喜欢”只是中间结果。',22,BLUE,bold=True)
    elif kind=='embedding':
        e=review['embedding']; real=review['real_tokenizer']
        panel(c,.60,2.08,5.85,3.25,'查表：ID 指定读取参数表的哪一行')
        c.text(.87,2.75,5.28,.36,'真实配置：“我喜欢喝牛奶。”',21,INK)
        c.text(.87,3.25,5.28,.38,'ID = '+str(real['ids']),23,BLUE,bold=True)
        c.text(.87,3.83,5.28,.36,'→ E[300]、E[304]、E[262]',22,TEAL)
        prose(c,.87,4.45,5.27,['相同 ID 在查表时读取相同参数；位置和上下文在后续参与计算。'],19,bottom=5.23)
        panel(c,6.65,2.08,6.05,3.25,'训练：误差推动参数改变')
        for i,label in enumerate(('预测 → 计算 loss','backward → 计算梯度','optimizer.step → 更新参数')):
            c.text(6.93,2.73+i*.48,5.45,.36,label,21,INK)
        c.text(6.93,4.47,5.45,.35,'第二课 loss：'+f"{e['bigram_loss_start']:.4f} → {e['bigram_loss_end']:.4f}",22,GREEN,bold=True)
        code(c,['vectors = embedding(ids)','loss.backward(); optimizer.step()'])
    elif kind=='positions':
        panel(c,.60,2.08,5.85,4.18,'方案 A：给输入加上位置向量')
        equation(c,r'\mathbf{x}_p=\mathbf{E}[u_p]+\mathbf{P}_p',.90,2.74,5.25,.66,size=30)
        prose(c,.88,3.67,5.28,[
            '同一个 token 的内容向量可以相同，位置向量随序号改变。',
            '位置表可以训练学习，也可以由多频率 sin/cos 公式生成。',
            '位置 0、1、2 指 token 序号。'],20,gap=.17,bottom=5.99)
        panel(c,6.65,2.08,6.05,4.18,'方案 B：RoPE 按位置旋转 Q/K')
        arrow_pair(c,8.10,4.05,(1,3))
        c.text(9.45,2.83,2.93,.37,'每格多转 30°',21,BLUE,bold=True)
        c.text(9.45,3.43,2.93,.37,'位置 1 / 3',21,INK)
        c.text(9.45,3.93,2.93,.37,'转 30° / 90°',21,TEAL)
        c.text(9.45,4.43,2.93,.37,'相差 60°',22,BLUE,bold=True)
        c.text(6.95,5.23,5.44,.70,'一起移到位置 3 / 5：转 90° / 150°，仍相差 60°。',20,INK)
    elif kind=='qkv_roles':
        formula_strip(c,r'\mathbf Q=\mathbf X\mathbf W_Q,\quad \mathbf K=\mathbf X\mathbf W_K,\quad \mathbf V=\mathbf X\mathbf W_V')
        cards=[(.60,'Q：用于发起匹配',BLUE,'当前位置用哪些特征寻找需要的信息？','“喝”的 Q 会分别与三个位置的 K 比较。'),
               (4.70,'K：用于接受匹配',TEAL,'每个位置提供哪些特征，供其他位置匹配？','与 Q 的点积，决定这个位置的匹配分数。'),
               (8.80,'V：提供被读取内容',ORANGE,'匹配完成后，实际从每个位置拿走什么？','按注意力权重混合 V，形成新的上下文向量。')]
        for x,title,color,a,bb in cards:
            panel(c,x,2.99,3.90,2.64,title,color)
            prose(c,x+.23,3.62,3.42,[a,bb],19,gap=.15,bottom=5.50)
        c.text(.88,5.94,11.5,.37,'训练学习 W_Q / W_K / W_V；每次前向计算都根据输入生成新的匹配权重。',20,INK)
    elif kind=='projections':
        # Each branch shows the actual 2x2 parameter matrix and its 3x2 result.
        for i,(tag,key,color) in enumerate((('Q','q',BLUE),('K','k',TEAL),('V','v',ORANGE))):
            x=.60+i*4.10
            panel(c,x,2.08,3.90,3.27,'W_'+tag+'  →  '+tag,color)
            table(c,x+.20,2.70,3.50,b['w_'+key],cell_h=.43,size=21)
            c.text(x+.23,3.63,3.40,.32,tag+' = X @ W_'+tag,19,color,bold=True)
            table(c,x+.20,4.03,3.50,b[key],labels=['我','喜欢','喝'],cell_h=.39,size=19)
        c.rect(.60,5.54,12.10,.73,PALE)
        c.text(.85,5.61,11.62,.33,'X 的三行：我 [1,0] ／ 喜欢 [0,1] ／ 喝 [1,1]',19,BLUE,bold=True)
        c.text(.85,5.93,11.62,.33,'手算“喝”的 V：[1,1] @ W_V = [3,-1]',19,INK)
    elif kind=='one_query':
        formula_strip(c,r'\mathbf q_2=[1,2],\qquad s_{2,j}=\mathbf q_2\cdot\mathbf k_j')
        two_panels(c,'“喝”的 Q 与三个 K 逐个匹配','缩放、归一化：得到一行读取比例')
        lines=['[1,2] · [1,1] = 3','[1,2] · [0,1] = 2','[1,2] · [1,2] = 5']
        for i,line in enumerate(lines):
            c.text(.89,3.61+i*.48,5.25,.38,line,23,INK)
        c.text(6.93,3.59,5.44,.37,'分数：[3, 2, 5] / √2',23,INK)
        c.text(6.93,4.12,5.44,.37,'权重 ≈ '+vec(b['weights'][2]),23,BLUE,bold=True)
        c.text(6.93,4.73,5.44,.35,'分别读取：我 / 喜欢 / 喝',20,MUTED)
        code(c,['scores = q[-1] @ k.T / math.sqrt(2)','last_weights = scores.softmax(dim=-1)'])
    elif kind=='score_matrix':
        formula_strip(c,r'\mathbf{Q}\mathbf{K}^{\mathsf{T}}=\mathbf{S}_{\mathrm{raw}}')
        two_panels(c,'一个格子 = 一对 Q/K 的点积','完整原始分数表：行查询，列被读取')
        prose(c,.88,3.61,5.27,[
            '第 i 行：位置 i 正在查询。第 j 列：它正在读取位置 j。',
            '“喜欢”查询“喝”：[0,2] · [1,2] = 4。',
            '形状：[3,2] @ [2,3] → [3,3]。'],19,gap=.09,bottom=5.25)
        table(c,6.91,3.59,5.52,b['raw_scores'],labels=['我','喜欢','喝'],
              columns=['我','喜欢','喝'],cell_h=.41,size=21,highlight_row=2)
        code(c,['raw_scores = q @ k.T'])
    elif kind=='scale':
        formula_strip(c,r'\mathbf S=\frac{\mathbf Q\mathbf K^\mathsf{T}}{\sqrt{d_k}},\qquad d_k=2')
        two_panels(c,'为何缩放：控制点积的数值尺度','同一行分数：缩放前后比较')
        prose(c,.88,3.61,5.27,[
            '点积累加 d_k 个乘积，维度增大时，典型波动幅度会增大。',
            '较大的分数差会让 Softmax 更集中；缩放缓和这种集中程度。'],19,gap=.15,bottom=5.24)
        c.text(6.92,3.61,5.47,.35,'不缩放：[3, 2, 5]',21,INK)
        c.text(6.92,4.03,5.47,.36,'→ '+vec(r['scale']['unscaled_weights'][2]),22,ORANGE)
        c.text(6.92,4.51,5.47,.35,'除以 √2 后：',21,INK)
        c.text(6.92,4.91,5.47,.35,'→ '+vec(b['weights'][2]),22,BLUE,bold=True)
        code(c,['scores = raw_scores / math.sqrt(q.shape[-1])'])
    elif kind=='mask':
        panel(c,.60,2.08,5.85,3.25,'先限制可读范围，再做 Softmax')
        prose(c,.88,2.74,5.27,[
            '行 i 只能读取列 j≤i：过去与当前位置。',
            '未来格子的分数设为 -∞，其指数为 0，归一化后权重也为 0。',
            '对角线保留：当前输入用于预测下一 token。'],19,gap=.15,bottom=5.18)
        panel(c,6.65,2.08,6.05,3.25,'缩放分数 + 因果 mask')
        rows=[['-∞' if b['forbidden'][i][j] else b['scores'][i][j] for j in range(3)] for i in range(3)]
        table(c,6.92,2.78,5.49,rows,labels=['我','喜欢','喝'],
              columns=['我','喜欢','喝'],cell_h=.53,size=22)
        c.text(6.92,5.00,5.49,.31,'设为 0 仍有 exp(0)=1，无法屏蔽。',18,MUTED)
        code(c,['future = torch.ones(3, 3, dtype=torch.bool).triu(1)',
                'masked = scores.masked_fill(future, -torch.inf)'])
    elif kind=='mix':
        formula_strip(c,r'\mathbf A=\operatorname{softmax}_{j}(\mathbf S+\mathbf M),\qquad\mathbf Z=\mathbf A\mathbf V')
        two_panels(c,'A：每个查询在可见位置间分配权重','Z：每行都是加权后的新向量')
        table(c,.85,3.53,5.34,b['weights'],labels=['我','喜欢','喝'],
              columns=['我','喜欢','喝'],cell_h=.43,size=20,heat=True)
        table(c,6.92,3.53,5.49,b['output'],labels=['我','喜欢','喝'],
              columns=['分量 0','分量 1'],cell_h=.43,size=21)
        code(c,['weights = masked.softmax(dim=-1)','output = weights @ v',
                '# Row 1: 0.5*v[0] + 0.5*v[1] = [1.5, -0.5]'],y=5.35,height=1.04)
    elif kind in ('change_v','change_k'):
        d=r['interventions']['value' if kind=='change_v' else 'key']
        panel(c,.60,2.08,3.65,3.25,'只改一个中间张量')
        if kind=='change_v':
            prose(c,.85,2.75,3.15,[
                'Q、K 固定。',
                'V[0]：[1,-1] → [3,-1]。',
                '权重不变。输出变化 = 第 0 列权重 × [2,0]。'],19,gap=.17,bottom=5.22)
        else:
            prose(c,.85,2.75,3.15,[
                'Q、V 固定。',
                'K[0]：[1,1] → [2,1]。',
                '观察最后一行：第 0 个匹配分数从 3 变为 4。'],19,gap=.17,bottom=5.22)
        panel(c,4.45,2.08,8.25,3.25,'同一位置的输出：修改前 → 修改后')
        c.text(5.47,2.77,3.24,.34,'修改前 Z',19,MUTED,center=True)
        c.text(9.08,2.77,3.24,.34,'修改后 Z',19,BLUE,center=True)
        for i,label in enumerate(b['tokens']):
            y=3.28+i*.47
            c.text(4.71,y,.65,.34,label,18,MUTED,center=True)
            c.text(5.47,y,3.24,.37,vec(d['output_before'][i]),22,INK,center=True)
            c.text(9.08,y,3.24,.37,vec(d['output_after'][i]),22,BLUE,bold=True,center=True)
        caption=('第三行变化：'+vec(d['delta_output'][2]) if kind=='change_v' else
                 '最后一行权重：'+vec(d['weights_before'][2])+' → '+vec(d['weights_after'][2]))
        c.text(4.72,4.91,7.66,.34,caption,18,GREEN)
        code(c,['changed_v = v.clone(); changed_v[0] += v.new_tensor([2., 0.])',
                'changed = attention(q, k, changed_v)'] if kind=='change_v' else
               ['changed_k = k.clone(); changed_k[0] += k.new_tensor([1., 0.])',
                'changed = attention(q, changed_k, v)'])
    elif kind=='causality':
        d=r['interventions']['causality']
        panel(c,.60,2.08,4.12,3.25,'改输入，再重算全部投影')
        c.text(.88,2.75,3.56,.34,'只改最后位置 X[2]：',19,INK)
        c.text(.88,3.20,3.56,.38,'[1,1] → [4,-1]',23,BLUE,bold=True)
        prose(c,.88,3.89,3.56,[
            '位置与参数固定。',
            '重新生成 Q/K/V。',
            '比较每行的最大绝对差。'],19,gap=.10,bottom=5.24)
        panel(c,4.92,2.08,7.78,3.25,'对照实验：打开 / 关闭因果 mask')
        delta=lambda a,b:[max(abs(x-y) for x,y in zip(aa,bb)) for aa,bb in zip(a,b)]
        on=delta(d['masked_before'],d['masked_after'])
        off=delta(d['unmasked_before'],d['unmasked_after'])
        table(c,5.16,2.75,7.27,list(zip(on,off)),labels=b['tokens'],
              columns=['mask 打开','mask 关闭'],cell_h=.53,size=23,highlight_row=2)
        c.text(5.22,5.00,7.1,.31,'前两行变化为 0；最后位置确实改变。',18,GREEN,bold=True)
        code(c,['changed_x = x.clone(); changed_x[-1] += x.new_tensor([3., -2.])',
                'q2, k2, v2 = project(changed_x)'])
    elif kind=='core_code':
        c.rect(.60,2.08,12.10,2.75,DARK)
        lines=[
            'q, k, v = x @ w_q, x @ w_k, x @ w_v',
            'scores = q @ k.T / math.sqrt(q.shape[-1])',
            'future = torch.ones(3, 3, dtype=torch.bool).triu(1)',
            'masked = scores.masked_fill(future, -torch.inf)',
            'weights = masked.softmax(dim=-1)',
            'output = weights @ v']
        for i,line in enumerate(lines):
            c.text(.87,2.31+i*.39,11.52,.35,line,19,WHITE,mono=True)
        for i,(title,body) in enumerate((
            ('① 投影与匹配','[3,2] → 分数 [3,3]'),
            ('② 可读范围与比例','mask → 每行权重和为 1'),
            ('③ 汇总内容','[3,3] @ [3,2] → [3,2]'))):
            x=.60+i*4.10; panel(c,x,5.04,3.90,1.20,title)
            c.text(x+.22,5.66,3.46,.35,body,18,INK)
    elif kind=='multihead':
        panel(c,.60,2.08,12.10,4.16,'结构示意：D=4，H=2，每个头宽度为 2')
        c.rect(.90,3.63,2.15,.68,PALE)
        c.text(1.00,3.78,1.95,.38,'X [1,3,4]',22,BLUE,bold=True,center=True)
        c.line(3.08,3.97,3.60,3.97)
        c.line(3.60,3.97,3.60,3.18)
        c.line(3.60,3.18,4.05,3.18,arrow=True)
        c.line(3.60,3.97,3.60,4.77)
        c.line(3.60,4.77,4.05,4.77,arrow=True)
        for i,y in enumerate((2.73,4.32)):
            c.rect(4.08,y,3.77,.94,PALE if i==0 else 'DFF0ED')
            c.text(4.22,y+.13,3.49,.34,f'Head {i+1}：投影 → Attention',18,BLUE if i==0 else TEAL,bold=True,center=True)
            c.text(4.22,y+.55,3.49,.30,'输出 [1,3,2]',17,MUTED,center=True)
            c.line(7.88,y+.45,8.48,y+.45)
        c.line(8.48,3.18,8.48,4.77)
        c.line(8.48,3.97,8.96,3.97,arrow=True)
        c.rect(8.99,3.54,3.37,.88,PALE)
        c.text(9.12,3.67,3.11,.34,'拼接 → W_O',20,BLUE,bold=True,center=True)
        c.text(9.12,4.06,3.11,.30,'输出 [1,3,4]',17,MUTED,center=True)
        c.text(.94,5.72,11.32,.36,'每头学习不同投影；下一课实现拆头、合头，再接残差、Norm 和 FFN。',20,INK)
    elif kind=='quiz':
        panel(c,.60,2.08,5.85,3.89,'先看图：任选一行，解释它在做什么')
        table(c,.87,2.77,5.30,b['weights'],labels=b['tokens'],columns=b['tokens'],
              cell_h=.58,size=22,heat=True)
        c.text(.90,5.28,5.20,.37,'这一行由哪个 Q 查询哪些 K？',22,BLUE,bold=True)
        panel(c,6.65,2.08,6.05,3.89,'再回答：先预测，再运行')
        prose(c,6.92,2.76,5.49,[
            '① 只改 V：哪一步开始变化？',
            '② 修改未来输入：前两行会变吗？',
            '③ 为什么保留对角线？',
            '④ Attention 最后输出什么？'],21,gap=.17,bottom=5.80)
        c.text(.89,6.13,11.6,.32,'下一课：多头 + 输出投影 + 残差 + Norm + FFN → 一个完整 Transformer Block',18,MUTED)
    else:
        raise ValueError(kind)
