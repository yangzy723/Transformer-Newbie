"""前15页：保留关键计算，合并重复解释和外围铺垫。"""
from layout import *


def draw(c,item,r):
    k=item['kind']
    if k=='opening':
        label(c,.86,2.05,11.50,'已有上下文：我  /  喜欢  /  喝',size=24)
        for x,title,detail,col,fill in [(.87,'更新每个位置的表示','Attention → FFN → 后续层',BLUE,PALE),(4.97,'算出词表概率','向量 → 分数 → 概率',TEAL,MINT),(9.07,'选出一个 ID','解码选择 → 追加到上下文',ORANGE,SAND)]:
            node(c,x,2.82,3.37,1.10,title,detail,fill,col,size=23 if x<9 else 22)
        arrow(c,4.41,3.37,4.79,3.37);arrow(c,8.51,3.37,8.89,3.37,TEAL)
        rowtable(c,.87,4.65,11.57,['计算对象','向量','词表分数','概率','token ID'],[
            ['大小','D 个特征','V 个分数','V 个概率值','1 个整数'],
            ['含义','内部表示','候选相对得分','总和为 1','词表地址'],
        ],widths=[.15,.22,.23,.25,.15],row_h=.62,size=21)
        note(c,.90,6.34,11.35,'D 表示模型宽度，V 表示词表大小；生成时使用最后一个位置的最终表示。',20)
    elif k=='block':
        specs=[(.85,1.0,'x','输入'),(2.55,1.75,'Norm₁','按 D 维'),(5.03,3.20,'Causal Attention','Q/K + RoPE、Mask'),(8.97,1.60,'+ x','得到 h'),(11.27,1.05,'h','')]
        for x,w,title,sub in specs:node(c,x,2.73,w,.94,title,sub,size=20)
        for a,b in [(2.01,2.36),(4.47,4.84),(8.40,8.78),(10.74,11.08)]:arrow(c,a,3.20,b,3.20)
        c.path([(1.35,2.57),(1.35,2.13),(9.77,2.13),(9.77,2.57)],color=MUTED,width=1.5)
        for x,w,title,sub in [(8.97,1.95,'Norm₂','按 D 维'),(5.03,3.20,'FFN','升维 → 激活 → 降维'),(2.55,1.75,'+ h','第二次残差'),(.85,1.0,'y','输出')]:node(c,x,4.64,w,.94,title,sub,fill=MINT if title=='FFN' else PALE,color=TEAL if title=='FFN' else BLUE,size=20)
        c.path([(11.79,3.86),(11.79,5.11),(11.10,5.11)],color=BLUE,width=2.3)
        for a,b in [(8.78,8.42),(4.84,4.47),(2.36,2.02)]:arrow(c,a,5.11,b,5.11)
        c.path([(11.79,4.06),(3.43,4.06),(3.43,4.48)],color=MUTED,width=1.5)
        eq(c,r'h=x+\operatorname{Attention}(\operatorname{Norm}_1(x)),\qquad y=h+\operatorname{FFN}(\operatorname{Norm}_2(h))',x=.85,y=5.95,w=11.45,h=.47,size=24)
        note(c,.88,6.35,11.45,'单个位置示例：x=[1,2]，Attention 分支输出 [0.3,-0.2]，相加后 h=[1.3,1.8]。',18)
    elif k=='heads':
        label(c,.87,2.06,11.50,'设模型宽度 D=4，头数 H=2，每头宽度 d_h=2；下图为位置 i 的两个头输出。',size=21)
        node(c,.90,2.95,2.48,.91,'头 1：[1, 2]','读取可见位置后得到',size=22)
        node(c,.90,4.43,2.48,.91,'头 2：[-1, 3]','独立 Q/K/V 投影',fill=MINT,color=TEAL,size=21)
        c.path([(3.56,3.40),(3.89,3.40),(3.89,4.13),(4.18,4.13)],color=BLUE,width=2.2)
        c.path([(3.56,4.88),(3.89,4.88),(3.89,4.13)],color=TEAL,width=2.2,arrow=False)
        node(c,4.37,3.61,3.17,1.05,'[1, 2, -1, 3]','Concat：沿特征轴拼接',size=25)
        node(c,8.25,3.61,1.58,1.05,'× W_O','[4,4]',size=22)
        node(c,10.55,3.61,1.80,1.05,'输出 [D]','本例 D=4',size=22)
        arrow(c,7.72,4.13,8.06,4.13);arrow(c,10.01,4.13,10.36,4.13)
        rowtable(c,.90,5.69,11.45,['各头输出拼接','输出投影 W_O','最终形状'],[
            ['2 + 2 = 4 个特征','学习各头特征的线性组合','每个位置仍为 D=4 维'],
        ],widths=[.30,.38,.32],row_h=.60,size=20)
    elif k=='norm':
        eq(c,r'\mu=\frac{1}{D}\sum_jh_j,\quad v=\frac{1}{D}\sum_j(h_j-\mu)^2,\quad u_j=\gamma_j\frac{h_j-\mu}{\sqrt{v+\epsilon}}+\beta_j',y=2.02,h=.66,size=27)
        rowtable(c,.88,3.12,7.25,['计算','h=[1.3,1.8] 的实际结果'],[
            ['均值 μ','(1.3+1.8)/2 = 1.55'],
            ['方差 v','((-0.25)²+0.25²)/2 = 0.0625'],
            ['归一化 u','[-0.99992, 0.99992]'],
        ],widths=[.24,.76],row_h=.73,size=21)
        card(c,8.65,3.11,3.70,2.70,'归一化后的计算','FFN 的输入为 u。\n\n残差结果为 h+FFN(u)。\n\n各位置分别计算均值和方差。',fill=MINT,color=TEAL,body_size=20)
        note(c,.91,6.20,11.35,'本例 γ=[1,1]、β=[0,0]、ε=0.00001；γ、β 分别是可学习的缩放与偏移参数。',20)
    elif k=='ffn_shape':
        eq(c,r'\operatorname{FFN}(u)=\phi(uW_1+b_1)W_2+b_2',y=2.01,h=.70,size=33)
        for x,w,title,sub,col,fill in [(.88,1.9,'输入 u','D 维',BLUE,PALE),(3.55,2.25,'投影 W_1','D → d_ff',BLUE,PALE),(6.56,1.85,'激活 φ','逐元素',TEAL,MINT),(9.17,3.18,'投影 W_2','d_ff → D',BLUE,PALE)]:node(c,x,3.22,w,1.09,title,sub,fill,col,size=23)
        for a,b in [(2.95,3.36),(5.97,6.37),(8.58,8.98)]:arrow(c,a,3.77,b,3.77)
        rowtable(c,.90,4.91,11.45,['形状','本课取值','含偏置的参数量'],[
            ['[B,T,D] → [B,T,d_ff] → [B,T,D]','D=2，d_ff=4','2D·d_ff + d_ff + D = 22'],
        ],widths=[.47,.21,.32],row_h=.67,size=20)
        note(c,.92,6.27,11.30,'B：批大小；T：位置数；D：模型宽度。矩阵与偏置在所有位置共享。',20)
    elif k=='weights':
        label(c,.90,2.13,3.0,'FFN 输入 u [2]',size=21)
        text(c,.91,3.03,2.90,.60,'[1, -2]',34,BLUE,True)
        label(c,4.10,2.13,3.75,'W_1 [2,4]',size=21)
        table(c,4.08,2.93,3.88,r['ffn']['w1'],cell_h=.67,size=28)
        label(c,9.0,2.13,3.3,'W_2 [4,2]',size=21)
        table(c,9.0,2.93,3.20,r['ffn']['w2'],cell_h=.52,size=27)
        c.rect(.90,5.39,11.44,.60,PALE,False)
        text(c,1.05,5.51,11.10,.37,'W_1 的第 2 列（索引 1）为 [-1,0]：1×(-1)+(-2)×0 = -1。',21,BLUE)
        note(c,.92,6.28,11.30,'本例输入为 u=[1,-2]，偏置 b₁、b₂ 均为 0；每一列决定一个输出分量。',19)
    elif k=='expand':
        rowtable(c,.88,2.48,7.62,['分量','取出的列','逐项乘加'],[
            ['a₀','[1,0]','1×1 + (-2)×0 = 1'],
            ['a₁','[-1,0]','1×(-1) + (-2)×0 = -1'],
            ['a₂','[0,1]','1×0 + (-2)×1 = -2'],
            ['a₃','[0,-1]','1×0 + (-2)×(-1) = 2'],
        ],widths=[.14,.24,.62],row_h=.72,size=21)
        node(c,9.04,3.20,3.30,1.22,'[1, -1, -2, 2]','中间向量 a [4]',size=25)
        para(c,9.10,4.85,3.18,'每个中间分量都是输入分量的线性组合。',21)
        note(c,.92,6.30,11.30,'向量从 2 维扩展为 4 维：同一个位置产生四个中间特征。',21)
    elif k=='relu':
        curve(c,1.33,2.87,5.25,2.77,[('ReLU',lambda x:max(0,x),TEAL)],xlim=(-3,3),ylim=(-1,3))
        eq(c,r'\operatorname{ReLU}(a)=\max(0,a)',x=7.10,y=2.01,w=5.20,h=.64,size=29)
        rowtable(c,7.09,3.07,5.25,['输入 a','ReLU(a)'],[['1','1'],['-1','0'],['-2','0'],['2','2']],row_h=.61,size=23)
        note(c,.94,6.28,11.28,'本例输出 r=[1,0,0,2]：负分量被置零，正分量保持原值。',21)
    elif k=='project':
        label(c,.91,2.08,3.80,'激活向量 r=[1,0,0,2]',size=22)
        table(c,5.28,2.55,2.40,r['ffn']['w2'],cell_h=.49,size=25)
        text(c,4.48,3.14,.50,.48,'×',30,MUTED,center=True)
        arrow(c,8.03,3.56,8.59,3.56)
        node(c,8.91,3.04,3.40,1.05,'f = [1, 2]','FFN 分支输出 [D]',size=29)
        rowtable(c,.90,4.65,11.45,['第 1 个输出分量（索引 0）','第 2 个输出分量（索引 1）'],[
            ['1×1 + 0×1 + 0×0 + 2×0 = 1','1×0 + 0×0 + 0×1 + 2×1 = 2'],
        ],row_h=.65,size=20)
        text(c,.94,5.99,11.27,.39,'[1,-2] → [1,-1,-2,2] → [1,0,0,2] → [1,2]',23,BLUE,True)
        note(c,.95,6.38,11.25,'最终投影可使用正、负权重，因此 FFN 的输出分量也可以为负。',18)
    elif k=='positionwise':
        label(c,.91,2.02,11.40,'本例 FFN([a,b])=[|a|,|b|]；参数固定，比较修改一行输入前后的结果。',size=20)
        rowtable(c,.90,2.44,11.45,['位置','原输入','改后输入','原输出','改后输出'],[
            ['0','[1,-2]','[1,-2]','[1,2]','[1,2]'],
            ['1','[-1,2]','[-3,2]','[1,2]','[3,2]'],
            ['2','[.5,.25]','[.5,.25]','[.5,.25]','[.5,.25]'],
        ],widths=[.09,.22,.25,.22,.22],row_h=.76,size=22)
        label(c,.94,5.64,5.25,'逐位置计算',size=21)
        para(c,.94,5.98,5.25,'修改位置 1 的输入，仅该位置的 FFN 输出改变。',20)
        label(c,6.91,5.64,5.29,'上下文来自前面的 Attention',color=TEAL,size=21)
        para(c,6.91,5.98,5.29,'进入 FFN 前，各行表示已融合可见位置的信息。',20)
    elif k=='nonlinearity':
        eq(c,r'(uW_1+b_1)W_2+b_2=u(W_1W_2)+(b_1W_2+b_2)',y=2.01,h=.58,size=29)
        label(c,.93,2.74,11.37,'本例 FFN([a,b])=[|a|,|b|]；取 u=[x,0]，比较第一个输出分量。',size=19.5)
        curve(c,1.31,3.64,5.18,2.05,[('ReLU: |x|',abs,TEAL),('No activation: 0',lambda x:0,ORANGE)],xlim=(-3,3),ylim=(-1,3))
        rowtable(c,7.05,3.43,5.24,['x','有 ReLU','无激活'],[['-2','2','0'],['0','0','0'],['2','2','0']],widths=[.22,.39,.39],row_h=.61,size=23)
        para(c,7.09,5.94,5.16,'一条直线若经过 (-2,2)、(2,2)，在 x=0 时应输出 2，而非 0。',20)
        note(c,1.00,6.25,5.70,'本组参数：W₁W₂=0，b₁=b₂=0。',17.5)
    elif k=='activations':
        ge=lambda x:.5*x*(1+math.erf(x/math.sqrt(2)));si=lambda x:x/(1+math.exp(-x))
        curve(c,1.33,2.93,5.14,2.90,[('ReLU',lambda x:max(0,x),'8B98A8'),('GELU',ge,BLUE),('SiLU',si,TEAL)],xlim=(-3,3),ylim=(-1,3))
        eq(c,r'\operatorname{GELU}(x)=x\Phi(x)',x=7.05,y=2.00,w=5.25,h=.39,size=26)
        eq(c,r'\operatorname{SiLU}(x)=x\sigma(x)',x=7.05,y=2.48,w=5.25,h=.39,size=26)
        eq(c,r'\sigma(x)=\frac{1}{1+e^{-x}}',x=7.05,y=2.95,w=5.25,h=.33,size=23)
        rowtable(c,7.04,3.39,5.25,['x','ReLU','GELU','SiLU'],[
            [str(x),str(max(0,x)),f'{ge(x):.4f}',f'{si(x):.4f}'] for x in [-1,0,1]
        ],widths=[.13,.25,.31,.31],row_h=.60,size=21)
        note(c,7.09,5.86,5.17,'Φ 为标准正态分布的累积分布函数。',18)
        note(c,.95,6.39,11.26,'GELU、SiLU 在负半轴也可输出小负值；本课代码使用 nn.GELU()。',18)
    elif k=='swiglu':
        eq(c,r'\operatorname{SwiGLU}(u)=[\operatorname{SiLU}(uW_g)\odot(uW_v)]W_d',y=1.99,h=.53,size=28)
        node(c,.90,3.44,1.73,.85,'u=[1,-2]',size=22)
        node(c,3.48,2.80,3.24,1.06,'SiLU(uW_g)','[0.7311, -0.2384]',fill=MINT,color=TEAL,size=23)
        node(c,3.48,4.48,3.24,.84,'uW_v = [1,-1]',size=22)
        c.path([(2.79,3.87),(3.12,3.87),(3.12,3.33),(3.30,3.33)],color=TEAL,width=2.2)
        c.path([(3.12,3.87),(3.12,4.90),(3.30,4.90)],color=BLUE,width=2.2)
        node(c,7.53,3.44,1.26,.85,'×',fill=SAND,color=ORANGE,size=30)
        text(c,7.14,2.91,2.03,.31,'逐元素相乘',17,ORANGE,center=True)
        c.path([(6.89,3.33),(7.19,3.33),(7.19,3.87),(7.35,3.87)],color=TEAL,width=2.2)
        c.path([(6.89,4.90),(7.19,4.90),(7.19,3.87)],color=BLUE,width=2.2,arrow=False)
        node(c,9.55,3.36,2.79,1.03,'[.7311,.2384]','× W_d 后输出 [D]',size=22)
        arrow(c,8.96,3.87,9.36,3.87,ORANGE)
        text(c,.95,5.70,11.27,.38,'本例 D=m=2，W_g=W_d=I；W_v 的两行为 [1,1]、[0,1]，偏置为 0。',19,INK)
        note(c,.95,6.26,11.27,'m 为中间宽度。忽略偏置时，SwiGLU 参数量为 3Dm，普通 FFN 为 2D·d_ff。',19)
    elif k=='residual':
        n=r['norm']
        label(c,.92,2.02,11.40,'h=[1.3,1.8]；γ=[1,1]，β=[0,0]，ε=0.00001；b₁=b₂=0。',size=20)
        rowtable(c,.90,2.73,7.40,['步骤','计算结果'],[
            ['原表示 h','[1.3,1.8]'],
            ['u=LayerNorm(h)',vec(n['normalized'],5)],
            ['uW_1',vec(n['expanded'],4)],
            ['ReLU(uW_1)',vec(n['activated'],4)],
            ['f=FFN(u)',vec(n['ffn_output'],5)],
            ['y=h+f',vec(n['residual_output'],5)],
        ],widths=[.36,.64],row_h=.55,size=20)
        label(c,8.89,2.73,3.45,'W₁ [2,4]',size=20)
        table(c,8.87,3.20,3.47,r['ffn']['w1'],cell_h=.44,size=23)
        label(c,8.89,4.30,3.45,'W₂ [4,2]',size=20)
        table(c,8.87,4.72,3.47,r['ffn']['w2'],cell_h=.42,size=23)
    elif k=='block_code':
        code(c,[
            'self.ffn = nn.Sequential(',
            '    nn.Linear(D, 4 * D), nn.GELU(),',
            '    nn.Linear(4 * D, D),',
            ')',
            '',
            'context, weights = self.attention(self.ln1(x))',
            'h = x + context',
            'y = h + self.ffn(self.ln2(h))',
        ],x=.99,y=2.15,w=8.30,size=19.5,line_h=.40)
        for y,title,sub,fill,col in [(2.30,'输入 [B,T,D]','模型宽度 D',PALE,BLUE),(3.46,'中间 [B,T,4D]','本例中间宽度 4D',MINT,TEAL),(4.62,'输出 [B,T,D]','形状相同才可相加',PALE,BLUE)]:node(c,9.70,y,2.66,.94,title,sub,fill,col,size=19.5)
        note(c,.99,5.90,11.20,'公式写 uW；nn.Linear.weight 存为 [out,in]，前向实际计算 u @ weight.T + bias。',20)
        text(c,.99,6.40,11.20,.34,'本页权重形状：第一层 [4D,D]，第二层 [D,4D]。',19,BLUE,True)
    else:return False
    return True
