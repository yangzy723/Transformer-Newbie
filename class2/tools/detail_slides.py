"""课堂阅读版：定义、计算过程、实际结果与关键代码同页对应。"""
from PIL import ImageFont

from formulas import place_formula
from slide_design import INK, MUTED, BLUE, TEAL, WHITE, PALE, GREEN, DARK, token


def vector(values,digits=4):
    return "["+", ".join(f"{v:.{digits}f}".rstrip("0").rstrip(".") if v else "0" for v in values)+"]"


def prose(c, x, y, width, texts, size=18, gap=.07):
    font=ImageFont.truetype(c.font_path,round(size*120/72))
    for text in texts:
        n,current=1,""
        for char in text:
            if c.draw.textlength(current+char,font=font)>width*120-5:
                n+=1
                current=char
            else:
                current+=char
        height=n*size/72*1.22
        c.text(x,y,width,height+.015,text,size,INK)
        y+=height+gap
    return y


def panel(c,x,y,w,h,title):
    c.rect(x,y,w,h,WHITE)
    c.text(x+.20,y+.15,w-.4,.31,title,18,BLUE,bold=True)


def eq(c,root,kind,expr,x,y,w,h,size=32):
    # 文件名由公式内容决定，同页可以有独立的中间推导。
    import hashlib
    name=hashlib.sha256(expr.encode()).hexdigest()[:10]
    place_formula(c,expr,root/f"assets/formulas/detail_{kind}_{name}.png",x,y,w,h,font_size=size)


def code_band(c,lines):
    c.rect(.60,5.53,12.10,.74,DARK)
    c.text(.78,5.65,.55,.3,"PY",15,"8BCFD2",bold=True)
    for i,line in enumerate(lines[:2]):
        line=line.partition("#")[0].rstrip()
        c.text(1.35,5.57+i*.30,11.06,.32,line,18,WHITE,mono=True)


KEY_FORMULAS={
    "onehot":[r"\mathbf{X}=\mathbf{E}[I]=\operatorname{one\_hot}(I)\,\mathbf{E}"],
    "grad":[r"\frac{\partial L}{\partial\mathbf{E}_j}=\sum_{(b,t):\,I_{b,t}=j}\frac{\partial L}{\partial\mathbf{X}_{b,t}}\quad(j\ne\mathrm{pad})"],
    "chain":[r"\frac{\partial L}{\partial\mathbf{e}}=(\mathbf{p}-\operatorname{one\_hot}(y))\mathbf{W}_{\rm out}"],
    "training":[r"L=-\frac{1}{N}\sum_{r=1}^{N}\log p(y_r\mid x_r)"],
    "tie":[r"\mathbf{z}=\mathbf{h}\mathbf{E}^{\mathsf{T}}+\mathbf{b},\qquad\nabla_{\mathbf{E}}L=\nabla_{\mathbf{E}}^{\rm lookup}L+\nabla_{\mathbf{E}}^{\rm output}L"],
    "permutation":[r"\operatorname{Attn}(\Pi\mathbf{X})=\Pi\operatorname{Attn}(\mathbf{X})"],
    "absolute":[r"\mathbf{X}_{b,t}=\mathbf{E}[I_{b,t}]+\operatorname{PE}(t)"],
    "frequencies":[r"\omega_i=10000^{-2i/D},\quad\theta_{t,i}=t\omega_i",r"\operatorname{PE}(t,2i)=\sin\theta_{t,i},\quad\operatorname{PE}(t,2i+1)=\cos\theta_{t,i}"],
    "vector_rope":[r"x'_{2i}=x_{2i}\cos\theta-x_{2i+1}\sin\theta",r"x'_{2i+1}=x_{2i}\sin\theta+x_{2i+1}\cos\theta"],
    "mask":[r"\mathbf{A}=\operatorname{softmax}\!\left(\frac{\widetilde{\mathbf{Q}}\widetilde{\mathbf{K}}^{\mathsf{T}}}{\sqrt{d_h}}+\mathbf{M}\right)"],
}


CODE={
    "onehot":["vectors = embedding(ids)","one_hot @ embedding.weight  # 与查表前向等价"],
    "grad":["loss = (embedding(ids) * upstream).sum()","loss.backward()  # worked_examples.gradient_example"],
    "chain":["loss = F.cross_entropy(logits, target)","loss.backward()  # 将概率误差传回 embedding_vector"],
    "training":["logits = output(embedding(x)); loss = F.cross_entropy(logits, y)","optimizer.zero_grad(); loss.backward(); optimizer.step()"],
    "tie":["head.weight = tied_embedding.weight","# 必须在创建 optimizer 前绑定；避免重复传入同一个参数"],
    "permutation":["rearranged = attention(x[permutation])","torch.allclose(rearranged, original[permutation])"],
    "absolute":["pe = sinusoidal(torch.arange(3), 4)","absolute_original = attention(x + pe)"],
    "frequencies":["angles = positions[:, None] * frequencies[None, :]","result[:, 0::2], result[:, 1::2] = angles.sin(), angles.cos()"],
    "rotation":["qr, kr = apply_rope(q, [1]), apply_rope(k, [3])","score = (qr * kr).sum()  # 同时平移两个位置，点积保持"],
    "vector_rope":["result[..., 0::2] = even * cos - odd * sin","result[..., 1::2] = even * sin + odd * cos"],
    "mask":["forbidden = future | ~valid[None, :]  # 四位置算例","weights = scores.masked_fill(forbidden, -torch.inf).softmax(-1)"],
}


def formula_detail(c,item,link,root,report,worked):
    kind=item["kind"]
    c.rect(.60,2.06,12.10,.98,WHITE)
    if kind=="rotation":
        for expression,bounds in zip(link["formulas"],((.8,2.12,3.5,.81),(4.6,2.12,7.7,.81))):
            eq(c,root,kind,expression,*bounds,size=32)
    else:
        expressions=KEY_FORMULAS[kind]
        h=.88/len(expressions)
        for i,expression in enumerate(expressions):
            eq(c,root,kind,expression,.92,2.10+i*h,11.46,h-.02,size=32)
    panel(c,.60,3.21,5.86,2.12,"机制：每个量在做什么")
    title="S全0、末列为pad：注意力权重" if kind=="mask" else "算例：跟着数值走一遍"
    panel(c,6.66,3.21,6.04,2.12,title)
    if kind not in ("rotation","frequencies"):
        end=prose(c,.82,3.75,5.40,item["explanation"][:2])
        if end>5.30:
            raise ValueError(f"解释过长：{kind}，结束于{end}")
    x,y=6.90,3.73
    if kind=="onehot":
        data=worked["lookup"]
        c.text(x,y,5.5,.34,"算术小表 E 的第 1、2 行：",19,MUTED)
        c.text(x,y+.41,5.5,.36,"E[1] = "+vector(data["table"][1]),21,BLUE)
        c.text(x,y+.82,5.5,.36,"E[2] = "+vector(data["table"][2]),21,TEAL)
        c.text(x,y+1.25,5.5,.34,"I=[1,2,1] → 依次取第1、第2、第1行",18,INK)
    elif kind=="grad":
        d=worked["gradient"]
        c.text(x,y,5.5,.35,"ID 1：[1,0]+[0,2] = "+vector(d["row_gradients"][1]),21,BLUE)
        c.text(x,y+.46,5.5,.35,"ID 2：第2行梯度 = "+vector(d["row_gradients"][2]),20,TEAL)
        prose(c,x,y+.96,5.45,["pad 行的 lookup 梯度为 [0,0]。上游梯度是向量，累加不是数出现次数这么简单。"],18)
    elif kind=="chain":
        d=worked["cross_entropy"]
        c.text(x,y,5.5,.33,"p−one_hot(y) = "+vector(d["error"]),20,BLUE)
        c.text(x,y+.40,5.5,.34,"W_out 三行：[1,0]、[0,1]、[−1,0.5]",18,INK)
        c.text(x,y+.80,5.5,.37,"向量梯度 = "+vector(d["vector_gradient"]),23,GREEN,bold=True)
        c.text(x,y+1.28,5.5,.32,"这只是三类算术例；真实类别数为 V。",18,MUTED)
    elif kind=="training":
        rows=report["history"]
        c.text(x,y,5.5,.35,"步数       0          30         120",20,MUTED)
        values=[rows[i]["loss"] for i in (0,1,-1)]
        c.text(x,y+.43,5.5,.37,"loss   "+"   ".join(f"{v:.4f}" for v in values),22,BLUE)
        prose(c,x,y+.94,5.45,["26 行参数改变；模型只看当前 token。同一输入存在多种后续，loss 不必归零。"],18)
    elif kind=="tie":
        prose(c,x,y,5.45,["输入路径：只读取当前出现的行。","输出路径：softmax 对全词表候选求导。","两条贡献相加：没出现在输入里的行，也可能因输出分类收到梯度。"],18)
    elif kind=="permutation":
        c.text(x,y,5.5,.35,"内容顺序：我 / 喜欢 / 你",22,BLUE)
        c.text(x,y+.42,5.5,.35,"重排为：你 / 喜欢 / 我",22,TEAL)
        prose(c,x,y+.92,5.45,["Q/K/V 都随行重排，scores 的行列一起重排。加上固定位置坐标后，一般不再满足这个等式。"],18)
    elif kind=="absolute":
        c.text(x,y,5.5,.33,"D=4，位置 t=0 的正弦向量",19,MUTED)
        c.text(x,y+.40,5.5,.35,"内容：[0.2, −0.1, 0.4, 0]",21,INK)
        c.text(x,y+.80,5.5,.35,"位置：[0, 1, 0, 1]",21,TEAL)
        c.text(x,y+1.20,5.5,.37,"相加：[0.2, 0.9, 0.4, 1]",23,BLUE,bold=True)
    elif kind=="frequencies":
        prose(c,.82,3.75,5.40,["D=4：两对维度的频率为 [1,0.01]。t=1 时角度为 [1,0.01]。","[T,1] × [1,D/2] 广播得到 [T,D/2]；不是逐个 token 写循环。"],18)
        c.text(x,y,5.5,.34,"PE(0) = [0, 1, 0, 1]",22,BLUE)
        c.text(x,y+.46,5.5,.33,"PE(1) ≈ [0.8415, 0.5403,",21,TEAL)
        c.text(x+.88,y+.83,4.4,.34,"0.0100, 1.0000]",21,TEAL)
        c.text(x,y+1.28,5.5,.32,"高低频率提供不同的位置变化尺度。",18,MUTED)
    elif kind=="rotation":
        eq(c,root,kind,r"R(m\omega)^{\mathsf{T}}=R(-m\omega)",.88,3.74,5.35,.48,size=27)
        eq(c,root,kind,r"R(-m\omega)R(n\omega)=R((n-m)\omega)",.88,4.25,5.35,.46,size=25)
        c.text(.85,4.91,5.35,.33,"固定未旋转 q/k；只讨论位置变化。",18,MUTED)
        c.text(x,y,5.5,.34,"二维算例：q=k=[1,0]，ω=1",20,INK)
        c.text(x,y+.44,5.5,.35,"m=1，n=3 → 点积 cos(2)",21,BLUE)
        c.text(x,y+.86,5.5,.36,f"数值 ≈ {worked['position']['score']:.4f}",24,GREEN,bold=True)
        c.text(x,y+1.28,5.5,.33,"m=6，n=8：差仍为2，点积相同。",18,MUTED)
    elif kind=="vector_rope":
        c.text(x,y,5.5,.34,"一对分量：[1,0]，旋转角 θ=1",20,INK)
        c.text(x,y+.46,5.5,.38,"→ "+vector(worked["position"]["q_rotated"]),24,BLUE,bold=True)
        prose(c,x,y+.98,5.45,["平方和仍为1，所以范数保持。实现按相邻维度配对；split-half 需要匹配的坐标排列。"],18)
    elif kind=="mask":
        from fractions import Fraction
        for i,row in enumerate(worked["mask"]["weights"]):
            for j,value in enumerate(row):
                text=str(Fraction(value).limit_denominator(12))
                c.text(x+.1+j*1.13,y+i*.29,1.0,.30,text,18,BLUE,mono=True,center=True)
        c.text(x,5.02,5.5,.30,"pad query 的输出另行清零。",18,MUTED)
    else:
        raise ValueError(kind)
    code_band(c,CODE[kind])


def worked_bpe(c,root,worked,training):
    d=worked["bpe"]
    if training:
        panel(c,.60,2.08,5.50,2.55,"① 语料与权重：不是每句话只算一次")
        for i,(text,n) in enumerate(d["corpus"].items()):
            c.text(.86,2.65+i*.45,4.95,.38,f"{text}  × {n}",23,INK)
        c.text(.85,4.12,4.98,.34,"喜+欢：5+3+2=10；我+喜：5+2=7",18,BLUE)
        panel(c,6.30,2.08,6.40,2.55,"② 每次合并后重算；同频按首次出现处理")
        for i,row in enumerate(d["history"]):
            pair=" + ".join(row["pair"])
            c.text(6.55,2.62+i*.43,5.92,.37,f"rank {i}：{pair}   频次 {row['count']}",20,INK)
        c.rect(.60,4.80,12.10,.55,PALE)
        c.text(.83,4.90,11.6,.38,"先从字符出发，学出“喜欢喝奶茶”；这是高频片段，不是人工指定的语言学词语。",20,BLUE)
        code_band(c,["counts[pair] += corpus[text]  # 当前相邻 pair 出现一次，加一次句子频次","pair = max(counts, key=counts.get)  # 合并后重新统计"])
    else:
        panel(c,.60,2.08,4.26,2.76,"① 冻结规则：不再计算语料频次")
        for i,pair in enumerate(d["merges"]):
            c.text(.85,2.63+i*.43,3.82,.37,f"{i}：{' + '.join(pair)}",21,INK)
        panel(c,5.06,2.08,7.64,2.76,"② 编码“你喜欢喝奶茶”：按最小可用 rank")
        c.text(5.30,2.58,7.1,.36,"你 / 喜 / 欢 / 喝 / 奶 / 茶",21,MUTED)
        for i,row in enumerate(d["trace"]):
            c.text(5.30,3.02+i*.40,7.1,.35,f"{i} → "+" / ".join(row["after"]),21,BLUE)
        c.text(.84,5.02,11.6,.34,"重叠反例：bc 的 rank=0、ab 的 rank=1；abc 先合并 bc → a/bc，而非 ab/c。",18,INK)
        code_band(c,["rank, pair = min(candidates, key=lambda item: item[0])","parts = merge_pair(parts, pair)  # 无候选时停止；不按最长片段选"])


def concept_slide(c,item,worked):
    kind=item["kind"]
    if kind=="opening":
        architecture_review(c)
    elif kind=="lesson_scope":
        lesson_scope(c)
    elif kind=="hf_pipeline":
        panel(c,.60,2.08,5.85,3.25,"合并规则以外，还要固定哪些步骤？")
        prose(c,.84,2.66,5.37,item["explanation"],18)
        c.text(.84,5.40,5.37,.69,"基础字节负责覆盖；预切分规定边界；merges 决定合并次序。",19,BLUE)
        panel(c,6.65,2.08,6.05,3.25,"API位置示意；完整参数见实验3")
        lines=["Tokenizer(models.BPE(...))","pre_tokenizers.ByteLevel(...)","trainers.BpeTrainer(...)","tokenizer.train_from_iterator(...)","tokenizer.save(...)  # 完整 JSON"]
        for i,line in enumerate(lines):
            c.text(6.89,2.67+i*.46,5.60,.34,line,18,INK,mono=True)
        code_band(c,["tokenizer = load_tokenizer()  # 加载冻结的完整配置","enc = tokenizer.encode(text)  # 查看 tokens、ids、offsets"])
    elif kind=="artifact":
        panel(c,.60,2.08,12.10,1.25,"真实配置：普通文本的 ID 只对当前词表有意义")
        c.text(.90,2.65,11.5,.40,"我喜欢喝牛奶。 → [300,304,262] → E[300]、E[304]、E[262]",23,BLUE,bold=True)
        panel(c,.60,3.53,5.85,1.80,"地址相同、含义不同：假设两份词表")
        c.text(.84,4.09,5.35,.40,"词表 A：奶茶 → 7",23,INK)
        c.text(.84,4.59,5.35,.40,"词表 B：咖啡 → 7",23,TEAL)
        panel(c,6.65,3.53,6.05,1.80,"只换词表会怎样？")
        prose(c,6.89,4.09,5.57,["模型原先把 E[7] 当作奶茶的表示。只替换词表后，咖啡会读到原来的参数行。","这不是向量算错，而是 ID 与参数地址约定失配。"],18)
        c.text(.86,5.70,11.5,.38,"默认特殊符号：pad=0，unk=1，bos=2，eos=3；本课显式添加 bos/eos。",20,MUTED)
    elif kind=="lookup":
        panel(c,.60,2.08,12.10,2.0,"B=2，T=6：t₁…t₄ 是普通 token；起止标记也占位置")
        for r,labels in enumerate((("bos","t₁","t₂","t₃","eos","pad"),("bos","t₁","t₂","t₃","t₄","eos"))):
            y=2.69+r*.72
            c.text(.86,y+.12,1.1,.37,f"句{r+1}",22,INK)
            for j,t in enumerate(labels):
                token(c,2.0+j*1.68,y,t,1.42,fill="E2E8F0" if t=="pad" else PALE,size=23)
        panel(c,.60,4.27,12.10,1.98,"三个接口分别负责什么？")
        for i,(a,b) in enumerate((("padding_idx","抑制 pad 行的 lookup 梯度"),("padding key mask","Attention 不读取 pad key"),("ignore_index","损失不计算 pad 标签"))):
            yy=4.80+i*.40
            c.text(.87,yy,4.0,.33,a,20,BLUE)
            c.text(5.1,yy,7.1,.33,b,20,INK)
    elif kind=="pipeline":
        panel(c,.60,2.08,12.10,1.12,"从本地 ID 到多头注意力输出：每个轴都有含义")
        c.text(.85,2.63,11.55,.38,"[2,6] → [2,6,8] → Q/K [2,2,6,4] → scores [2,2,6,6] → [2,6,8]",23,BLUE,bold=True)
        panel(c,.60,3.40,12.10,2.85,"对应实现与每步目的")
        rows=[("1  查表","embedding(ids)","离散地址 → D维参数"),("2  投影、拆头","Linear → view → transpose","D=8 拆成 H=2、d_h=4"),("3  位置、遮罩","apply_rope → masked_fill","位置进入Q/K，mask限制key"),("4  混合、合头","softmax → @v → reshape","沿key归一化，再合并各head")]
        for i,(a,b,d) in enumerate(rows):
            y=3.98+i*.46
            c.text(.85,y,2.45,.35,a,20,BLUE)
            c.text(3.4,y,4.8,.35,b,19,INK)
            c.text(8.45,y,3.95,.35,d,18,MUTED)
    elif kind=="quiz":
        panel(c,.60,2.08,12.10,4.17,"先独立回答，再用下面的机制核对")
        rows=[("编码","为什么abc不是最长前缀ab/c？","可合并的当前相邻候选中，选最小rank。"),("梯度","同一ID出现三次，第j行怎样更新？","把三个位置的上游向量梯度相加。"),("位置","RoPE已有相对位移，mask还做什么？","位置参与打分，mask限制哪些key可读。")]
        for i,(a,q,answer) in enumerate(rows):
            y=2.68+i*1.08
            c.text(.86,y,1.4,.43,a,25,BLUE,bold=True)
            c.text(2.50,y,9.7,.42,q,23,INK)
            c.text(2.50,y+.48,9.7,.35,"核对："+answer,20,MUTED)
    else:
        raise ValueError(kind)


def architecture_review(c):
    panel(c,.60,2.08,5.45,4.16,"生成数据流：每轮预测一个新 token")
    rows=["文本 → Tokenizer → IDs [B,T]",
          "Embedding → X [B,T,D]",
          "Decoder Block × L → H [B,T,D]",
          "Norm + LM Head → logits [B,T,V]",
          "最后位置：Softmax → 选下一 token"]
    for i,text in enumerate(rows):
        y=2.65+i*.64
        fill=PALE if i!=2 else "DFF0ED"
        c.rect(.98,y,4.76,.48,fill)
        c.text(1.07,y+.08,4.58,.35,text,18,BLUE if i!=2 else GREEN,center=True)
        if i<4:
            c.line(3.36,y+.49,3.36,y+.62,arrow=True)
    # 将采样结果接回前文；回路留在节点左侧，避免跨过文字。
    c.line(3.36,5.70,3.36,5.95)
    c.line(3.36,5.95,.79,5.95)
    c.line(.79,5.95,.79,2.57)
    c.line(.79,2.57,5.12,2.57)
    c.line(5.12,2.57,5.12,2.63,arrow=True)
    c.text(1.17,6.00,4.35,.31,"生成的 ID 接回序列，继续下一轮。",18,MUTED)
    panel(c,6.25,2.08,6.45,4.16,"单层 Decoder Block 展开（Pre-Norm）")
    boxes=["归一化 Norm", "因果 MHA：Q/K 用 RoPE", "+ 输入 x（残差）",
           "归一化 Norm", "FFN / MLP：逐位置加工", "+ 中间 h（残差）"]
    for i,text in enumerate(boxes):
        y=2.70+i*.50
        color="DFF0ED" if i==1 else PALE
        c.rect(8.00,y,3.62,.42,color)
        c.text(8.08,y+.045,3.46,.34,text,18,GREEN if i==1 else INK,center=True)
        if i<5:
            c.line(9.81,y+.425,9.81,y+.49,arrow=True)
    # x -> Norm -> Attention -> Add(x)，跳过的仍是同一个输入 x。
    c.line(9.81,2.51,9.81,2.68,arrow=True)
    c.line(9.81,2.56,7.22,2.56,TEAL)
    c.line(7.22,2.56,7.22,3.91,TEAL)
    c.line(7.22,3.91,7.98,3.91,TEAL,arrow=True)
    c.text(6.48,3.22,.62,.34,"x",20,TEAL,center=True)
    # h = x + Attention(Norm(x))；h 同时进入 FFN 路径与第二条残差。
    c.line(9.81,4.16,12.15,4.16,TEAL)
    c.line(12.15,4.16,12.15,5.41,TEAL)
    c.line(12.15,5.41,11.64,5.41,TEAL,arrow=True)
    c.text(12.23,4.35,.34,.34,"h",18,TEAL,center=True)
    c.line(9.81,5.63,9.81,5.82,arrow=True)
    c.text(7.75,5.91,4.15,.33,"输出仍为 [B,T,D]，再送下一层。",18,MUTED,center=True)
    c.text(.80,6.37,11.70,.30,"B：句子数；T：token数；D：每个token的向量维度；V：词表大小。",18,MUTED)


def lesson_scope(c):
    panel(c,.60,2.08,12.10,1.25,"上节课的人工算例：“我 / 喜欢 / 喝” → 下一 token")
    c.text(.85,2.62,11.58,.35,"Q/K 匹配 → Softmax 权重 ≈ [0.248,0.248,0.503] → 加权混合 V",21,INK)
    c.text(.85,3.02,11.58,.31,"混合向量 ≈ [0.752,0.752] → 输出映射 → 酒 / 牛奶的概率 ≈ [0.680,0.320]",19,BLUE)
    cards=[("01  Tokenizer","文本 → IDs [B,T]","上节课：按空格切分。","今天：BPE 怎样学规则、怎样执行？"),
           ("02  Embedding","ID → 向量 [B,T,D]","上节课：人工二维向量。","今天：怎样查表、怎样通过训练更新？"),
           ("03  位置编码","位置 t → 注意力的位置关系","上节课：认识顺序的作用。","今天：输入加 PE，或在 Q/K 上用 RoPE。")]
    for i,(title,interface,before,today) in enumerate(cards):
        x=.60+i*4.10
        panel(c,x,3.55,3.90,2.40,title)
        c.text(x+.21,4.10,3.48,.66,interface,20,BLUE,bold=True)
        prose(c,x+.21,4.83,3.48,[before,today],18,gap=.06)
    c.text(.85,6.08,11.55,.31,"位置的两条路径是不同方案；上节课的数值来自人工参数，今天逐步补全真实机制。",18,MUTED)


def draw_detail(c,item,link,root,report,worked):
    kind=item["kind"]
    if kind in ("bpe_train","rank"):
        worked_bpe(c,root,worked,kind=="bpe_train")
    elif kind in KEY_FORMULAS or kind=="rotation":
        formula_detail(c,item,link,root,report,worked)
    else:
        concept_slide(c,item,worked)
