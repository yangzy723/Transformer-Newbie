"""第二课投影版：大公式、可读代码和解释机制的原生图形。"""
from PIL import ImageFont
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from formulas import place_formula

INK, MUTED, BLUE, TEAL = "14263D", "475569", "2458C6", "087F8C"
BG, WHITE, PALE, LINE = "F4F7FB", "FFFFFF", "E8F0FC", "CBD5E1"
GREEN, RED, DARK = "087568", "B6453B", "14263D"

# 标题表达本页唯一的问题，细节进入图示和讲者备注。
DESIGN = {
    "opening": ("从文本到模型输入", "第二课  ·  Tokenizer、Embedding 与位置机制", "今天沿同一条链路，解释规则、参数和顺序。"),
    "budget": ("分得更细，还是词表更大？", "词表大小 V 与序列长度 T，会影响不同部分的计算成本。", "选择切分粒度，也是在选择参数与计算的开销。"),
    "bpe_train": ("BPE：先统计，再合并", "每轮重新统计当前相邻片段，选择最高频的一对。", "频次来自语料；合并后要重新统计。"),
    "rank": ("BPE 编码：谁的 rank 更小？", "规则已经冻结；每次在当前候选中选择优先级最高的一对。", "merge rank 决定合并顺序，token ID 决定查表地址。"),
    "hf_pipeline": ("把 BPE 接到真实文本上", "ByteLevel 处理字节与边界；BPE 处理合并规则。", "先训练并保存配置，再加载同一配置进行编码。"),
    "byte": ("同一句话，为什么 token 数不同？", "真实对照：同一份语料，分别训练 320 项与 384 项词表。", "词表变大不保证每句话都缩短；中文样例均为 3 项。"),
    "artifact": ("ID 必须与参数表配套", "同样大小的词表，也可能把同一片段映射到不同的地址。", "更换 tokenizer 时，必须核对 ID 与模型参数的对应关系。"),
    "onehot": ("Embedding：一个 ID 选择一行", "查表与 one-hot 乘法前向等价；实际实现直接选行。", "ID 是地址；可训练的表示存放在 E 的每一行。"),
    "lookup": ("不同长度，怎样组成 batch？", "先添加起止标记，再补齐；pad 只是占位。", "padding_idx、注意力 mask 和损失忽略，需要分别处理。"),
    "grad": ("读了同一行，梯度就要相加", "共享参数接收所有读取位置的贡献；这里先看 lookup 路径。", "反向是 scatter-add：累加贡献，而不是覆盖旧值。"),
    "chain": ("预测误差怎样传回向量？", "单位置交叉熵：先得到类别误差，再乘回输出权重。", "先算每次出现的向量梯度，再按 ID 汇总到参数表。"),
    "training": ("让 Embedding 真正学一次", "57 对相邻 token；Embedding → Linear → CrossEntropy。", "这是只看当前 token 的 bigram；没有 Attention。"),
    "tie": ("一张表，两条梯度路径", "权重绑定：输入查表和输出分类器共享同一个 E。", "绑定后，没作为输入的行也可能收到输出梯度。"),
    "permutation": ("没有位置，重排会怎样？", "假设：没有位置机制，也没有因果 mask。", "等变表示“输出跟着重排”，不是“输出完全不变”。"),
    "absolute": ("把位置向量加进输入", "内容与位置使用同样的维度，相加后形状保持不变。", "位置是 token 序号，不是原文的字符 offset。"),
    "frequencies": ("位置 × 频率，得到角度", "不同维度对使用不同频率，再交错存放 sin 与 cos。", "positions → angles → sin / cos：每一步都有明确形状。"),
    "rotation": ("RoPE：相对位移从哪里来？", "q、k 分别按绝对位置旋转；点积里出现位置之差。", "转置是反向旋转；角度相加后，剩下 n−m。"),
    "vector_rope": ("把二维旋转写成两行代码", "相邻维度组成一对；同一公式广播到 batch 与 head。", "保持向量范数；点积仍会随相对位置改变。"),
    "mask": ("位置关系与可读范围", "RoPE 改变分数；mask 决定哪些 key 可以被读取。", "有了位置机制，仍要屏蔽未来 key 与 padding key。"),
    "pipeline": ("沿真实输入，把形状接起来", "本地 tokenizer → Embedding → 多头 Q/K → RoPE + mask。", "这是随机参数的接口实验，尚未训练完整 Transformer。"),
    "quiz": ("用三个问题，检查是否理解", "解释一次编码、一次反传、一次位置计算。", "课后：改词表、查梯度、验证旋转，并保留实验结果。"),
}


def canvas_class(base, font_path):
    class ProjectionCanvas(base):
        def __init__(self, slide):
            super().__init__(slide)
            self.rect(0, 0, 13.333333, 7.5, BG, False)
            self.text_metrics = []

        def text(self, x, y, w, h, content, size=22, color=INK, bold=False,
                 center=False, mono=False):
            name = "Menlo" if mono else "Arial Unicode MS"
            path = "/System/Library/Fonts/Menlo.ttc" if mono else font_path
            try:
                font = ImageFont.truetype(path, round(size*120/72))
            except OSError:
                font = ImageFont.truetype("DejaVuSansMono.ttf" if mono else font_path, round(size*120/72))
            lines = []
            for original in str(content).split("\n"):
                current = ""
                for char in original:
                    if self.draw.textlength(current+char, font=font) > w*120-5:
                        if mono:
                            raise ValueError(f"代码必须显式换行：{original}")
                        lines.append(current)
                        current = char
                    else:
                        current += char
                lines.append(current)
            step = size*120/72*1.22
            if len(lines) == 1:
                h = max(h, size/72*1.22)
            if len(lines)*step > h*120+2:
                raise ValueError(f"文本超出高度（{size} pt）：{content!r}")
            box = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            tf = box.text_frame
            tf.clear()
            tf.word_wrap = False
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            for i, line in enumerate(lines):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = line
                p.font.name = name
                p.font.size = Pt(size)
                p.font.bold = bold
                p.font.color.rgb = RGBColor.from_string(color)
                p.line_spacing = Pt(size*1.22)
                p.space_before = p.space_after = Pt(0)
                if center:
                    p.alignment = PP_ALIGN.CENTER
                left = x*120+(w*120-self.draw.textlength(line, font=font))/2 if center else x*120
                self.draw.text((round(left), round(y*120+i*step)), line,
                               font=font, fill="#"+color, stroke_width=1 if bold else 0)
            self.text_metrics.append({"text": str(content), "size": size, "mono": mono,
                                      "lines": len(lines), "box": [x,y,w,h]})

    return ProjectionCanvas


def card(c, x, y, w, h, title, value, detail, color=BLUE):
    c.rect(x,y,w,h,WHITE)
    c.rect(x,y,.055,h,color,False)
    c.text(x+.23,y+.25,w-.46,.4,title,22,MUTED)
    c.text(x+.23,y+.91,w-.46,.75,value,38,color,bold=True)
    c.text(x+.23,y+h-.83,w-.46,.7,detail,20,INK)


def token(c, x, y, text, w=.8, fill=PALE, color=BLUE, size=24):
    c.rect(x,y,w,.58,fill)
    c.text(x+.06,y+.07,w-.12,.48,text,size,color,center=True)


def header(c, item, number, start, end, total):
    kind = item["kind"]
    title, subtitle, takeaway = DESIGN[kind]
    active = 0 if number <= 7 else 1 if number <= 13 else 2
    for i, text in enumerate(("01  分词", "02  向量", "03  位置")):
        x = .60+i*1.52
        c.rect(x,.27,1.34,.34,BLUE if i == active else "E6ECF4")
        c.text(x,.31,1.34,.26,text,14,WHITE if i == active else MUTED,center=True)
    c.text(10.0,.30,2.7,.28,f"{start:02d}–{end:02d} min  /  60",14,MUTED,center=True)
    c.text(.60,.84,12.10,.66,title,34,INK,bold=True)
    c.text(.62,1.54,12.0,.39,subtitle,20,MUTED)
    c.rect(.60,6.70,12.10,.47,PALE)
    c.text(.79,6.78,11.7,.36,takeaway,19,BLUE,bold=True)
    c.text(.62,7.27,8.8,.21,"TRANSFORMER NEWBIE  ·  CLASS 2",11,MUTED)
    c.text(11.5,7.25,1.2,.24,f"{number:02d} / {total:02d}",12,MUTED,center=True)


def reference(c, link):
    if not link:
        return
    experiment = f"实验 {link['experiment']:02d}" if link["experiment"] else "课后扩展"
    file = link["file"].replace("../tools/", "tools/")
    c.text(.65,6.36,12,.27,f"{experiment}   ·   {file} / {link['function']}",14,MUTED)


def code_lines(kind, link):
    if kind == "rank":
        return ["rank, pair = min(", "    candidates, key=lambda item: item[0]", ")", "parts = merge_pair(parts, pair)"]
    if kind == "mask":
        return ["masked = scores.masked_fill(", "    forbidden, -torch.inf", ")", "weights = masked.softmax(dim=-1)"]
    if kind == "tie":
        return ["head.weight = tied_embedding.weight"]
    if kind == "absolute":
        return ["pe = sinusoidal(torch.arange(3), 4)", "absolute_original = attention(x + pe)"]
    if kind == "frequencies":
        return ["angles = positions[:,None] * frequencies[None,:]",
                "result[:, 0::2] = angles.sin()", "result[:, 1::2] = angles.cos()"]
    if kind == "vector_rope":
        return ["even, odd = x[..., 0::2], x[..., 1::2]",
                "angles = positions[:,None] * frequencies[None,:]"]
    return link.get("code", [])


def formula_slide(c, item, link, root, report):
    kind = item["kind"]
    c.rect(.60,2.08,12.10,2.00,WHITE)
    formulas = link["formulas"]
    if kind == "rotation":
        slots = [(.83,2.20,4.35,1.20),(5.40,2.23,6.98,1.17)]
    elif kind == "grad":
        slots = [(.90,2.20,3.25,1.20),(4.40,2.20,7.85,1.20)]
    else:
        height = 1.40/len(formulas)
        slots = [(.92,2.17+i*height,11.46,height-.035) for i in range(len(formulas))]
    for i, (expression, bounds) in enumerate(zip(formulas, slots),1):
        place_formula(c, expression, root/f"assets/formulas/{kind}_{i}.png", *bounds, font_size=34)
    c.text(.88,3.68,11.5,.35,link["symbols"],18,MUTED,center=True)
    c.rect(.60,4.30,8.0,1.96,DARK)
    c.text(.86,4.43,7.5,.29,"PYTHON  /  对应实现" if kind not in ("tie", "absolute") else "PYTHON  /  补充接口",15,"8BCFD2",bold=True)
    lines = code_lines(kind,link)
    for i,line in enumerate(lines):
        c.text(.86,4.83+i*.30,7.50,.32,line,18,WHITE,mono=True)
    c.rect(8.80,4.30,3.90,1.96,WHITE)
    evidence(c,kind,report)


def evidence(c, kind, report):
    x,y = 9.02,4.41
    if kind == "bpe_train":
        c.text(x,y,3.4,.32,"“喜 + 欢”的首轮频次",18,MUTED)
        c.text(x,y+.58,3.4,.6,"5 + 3 + 2 = 10",26,BLUE,bold=True)
        c.text(x,y+1.19,3.4,.36,"四轮：10 → 10 → 8 → 8",18,INK)
    elif kind == "rank":
        c.text(x,y,3.4,.33,"bc 的 rank 0 更优先",20,MUTED)
        for i,t in enumerate(("a","b","c")):
            token(c,x+i*.88,y+.45,t,.70)
        c.line(x+1.05,y+1.05,x+1.05,y+1.16,arrow=True)
        token(c,x+.1,y+1.21,"a",.70)
        token(c,x+1.0,y+1.21,"bc",1.1,fill="D8F0EA",color=GREEN)
    elif kind == "onehot":
        c.text(x,y,3.4,.35,"前向等价检查",20,MUTED)
        c.text(x,y+.52,3.4,.63,"True",34,GREEN,bold=True)
        c.text(x,y+1.24,3.4,.35,"[2,6] → [2,6,8]",21,INK)
    elif kind == "grad":
        c.text(x,y,3.4,.35,"[a, a, b, pad] 求和",20,MUTED)
        for i,(k,v,color) in enumerate((("a","2",BLUE),("b","1",TEAL),("pad","0",MUTED))):
            c.text(x+i*1.10,y+.52,1.0,.66,v,36,color,center=True,bold=True)
            c.text(x+i*1.10,y+1.23,1.0,.35,k,20,MUTED,center=True)
    elif kind == "chain":
        c.text(x,y,3.4,.35,"同一个输入向量梯度",20,MUTED)
        c.text(x,y+.55,3.4,.55,"手算 = autograd",25,GREEN,bold=True)
        c.text(x,y+1.21,3.4,.35,"单位置；不含 batch 平均",18,INK)
    elif kind == "training":
        history = report["history"]
        c.text(x,y,3.4,.34,"训练 loss / 120 步",20,MUTED)
        c.text(x,y+.43,3.5,.50,"5.7565 → 0.3559",25,BLUE,bold=True)
        for a,b in zip(history,history[1:]):
            c.line(x+.15+2.9*a["step"]/120,y+1.45-.40*a["loss"]/6,
                   x+.15+2.9*b["step"]/120,y+1.45-.40*b["loss"]/6,BLUE)
        c.text(x,y+1.53,3.4,.28,"26 个输入表行发生更新",16,MUTED)
    elif kind == "tie":
        c.text(x,y,3.4,.34,"两条路径指向同一个 E",20,MUTED)
        token(c,x,y+.55,"lookup",1.45,size=20)
        token(c,x+1.70,y+.55,"output",1.45,size=20)
        c.line(x+.73,y+1.18,x+1.55,y+1.43,arrow=True)
        c.line(x+2.43,y+1.18,x+1.77,y+1.43,arrow=True)
        c.text(x+1.15,y+1.42,1.1,.38,"E",25,BLUE,center=True,bold=True)
    elif kind == "permutation":
        c.text(x,y,3.4,.35,"输入重排 → 输出重排",20,MUTED)
        c.text(x,y+.55,3.4,.5,"A B C → C A B",26,BLUE,bold=True)
        c.text(x,y+1.2,3.4,.38,"无位置时，数值验证为 True",18,INK)
    elif kind == "absolute":
        c.text(x,y,3.4,.35,"每个位置加一个向量",20,MUTED)
        c.text(x,y+.55,3.4,.5,"内容 + 位置",28,BLUE,bold=True)
        c.text(x,y+1.2,3.4,.4,"两者都是 D 维",22,INK)
    elif kind == "frequencies":
        c.text(x,y,3.4,.35,"D = 4：两组频率",20,MUTED)
        c.text(x,y+.50,3.4,.55,"1       0.01",30,BLUE,bold=True)
        c.text(x,y+1.20,3.4,.4,"t=0 → [0,1,0,1]",22,INK)
    elif kind == "rotation":
        c.text(x,y,3.4,.35,"固定未旋转的 q / k",20,MUTED)
        c.text(x,y+.55,3.4,.53,"n − m",34,BLUE,center=True,bold=True)
        c.text(x,y+1.22,3.4,.36,"点积由相对位移调节",20,INK)
    elif kind == "vector_rope":
        c.text(x,y,3.4,.35,"两项数值检查",20,MUTED)
        c.text(x,y+.52,3.4,.43,"向量范数保持",25,GREEN,bold=True)
        c.text(x,y+1.16,3.4,.4,"平移误差约 10⁻¹⁵",20,INK)
    elif kind == "mask":
        c.text(x,y,3.4,.33,"因果 mask 示意：T=4",18,MUTED)
        for row in range(4):
            for col in range(4):
                c.rect(x+.06+col*.32,y+.44+row*.29,.27,.24,BLUE if col<=row else "F1D9D5",False)
        c.text(x+1.65,y+.58,1.75,.8,"蓝：可读\n红：未来",19,INK)
    else:
        raise ValueError(kind)


def large_slide(c,item):
    kind = item["kind"]
    if kind == "opening":
        specs = [("01","先切片段","Tokenizer","文本 → ID",BLUE),
                 ("02","再学表示","Embedding","ID → 向量",TEAL),
                 ("03","最后加顺序","位置机制","Q/K → 位置关系",BLUE)]
        for i,(n,title,english,detail,color) in enumerate(specs):
            x=.6+i*4.10
            c.rect(x,2.35,3.9,2.67,WHITE)
            c.text(x+.25,2.58,3.3,.48,n,28,color,bold=True)
            c.text(x+.25,3.22,3.4,.52,title,30,INK,bold=True)
            c.text(x+.25,3.89,3.4,.4,english,24,color)
            c.text(x+.25,4.48,3.4,.36,detail,20,MUTED)
        c.text(.80,5.51,11.7,.48,"同一句话：我喜欢喝牛奶。",28,INK,center=True)
    elif kind == "budget":
        card(c,.6,2.30,3.9,3.5,"词表参数","V × D","V 越大，参数表越大。")
        card(c,4.7,2.30,3.9,3.5,"稠密分数矩阵","T × T","T 越大，分数矩阵越大。",TEAL)
        card(c,8.8,2.30,3.9,3.5,"输出投影计算","T × D × V","整个模型的成本不只有 T²。")
        c.text(.85,6.02,11.5,.34,"单份 E：32768 × 4096 × 2 字节 = 256 MiB（FP16）",21,MUTED,center=True)
    elif kind == "hf_pipeline":
        labels=[("本地语料","28 行中英文"),("ByteLevel","字节与边界"),("BPE 训练","学习 merges"),("保存 JSON","词表与配置"),("编码文本","tokens / IDs")]
        for i,(a,b) in enumerate(labels):
            x=.6+i*2.46
            c.rect(x,2.65,2.22,1.70,WHITE)
            c.text(x+.1,3.0,2.02,.43,a,24,BLUE,center=True,bold=True)
            c.text(x+.1,3.63,2.02,.37,b,20,MUTED,center=True)
            if i<4:
                c.line(x+2.24,3.47,x+2.42,3.47,arrow=True)
        c.rect(.6,4.84,5.90,1.13,PALE)
        c.rect(6.8,4.84,5.90,1.13,"E1F2EE")
        c.text(.85,5.13,5.40,.7,"256 种基础字节 + 4 个特殊符号",22,BLUE)
        c.text(7.05,5.13,5.40,.7,"分词器训练：统计与合并规则",22,GREEN)
    elif kind == "byte":
        c.text(.9,2.35,11.5,.65,"Hello world!",38,INK,bold=True)
        for y,vocab,n,color in ((3.50,320,9,BLUE),(4.63,384,5,TEAL)):
            c.text(.9,y+.12,2.15,.47,f"V = {vocab}",26,INK)
            c.rect(3.14,y,7.4*n/9,.70,color)
            c.text(3.14+7.4*n/9+.26,y+.04,1.60,.65,str(n),34,color,bold=True)
        c.text(.92,5.78,11.4,.42,"同时观察：前导空格、标点、未见字符与整串回解。",23,MUTED)
    elif kind == "artifact":
        c.rect(.6,2.40,5.35,3.50,WHITE)
        c.text(.9,2.70,4.7,.40,"冻结的 tokenizer：V = 320",22,MUTED)
        c.text(.9,3.38,4.7,.52,"我喜欢喝牛奶。",31,INK,bold=True)
        for i,v in enumerate(("300","304","262")):
            token(c,.95+i*1.50,4.42,v,1.24,size=27)
        c.line(6.08,4.22,6.63,4.22,arrow=True)
        c.rect(6.8,2.40,5.90,3.50,WHITE)
        c.text(7.1,2.70,5.2,.40,"配套参数表 E：[320,D]",25,BLUE,bold=True)
        for i,(row,name) in enumerate((("300","E[300]"),("304","E[304]"),("262","E[262]"))):
            yy=3.44+i*.65
            c.text(7.15,yy,1.1,.42,row,24,MUTED)
            c.rect(8.35,yy-.02,3.8,.52,PALE)
            c.text(8.48,yy+.05,3.5,.38,name+"  …",24,BLUE)
        c.text(.9,6.04,11.4,.34,"特殊 ID：pad=0  ·  unk=1  ·  bos=2  ·  eos=3",21,MUTED,center=True)
    elif kind == "lookup":
        c.rect(.6,2.40,12.1,3.58,WHITE)
        for row,labels in enumerate((("bos","t₁","t₂","t₃","eos","pad"),("bos","t₁","t₂","t₃","t₄","eos"))):
            y=3.10+row*1.20
            c.text(.92,y+.14,1.2,.42,f"句 {row+1}",25,INK)
            for col,text in enumerate(labels):
                token(c,2.35+col*1.64,y,text,1.40,fill="E2E8F0" if text=="pad" else PALE,
                      color=MUTED if text=="pad" else BLUE,size=25)
            c.text(2.35,y+.70,9.7,.35,"5 个有效 token + 1 个 pad" if row==0 else "6 个有效 token",21,MUTED)
        c.text(.91,2.58,11.3,.39,"B = 2，T = 6；t₁…t₄ 表示普通 token",23,BLUE)
    elif kind == "pipeline":
        steps=[("ID","[2,6]"),("Embedding","[2,6,8]"),("Q / K","[2,2,6,4]"),("scores","[2,2,6,6]"),("输出","[2,6,8]")]
        for i,(name,shape) in enumerate(steps):
            x=.6+i*2.46
            c.rect(x,2.72,2.22,1.85,WHITE)
            c.text(x+.08,3.02,2.06,.42,name,24,BLUE,center=True,bold=True)
            c.text(x+.08,3.75,2.06,.47,shape,24,INK,center=True)
            if i<4:
                c.line(x+2.24,3.65,x+2.42,3.65,arrow=True)
        c.text(.95,5.10,11.25,.50,"RoPE 作用于 Q/K；mask 在 scores 上限制可读 key。",25,INK,center=True)
        c.text(.95,5.82,11.25,.35,"B=2  ·  T=6  ·  D=8  ·  H=2  ·  d_h=4",22,MUTED,center=True)
    elif kind == "quiz":
        questions=[("01","编码","为什么 abc 会得到 a / bc？"),
                   ("02","梯度","同一 token 出现三次，表行梯度怎么算？"),
                   ("03","位置","RoPE 已有位置信息，为什么还需要 mask？")]
        for i,(n,topic,question) in enumerate(questions):
            y=2.30+i*1.22
            c.rect(.6,y,12.10,1.02,WHITE)
            c.text(.87,y+.24,.8,.49,n,28,BLUE,bold=True)
            c.text(1.90,y+.29,1.4,.43,topic,24,MUTED)
            c.text(3.40,y+.29,8.90,.48,question,25,INK)
    else:
        raise ValueError(kind)
