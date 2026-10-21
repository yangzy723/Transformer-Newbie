"""第二课机制图示。"""
TEAL, LIGHT, BLUE, MUTED = "087F8C", "E6EFEE", "DEE8F0", "5C6F7B"


def flow(c, labels, size=18, height=.55, gap=.22, y=2.2):
    for i, label in enumerate(labels):
        top = y + i * (height + gap)
        c.node(7.5, top, 4.8, height, label, LIGHT if i % 2 == 0 else BLUE, size)
        if i + 1 < len(labels):
            c.line(9.9, top + height + .01, 9.9, top + height + gap - .01, arrow=True)


def draw(c, kind):
    if kind == "opening":
        flow(c, ["离散规则：BPE 的统计与 rank", "可训练参数：Embedding 的反传", "数值结构：位置编码与旋转"], size=19, y=2.6, height=.65, gap=.3)
    elif kind == "budget":
        flow(c, ["E：[V,D] → V×D 个参数", "Attention scores：T×T", "输出投影：T×D×V", "同一段文本的 V / T 取舍"], size=20)
    elif kind == "bpe_train":
        for i, text in enumerate(("第0轮：喜 + 欢，频次10", "第1轮：喜欢 + 喝，频次10", "第2轮：喜欢喝 + 牛，频次8", "第3轮：喜欢喝牛 + 奶，频次8")):
            c.node(7.4, 2.5 + i*.76, 4.95, .6, text, BLUE, 18)
        c.text(7.4, 5.65, 5, .25, "每轮在当前片段上重算；同频规则固定", 13, MUTED)
    elif kind == "rank":
        flow(c, ["输入：a / b / c", "候选：ab rank1；bc rank0", "先选 bc：a / bc", "最长前缀 ab / c 在此例不正确"], size=19)
    elif kind == "hf_pipeline":
        flow(c, ["文本 + ByteLevel 预切分", "256 基础字节 + 4 特殊符号", "BPE trainer → 320 项词表", "保存完整 tokenizer.json", "加载 → encode / decode"], size=17, y=2.05, height=.49, gap=.2)
    elif kind == "byte":
        c.text(7.4, 2.2, 5, .4, "Hello world!：本次实际编码", 19, TEAL)
        c.node(7.4, 3.1, 4.95, .65, "词表 320 → 9 个 token", BLUE, 22)
        c.node(7.4, 4.15, 4.95, .65, "词表 384 → 5 个 token", LIGHT, 22)
        c.text(7.4, 5.35, 4.95, .4, "中文样例均为3项，不保证每句都缩短", 16, MUTED)
    elif kind == "artifact":
        flow(c, ["我喜欢喝牛奶。", "冻结配置 V=320", "ID：[300, 304, 262]", "与该配置配套的 E：[320,D]"], size=19)
    elif kind == "onehot":
        flow(c, ["I：[2,6]", "one-hot(I)：[2,6,320]", "@ E：[320,8]", "结果 = gather(E,I)：[2,6,8]"], size=18)
    elif kind == "lookup":
        c.text(7.4, 2.2, 5, .4, "B=2，T=6，pad=0", 21, TEAL)
        c.node(7.4, 3.0, 4.95, .65, "句1：bos + 3 tokens + eos + pad", BLUE, 17)
        c.node(7.4, 4.1, 4.95, .65, "句2：bos + 4 tokens + eos", LIGHT, 18)
        c.text(7.4, 5.35, 4.95, .4, "补齐不代表末尾出现了真实文本", 17, MUTED)
    elif kind == "grad":
        flow(c, ["位置梯度：1 / 1 / 1 / 1", "输入 ID：a / a / b / pad", "scatter-add 到共享表行", "行梯度：a=2，b=1，pad=0"], size=18)
    elif kind == "chain":
        flow(c, ["e：[D]", "z=e W^T+b：[V]", "g_z = p − one_hot(y)", "g_e = g_z W：[D]"], size=21)
    elif kind == "training":
        flow(c, ["57 个相邻 token 训练 pair", "Embedding(320,16)", "Linear(16,320) → CE", "120步：5.7565 → 0.3559"], size=19)
    elif kind == "tie":
        flow(c, ["输入路径：gather(E,I)", "输出路径：h @ E^T", "同一个参数 E", "两条梯度路径在 E 上相加"], size=20)
    elif kind == "permutation":
        flow(c, ["X → Q,K,V", "PX → PQ,PK,PV", "scores′ = P scores P^T", "Attn(PX) = P Attn(X)"], size=20)
        c.text(7.5, 5.52, 4.8, .27, "仅讨论无位置且无遮罩的标准注意力", 13, MUTED)
    elif kind == "absolute":
        flow(c, ["内容表 E：[V,D]", "学习式位置表 P：[L,D]", "或用固定函数计算 PE(p)", "X_p = E[I_p] + P_p"], size=20)
    elif kind == "frequencies":
        flow(c, ["D=4 → 频率 [1, 0.01]", "positions：[T]", "广播相乘 → angles：[T,2]", "sin / cos 交错 → PE：[T,4]"], size=18)
    elif kind == "rotation":
        flow(c, ["q′ = R(mθ) q", "k′ = R(nθ) k", "R(mθ)^T R(nθ) = R((n−m)θ)", "点积依赖相对位移 n−m"], size=18)
    elif kind == "vector_rope":
        flow(c, ["Q/K：[B,H,T,d_h]", "按相邻维度拆 even / odd", "分别计算旋转的两个分量", "形状与每个向量范数保持不变"], size=18)
    elif kind == "mask":
        flow(c, ["位置机制 → 数值中的位置关系", "causal：[1,1,T,T]", "padding key：[B,1,1,T]", "forbidden = future OR padding"], size=18)
    elif kind == "pipeline":
        flow(c, ["ID：[2,6]", "X：[2,6,8]", "Q/K：[2,2,6,4]", "scores：[2,2,6,6]", "合并 heads：[2,6,8]"], size=19, y=2.05, height=.49, gap=.2)
    elif kind == "quiz":
        flow(c, ["选 pair：rank 还是长度？", "更新表：覆盖还是累加？", "位置关系：旋转还是遮罩？"], size=21, y=2.6, height=.65, gap=.3)
    else:
        raise ValueError(kind)
