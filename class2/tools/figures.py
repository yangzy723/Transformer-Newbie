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
    elif kind == "hf_pipeline":
        flow(c, ["文本 + ByteLevel 预切分", "256 基础字节 + 4 特殊符号", "BPE trainer → 320 项词表", "保存完整 tokenizer.json", "加载 → encode / decode"], size=17, y=2.05, height=.49, gap=.2)
    elif kind == "byte":
        c.text(7.4, 2.2, 5, .4, "Hello world!：本次实际编码", 19, TEAL)
        c.node(7.4, 3.1, 4.95, .65, "词表 320 → 9 个 token", BLUE, 22)
        c.node(7.4, 4.15, 4.95, .65, "词表 384 → 5 个 token", LIGHT, 22)
        c.text(7.4, 5.35, 4.95, .4, "中文样例均为3项，不保证每句都缩短", 16, MUTED)
    elif kind == "artifact":
        flow(c, ["我喜欢喝牛奶。", "冻结配置 V=320", "ID：[300, 304, 262]", "与该配置配套的 E：[320,D]"], size=19)
    elif kind == "lookup":
        c.text(7.4, 2.2, 5, .4, "B=2，T=6，pad=0", 21, TEAL)
        c.node(7.4, 3.0, 4.95, .65, "句1：bos + 3 tokens + eos + pad", BLUE, 17)
        c.node(7.4, 4.1, 4.95, .65, "句2：bos + 4 tokens + eos", LIGHT, 18)
        c.text(7.4, 5.35, 4.95, .4, "补齐不代表末尾出现了真实文本", 17, MUTED)
    elif kind == "pipeline":
        flow(c, ["ID：[2,6]", "X：[2,6,8]", "Q/K：[2,2,6,4]", "scores：[2,2,6,6]", "合并 heads：[2,6,8]"], size=19, y=2.05, height=.49, gap=.2)
    elif kind == "quiz":
        flow(c, ["选 pair：rank 还是长度？", "更新表：覆盖还是累加？", "位置关系：旋转还是遮罩？"], size=21, y=2.6, height=.65, gap=.3)
    else:
        raise ValueError(kind)
