"""课程图示。"""
import math

TEAL, LIGHT, BLUE, MUTED = "087F8C", "E6EFEE", "DEE8F0", "5C6F7B"


def flow(c, labels, size=18, height=.55, gap=.22, y=2.2):
    for i, label in enumerate(labels):
        top = y + i * (height + gap)
        c.node(7.5, top, 4.8, height, label, LIGHT if i % 2 == 0 else BLUE, size)
        if i + 1 < len(labels):
            c.line(9.9, top + height + .01, 9.9, top + height + gap - .01, arrow=True)


def draw(c, kind):
    if kind == "opening":
        c.text(7.45, 2.35, 4.85, .4, "第一课的输入，还缺哪些细节？", 18, TEAL)
        c.text(7.45, 3.18, 4.85, .75, "我喜欢喝牛奶", 31, bold=True)
        c.node(7.45, 4.42, 4.85, .7, "切分？   数字？   顺序？", BLUE, 23)
    elif kind == "roadmap":
        flow(c, ["文本 → token ID", "ID → embedding", "表示中引入位置信息"], size=23, y=2.6, height=.65, gap=.3)
    elif kind == "units":
        flow(c, ["字符：我 / 喜 / 欢 / 喝 / 牛 / 奶", "词：我 / 喜欢 / 喝 / 牛奶", "本例 BPE：我 / 喜欢喝牛奶"], size=17, y=2.6, height=.65, gap=.3)
    elif kind == "bpe_train":
        c.text(7.4, 2.12, 5, .35, "句子频次：5 / 3 / 2", 19, TEAL)
        for i, (label, count) in enumerate((("喜 + 欢", 10), ("喜欢 + 喝", 10), ("喜欢喝 + 牛", 8), ("喜欢喝牛 + 奶", 8))):
            c.node(7.4, 2.78 + .65*i, 4.95, .52, f"第 {i+1} 轮   {label}   ×{count}", BLUE, 18)
        c.text(7.4, 5.5, 4.95, .3, "每合并一轮，都重新统计当前片段", 15, MUTED)
    elif kind == "bpe_apply":
        flow(c, ["我 / 喜 / 欢 / 喝 / 牛 / 奶", "我 / 喜欢 / 喝 / 牛 / 奶", "我 / 喜欢喝 / 牛 / 奶", "我 / 喜欢喝牛奶", "ID：[4, 11]"], size=17, height=.48, gap=.2, y=2.1)
    elif kind == "bytes":
        flow(c, ["牛奶：2 个汉字", "UTF-8：6 个字节", "基础单位可继续合并", "最终 token 数由 tokenizer 决定"], size=18)
    elif kind == "id":
        flow(c, ["我 → ID 4", "4 不是词义，也不是语义距离", "选择参数表中索引为 4 的行", "每行 4 个可学习的数"], size=19)
    elif kind == "lookup":
        c.text(7.4, 2.15, 5, .4, "Embedding 表：12 行 × 4 列", 20, TEAL)
        c.node(7.4, 3.0, 4.95, .58, "4  我 → [0.127, 0.448, …]", BLUE, 18)
        c.node(7.4, 3.92, 4.95, .58, "11 → [−0.256, 0.074, …]", BLUE, 18)
        c.node(7.4, 4.95, 4.95, .58, "两个 ID → 两行四维向量", LIGHT, 20)
    elif kind == "dense":
        c.text(7.4, 2.2, 5, .4, "one-hot：长度 V=12", 20, TEAL)
        for i in range(12):
            c.rect(7.4 + i*.4, 3.0, .34, .44, TEAL if i == 4 else BLUE)
        c.text(7.4, 3.7, 5, .4, "查表：D=4，数值由训练调整", 20, TEAL)
        for i, value in enumerate((".127", ".448", ".077", "−.103")):
            c.node(7.4+i*1.25, 4.5, 1.12, .6, value, LIGHT, 17)
        c.text(7.4, 5.45, 4.95, .3, "ID 指定哪一行；行内容是模型参数", 16, MUTED)
    elif kind == "learning":
        flow(c, ["初始化 embedding 表", "预测标签，计算交叉熵", "更新参数，重新预测", "本例 loss：0.6981 → 0.0656"], size=19)
    elif kind == "context":
        flow(c, ["同一 ID → 同一输入表行", "加入位置机制", "进入 Transformer，结合上下文", "后续隐藏表示可以不同"], size=18)
    elif kind == "order":
        flow(c, ["矩阵行序仍然存在", "无位置、无遮罩的 self-attention", "重排输入 → 对应重排输出", "需要可参与计算的位置特征"], size=17)
    elif kind == "add":
        flow(c, ["内容 E[id]：4 维", "位置 P[p]：4 维", "逐项相加 E[id] + P[p]", "结果仍是 4 维"], size=21)
    elif kind == "sinus":
        c.text(7.4, 2.12, 5, .4, "D=4：sin p / cos p / sin(p/100) / cos(p/100)", 13, TEAL)
        for i, (p, values) in enumerate(((0, "[0, 1, 0, 1]"), (1, "[.8415, .5403, .0100, 1.0000]"), (2, "[.9093, −.4161, .0200, .9998]"))):
            c.node(7.4, 2.95+i*.7, 4.95, .57, f"p={p}   {values}", BLUE, 16)
        c.text(7.4, 5.35, 4.95, .38, "同一 token，不同 p → 不同组合输入", 17, MUTED)
    elif kind == "rope":
        c.text(7.4, 2.15, 5, .4, "各自改变什么？", 20, TEAL)
        for i, text in enumerate(("因果遮罩 → 可读取的范围", "绝对位置 → 输入中的位置信号", "RoPE → Q/K 的位置相关旋转")):
            c.node(7.4, 3.0+i*.78, 4.95, .6, text, BLUE, 18)
        c.text(7.4, 5.58, 5, .25, "能够计算更远的位置 ≠ 保证长上下文效果", 13, MUTED)
    elif kind == "pipeline":
        flow(c, ["我喜欢喝牛奶", "token：我 / 喜欢喝牛奶", "ID：[4, 11]", "查表：[2, 4]", "组合位置 0、1 → [2, 4]"], size=18, height=.48, gap=.2, y=2.1)
    elif kind == "quiz":
        flow(c, ["边界由什么决定？", "映射的数值由什么决定？", "位置在何处参与计算？"], size=21, y=2.6, height=.65, gap=.3)
    elif kind == "homework":
        flow(c, ["改输入，保留 tokenizer", "改编号，同时对应重排表行", "改位置，保持内容向量不变"], size=19, y=2.6, height=.65, gap=.3)
    else:
        raise ValueError(kind)
