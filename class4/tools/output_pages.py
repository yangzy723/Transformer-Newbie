"""第四课后半：从词表分数到一次选择，连续的十三页课程。"""
import math
import random

from layout import *
from content import SOURCES

WORDS = ['牛奶', '茶', '水']
BASE = [.5, .3, .2]
PROB_COLORS = [ORANGE, 'CC8B4A', 'F2D7B7']


def softmax_local(z):
    maximum = max(z)
    weights = [math.exp(value - maximum) for value in z]
    return [value / math.fsum(weights) for value in weights]


def bullets(c, x, y, w, items, size=21, gap=.14):
    for value in items:
        y = para(c, x, y, w, value, size) + gap
    return y


def hyperlink(c, x, y, w, title, url, size=17.5):
    text(c, x, y, w, .4, title, size, BLUE, True)
    for paragraph in c.slide.shapes[-1].text_frame.paragraphs:
        for run in paragraph.runs:
            run.hyperlink.address = url


def probability_strip(c, x, y, w, probabilities, height=.55, digits=4):
    offset = 0.0
    for i, probability in enumerate(probabilities):
        width = w * probability
        c.rect(x + offset * w, y, width, height, PROB_COLORS[i], False)
        text(c, x + offset * w + .025, y + .13, width - .05, .35,
             f'{probability:.{digits}f}', 21,
             WHITE if i < 2 else INK, True, True)
        offset += probability


def draw(c, item, r):
    kind = item['kind']
    if kind == 'lm_head':
        stages = [
            (.85, 2.35, '后续 Block', '每个位置仍为 D 维'),
            (3.60, 2.35, '最终 Norm', '逐位置归一化'),
            (6.35, 2.35, '最后位置 g', '取序列最后一行'),
            (9.10, 3.25, 'LM Head', 'D 维 → V 个分数'),
        ]
        for x, w, title, detail in stages:
            node(c, x, 2.02, w, .88, title, detail, size=21)
        for start, end in [(3.2, 3.6), (5.95, 6.35), (8.7, 9.1)]:
            arrow(c, start, 2.46, end, 2.46)
        label(c, .9, 3.13, 4.35, '给定 g=[1,2]，偏置为 0', size=20)
        text(c, .9, 3.49, 4.35, .25, 'W_LM：2 维表示 → 3 个词表分数', 16, MUTED)
        table(c, .9, 3.72, 4.35, [[1, 0, -1], [0, 1, 0]],
              labels=['g[0]', 'g[1]'], columns=WORDS, cell_h=.59, size=24)
        eq(c, r'\mathbf{z}=\mathbf{g}W_{\mathrm{LM}}=[1,\ 2,\ -1]',
           x=.9, y=5.42, w=4.35, h=.61, size=25)
        rowtable(c, 5.78, 3.2, 6.57, ['候选 / ID', '按矩阵列求和', 'logit'], [
            ['牛奶 / 0', '1×1 + 2×0', '1'],
            ['茶 / 1', '1×0 + 2×1', '2'],
            ['水 / 2', '1×(-1) + 2×0', '-1'],
        ], [.25, .55, .20], size=20, row_h=.64)
        para(c, 5.78, 5.83, 6.57,
             '每一列对应一个候选 token。logit 可正可负，全部分数的总和无需等于 1。', 20)
        note(c, .9, 6.15, 4.35, '本页与下页使用同组参数。', 19)
    elif kind == 'softmax':
        logits = [1, 2, -1]
        shifted = [value - max(logits) for value in logits]
        exponentials = [math.exp(value) for value in shifted]
        probabilities = softmax_local(logits)
        eq(c, r'p_i=\frac{e^{z_i-\max(\mathbf{z})}}{\sum_j e^{z_j-\max(\mathbf{z})}}',
           x=.85, y=2.0, w=7.20, h=.75, size=28)
        rowtable(c, .85, 3.0, 7.3, ['候选', 'logit', '减最大值', '指数', '概率'], [
            [word, str(z), str(a), f'{weight:.4f}', f'{p:.4f}']
            for word, z, a, weight, p in zip(WORDS, logits, shifted, exponentials, probabilities)
        ], [.17, .13, .22, .23, .25], size=19.5, row_h=.67)
        label(c, 8.7, 2.52, 3.65, '输出词表的概率', ORANGE, 20)
        bars(c, 8.7, 3.1, 3.65, WORDS, probabilities, color=ORANGE, row_h=.78)
        para(c, .9, 5.59, 11.45,
             f'指数之和为 {sum(exponentials):.4f}，逐项相除得到概率。共同减去最大值不改变 Softmax 结果，并使指数的最大值为 1。', 20)
        c.line(.9, 6.36, 12.35, 6.36, LINE)
        text(c, .9, 6.47, 5.45, .37, 'Attention：沿 T 个位置分配读取权重。', 19, BLUE)
        text(c, 6.7, 6.47, 5.65, .37, '输出：沿 V 个词表候选分配概率。', 19, ORANGE)
    elif kind == 'selection':
        para(c, .85, 2.02, 11.5,
             '本例将 [0,1) 划为三个区间，区间长度依次为 0.5、0.3、0.2，分别对应牛奶、茶、水。', 20)
        text(c, .9, 2.8, 10.8, .46,
             '贪心选择：argmax(P) → ID 0「牛奶」', 22, ORANGE, True)
        x, y, w = .95, 3.93, 11.25
        offset = 0.0
        for i, (word, probability) in enumerate(zip(WORDS, BASE)):
            c.rect(x + w * offset, y, w * probability, .61, PROB_COLORS[i], False)
            text(c, x + w * offset, y + .13, w * probability, .4, word, 23,
                 WHITE if i < 2 else INK, True, True)
            offset += probability
        marker = x + w * .64
        arrow(c, marker, 3.52, marker, 3.88, ORANGE)
        text(c, marker - 2.85, 3.19, 5.7, .4, '随机抽样：r=.64 → ID 1「茶」', 21, ORANGE, True, True)
        for value in [0, .5, .8, 1]:
            text(c, x + w * value - .25, 4.68, .6, .37, f'{value:g}', 20, MUTED, center=True)
        for xpos, width, value in [(1.0, 4.95, '[0,.5) → 牛奶'), (6.4, 2.80, '[.5,.8) → 茶'), (9.5, 2.70, '[.8,1) → 水')]:
            text(c, xpos, 5.28, width, .42, value, 20, INK, center=True)
        para(c, .95, 5.96, 11.25,
             '从 [0,1) 均匀抽取 r，选择 r 所在区间对应的 token。区间长度等于选择概率；左闭右开，r=.5 对应「茶」。', 20)
    elif kind == 'frequency':
        rng = random.Random(7)
        counts = [0, 0, 0]
        for _ in range(1000):
            value = rng.random()
            counts[0 if value < .5 else 1 if value < .8 else 2] += 1
        label(c, .85, 2.02, 11.5, '固定 P(牛奶,茶,水)=[.5,.3,.2]；独立抽样 1000 次，seed=7', ORANGE, 20)
        rowtable(c, .85, 2.67, 7.05, ['候选', '理论概率', '真实次数', '真实频率'], [
            [word, f'{probability:.1f}', str(count), f'{count/1000:.3f}']
            for word, probability, count in zip(WORDS, BASE, counts)
        ], [.20, .26, .27, .27], size=21, row_h=.64)
        bars(c, 8.6, 2.97, 3.75, WORDS, [count / 1000 for count in counts],
             color=ORANGE, row_h=.87, digits=3)
        para(c, .9, 5.43, 11.45,
             '期望次数为 1000×P=[500,300,200]；本次实际次数为 [537,260,203]。\n每次选择都使用同一分布，实际频数随抽到的随机数而变化。', 20.5)
        para(c, .9, 6.29, 11.45,
             'seed=7 初始化随机数序列；连续调用得到不同的随机数，用于这 1000 次选择。', 19.5)
    elif kind == 'temperature':
        eq(c, r'p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)},\quad\tau>0',
           x=.85, y=2.0, w=8.1, h=.70, size=28)
        text(c, 9.1, 2.17, 3.3, .4, '固定 z=log([.5,.3,.2])', 19, MUTED)
        for i, word in enumerate(WORDS):
            xpos = 2.1 + i * 3.4
            c.rect(xpos, 2.87, .18, .18, PROB_COLORS[i], False)
            text(c, xpos + .35, 2.80, 2.2, .35, word, 19, ORANGE, True)
        for index, temperature in enumerate([.5, 1, 2]):
            probabilities = softmax_local([math.log(p) / temperature for p in BASE])
            yy = 3.35 + index * .80
            text(c, .90, yy + .1, 1.0, .42, f'τ={temperature:g}', 22, ORANGE, True)
            probability_strip(c, 2.05, yy, 10.15, probabilities)
        para(c, .9, 5.98, 11.45,
             '较低的正温度扩大 logits 差异，使概率更集中；较高温度缩小差异，使概率更均匀。正温度保持排序；τ=0 使用贪心分支。', 20)
    elif kind == 'topk':
        para(c, .85, 2.02, 11.5,
             'k=2：按概率降序保留两项，将其余候选置零，再除以被保留的总概率。', 21)
        rowtable(c, .85, 2.90, 7.25, ['候选', '原概率', '是否保留', '重新归一化'], [
            ['牛奶', '0.5', '保留', '0.5 / 0.8 = 0.625'],
            ['茶', '0.3', '保留', '0.3 / 0.8 = 0.375'],
            ['水', '0.2', '舍弃', '0'],
        ], [.15, .18, .23, .44], size=20, row_h=.75)
        label(c, 8.7, 2.90, 3.65, '截断后的最终分布', ORANGE, 19)
        bars(c, 8.7, 3.53, 3.65, WORDS, [.625, .375, 0], color=ORANGE, row_h=.72, digits=3)
        eq(c, r'm_{\mathrm{keep}}=0.5+0.3=0.8', x=.9, y=5.93, w=5.8, h=.48, size=24)
        para(c, 7.1, 5.93, 5.25,
             '截断后总概率为 0.8；除以 0.8，恢复总和为 1 的概率分布。', 20)
    elif kind == 'topp':
        text(c, .85, 2.02, 11.5, .32,
             '输入 P=[.5,.3,.2]，总和为 1；先按概率降序排列，再计算累计概率。', 19.5, INK)
        rowtable(c, .85, 2.46, 7.70, ['降序候选', '原概率', '累计概率', '阈值 p=.7'], [
            ['牛奶', '0.5', '0.5 < 0.7', '继续加入'],
            ['茶', '0.3', '0.8 ≥ 0.7', '保留后停止'],
            ['水', '0.2', '1.0', '不再加入'],
        ], [.23, .17, .25, .35], size=19.5, row_h=.43)
        para(c, 9.0, 2.48, 3.35,
             '保留累计概率首次达到\n阈值的最短前缀。\n「牛奶、茶」共 0.8，\n除以 0.8 后再抽样。', 19)
        text(c, 9.0, 4.00, 3.35, .42, 'P=[.8,.1,.1] 时只留第一项。', 17.5, ORANGE)
        label(c, .9, 4.47, 11.45, '仍使用 P=[.5,.3,.2]：不同阈值对应的保留项与最终概率', ORANGE, 19.5)
        rowtable(c, .85, 4.94, 11.5, ['阈值 p', '.5', '.7', '.8', '1.0'], [
            ['保留项', '牛奶', '牛奶、茶', '牛奶、茶', '全部三项'],
            ['保留总概率', '.5', '.8', '.8', '1.0'],
            ['最终概率', '[1,0,0]', '[.625,.375,0]', '[.625,.375,0]', '[.5,.3,.2]'],
        ], [.16, .18, .22, .22, .22], size=19, row_h=.39)
    elif kind == 'decode':
        stages = ['原始 logits', 'Softmax(z/τ)', 'Top-p 截断', '归一化', '抽样 ID']
        for i, title in enumerate(stages):
            x = .85 + i * 2.36
            node(c, x, 2.02, 2.10, .51, title, fill=SAND, color=ORANGE, size=20)
            if i < 4:
                arrow(c, x + 2.10, 2.275, x + 2.36, 2.275, ORANGE)
        text(c, .9, 2.66, 11.45, .26,
             'softmax 按温度计算概率；sample_index 按累计概率区间抽取 ID。', 16.5, MUTED)
        code(c, [
            'from lesson import softmax, sample_index',
            'def decode(z, tau, p, rng):',
            '    if tau == 0:',
            '        return max(range(len(z)), key=z.__getitem__)',
            '    q = softmax(z, tau)',
            '    order = sorted(range(len(q)), key=lambda i: -q[i])',
            '    kept, mass = [], 0.0',
            '    for i in order:',
            '        kept.append(i); mass += q[i]',
            '        if mass + 1e-15 >= p: break',
            '    q = [q[i]/mass if i in kept else 0 for i in range(len(q))]',
            '    return sample_index(q, rng)',
        ], x=.98, y=3.04, w=11.22, size=18, line_h=.265)
        text(c, .9, 6.48, 11.45, .28,
             '输入：logits 列表 z，tau≥0，0<p≤1，随机数发生器 rng；1e-15 用于累计概率的边界舍入。',
             16.5, MUTED)
    elif kind == 'generation':
        for i, (title, detail) in enumerate([
            ('前向计算', '当前上下文 → 最后位置 logits'),
            ('解码选择', '当前 logits → 一个 ID'),
            ('追加与检查', '追加 ID；EOS / 长度停止'),
        ]):
            x = .85 + i * 3.95
            node(c, x, 2.02, 3.55, .86, title, detail, size=21,
                 fill=SAND if i > 0 else PALE, color=ORANGE if i > 0 else BLUE)
            if i < 2:
                arrow(c, x + 3.55, 2.45, x + 3.95, 2.45, ORANGE)
        code(c, [
            'def generate(forward, ids, choose, eos, limit):',
            '    ids = list(ids)',
            '    for _ in range(limit):',
            '        z = forward(ids)',
            '        token = choose(z)',
            '        ids.append(token)',
            '        if token == eos: break',
            '    return ids',
        ], x=.98, y=3.22, w=7.65, size=18.8, line_h=.33)
        label(c, 9.1, 3.31, 3.25, 'forward 的接口约定', BLUE, 19)
        para(c, 9.1, 3.91, 3.25,
             '输入：ID 列表，长度为 T。\n\n输出：最后位置的分数。\n形状：[V]。\n\n从模型输出 [B,T,V] 中\n取 [0,-1,:]（本例 B=1）。', 19)
        text(c, .98, 6.09, 7.65, .31,
             '我 喜欢 喝 → 追加「牛奶」→ 以新上下文重新前向计算', 19, BLUE)
        text(c, .9, 6.43, 11.45, .32,
             'choose(z) 返回一个 ID；eos 为结束符 ID；limit 为本次最多新增的 token 数。',
             17.5, MUTED)
    elif kind == 'train_infer':
        label(c, .9, 2.04, 5.35, '训练：已有正确后续 token', BLUE, 22)
        label(c, 6.85, 2.04, 5.4, '普通生成：参数固定', ORANGE, 22)
        c.line(6.48, 2.05, 6.48, 5.32, LINE)
        bullets(c, .9, 2.80, 5.35, [
            '输入：[我, 喜欢, 喝]',
            '目标：[喜欢, 喝, 牛奶]',
            '前向 → logits → 交叉熵损失',
            '反向传播 → 优化器更新参数',
        ], 21, .18)
        bullets(c, 6.85, 2.80, 5.4, [
            '输入：当前已有上下文。',
            '前向 → 最后位置的 logits',
            '解码 → 追加一个新 token ID',
            '继续前向；没有参数更新。',
        ], 21, .18)
        code(c, [
            'from torch.nn import functional as F',
            'loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())',
            '# logits: [B,T,V] -> [B*T,V] ; target: [B,T] -> [B*T]',
        ], x=.98, y=5.57, w=11.22, size=18, line_h=.32)
    elif kind == 'experiments':
        rowtable(c, .85, 2.08, 11.50, ['Notebook 实验', '保持不变', '改变的因素', '计算结果与原因'], [
            ['05 · 逐位置 FFN', 'FFN 参数、其他行', '第二行 [-1,2]\n改为 [-3,2]', '第二行输出 [1,2] → [3,2]；\nFFN 的各行计算互不混合。'],
            ['06 · 激活函数', 'u=[1,-2]、W₁/W₂', 'ReLU 改为恒等', '输出 [1,2] → [0,0]；\n这组参数满足 W₁W₂=0。'],
            ['13 · 温度', 'z=log([.5,.3,.2])', 'τ：1 → .5', '牛奶概率 .5000 → .6579；\n最大项更集中，候选排序不变。'],
            ['11–12 · 选择', 'P=[.5,.3,.2]', '贪心改为抽样', '1000 次：贪心 [1000,0,0]；\nseed=7 抽样 [537,260,203]。'],
        ], [.19, .23, .21, .37], size=18.5, row_h=.81)
        para(c, .9, 6.20, 11.45,
             'FFN 更新各位置的表示；温度改变概率分布；选择规则返回一个 token ID。', 20)
    elif kind == 'recap':
        c.line(6.46, 2.05, 6.46, 6.47, LINE)
        c.line(.9, 4.25, 12.35, 4.25, LINE)
        text(c, .9, 2.12, 5.15, .55, 'FFN：输入 u=[−1,2]，两层偏置为 0', 20.5, BLUE, True)
        para(c, .9, 2.71, 5.15,
             '升维：uW₁=[u₀,−u₀,u₁,−u₁]\n激活：a=ReLU(uW₁)=[0,1,2,0]\n降维：aW₂=[a₀+a₁,a₂+a₃]\nFFN(u)=[1,2]，输出为更新向量。', 19)
        text(c, 6.9, 2.12, 5.45, .75, 'ReLU 非负，FFN 输出是否也非负？', 21, TEAL, True)
        para(c, 6.9, 2.95, 5.45,
             '最终线性层仍可产生负数。\n例如激活 a=[1,0]，某个输出列\nw=[−1,2]，则 a·w=−1。', 20.5)
        text(c, .9, 4.48, 5.15, .75, 'Top-p：P=[.5,.3,.2]，阈值为 .7', 20.5, ORANGE, True)
        para(c, .9, 5.29, 5.15,
             '候选顺序：牛奶、茶、水。\n前两项累计 .8，保留牛奶与茶。\n归一化：[.625,.375,0]。', 20.5)
        text(c, 6.9, 4.48, 5.45, .75, '普通生成：追加 token 是否更新参数？', 20.5, BLUE, True)
        para(c, 6.9, 5.29, 5.45,
             '普通生成不改参数，只增长上下文。\n训练才通过损失、反向传播和\n优化器来更新参数。', 20.5)
    elif kind == 'closing':
        flow = [('上下文向量', BLUE), ('Block / FFN', BLUE), ('LM Head 分数', BLUE), ('概率与解码', ORANGE), ('新 token ID', ORANGE)]
        for i, (title, color) in enumerate(flow):
            x = .85 + i * 2.36
            node(c, x, 2.03, 2.10, .72, title,
                 fill=SAND if color == ORANGE else PALE, color=color, size=20)
            if i < 4:
                arrow(c, x + 2.10, 2.39, x + 2.36, 2.39, color)
        text(c, .9, 3.13, 11.45, .5,
             'Attention 汇总信息，FFN 变换特征；LM Head 计算分数，解码选择 ID。', 23, INK, True)
        label(c, .9, 3.97, 11.45, '下一课：通过预测误差学习模型参数', BLUE, 21)
        text(c, .9, 4.53, 11.45, .45,
             '预测 logits 与目标 ID → 交叉熵损失 → 梯度 → 优化器更新参数', 21, BLUE)
        c.line(.9, 5.17, 12.35, 5.17, LINE)
        text(c, .9, 5.31, 11.45, .34, '参考文献与接口文档', 17.5, MUTED, True)
        hyperlink(c, .9, 5.83, 7.6, 'Attention Is All You Need · §3.3：FFN', SOURCES['ffn'])
        hyperlink(c, .9, 6.22, 7.6, 'GLU Variants Improve Transformer：门控结构', SOURCES['glu'])
        hyperlink(c, 8.55, 5.83, 3.8, 'PyTorch · nn.Linear', SOURCES['linear'])
        hyperlink(c, 8.55, 6.22, 3.8, 'PyTorch · nn.LayerNorm', SOURCES['norm'])
        # Long paper title stays readable on a separate line above the two-column source list.
        hyperlink(c, 5.7, 5.31, 6.65, 'Neural Text Degeneration：Top-p 定义', SOURCES['sampling'], 17.5)
    else:
        return False
    return True
