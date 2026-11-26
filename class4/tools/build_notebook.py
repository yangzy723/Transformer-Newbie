"""生成第四课 Notebook：FFN、激活函数与采样。"""

import json
from pathlib import Path
from textwrap import dedent


def build(root):
    """写入 class4/demo.ipynb 并返回路径；root 可为 class4 或仓库根。"""
    root = Path(root)
    if (root / "class4").is_dir():
        root = root / "class4"
    cells = []

    def cell(kind, source):
        value = {
            "cell_type": kind,
            "id": f"class4-{len(cells):02d}",
            "metadata": {},
            "source": dedent(source).strip() + "\n",
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    def md(source):
        cell("markdown", source)

    def code(source):
        cell("code", source)

    md(r"""
    # 第四课：FFN、激活函数与采样

    本 Notebook 与 28 页 PPT 对应，依次计算多头汇合、FFN、激活函数、词表分数与采样。
    核心计算使用 Python 标准库；最后一节使用 PyTorch 核对权重方向与 LayerNorm。

    **数值约定**：采用行向量右乘 `u @ W`，矩阵写成 Python 的行列表；位置从 0 开始。
    小矩阵由课程给定。主要算例分为三组：FFN 使用 `u=[1,-2]`；残差计算沿用
    第三课的 `h=[1.3,1.8]`，使用本课 FFN 参数计算更新；输出与采样分别使用
    `g=[1,2]` 和固定概率 `[.5,.3,.2]`。各节列明输入及参数。
    """)
    md("""
    ## 准备：向量与矩阵运算

    `vecmat` 将输入向量与矩阵的每一列分别做点积，得到各个输出分量。
    `softmax` 先减去最大分数再计算指数；`layer_norm` 沿单个位置的特征维计算均值与总体方差。
    """)
    code("""
    import math
    import random
    from collections import Counter

    def vecmat(vector, matrix):
        assert len(vector) == len(matrix)
        return [sum(vector[i] * matrix[i][j] for i in range(len(vector)))
                for j in range(len(matrix[0]))]

    def matmul(left, right):
        return [vecmat(row, right) for row in left]

    def relu(x):
        return max(0.0, x)

    def gelu(x):
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

    def silu(x):
        # 课堂数值范围较小，直接计算 sigmoid。
        return x / (1.0 + math.exp(-x))

    def softmax(logits):
        maximum = max(logits)
        exponents = [math.exp(value - maximum) for value in logits]
        total = math.fsum(exponents)
        return [value / total for value in exponents]

    def layer_norm(vector, eps=1e-5):
        mean = math.fsum(vector) / len(vector)
        variance = math.fsum((x - mean) ** 2 for x in vector) / len(vector)
        normalized = [(x - mean) / math.sqrt(variance + eps) for x in vector]
        return normalized, mean, variance  # 本课 gamma=1、beta=0。

    def rounded(values, digits=4):
        return [round(value, digits) for value in values]

    def close_vectors(left, right, tolerance=1e-10):
        return len(left) == len(right) and all(
            abs(a - b) <= tolerance for a, b in zip(left, right))

    print('标准库工具已准备；矩阵采用 [in, out] 的右乘约定。')
    """)
    md("""
    ## 实验 01 · 多头拼接与输出投影 · PPT 3

    给定同一位置两个头的 Attention 输出 `[1,2]` 与 `[-1,3]`。
    沿特征轴拼接后得到四维向量 `[1,2,-1,3]`，再右乘输出投影矩阵 `W_O`。

    `W_O` 混合不同头的特征，输出宽度为模型宽度 D=4，与残差向量保持一致。
    """)
    code("""
    head_0 = [1.0, 2.0]
    head_1 = [-1.0, 3.0]
    concatenated = head_0 + head_1  # 列表拼接；不是向量逐项相加。
    w_o = [[1.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 0.0],
           [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0]]
    attention_update = vecmat(concatenated, w_o)
    print('头 1：', head_0, '头 2：', head_1)
    print('沿特征轴拼接：', concatenated)
    print('右乘 W_O：', attention_update)
    assert attention_update == [0.0, 5.0, -1.0, 3.0]
    """)
    md("""
    ## 实验 02 · FFN 第一步：升维 · PPT 5–7

    给定二维输入 `u=[1,-2]` 和 2×4 的矩阵 `W1`，输出由四个点积组成。
    例如第三列为 `[0,1]`，对应结果是 `1×0 + (-2)×1=-2`。

    两层偏置均设为 0，后续各位置共享同一组 `W1`、`W2`。
    输出的四个数是中间特征；它们的维度由 FFN 宽度决定。
    """)
    code("""
    u = [1.0, -2.0]
    W1 = [[1.0, -1.0, 0.0, 0.0],
          [0.0,  0.0, 1.0, -1.0]]
    W2 = [[1.0, 0.0],
          [1.0, 0.0],
          [0.0, 1.0],
          [0.0, 1.0]]

    expanded = vecmat(u, W1)
    print('输入 u：', u)
    print('uW1：', expanded)
    for j in range(4):
        products = [u[i] * W1[i][j] for i in range(2)]
        print('输出分量', j, '的贡献：', products, '→', sum(products))
    assert expanded == [1.0, -1.0, -2.0, 2.0]
    """)
    md("""
    ## 实验 03 · FFN 第二步：逐元素激活 · PPT 8

    ReLU 定义为 `max(0,x)`，逐个保留正值、将负值置零。
    它不进行归一化，输出之和没有固定要求。
    """)
    code("""
    activated = [relu(value) for value in expanded]
    print('激活前：', expanded)
    print('激活后：', activated)
    print('总和：', sum(activated), '；ReLU 不要求总和为 1。')
    assert activated == [1.0, 0.0, 0.0, 2.0]
    """)
    md("""
    ## 实验 04 · FFN 第三步：降维 · PPT 9

    四维激活向量右乘 4×2 的 `W2`，得到二维 FFN 输出，用于后续残差相加。
    主算例得到 `[1,2]`。再将 `W2[3][1]` 改为 -1，输出变为 `[1,-2]`：
    下投影的权重可以为负，因此 FFN 的最终输出也可以为负。
    """)
    code("""
    ffn_update = vecmat(activated, W2)
    print('激活向量：', activated)
    print('FFN 分支输出：', ffn_update)
    assert ffn_update == [1.0, 2.0]

    negative_W2 = [row[:] for row in W2]
    negative_W2[3][1] = -1.0
    print('仅将 W2[3][1] 改为 -1 后：', vecmat(activated, negative_W2))
    assert vecmat(activated, negative_W2) == [1.0, -2.0]

    def ffn(vector, activation=relu):
        hidden = vecmat(vector, W1)
        return vecmat([activation(value) for value in hidden], W2)
    """)
    md("""
    ## 实验 05 · 逐位置与共享参数 · PPT 10

    三个位置使用相同的 `W1`、`W2`，分别计算自己的 FFN 输出。
    参数保持不变，只将位置 1 的输入从 `[-1,2]` 改为 `[-3,2]`，比较各行输出。

    FFN 本身不跨位置汇总；此前 Attention 的输出已可包含其他可见位置的信息。
    """)
    code("""
    sequence = [[1.0, -2.0], [-1.0, 2.0], [0.5, 0.25]]
    before = [ffn(row) for row in sequence]
    changed_sequence = [row[:] for row in sequence]
    changed_sequence[1] = [-3.0, 2.0]
    after = [ffn(row) for row in changed_sequence]
    changed_positions = []
    for position, (old, new) in enumerate(zip(before, after)):
        print('位置', position, '输出：', old, '→', new)
        if not close_vectors(old, new):
            changed_positions.append(position)
    print('改变的位置：', changed_positions)
    assert changed_positions == [1]
    """)
    md(r"""
    ## 实验 06 · 无激活时两层仿射变换的合并 · PPT 11

    用恒等映射替换 ReLU，连续两层仿射变换可以合并为一层：

    $$(uW_1+b_1)W_2+b_2=u(W_1W_2)+(b_1W_2+b_2).$$

    本例 `W1W2` 为零矩阵；保留 ReLU 后，FFN 对输入逐分量取绝对值。
    比较一对相反输入及其中点：仿射函数保留中点关系，绝对值函数不满足这一性质。
    这一合并只涉及 FFN 的两层仿射变换；Attention 中仍有 Softmax 等计算。
    """)
    code("""
    combined = matmul(W1, W2)
    print('W1W2：', combined)
    for vector in ([1.0, -2.0], [-1.0, 2.0], [0.0, 0.0]):
        no_activation = ffn(vector, activation=lambda value: value)
        print('输入', vector, 'ReLU FFN：', ffn(vector), '去激活：', no_activation)
        assert close_vectors(no_activation, vecmat(vector, combined))

    midpoint_input = [0.0, 0.0]
    output_at_midpoint = ffn(midpoint_input)
    average_output = [(a + b) / 2.0 for a, b in zip(ffn(u), ffn([-x for x in u]))]
    print('中点的输出：', output_at_midpoint, '两端输出的中点：', average_output)
    assert not close_vectors(output_at_midpoint, average_output)
    """)
    md("""
    ## 实验 07 · ReLU、GELU、SiLU 数值对照 · PPT 12

    GELU 使用精确 `erf` 公式；SiLU 为 `x·sigmoid(x)`。
    两者在负数区域可以输出非零值，且相同输入总会得到相同输出。
    下表比较三个函数在 -2 到 2 的取值，再将主 FFN 的 ReLU 替换为 GELU。
    """)
    code("""
    print('     x      ReLU      GELU      SiLU')
    for value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        print(f'{value:6.1f} {relu(value):9.4f} {gelu(value):9.4f} {silu(value):9.4f}')
    print('同一 u 换为 GELU：', rounded(ffn(u, activation=gelu)))
    assert gelu(-1.0) < 0.0 and silu(-1.0) < 0.0
    assert gelu(-1.0) == gelu(-1.0)
    """)
    md("""
    ## 实验 08 · SwiGLU 的两条支路 · PPT 13

    输入分别经过门控投影和升维投影，两条支路的对应分量相乘，再做下投影。
    SiLU 产生连续的门控系数，可取负值或大于 1 的值。
    本节给定一组独立的无偏置参数，门控宽度 m=2。
    """)
    code("""
    gate_input = [1.0, -2.0]
    W_gate = [[1.0, 0.0], [0.0, 1.0]]
    W_up = [[1.0, 1.0], [0.0, 1.0]]
    W_down = [[1.0, 0.0], [0.0, 1.0]]
    gate_pre = vecmat(gate_input, W_gate)
    gate = [silu(value) for value in gate_pre]
    up = vecmat(gate_input, W_up)
    product = [a * b for a, b in zip(gate, up)]
    gated_output = vecmat(product, W_down)
    for name, values in [('gate 投影', gate_pre), ('SiLU 后', gate),
                         ('up 投影', up), ('逐元素乘积', product), ('输出', gated_output)]:
        print(name, rounded(values))
    assert len(product) == 2
    """)
    md("""
    ## 实验 09 · LayerNorm、FFN 与残差相加 · PPT 2、4、14

    沿用第三课 `h=[1.3,1.8]`。残差旁路保留 `h`，分支依次计算 `Norm(h)` 和 FFN，
    最后得到 `y=h+FFN(Norm(h))`。本节使用实验 02 的 `W1`、`W2`。

    LayerNorm 的均值为 1.55，总体方差为 0.0625；γ=1、β=0、ε=1e-5。
    第三课为演示残差加法而给定的更新是 `[-0.1,0.4]`，本节重新计算分支输出。
    """)
    code("""
    h = [1.3, 1.8]
    normalized_h, mean_h, variance_h = layer_norm(h)
    computed_update = ffn(normalized_h)
    y = [base + update for base, update in zip(h, computed_update)]
    print('h：', h)
    print('均值：', round(mean_h, 6), '总体方差：', round(variance_h, 6))
    print('Norm(h)：', rounded(normalized_h, 6))
    print('FFN(Norm(h))：', rounded(computed_update, 6))
    print('h + 更新：', rounded(y, 6))
    print('第三课给定更新：[-0.1,0.4]；本节更新由上述 FFN 参数计算。')
    assert len(h) == len(y) == 2

    # LayerNorm 沿特征维计算，每个位置分别统计。
    norm_rows = [layer_norm(row)[0] for row in sequence]
    print('三行各自的 Norm：', [rounded(row) for row in norm_rows])
    """)
    md("""
    ## 实验 10 · LM Head 与稳定 Softmax · PPT 16–17

    给定最终 Norm 后最后位置的二维表示 `g=[1,2]`，右乘 2×3 的词表矩阵，
    得到三个候选 token 的分数。词表为“牛奶、茶、水”，本例 ID 依次为 0、1、2。

    Softmax 将分数归一化为词表概率，下一节再进行选择。
    Attention 的 Softmax 沿输入位置归一化，此处则沿词表候选归一化。
    """)
    code("""
    labels = ['牛奶', '茶', '水']
    g = [1.0, 2.0]
    W_vocab = [[1.0, 0.0, -1.0],
               [0.0, 1.0,  0.0]]
    lm_logits = vecmat(g, W_vocab)
    shifted = [value - max(lm_logits) for value in lm_logits]
    exponents = [math.exp(value) for value in shifted]
    lm_probabilities = softmax(lm_logits)
    print('LM Head logits：', lm_logits)
    print('减最大值：', shifted)
    print('指数：', rounded(exponents))
    print('归一化：', rounded(lm_probabilities), '和：', sum(lm_probabilities))
    print('对应候选：', list(zip(labels, rounded(lm_probabilities))))
    assert lm_logits == [1.0, 2.0, -1.0]
    assert close_vectors(lm_probabilities, softmax([x + 1000 for x in lm_logits]))
    """)
    md("""
    ## 实验 11 · 贪心选择与累计概率抽样 · PPT 18、23

    后续解码实验统一使用固定概率 `[0.5,0.3,0.2]`，对应 logits 为各项概率的对数。
    贪心选择取最大分数项；累计概率抽样将 `[0,1)` 分为
    `[0,.5)`、`[.5,.8)`、`[.8,1)`，由均匀随机数所在区间确定 token。
    例如 `r=0.64` 落在第二段，选中 ID=1 的“茶”。
    """)
    code("""
    sample_probabilities = [0.5, 0.3, 0.2]
    sample_logits = [math.log(value) for value in sample_probabilities]

    def choose_from_cdf(probabilities, r):
        assert 0.0 <= r < 1.0
        cumulative = 0.0
        for token_id, probability in enumerate(probabilities):
            cumulative += probability
            if r < cumulative:
                return token_id
        return len(probabilities) - 1  # 仅处理浮点数尾差。

    def sample_cdf(probabilities, rng):
        return choose_from_cdf(probabilities, rng.random())

    greedy_id = max(range(len(sample_logits)), key=lambda i: sample_logits[i])
    print('Greedy：', greedy_id, labels[greedy_id])
    for r in [0.0, 0.49, 0.5, 0.64, 0.8, 0.999]:
        chosen = choose_from_cdf(sample_probabilities, r)
        print('r =', r, '→ ID', chosen, labels[chosen])
    assert choose_from_cdf(sample_probabilities, 0.64) == 1
    """)
    md("""
    ## 实验 12 · 固定分布的频数与随机种子 · PPT 19

    对 `[0.5,0.3,0.2]` 独立抽样 1000 次，统计各候选的次数和频率。
    概率规定抽样规则，有限次实验的频率通常会与概率存在偏差。
    使用独立的 `random.Random(7)`；在相同环境与调用顺序下，同种子可复现相同序列。
    """)
    code("""
    rng = random.Random(7)
    draws = [sample_cdf(sample_probabilities, rng) for _ in range(1000)]
    counts = Counter(draws)
    print('候选    理论概率   抽样次数   观测频率')
    for token_id, label in enumerate(labels):
        print(f'{label:4s}    {sample_probabilities[token_id]:.3f}'
              f'      {counts[token_id]:4d}      {counts[token_id] / len(draws):.3f}')
    same_seed = random.Random(7)
    assert draws == [sample_cdf(sample_probabilities, same_seed) for _ in range(1000)]
    print('相同环境与调用顺序下，同种子可复现本实验的抽样序列。')
    """)
    md("""
    ## 实验 13 · 温度与概率分布 · PPT 20

    用相同 logits 计算 `softmax(logits / tau)`，比较 τ=0.5、1、2。
    正温度保留候选排序：较低温度使概率更集中，较高温度使分布更平缓。
    贪心选择直接取 argmax，作为单独分支实现。
    """)
    code("""
    original_order = sorted(range(3), key=lambda i: sample_logits[i], reverse=True)
    for tau in [0.5, 1.0, 2.0]:
        probabilities_tau = softmax([value / tau for value in sample_logits])
        current_order = sorted(range(3), key=lambda i: probabilities_tau[i], reverse=True)
        print('tau =', tau, '→', rounded(probabilities_tau))
        assert current_order == original_order
    """)
    md("""
    ## 实验 14 · top-k、top-p 与重新归一化 · PPT 21–22

    top-k 保留概率最高的 k 项；top-p 保留按概率降序排列后，累计概率首次达到或超过阈值的前缀。
    两者都将其余项置零，再对保留项重新归一化。

    本实现中，概率相等时按原 ID 排序，输出按原词表顺序排列。
    判断累计概率是否达到阈值时，使用 `isclose` 处理 1e-12 以内的浮点误差。
    """)
    code("""
    def renormalize_kept(probabilities, kept):
        total = math.fsum(probabilities[i] for i in kept)
        return [probability / total if i in kept else 0.0
                for i, probability in enumerate(probabilities)]

    def keep_top_k(probabilities, k):
        assert isinstance(k, int) and 1 <= k <= len(probabilities)
        order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
        return renormalize_kept(probabilities, set(order[:k]))

    def keep_top_p(probabilities, threshold):
        assert 0.0 < threshold <= 1.0
        order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
        cumulative, kept = 0.0, set()
        for i in order:
            kept.add(i)
            cumulative += probabilities[i]
            if cumulative >= threshold or math.isclose(
                    cumulative, threshold, rel_tol=0.0, abs_tol=1e-12):
                break
        return renormalize_kept(probabilities, kept)

    print('top-k=2：', rounded(keep_top_k(sample_probabilities, 2)))
    expected_supports = {0.5: [0], 0.7: [0, 1], 0.8: [0, 1], 1.0: [0, 1, 2]}
    for threshold, expected_support in expected_supports.items():
        filtered = keep_top_p(sample_probabilities, threshold)
        support = [i for i, value in enumerate(filtered) if value > 0.0]
        print('top-p =', threshold, '保留', support, '概率', rounded(filtered))
        assert support == expected_support

    different_distribution = [0.4, 0.25, 0.15, 0.12, 0.08]
    print('另一个分布，top-p=.7：', rounded(keep_top_p(different_distribution, 0.7)))
    assert close_vectors(keep_top_p(different_distribution, 0.7), [0.5, 0.3125, 0.1875, 0.0, 0.0])
    """)
    md("""
    ## 实验 15 · 组合温度与候选截断 · PPT 23

    本实现的顺序为：原始 logits → 正温度 → top-k（可选）→ top-p（可选）→ 抽样。
    两种截断连续使用时，top-p 作用于 top-k 后重新归一化的分布。
    正温度不会改变 top-k 的排序，但会改变 top-p 的累计概率和保留集合。
    """)
    code("""
    def decode_one(logits, rng, tau=1.0, top_k=None, top_p=None, greedy=False):
        if greedy:
            return max(range(len(logits)), key=lambda i: logits[i])
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('概率抽样要求 tau>0；greedy 请使用独立分支。')
        probabilities = softmax([value / tau for value in logits])
        if top_k is not None:
            probabilities = keep_top_k(probabilities, top_k)
        if top_p is not None:
            probabilities = keep_top_p(probabilities, top_p)
        return sample_cdf(probabilities, rng)

    supports_at_temperature = []
    for tau in [0.5, 1.0, 2.0]:
        values = softmax([value / tau for value in sample_logits])
        kept = keep_top_p(values, 0.6)
        support = [i for i, p in enumerate(kept) if p > 0]
        supports_at_temperature.append(support)
        print('tau', tau, 'top-p=.6 保留', support)
    assert supports_at_temperature == [[0], [0, 1], [0, 1]]
    sampled_id = decode_one(sample_logits, random.Random(7), tau=1.0, top_p=0.7)
    print('组合解码选择的 ID：', sampled_id, labels[sampled_id])
    """)
    md("""
    ## 实验 16 · 自回归生成循环 · PPT 24–25

    每一轮根据当前上下文计算 logits，选择一个 ID，追加到输入，再进行下一轮。
    达到 EOS 或长度上限时停止。下方 `autoregressive` 函数接收模型前向函数和解码函数，
    可与训练完成的模型配合使用；随后用给定 ID=4 演示列表追加。

    普通生成时模型参数固定，输入上下文逐步增长。
    列表演示使用下方的八项局部词表，包含起始和结束标记。
    """)
    code("""
    def autoregressive(model_forward, initial_ids, decode, eos_id, max_new_tokens=6):
        ids = list(initial_ids)
        for _ in range(max_new_tokens):
            next_logits = model_forward(ids)  # 当前上下文最后位置的词表分数。
            next_id = decode(next_logits)
            ids.append(next_id)
            if next_id == eos_id:
                break
        return ids

    loop_vocab = ['<bos>', '我', '喜欢', '喝', '牛奶', '茶', '水', '<eos>']
    context_ids = [0, 1, 2, 3]
    next_id_given = 4  # 给定本步选择结果，演示上下文追加。
    print('追加前：', context_ids, '→', ' '.join(loop_vocab[i] for i in context_ids))
    context_ids.append(next_id_given)
    print('追加后：', context_ids, '→', ' '.join(loop_vocab[i] for i in context_ids))
    print('下一步调用 model_forward(context_ids)，根据新上下文计算词表分数。')
    """)
    md("""
    ## 可选核对 · PyTorch Linear 的权重形状 · PPT 15

    公式右乘矩阵是 `[in,out]`，PyTorch 存储 `[out,in]`，计算 `x @ weight.T + bias`。
    因此 `nn.Linear(2,4).weight` 的形状为 `[4,2]`，复制本课参数时使用 `W1.T`、`W2.T`。
    核对使用 float64 和零偏置，与手算设置相同；未安装 torch 时跳过这一单元。
    """)
    code("""
    try:
        import torch
    except ImportError:
        print('跳过可选 PyTorch 核对：当前环境未安装 torch；全部标准库计算已经完成。')
    else:
        layer1 = torch.nn.Linear(2, 4, dtype=torch.float64)
        layer2 = torch.nn.Linear(4, 2, dtype=torch.float64)
        with torch.no_grad():
            layer1.weight.copy_(torch.tensor(W1, dtype=torch.float64).T)
            layer2.weight.copy_(torch.tensor(W2, dtype=torch.float64).T)
            layer1.bias.zero_()
            layer2.bias.zero_()
            tensor_u = torch.tensor(u, dtype=torch.float64)
            torch_result = layer2(torch.relu(layer1(tensor_u)))
        print('Linear(2,4).weight.shape：', tuple(layer1.weight.shape))
        print('PyTorch 输出：', torch_result.tolist(), '标准库输出：', ffn(u))
        torch.testing.assert_close(torch_result, torch.tensor(ffn(u), dtype=torch.float64))

        norm = torch.nn.LayerNorm(2, eps=1e-5, dtype=torch.float64)
        with torch.no_grad():
            torch_norm = norm(torch.tensor(h, dtype=torch.float64))
        torch.testing.assert_close(torch_norm, torch.tensor(normalized_h, dtype=torch.float64))
        print('LayerNorm(D) 与沿单行总体方差的计算相符。')
    """)
    md("""
    ## 综合练习 · PPT 26–28

    1. 为什么共享同一套 FFN 参数，不代表各位置输出相同？
    2. 为什么 ReLU 后非负，FFN 最终输出仍可能为负？回到实验 04 找到证据。
    3. 去掉 FFN 激活能合并哪两层？这是否说明 Attention 也变线性？
    4. top-p=.7 为什么留下概率 .3 的“茶”？被保留的原始概率总和是多少？
    5. 固定分布频数实验与自回归生成，对上下文的处理有什么区别？

    回答时引用相应算例或公式。下一课将把 logits 与下一 token 的目标对齐，
    通过损失、反向传播和优化器讲解参数学习。
    """)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = root / "demo.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1]))
