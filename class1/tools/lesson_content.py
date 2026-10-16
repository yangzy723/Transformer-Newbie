"""简明版的教学图示和 Notebook 内容；课堂只阅读 code/lesson.py。"""

import re

import nbformat as nbf


TEAL, LIGHT, BLUE, MUTED = "087F8C", "E6EFEE", "DEE8F0", "5C6F7B"


def flow(c, labels, y=2.15, height=.56, gap=.24, size=18):
    for i, label in enumerate(labels):
        top = y+i*(height+gap)
        c.node(7.55, top, 4.75, height, label, LIGHT if i % 2 == 0 else BLUE, size)
        if i < len(labels)-1:
            c.line(9.925, top+height+.01, 9.925, top+height+gap-.01, arrow=True)


def draw_diagram(c, kind):
    if kind == "opening":
        c.text(7.5, 2.35, 4.8, .4, "只用这一句例子", 20, TEAL)
        c.text(7.5, 3.2, 4.8, 1.4, "我 喜欢 喝\n____", 35, bold=True)
        c.node(7.5, 4.85, 2.05, .58, "酒", BLUE, 23)
        c.node(9.9, 4.85, 2.1, .58, "牛奶", LIGHT, 23)
    elif kind == "roadmap":
        flow(c, ["① 文字变数字", "② 按权重混合信息", "③ 分数 → 概率 → 下一项"], y=2.55, height=.65, gap=.32, size=21)
    elif kind == "tokens":
        flow(c, ["我 喜欢 喝", '["我", "喜欢", "喝"]', "[0, 1, 2]"], y=2.55, height=.65, gap=.32, size=22)
    elif kind == "embedding":
        c.text(7.5, 2.25, 4.8, .4, "输入编号 → 查表 → 向量", 20, TEAL)
        for i, (word, idx, vector) in enumerate((("我", 0, "[1, 0]"), ("喜欢", 1, "[0, 1]"), ("喝", 2, "[1, 1]"))):
            c.node(7.5, 3.05+i*.67, 4.65, .55, f"{word}    ID={idx}    →    {vector}", BLUE, 20)
        c.text(7.5, 5.33, 4.8, .4, "人工数字；每个位置仅有 2 维", 17, MUTED)
    elif kind == "position":
        flow(c, ["我 喜欢 喝", "与“喝 喜欢 我”不同", "真实模型加入位置信息"], y=2.55, height=.65, gap=.32, size=22)
    elif kind in ("mixing_intro", "mixing_result"):
        weights = "[0.2, 0.3, 0.5]" if kind == "mixing_intro" else "[0.248, 0.248, 0.503]"
        result = "[0.7, 0.8]" if kind == "mixing_intro" else "[0.752, 0.752]"
        flow(c, [f"权重：{weights}", "V：[1,0] / [0,1] / [1,1]", "分别对两个数字加权求和", f"输出向量：{result}"], y=2.18, height=.58, gap=.25, size=18)
    elif kind == "qkv_simple":
        flow(c, ["Q：当前位置“喝” → [1,1]", "K：用来算匹配分数", "V：按权重混合的信息"], y=2.55, height=.65, gap=.32, size=19)
        c.text(7.5, 5.48, 4.8, .3, "本例为恒等投影；实际常分别学习投影", 13, MUTED)
    elif kind == "scores_simple":
        flow(c, ["Q：[1,1]", "点积：[1, 1, 2]", "除以 √2", "分数：[0.707,0.707,1.414]"], height=.58, size=18)
    elif kind == "weights_simple":
        c.text(7.5, 2.2, 4.8, .4, "Softmax 后的混合权重", 20, TEAL)
        for i, (word, weight) in enumerate((("我", .248), ("喜欢", .248), ("喝", .503))):
            y=3+i*.72
            c.text(7.5, y, 1, .4, word, 19)
            c.rect(8.65, y, 3*weight, .37, TEAL)
            c.text(11.55, y, .7, .4, f"{weight:.3f}", 18)
        c.text(7.5, 5.38, 4.8, .3, "每个已知输入位置得到一个权重", 16, MUTED)
    elif kind == "mask_simple":
        flow(c, ["如果当前在第一个位置", "分数：[0.707, −∞, −∞]", "未来位置不可读取", "权重：[1, 0, 0]"], height=.58, size=20)
    elif kind == "next_simple":
        flow(c, ["混合向量：[0.752,0.752]", "人工映射 → 两个候选分数", "[1.503, 0.752]", "酒：0.680 / 牛奶：0.320"], height=.58, size=18)
    elif kind == "generation_simple":
        flow(c, ["前文：我 喜欢 喝", "选择下一 token：酒", "新前文：我 喜欢 喝 酒", "真实生成会重复预测"], height=.58, size=20)
    elif kind == "architecture_simple":
        flow(c, ["Token 表示 + 位置信息", "Block：Attention + FFN", "内部包含残差与归一化", "最终归一化 + 输出映射", "整个词表的概率 → 下一 token"], y=2.05, height=.49, gap=.2, size=16)
        c.text(7.55, 5.63, 4.75, .25, "文本生成的 decoder-only 架构示意", 12, MUTED)
    elif kind == "training_simple":
        c.text(7.5, 2.25, 4.8, .4, "训练：改变参数", 22, TEAL)
        c.node(7.5, 2.95, 4.7, .64, "句子 → 预测 → 误差 → 更新", BLUE, 18)
        c.text(7.5, 4.05, 4.8, .4, "生成：增加上下文", 22, TEAL)
        c.node(7.5, 4.75, 4.7, .64, "前文 → 预测 → 接上新 token", LIGHT, 18)
    elif kind == "quiz_simple":
        flow(c, ["文字 → 数字", "权重 → 混合向量", "分数 → 概率 → 下一 token"], y=2.55, height=.65, gap=.32, size=20)
    else:
        raise ValueError(kind)


def build_simple_notebook(root):
    descriptions = [
        ("准备：只用 Python 标准库", "先定义一个短函数 softmax：分数变成非负、总和为 1 的比例。减去最大值用于数值稳定。本课参数人工设定，展示一次计算；完整训练实现放在 optional/。"),
        ("① 文字变数字", "输入固定为“我 喜欢 喝”。先按空格切分，再查 ID，再查二维 embedding 表。预测：最后一个向量是什么？本课 ID 仅是编号，数字大小没有语义。"),
        ("② 从最后一个位置提出查询", "本例 Q/K/V 采用恒等投影，数字相同但角色不同。Q/K 用于确定权重，V 用于混合。实际模型一般分别学习三个线性投影。只算“喝”这个位置的一行。"),
        ("② 算匹配分数与权重", "逐项相乘再相加是点积，除以 √2 控制分数尺度。运行前先算三个点积：1、1、2。Softmax 后权重约为 [0.248,0.248,0.503]。显示值的和可能有四舍五入差异。"),
        ("② 按权重混合信息", "分别计算两个维度的加权和。预期输出约 [0.752,0.752]。它是向量，尚未选出 token。"),
        ("③ 候选分数变概率，再选择", "本例用人工系数把混合向量映射到两个候选的分数，Softmax 得到约 [0.680,0.320]。真实模型会输出整个词表的分数，且在完整 Block 后做输出映射。本例只演示一次选择，无训练与生成循环。"),
    ]
    source = (root / "code/lesson.py").read_text(encoding="utf-8")
    blocks = [block.strip() for block in re.split(r"(?m)^# %% .*\n", source) if block.strip()]
    if len(blocks) != len(descriptions):
        raise ValueError("课堂脚本与 Notebook 步骤数量不一致")
    cells = [nbf.v4.new_markdown_cell("# Class 1 简明版：同一句例子，走通三个步骤\n\n文字变数字 → Attention 混合信息 → 下一 token。下面代码与 `code/lesson.py` 相同；无需 PyTorch、NumPy 或网络。保留的输出来自实际执行。")]
    for (title, explanation), block in zip(descriptions, blocks):
        compile(block, "<lesson-cell>", "exec")
        cells += [nbf.v4.new_markdown_cell(f"## {title}\n\n{explanation}"), nbf.v4.new_code_cell(block)]
    cells += [
        nbf.v4.new_markdown_cell("## 小实验：只改 V，先猜结果\n\n保持 Q/K 和权重不变，仅把第三个 V 改成 [3,1]。新输出约为 [1.759,0.752]。这里新建一个列表，避免同时修改共享的 K。"),
        nbf.v4.new_code_cell('changed_values = [[1, 0], [0, 1], [3, 1]]\nchanged_mixed = [0.0, 0.0]\nfor i in range(len(tokens)):\n    changed_mixed[0] += weights[i] * changed_values[i][0]\n    changed_mixed[1] += weights[i] * changed_values[i][1]\nprint("权重仍是：", [round(w, 3) for w in weights])\nprint("新输出：", [round(v, 3) for v in changed_mixed])'),
        nbf.v4.new_markdown_cell("## 因果遮罩：只看第一个位置\n\n主例子中的最后一个位置可以读三个已知输入。如果换成第一个位置，未来位置必须遮住；在 Softmax 前设成负无穷，权重变成 0。"),
        nbf.v4.new_code_cell('first_position_scores = [0.707, -math.inf, -math.inf]\nfirst_position_weights = softmax(first_position_scores)\nprint("第一个位置的权重：", first_position_weights)'),
        nbf.v4.new_markdown_cell("## 把代码放回完整架构\n\n真实 decoder-only 模型会加入位置信息，并使用含 Attention、FFN、残差和归一化的 Block；可以叠多层。本课脚本省略这些模块，不能直接称为完整 Transformer。训练会学习参数，生成不更新参数。\n\n检查：ID 和向量有什么区别？Attention 输出为何还不能直接当作 token？两次 Softmax 的对象有什么不同？完整训练另见 [optional/](optional/README.md)。"),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "file_extension": ".py"}}
    nbf.write(nb, root / "demo.ipynb")
