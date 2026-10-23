"""机制版课堂 Notebook：公式、真实接口与可执行验证相互对应。"""
import ast
import inspect
import json
import re

import nbformat as nbf
from lesson_alignment import LINKS, source_label


def function_source(root, file, name):
    source = (root / "code" / file).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def build(root):
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(inspect.cleandoc(text)))

    def code(text):
        source = inspect.cleandoc(text)
        compile(source, "<lesson-cell>", "exec")
        cells.append(nbf.v4.new_code_cell(source))

    md("""# 第二课：Tokenizer、Embedding 与位置机制

    通过 BPE 算法、真实 ByteLevel 配置、自动求导与旋转运算，解释模型输入。
    先按 README 安装依赖，使用项目 Python 内核。本地配置已保存，不下载预训练模型。
    课堂主讲标注的核心语句；辅助函数的完整实现可在 code/ 中查看。""")
    code('''from pathlib import Path
    from collections import Counter
    import math
    import sys
    import torch
    from torch import nn
    from torch.nn import functional as F
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    ROOT = next(p for p in [Path.cwd(), Path.cwd() / "class2", Path.cwd().parent]
                if (p / "code/bpe_core.py").exists())
    sys.path.insert(0, str(ROOT / "code"))
    from bpe_core import merge_pair, BPE_CORPUS
    from runtime import load_tokenizer, make_batch
    from embedding_train import bigram_batch
    from worked_examples import lookup_example, gradient_example, cross_entropy_example, rope_example, mask_example
    torch.set_num_threads(1)
    ''')
    md("""## 1. BPE 训练：频次怎样影响每轮选择？

    对每个当前相邻 pair 累加 `句子频次 × 出现次数`，选最高频，再重新统计。
    这里是可读的字符级算法，负责解释规则；真实 ByteLevel 库在后面另训练。""")
    code(function_source(root, "bpe_core.py", "train_bpe") + '\n\ncorpus = BPE_CORPUS\nprint("奶茶语料：", corpus)\nmerges, history = train_bpe(corpus, 4)\nfor row in history:\n    print(row)')
    md("""## 2. BPE 编码：rank 优先，而非最长匹配

    先执行上一页奶茶语料学到的四条规则，打印每一步切分；再看重叠候选 `ab`、`bc` 谁先合并。
    `bc×6、ab×5、abc×1` 会学到 `bc` 优先于 `ab`，编码 `abc` 得到 `a/bc`。""")
    code(function_source(root, "bpe_core.py", "encode_bpe") + '\n\nresult, trace = encode_bpe("你喜欢喝奶茶", merges)\nfor row in trace:\n    print(row["rank"], " / ".join(row["after"]))\nprint("新组合：", encode_bpe("你喜欢喝咖啡", merges)[0])\nconflict_merges, _ = train_bpe({"bc": 6, "ab": 5, "abc": 1}, 2)\nprint("重叠反例规则：", conflict_merges)\nprint("rank 编码：", encode_bpe("abc", conflict_merges))')
    md("""## 3. 用真实库训练并保存完整配置

    `ByteLevel` 预切分、256个基础字节符号、4个特殊符号、BPE合并组成配置。
    当前单元训练一个本地对象；后续实验使用仓库已保存的320项配置，以固定ID。
    小语料只用于机制观察，不是商用模型的 tokenizer。""")
    code('SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]\n\n' + function_source(root, "train_tokenizer.py", "train") + '\n\nfresh = train(320)\nprint("新训练词表大小：", fresh.get_vocab_size())\nprint("冻结配置词表大小：", load_tokenizer().get_vocab_size())')
    md("""## 4. 固定配置：文本、ID 与参数地址配套

    先结束字符级奶茶演示，再切换到真实 ByteLevel 配置。两者的片段、ID不能混用。
    查看默认文本的ID与四个特殊符号。bos/eos由batch函数显式添加，不假定encode自动插入。""")
    code('''tokenizer = load_tokenizer()
    text = "我喜欢喝牛奶。"
    enc = tokenizer.encode(text)
    print("tokens：", enc.tokens)
    print("IDs：", enc.ids)
    print("回解：", tokenizer.decode(enc.ids))
    print("特殊ID：", {s: tokenizer.token_to_id(s) for s in ["<pad>", "<unk>", "<bos>", "<eos>"]})
    tokenizer = load_tokenizer()
    assert enc.ids == [300, 304, 262]''')
    md("""## 5. batch 查表与 one-hot 前向等价

    IDs `[B,T]` → 表E `[V,D]` → 向量 `[B,T,D]`。
    这里只为了验证构造one-hot；实际训练不需要分配如此大的中间张量。""")
    code('''torch.manual_seed(7)
    ids, valid = make_batch(tokenizer, ["我喜欢喝牛奶。", "我喜欢喝酒。"])
    pad = tokenizer.token_to_id("<pad>")
    embedding = nn.Embedding(tokenizer.get_vocab_size(), 8, padding_idx=pad)
    vectors = embedding(ids)
    one_hot = F.one_hot(ids, tokenizer.get_vocab_size()).to(vectors.dtype)
    print("IDs / E / X：", list(ids.shape), list(embedding.weight.shape), list(vectors.shape))
    print("前向等价：", torch.allclose(vectors, one_hot @ embedding.weight))
    print("valid mask：", valid.tolist())
    print("PPT算术小表（独立ID）：", lookup_example())''')
    md("""## 6. gather 反传：同一行的梯度累加

    对 `[a,a,b,pad]` 的输出求和，上游每维都是1。a行收到两次贡献，b一次，pad的lookup梯度为0。
    这只是诊断loss。`padding_idx`不会自动屏蔽Attention或loss；权重绑定时还有输出梯度路径。""")
    code('''a, b = tokenizer.encode("我喜欢喝牛奶。").ids[:2]
    repeated = torch.tensor([[a, a, b, pad]])
    embedding.zero_grad()
    embedding(repeated).sum().backward()
    print("a行梯度：", embedding.weight.grad[a].tolist())
    print("b行梯度：", embedding.weight.grad[b].tolist())
    print("pad行梯度：", embedding.weight.grad[pad].tolist())
    print("非全1上游梯度：", gradient_example())''')
    md("""## 7. 单位置交叉熵梯度：手算与 autograd 对齐

    先算 logits 和类别概率，再将预测误差乘回输出权重。本例一个位置，不含 batch 平均因子。
    将手算结果与自动求导逐项比较。""")
    code('''torch.manual_seed(7)
    embedding_vector = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    output_weight = torch.randn(6, 4, dtype=torch.float64)
    target = torch.tensor([2])
    logits = embedding_vector @ output_weight.T
    loss = F.cross_entropy(logits, target)
    loss.backward()
    probabilities = logits.detach().softmax(dim=-1)
    error = probabilities - F.one_hot(target, 6)
    manual_gradient = error @ output_weight
    print("autograd：", embedding_vector.grad.tolist())
    print("手算：", manual_gradient.tolist())
    torch.testing.assert_close(embedding_vector.grad, manual_gradient)
    small_ce = cross_entropy_example()
    print("PPT三类CE：", round(small_ce["loss"], 4))
    print("PPT输入梯度：", [round(v, 4) for v in small_ce["vector_gradient"]])''')
    md("""## 8. 真实下一 token 损失更新 embedding

    七句文本构成57对相邻token。`Embedding → Linear → CrossEntropy` 是bigram基线：每次只看一个token。
    输出原始logits给交叉熵。没有验证集；观察训练拟合，不把它称为Transformer。""")
    code('''torch.manual_seed(7)
    x, y = bigram_batch(tokenizer)
    embedding = nn.Embedding(tokenizer.get_vocab_size(), 16)
    output = nn.Linear(16, tokenizer.get_vocab_size())
    optimizer = torch.optim.Adam(list(embedding.parameters()) + list(output.parameters()), lr=0.05)
    before = embedding.weight.detach().clone()
    for step in range(121):
        logits = output(embedding(x))
        loss = F.cross_entropy(logits, y)
        if step % 30 == 0:
            print(step, round(float(loss.detach()), 4))
        if step == 120:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("改变表行数：", int(((embedding.weight.detach()-before).norm(dim=1) > 0).sum()))''')
    md("""## 9. 无位置、无遮罩的注意力：置换等变

    重排输入后输出也对应重排。给重新排列的token使用新的当前位置编码，则通常不再满足这一等式。
    这里用Q=K=V=X做最小验证，因果mask不在该等变结论的假设中。""")
    code(function_source(root, "position_mechanics.py", "attention") + '''

torch.manual_seed(7)
x = torch.randn(3, 4, dtype=torch.float64)
permutation = torch.tensor([2, 1, 0])
original = attention(x)
rearranged = attention(x[permutation])
print("无位置时等变：", torch.allclose(rearranged, original[permutation]))
from positions import sinusoidal
pe = sinusoidal(torch.arange(3), 4)
print("加入位置后仍等变：",
      torch.allclose(attention(x[permutation]+pe), attention(x+pe)[permutation]))''')
    md("""## 10. 向量化正弦位置编码

    `[T,1] × [1,D/2] → [T,D/2]` 的广播产生各位置、各频率的角度。
    相邻两维分别放sin/cos。固定偏移可由一组二维旋转表达。""")
    code(function_source(root, "positions.py", "sinusoidal") + '\n\nprint(sinusoidal(torch.arange(3), 4).round(decimals=4))')
    md("""## 11. RoPE：成对旋转与相对位移

    q/k 分别按位置 m/n 旋转，利用旋转矩阵的转置和角度相加性质，点积中出现位置之差。
    实现采用相邻维度配对，输入可为`[...,T,D]`，广播到batch和head。整体平移位置测试固定未旋转的Q/K内容。""")
    code(function_source(root, "positions.py", "apply_rope") + '''

torch.manual_seed(7)
q, k = torch.randn(2, 3, 4, dtype=torch.float64)
p = torch.arange(3)
score = apply_rope(q, p) @ apply_rope(k, p).T
shifted = apply_rope(q, p+5) @ apply_rope(k, p+5).T
print("整体平移后最大差异：", float((score-shifted).abs().max()))
print("范数保持：", torch.allclose(q.norm(dim=-1), apply_rope(q, p).norm(dim=-1)))
small_rope = rope_example()
print("二维PPT算例：", round(small_rope["score"], 4), round(small_rope["shifted_score"], 4))''')
    md("""## 12. 接口整合：真实ID、多头Q/K、RoPE、两种mask

    打开`code/input_pipeline.py`，核对每步shape。此分支直接在Q/K使用RoPE，不叠加相加式绝对位置。
    参数是随机初始化；脚本不包含完整Block、语言模型训练或生成循环。""")
    code(function_source(root, "input_pipeline.py", "demo").replace("def demo():", "def input_pipeline():")
         + '\n\npipeline_result = input_pipeline()\nprint("四位置mask算例：", mask_example()["weights"])')
    md("""## 复习与作业

    解释：rank与最长前缀为何不同？重复token梯度为何累加？绑定输入输出权重后谁还会更新？
    RoPE与因果mask各控制什么？详见 [练习](exercises.md) 与 [讲义](student_handout.md)。""")
    # 同一索引驱动 PPT、对照表和 Notebook，避免实验编号与公式各自维护。
    primary = {1: ["bpe_train"], 2: ["rank"], 3: ["hf_pipeline"], 4: ["artifact"],
               5: ["onehot"], 6: ["grad"], 7: ["chain"], 8: ["training"],
               9: ["permutation"], 10: ["frequencies"],
               11: ["rotation", "vector_rope"], 12: ["mask", "pipeline"]}
    slides = json.loads((root / "slides.json").read_text(encoding="utf-8"))
    slide_numbers = {item["kind"]: i for i, item in enumerate(slides, 1)}
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        match = re.match(r"## (\d+)\.", cell.source)
        if not match:
            continue
        for kind in primary[int(match.group(1))]:
            link = LINKS[kind]
            cell.source += (f"\n\n**PPT {slide_numbers[kind]:02d}** · "
                            f"[{source_label(link)}](code/{link['file']})")
            for formula in link.get("formulas", []):
                cell.source += "\n\n$$\n" + formula + "\n$$"
            if link.get("symbols"):
                cell.source += "\n\n" + link["symbols"] + "。"
            if link.get("worked_function"):
                cell.source += f"\n\n**页内数值算例**：[worked_examples.py:{link['worked_function']}](code/worked_examples.py)。"
            cell.source += "\n\n**运行后核对**：" + link["result"] + "。"
    cells[0].source += "\n\n备课和复习可查看 [知识、公式与代码对照](knowledge_code_map.md)。"
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "file_extension": ".py"}}
    nbf.write(notebook, root / "demo.ipynb")
