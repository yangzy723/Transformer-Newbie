"""PPT、Notebook 和知识代码对照表共用的授课索引。"""

LINKS = {
    "bpe_train": {
        "experiment": 1, "file": "bpe_core.py", "function": "train_bpe",
        "formulas": [r"S(a,b)=\sum_s f_s\,\operatorname{count}_s(a,b)"],
        "symbols": "f_s：文本频次；count：当前切分中的相邻次数",
        "code": ["counts[pair] += corpus[text]", "pair = max(counts, key=counts.get)"],
        "result": "四轮合并频次：10 → 10 → 8 → 8",
        "question": "怎样用语料频次决定下一次合并？",
    },
    "rank": {
        "experiment": 2, "file": "bpe_core.py", "function": "encode_bpe",
        "formulas": [r"(a,b)^*=\underset{(a,b)\in\mathcal{C}}{\arg\min}\;\operatorname{rank}(a,b)"],
        "symbols": "C：当前相邻、且在 merges 中的候选 pair",
        "code": ["rank, pair = min(candidates, key=lambda item: item[0])", "parts = merge_pair(parts, pair)"],
        "result": "你喜欢喝奶茶 → 你 / 喜欢喝奶茶；重叠反例 abc → a / bc",
        "question": "规则学好以后，编码时先执行哪条？",
    },
    "hf_pipeline": {
        "experiment": 3, "file": "train_tokenizer.py", "function": "train",
        "result": "从本地语料训练，完整配置保存在 JSON 中",
        "question": "字符级算法怎样落到真实的字节级分词器？",
    },
    "artifact": {
        "experiment": 4, "file": "runtime.py", "function": "load_tokenizer",
        "result": "默认配置下：我喜欢喝牛奶。→ [300,304,262]",
        "question": "为什么分词器必须和参数表配套？",
    },
    "onehot": {
        "experiment": 5, "file": "embeddings.py", "function": "demo",
        "formulas": [r"\mathbf{E}\in\mathbb{R}^{V\times D},\quad\mathbf{X}=\mathbf{E}[I]",
                     r"\mathbf{X}=\operatorname{one\_hot}(I)\,\mathbf{E}"],
        "symbols": "I：[B,T]；E：[V,D]；X：[B,T,D]",
        "code": ["vectors = embedding(ids)"],
        "result": "[2,6] → [2,6,8]；与 one-hot 乘法前向等价",
        "question": "为什么 ID 要变成向量，查表对应什么运算？",
    },
    "lookup": {
        "experiment": 5, "file": "runtime.py", "function": "make_batch",
        "result": "第一句 5 项补齐到 6；第二句 6 项无需补齐",
        "question": "长短不同的句子怎样组成同一个 batch？",
    },
    "grad": {
        "experiment": 6, "file": "embeddings.py", "function": "demo",
        "formulas": [r"\mathbf{g}_{b,t}=\frac{\partial L}{\partial\mathbf{X}_{b,t}}",
                     r"\frac{\partial L}{\partial\mathbf{E}_j}=\sum_{(b,t):\,I_{b,t}=j}\mathbf{g}_{b,t}"],
        "symbols": "对非 pad 行 j；遍历 batch 中所有读取它的位置",
        "code": ["repeated = torch.tensor([[a, a, b, pad]])", "embedding(repeated).sum().backward()"],
        "result": "每维梯度：a 行 2，b 行 1，pad 行 0",
        "question": "同一个 token 出现多次，梯度覆盖还是累加？",
    },
    "chain": {
        "experiment": 7, "file": "gradient_chain.py", "function": "demo",
        "formulas": [r"\mathbf{z}=\mathbf{e}\mathbf{W}_{\rm out}^{\mathsf{T}},\quad L=-\log p_y",
                     r"\frac{\partial L}{\partial\mathbf{e}}=(\mathbf{p}-\operatorname{one\_hot}(y))\mathbf{W}_{\rm out}"],
        "symbols": "p=softmax(z)；W_out：[V,D]；向量梯度：[D]",
        "code": ["logits = embedding_vector @ output_weight.T", "loss = F.cross_entropy(logits, target)", "loss.backward()"],
        "result": "手算向量梯度与 autograd 逐项一致",
        "question": "预测错了，误差信号怎样传回输入向量？",
    },
    "training": {
        "experiment": 8, "file": "embedding_train.py", "function": "train",
        "formulas": [r"\mathbf{e}=\mathbf{E}[x],\quad\mathbf{z}=\mathbf{e}\mathbf{W}_{\rm out}^{\mathsf{T}}+\mathbf{b}",
                     r"L=-\frac{1}{N}\sum_{r=1}^{N}\log p(y_r\mid x_r)"],
        "symbols": "x/y：相邻 token；N=57；只用当前 token 预测下一项",
        "code": ["logits = output(embedding(x))", "loss = F.cross_entropy(logits, y)", "loss.backward()"],
        "result": "120 步：loss 5.7565 → 0.3559；26 行更新",
        "question": "把梯度放进训练循环后，参数是否真的变化？",
    },
    "tie": {
        "experiment": None, "file": "../tools/check_experiments.py", "function": "main",
        "formulas": [r"\mathbf{z}=\mathbf{h}\mathbf{E}^{\mathsf{T}}+\mathbf{b}",
                     r"\nabla_{\mathbf{E}}L=\nabla_{\mathbf{E}}^{\rm lookup}L+\nabla_{\mathbf{E}}^{\rm output}L"],
        "symbols": "输入与输出共用同一个参数；主训练实验保持未绑定",
        "result": "课后验证：没作为输入的行也可能得到输出梯度",
        "question": "输入表和输出分类器共用参数，会发生什么？",
    },
    "permutation": {
        "experiment": 9, "file": "position_mechanics.py", "function": "attention",
        "formulas": [r"\mathbf{S}'=\Pi\mathbf{S}\Pi^{\mathsf{T}}",
                     r"\operatorname{Attn}(\Pi\mathbf{X})=\Pi\operatorname{Attn}(\mathbf{X})"],
        "symbols": "Π：行置换矩阵；假设无位置机制、无 mask",
        "code": ["scores = x @ x.T / math.sqrt(x.shape[-1])", "return scores.softmax(dim=-1) @ x"],
        "result": "重排后输出也重排：True；加入位置后通常 False",
        "question": "为什么仅有 token 向量，还没有显式位置坐标？",
    },
    "absolute": {
        "experiment": 10, "file": "positions.py", "function": "sinusoidal",
        "formulas": [r"\mathbf{X}_{b,t}=\mathbf{E}[I_{b,t}]+\operatorname{PE}(t)"],
        "symbols": "t：token 序号；PE 来自学习式位置表或固定函数",
        "code": [],
        "result": "相加后形状保持 [B,T,D]；本课代码演示固定函数",
        "question": "怎样在不增加向量维度的前提下加入位置？",
    },
    "frequencies": {
        "experiment": 10, "file": "positions.py", "function": "sinusoidal",
        "formulas": [r"\omega_i=10000^{-2i/D},\quad\theta_{t,i}=t\omega_i",
                     r"\operatorname{PE}(t,2i)=\sin\theta_{t,i},\quad\operatorname{PE}(t,2i+1)=\cos\theta_{t,i}"],
        "symbols": "i：维度对编号；D=4 时 ω=[1,0.01]",
        "code": ["angles = positions[:, None] * frequencies[None, :]", "result[:, 0::2] = angles.sin()", "result[:, 1::2] = angles.cos()"],
        "result": "位置 0：[0,1,0,1]；不同维度对有不同变化尺度",
        "question": "正弦编码怎样从位置和频率生成一个向量？",
    },
    "rotation": {
        "experiment": 11, "file": "positions.py", "function": "apply_rope",
        "formulas": [r"R(\varphi)=\begin{bmatrix}\cos\varphi&-\sin\varphi\\\sin\varphi&\cos\varphi\end{bmatrix}",
                     r"\widetilde{\mathbf{q}}_m^{\mathsf{T}}\widetilde{\mathbf{k}}_n=\mathbf{q}^{\mathsf{T}}R((n-m)\omega)\mathbf{k}"],
        "symbols": "m/n：位置；ω：某一维度对的频率；φ=tω",
        "code": ["cos, sin = angles.cos(), angles.sin()", "result[..., 0::2] = even * cos - odd * sin", "result[..., 1::2] = even * sin + odd * cos"],
        "result": "固定 q/k 内容：点积由相对位置 n−m 调节",
        "question": "为什么按绝对位置旋转，却产生相对位移？",
    },
    "vector_rope": {
        "experiment": 11, "file": "positions.py", "function": "apply_rope",
        "formulas": [r"x'_{2i}=x_{2i}\cos\theta_{t,i}-x_{2i+1}\sin\theta_{t,i}",
                     r"x'_{2i+1}=x_{2i}\sin\theta_{t,i}+x_{2i+1}\cos\theta_{t,i}"],
        "symbols": "even/odd 是相邻分量；角度广播到 batch 和 head",
        "code": ["even, odd = x[..., 0::2], x[..., 1::2]", "angles = positions[:, None] * frequencies[None, :]"],
        "result": "范数保持；整体平移位置后分数误差约 10⁻¹⁵",
        "question": "二维旋转怎样成为一个多头张量运算？",
    },
    "mask": {
        "experiment": 12, "file": "input_pipeline.py", "function": "demo",
        "formulas": [r"\mathbf{S}=\frac{\widetilde{\mathbf{Q}}\widetilde{\mathbf{K}}^{\mathsf{T}}}{\sqrt{d_h}}",
                     r"\mathbf{A}=\operatorname{softmax}(\mathbf{S}+\mathbf{M})"],
        "symbols": "M：可见为 0，禁止为 −∞；softmax 沿 key 维",
        "code": ["masked = scores.masked_fill(forbidden, -torch.inf)", "weights = masked.softmax(dim=-1)"],
        "result": "未来 key 与 padding key 的权重均为 0",
        "question": "有了位置信息，为什么还需要 mask？",
    },
    "pipeline": {
        "experiment": 12, "file": "input_pipeline.py", "function": "demo",
        "result": "ID [2,6] → X [2,6,8] → Q/K [2,2,6,4] → 输出 [2,6,8]",
        "question": "真实输入的形状能否一路接到 Attention？",
    },
}


def source_label(link):
    file = "tools/"+link["file"][9:] if link["file"].startswith("../tools/") else "code/"+link["file"]
    return file + (":" + link["function"] if link["function"] else "")


def experiment_label(link):
    return f"Notebook 实验 {link['experiment']}" if link["experiment"] else "课后扩展验证"


for kind,function in {"onehot":"lookup_example","grad":"gradient_example",
                      "chain":"cross_entropy_example","absolute":"rope_example",
                      "frequencies":"rope_example","rotation":"rope_example",
                      "vector_rope":"rope_example","mask":"mask_example"}.items():
    LINKS[kind]["worked_function"]=function
