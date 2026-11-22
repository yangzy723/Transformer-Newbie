"""生成第三课课堂Notebook；核心计算直接展示，执行后保留真实输出。"""
from pathlib import Path
import nbformat as nbf


def build(root):
    root=Path(root)
    cells=[]
    md=lambda text:cells.append(nbf.v4.new_markdown_cell(text.strip()))
    code=lambda text:cells.append(nbf.v4.new_code_cell(text.strip()))
    md(r"""
# 第三课：重新走进 Transformer，理解 Attention

28页PPT、60分钟。实验01–03复习前两课；实验04–08展开主算例与三个干预；
实验09把RoPE接回计算。主例位置从0开始，第三个位置为2。
""")
    code("""
from pathlib import Path
import sys
import math
import torch

ROOT = Path.cwd()
if (ROOT / "class3" / "code").is_dir():
    ROOT = ROOT / "class3"
sys.path.insert(0, str(ROOT / "code"))
from attention import X, W_Q, W_K, W_V, project, attention, apply_rope
from review import build_review

torch.set_printoptions(precision=4, sci_mode=False)
review = build_review()
x = X.clone()
w_q, w_k, w_v = W_Q.clone(), W_K.clone(), W_V.clone()
""")
    md("""
## 实验01 · BPE与Embedding · PPT 3–5

每轮统计当前的相邻片段；编码执行已冻结的规则。对照最终两个token与中间五个token。

真实ByteLevel配置另行加载，查看“我喜欢喝牛奶。”的ID。梯度小表使用独立的算术ID，
人为指定上游梯度，以观察同一参数行的累加；第二课bigram记录展示预测任务的真实训练结果。
""")
    code("""
for row in review["bpe"]["training"]:
    print("训练rank", row["rank"], row["pair"], "频次", row["count"])
for row in review["bpe"]["encode_trace"]:
    print("编码rank", row["rank"], " / ".join(row["tokens"]))
print("冻结的真实配置：", review["real_tokenizer"])
e = review["embedding"]
print("算术小表IDs：", e["ids"], "查出：", e["vectors"])
print("参数行梯度：", e["gradient"])
print("SGD前：", e["weight"], "SGD后：", e["after"])
print("第二课bigram loss：", e["bigram_loss_start"], "→", e["bigram_loss_end"])
""")
    md("""
## 实验02 · 两支箭头一起转 · PPT 6

图解初始q=k=[1,0]，每位置转30°。位置1/3移到3/5，角度差仍60°，点积都为0.5。
相加式正弦PE则根据位置生成同维向量，直接加到内容向量上。
""")
    code("""
p = review["rope_intuition"]
print("原位置：", p["positions"], "方向：", p["angles_degrees"], "点积：", p["dot"])
print("移动后：", p["shifted_positions"], "方向：", p["shifted_angles_degrees"],
      "点积：", p["shifted_dot"])
print("四维相加式正弦PE：", p["pe_width4"])
""")
    md("""
## 实验03 · 逐项加权汇总 · PPT 8–9、22

先看三个位置各自贡献什么，再沿第0维求和，得到喝这个位置的新向量。
最后保留两候选概率，PPT第22页对比读取比例与预测概率。
这些旧例使用第一课相同的人工参数，后半节才切换到新的Q/K/V投影。
""")
    code("""
old = review["first_lesson"]
values = torch.tensor(old["values"], dtype=torch.float64)
weights_review = torch.tensor(old["weights"], dtype=torch.float64)
weighted = weights_review[:, None] * values
context = weighted.sum(dim=0)
print("每个位置的贡献：\\n", weighted)
print("按列求和，得到喝的新向量：", context)
for name in ("scores", "weights", "values", "context"):
    print(name, "=", old[name])
print("候选：", old["candidate_labels"])
print("候选分数：", old["lm_logits"])
print("候选概率：", old["lm_probabilities"])
""")
    md(r"""
## 实验04 · 三个可训练投影 · PPT 13–15

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

本课采用行向量右乘；三张2×2矩阵是便于手算的参数取值。实际训练会更新这些矩阵。
主算例暂不叠加位置，便于逐项核对；实验09接回真实RoPE。
""")
    code("""
q = x @ w_q
k = x @ w_k
v = x @ w_v
for name, tensor in {"X": X, "W_Q": W_Q, "W_K": W_K, "W_V": W_V,
                     "Q": q, "K": k, "V": v}.items():
    print(name, "\\n", tensor)
print("喝的V：", X[-1] @ W_V)
""")
    md(r"""
## 实验05 · 一行到完整矩阵 · PPT 16–21、26

行是query，列是key。第三行点积为[3,2,5]；每行按自己的可读范围归一化。

$$
S=QK^\mathsf{T}/\sqrt{d_k},\quad
A=\operatorname{softmax}_{j}(S+M),\quad Z=AV.
$$
""")
    code("""
raw_scores = q @ k.T
scores = raw_scores / math.sqrt(q.shape[-1])
future = torch.ones(3, 3, dtype=torch.bool).triu(1)
masked = scores.masked_fill(future, -torch.inf)
weights = masked.softmax(dim=-1)
output = weights @ v

for name, tensor in {"原始分数": raw_scores, "缩放分数": scores,
                     "mask后": masked, "权重": weights, "输出": output}.items():
    print(name, "\\n", tensor)
print("每行权重和：", weights.sum(-1))
print("第二行手算：", .5*v[0] + .5*v[1])
print("第三行不缩放/缩放后的权重：", raw_scores[-1].softmax(-1), weights[-1])
""")
    md(r"""
## 实验06 · 只改V · PPT 23

Q/K保持，修改已经生成的第0行V。先预测权重会不会变，再核对解析关系：

$$\Delta z_i=A_{i,0}\Delta v_0.$$
""")
    code("""
changed_v = v.clone()
delta_v = v.new_tensor([2., 0.])
changed_v[0] += delta_v
result_v = attention(q, k, changed_v)
print("权重保持：", torch.allclose(weights, result_v["weights"]))
print("实际输出变化：\\n", result_v["output"] - output)
print("解析预测：\\n", weights[:, 0:1] * delta_v)
torch.testing.assert_close(result_v["output"] - output, weights[:, 0:1] * delta_v)
""")
    md("""
## 实验07 · 只改K · PPT 24

Q/V保持，K[0]加[1,0]。主要观察第三行：它对第0列的原始分数从3变4。
第一行只有一个可读位置，第二行Q=[0,2]与本次改变量正交，因此这两行权重保持。
""")
    code("""
changed_k = k.clone()
changed_k[0] += k.new_tensor([1., 0.])
result_k = attention(q, changed_k, v)
print("原权重：\\n", weights)
print("新权重：\\n", result_k["weights"])
print("新输出：\\n", result_k["output"])
print("第三行读取第0列：", weights[2, 0].item(), "→",
      result_k["weights"][2, 0].item())
""")
    md("""
## 实验08 · 修改未来输入 · PPT 25

位置与参数固定，把最后输入从[1,1]改为[4,-1]，重新生成Q/K/V。
比较较早两行输出；同时检查最后一行确实改变，并运行无mask对照。
""")
    code("""
changed_x = x.clone()
changed_x[-1] += X.new_tensor([3., -2.])
q2, k2, v2 = project(changed_x)
masked_after = attention(q2, k2, v2)["output"]
plain_before = attention(q, k, v, causal=False)["output"]
plain_after = attention(q2, k2, v2, causal=False)["output"]
print("mask开，前两行最大变化：", (masked_after[:-1] - output[:-1]).abs().max().item())
print("mask关，前两行最大变化：",
      (plain_after[:-1] - plain_before[:-1]).abs().max().item())
print("最后位置：", output[-1], "→", masked_after[-1])
torch.testing.assert_close(output[:-1], masked_after[:-1])
assert not torch.allclose(output[-1], masked_after[-1])
""")
    md("""
## 实验09 · 接回第二课的真实RoPE接口 · PPT 6、25

apply_rope复用第二课实现。按固定位置旋转Q/K，V保留，再进行因果Attention。
固定未旋转内容，整体平移位置后点积保持；修改未来输入时，较早输出仍保持。
""")
    code("""
positions = torch.arange(3, dtype=X.dtype)
qr, kr = apply_rope(q, positions), apply_rope(k, positions)
rope_output = attention(qr, kr, v)["output"]
shifted_q = apply_rope(q, positions + 7)
shifted_k = apply_rope(k, positions + 7)
rope_future = attention(apply_rope(q2, positions),
                        apply_rope(k2, positions), v2)["output"]
print("RoPE输出：\\n", rope_output)
print("Q范数保持：", torch.allclose(q.norm(dim=-1), qr.norm(dim=-1)))
print("整体平移，点积最大差：", (qr @ kr.T - shifted_q @ shifted_k.T).abs().max().item())
print("RoPE下，未来修改不影响前两行：",
      torch.allclose(rope_output[:-1], rope_future[:-1]))
torch.testing.assert_close(rope_output[:-1], rope_future[:-1])
""")
    md("""
## 课堂检查与第四课

解释一行读取权重、计算一次加权V、预测三种修改的结果。
第27页多头图使用独立形状[1,3,4]，两头各2维；下一课实现多头和完整Block。
练习见 [exercises.md](exercises.md)，公式与页码对应见 [knowledge_code_map.md](knowledge_code_map.md)。
""")
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}})
    nbf.write(notebook,root/'demo.ipynb')
    return root/'demo.ipynb'


if __name__=='__main__':
    print(build(Path(__file__).resolve().parents[1]))
