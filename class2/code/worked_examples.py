"""PPT 的数值算例：从真实函数计算，避免手填结果与代码脱节。"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bpe_core import BPE_CORPUS, encode_bpe, train_bpe
from positions import apply_rope, sinusoidal
from runtime import load_tokenizer


def bpe_example():
    merges, history = train_bpe(BPE_CORPUS, 4)
    result, trace = encode_bpe("你喜欢喝奶茶", merges)
    return {"corpus": BPE_CORPUS, "merges": merges, "history": history,
            "text": "你喜欢喝奶茶", "result": result, "trace": trace,
            "new_combination": encode_bpe("你喜欢喝咖啡", merges)[0]}


def lookup_example():
    # 算术小表：ID 与默认 tokenizer 的 ID 没有关系。
    table = torch.tensor([[0., 0., 0.], [.2, -.1, .4],
                          [.5, .3, -.2], [-.4, .8, .1]], dtype=torch.float64)
    ids = torch.tensor([1, 2, 1])
    selected = F.embedding(ids, table)
    torch.testing.assert_close(selected, F.one_hot(ids, 4).to(table.dtype) @ table)
    return {"table": table.tolist(), "ids": ids.tolist(), "selected": selected.tolist()}


def gradient_example():
    embedding = nn.Embedding(4, 2, padding_idx=0, dtype=torch.float64)
    ids = torch.tensor([1, 1, 2, 0])
    upstream = torch.tensor([[1., 0.], [0., 2.], [3., 4.], [9., 9.]], dtype=torch.float64)
    loss = (embedding(ids) * upstream).sum()
    loss.backward()
    expected = torch.tensor([[0., 0.], [1., 2.], [3., 4.], [0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(embedding.weight.grad, expected)
    return {"ids": ids.tolist(), "upstream": upstream.tolist(),
            "row_gradients": embedding.weight.grad.tolist()}


def cross_entropy_example():
    embedding_vector = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    output_weight = torch.tensor([[1., 0.], [0., 1.], [-1., .5]], dtype=torch.float64)
    bias = torch.tensor([[.1, .7, .2]], dtype=torch.float64).log()
    logits = embedding_vector @ output_weight.T + bias
    target = torch.tensor([1])
    loss = F.cross_entropy(logits, target)
    loss.backward()
    probabilities = logits.detach().softmax(-1)
    error = probabilities - F.one_hot(target, 3)
    manual = error @ output_weight
    torch.testing.assert_close(manual, embedding_vector.grad)
    return {"probabilities": probabilities.tolist()[0], "target": 1,
            "loss": float(loss.detach()), "error": error.tolist()[0],
            "output_weight": output_weight.tolist(), "vector_gradient": manual.tolist()[0]}


def rotate_pair(x, position, degrees_per_step=30.):
    """二维图解：每前进一个位置转30°；标准多频率实现见 apply_rope。"""
    x = torch.as_tensor(x, dtype=torch.float64)
    angle = torch.tensor(position * degrees_per_step * torch.pi / 180, dtype=x.dtype)
    a, b = x.unbind(-1)
    return torch.stack((a * angle.cos() - b * angle.sin(),
                        a * angle.sin() + b * angle.cos()), dim=-1)


def rope_example():
    q = k = torch.tensor([[1., 0.]], dtype=torch.float64)
    qr, kr = apply_rope(q, [1]), apply_rope(k, [3])
    score = (qr * kr).sum()
    shifted = (apply_rope(q, [6]) * apply_rope(k, [8])).sum()
    torch.testing.assert_close(score, shifted)
    torch.testing.assert_close(score, torch.cos(torch.tensor(2., dtype=torch.float64)))
    q1, k3 = rotate_pair([1., 0.], 1), rotate_pair([1., 0.], 3)
    q3, k5 = rotate_pair([1., 0.], 3), rotate_pair([1., 0.], 5)
    intuition_score, intuition_shifted = q1 @ k3, q3 @ k5
    torch.testing.assert_close(intuition_score, intuition_shifted)
    torch.testing.assert_close(intuition_score, torch.tensor(.5, dtype=torch.float64))
    return {"intuition": {"degrees_per_step": 30, "positions": [1, 3],
                          "shifted_positions": [3, 5], "q1": q1.tolist(), "k3": k3.tolist(),
                          "q3": q3.tolist(), "k5": k5.tolist(),
                          "score": float(intuition_score), "shifted_score": float(intuition_shifted)},
            "q": q.tolist()[0], "k": k.tolist()[0], "m": 1, "n": 3,
            "q_rotated": qr.tolist()[0], "k_rotated": kr.tolist()[0],
            "score": float(score), "shifted_score": float(shifted),
            "pe_width4": sinusoidal(torch.arange(3), 4).tolist()}


def mask_example():
    scores = torch.zeros(4, 4, dtype=torch.float64)
    future = torch.ones(4, 4, dtype=torch.bool).triu(1)
    valid = torch.tensor([True, True, True, False])
    forbidden = future | ~valid[None, :]
    weights = scores.masked_fill(forbidden, -torch.inf).softmax(-1)
    assert torch.isfinite(weights).all()
    assert not torch.count_nonzero(weights[:, 3])
    torch.testing.assert_close(weights.sum(-1), torch.ones(4, dtype=torch.float64))
    return {"scores": scores.tolist(), "valid": valid.tolist(),
            "forbidden": forbidden.tolist(), "weights": weights.tolist()}


def build_report():
    tokenizer = load_tokenizer()
    report = {"bpe": bpe_example(), "lookup": lookup_example(),
              "gradient": gradient_example(), "cross_entropy": cross_entropy_example(),
              "position": rope_example(), "mask": mask_example(),
              "real_ids": tokenizer.encode("我喜欢喝牛奶。").ids}
    path = Path(__file__).resolve().parents[1] / "assets/worked_examples.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = build_report()
    print("BPE 四轮：", [(row["pair"], row["count"]) for row in report["bpe"]["history"]])
    print("奶茶编码：", report["bpe"]["result"])
    print("新组合：", report["bpe"]["new_combination"])
    print("小表查行：", report["lookup"]["selected"])
    print("重复行梯度：", report["gradient"]["row_gradients"][1])
    print("三类 CE / 向量梯度：", round(report["cross_entropy"]["loss"], 4), report["cross_entropy"]["vector_gradient"])
    print("30°图解点积 / 平移后：", round(report["position"]["intuition"]["score"], 4), round(report["position"]["intuition"]["shifted_score"], 4))
    print("RoPE 点积 / 平移后：", round(report["position"]["score"], 4), round(report["position"]["shifted_score"], 4))
    print("最后一列 padding key 权重：", [row[-1] for row in report["mask"]["weights"]])
