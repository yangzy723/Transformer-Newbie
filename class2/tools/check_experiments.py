"""检查课程的算法反例、梯度公式和位置不变量，不访问网络。"""
from collections import Counter
import contextlib
import io
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from bpe_core import encode_bpe, merge_pair, train_bpe
from embedding_train import bigram_batch, train
from gradient_chain import demo as gradient_demo
from input_pipeline import demo as pipeline_demo
from position_mechanics import attention
from positions import apply_rope, sinusoidal
from runtime import load_tokenizer


def main():
    merges, history = train_bpe({"bc": 6, "ab": 5, "abc": 1}, 2)
    assert merges == [("b", "c"), ("a", "b")]
    assert [r["count"] for r in history] == [7, 5]
    assert encode_bpe("abc", merges)[0] == ["a", "bc"]
    assert merge_pair(["a", "a", "a"], ("a", "a")) == ["aa", "a"]
    print("BPE：加权频次、rank重叠反例、非重叠合并通过")

    report = json.loads((ROOT / "assets/tokenizer_report.json").read_text(encoding="utf-8"))
    for size in (320, 384):
        tokenizer = load_tokenizer(size)
        assert tokenizer.get_vocab_size() == size
        for entry in report["models"][str(size)]["examples"]:
            result = tokenizer.encode(entry["text"])
            assert result.ids == entry["ids"] and result.tokens == entry["tokens"]
            assert tokenizer.decode(result.ids) == entry["text"]
        for text in ("咖啡👩‍💻", "", " e\u0301 ", "\n测试\tXYZ"):
            assert tokenizer.decode(tokenizer.encode(text).ids) == text
    tokenizer = load_tokenizer()
    print("Tokenizer：冻结配置、编码报告、未见字节与回解通过")

    torch.manual_seed(7)
    embedding = nn.Embedding(6, 4, padding_idx=0)
    ids = torch.tensor([[2, 2, 3, 0]])
    vectors = embedding(ids)
    one_hot = F.one_hot(ids, 6).to(vectors.dtype)
    torch.testing.assert_close(vectors, one_hot @ embedding.weight)
    vectors.sum().backward()
    torch.testing.assert_close(embedding.weight.grad[2], torch.full((4,), 2.0))
    torch.testing.assert_close(embedding.weight.grad[3], torch.ones(4))
    assert int(torch.count_nonzero(embedding.weight.grad[0])) == 0
    with contextlib.redirect_stdout(io.StringIO()):
        analytic, automatic = gradient_demo()
    torch.testing.assert_close(analytic, automatic)
    print("Embedding：one-hot前向、行梯度累加、单位置CE公式通过")

    torch.manual_seed(7)
    tied_embedding = nn.Embedding(6, 4, padding_idx=0)
    head = nn.Linear(4, 6, bias=False)
    head.weight = tied_embedding.weight
    F.cross_entropy(head(tied_embedding(torch.tensor([2]))), torch.tensor([3])).backward()
    assert float(tied_embedding.weight.grad[0].norm()) > 0
    assert float(tied_embedding.weight.grad[5].norm()) > 0
    print("权重绑定：未作输入的行及pad行可接收输出梯度")

    _, _, trained = train(verbose=False)
    x, y = bigram_batch(tokenizer)
    pairs, totals = Counter(zip(x.tolist(), y.tolist())), Counter(x.tolist())
    entropy = -sum(c / len(x) * math.log(c / totals[a]) for (a, b), c in pairs.items())
    final_loss = trained["history"][-1]["loss"]
    assert trained["changed_rows"] == trained["observed_rows"] == 26
    assert entropy - 1e-6 <= final_loss < entropy + .01
    assert final_loss < trained["history"][0]["loss"] / 2
    print("bigram：实际参数更新、loss下降、条件熵下界通过")

    torch.manual_seed(7)
    x = torch.randn(3, 4, dtype=torch.float64)
    p = torch.tensor([2, 0, 1])
    torch.testing.assert_close(attention(x[p]), attention(x)[p])
    pe = sinusoidal(torch.arange(3), 4)
    assert not torch.allclose(attention(x[p] + pe), attention(x + pe)[p])
    torch.testing.assert_close(pe[:, 0::2].square() + pe[:, 1::2].square(), torch.ones(3, 2, dtype=pe.dtype))
    print("位置：无位置等变、固定位置破坏该等式、多频率范数通过")

    q, k = torch.randn(2, 2, 2, 5, 4, dtype=torch.float64)
    positions = torch.arange(5)
    qr, kr = apply_rope(q, positions), apply_rope(k, positions)
    torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1))
    torch.testing.assert_close(qr @ kr.transpose(-2, -1),
        apply_rope(q, positions+9) @ apply_rope(k, positions+9).transpose(-2, -1))
    print("RoPE：多batch/head广播、范数与相对位移性质通过")

    with contextlib.redirect_stdout(io.StringIO()):
        result = pipeline_demo()
    assert torch.isfinite(result["output"]).all()
    torch.testing.assert_close(result["weights"].sum(-1), torch.ones(2, 2, 6))
    assert int(torch.count_nonzero(result["weights"].triu(1))) == 0
    assert int(torch.count_nonzero(result["output"][~result["valid"]])) == 0
    for b in range(2):
        assert int(torch.count_nonzero(result["weights"][b, :, :, ~result["valid"][b]])) == 0
    print("整合：shape、归一化、因果/padding遮罩、数值有限通过")


if __name__ == "__main__":
    main()
