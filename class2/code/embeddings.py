"""真实 nn.Embedding：batch 查表、one-hot 等价、梯度累加和 padding。"""
import torch
from torch import nn
from torch.nn import functional as F

from runtime import load_tokenizer, make_batch


def demo():
    torch.manual_seed(7)
    tokenizer = load_tokenizer()
    ids, valid = make_batch(tokenizer, ["我喜欢喝牛奶。", "我喜欢喝酒。"])
    pad = tokenizer.token_to_id("<pad>")
    embedding = nn.Embedding(tokenizer.get_vocab_size(), 8, padding_idx=pad)
    vectors = embedding(ids)
    one_hot = F.one_hot(ids, num_classes=embedding.num_embeddings).to(vectors.dtype)
    print("ID / E / X：", list(ids.shape), list(embedding.weight.shape), list(vectors.shape))
    print("lookup == one_hot @ E：", torch.allclose(vectors, one_hot @ embedding.weight))
    print("有效位置：", valid.tolist())

    a, b = tokenizer.encode("我喜欢喝牛奶。").ids[:2]
    assert a != b and a != pad and b != pad
    repeated = torch.tensor([[a, a, b, pad]])
    embedding.zero_grad()
    embedding(repeated).sum().backward()
    print("用于求和的 ID：", repeated.tolist())
    print("重复 token 行的梯度：", embedding.weight.grad[a].tolist())
    print("出现一次的行梯度：", embedding.weight.grad[b].tolist())
    print("padding 行梯度：", embedding.weight.grad[pad].tolist())
    print("此 loss 只是梯度诊断；它还不是语言模型的交叉熵。")
    return embedding, ids


if __name__ == "__main__":
    demo()
