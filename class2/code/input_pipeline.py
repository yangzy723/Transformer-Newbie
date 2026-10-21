"""真实 tokenizer → batch embedding → 多头 Q/K → RoPE → 遮罩 Attention。"""
import math

import torch
from torch import nn

from positions import apply_rope
from runtime import load_tokenizer, make_batch


def demo():
    torch.manual_seed(7)
    tokenizer = load_tokenizer()
    ids, valid = make_batch(tokenizer, ["我喜欢喝牛奶。", "我喜欢喝酒。"])
    batch, length = ids.shape
    width, heads = 8, 2
    embedding = nn.Embedding(tokenizer.get_vocab_size(), width,
                             padding_idx=tokenizer.token_to_id("<pad>"))
    x = embedding(ids)
    q, k, v = nn.Linear(width, 3 * width, bias=False)(x).chunk(3, dim=-1)
    q, k, v = [t.view(batch, length, heads, width // heads).transpose(1, 2) for t in (q, k, v)]
    positions = torch.arange(length)
    q, k = apply_rope(q, positions), apply_rope(k, positions)
    scores = q @ k.transpose(-2, -1) / math.sqrt(width // heads)
    future = torch.ones(length, length, dtype=torch.bool).triu(1)
    forbidden = future[None, None, :, :] | ~valid[:, None, None, :]
    weights = scores.masked_fill(forbidden, float("-inf")).softmax(dim=-1)
    output = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
    output = output.masked_fill(~valid[:, :, None], 0)
    print("ID / embedding：", list(ids.shape), list(x.shape))
    print("Q/K：", list(q.shape), "scores：", list(scores.shape))
    print("输出：", list(output.shape))
    print("未来与 padding key 权重均为0：", bool((weights.masked_select(forbidden.expand_as(weights)) == 0).all()))
    return {"ids": ids, "valid": valid, "weights": weights, "output": output}


if __name__ == "__main__":
    demo()
