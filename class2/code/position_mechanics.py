"""验证无位置、无遮罩 self-attention 的置换等变性。"""
import math

import torch

from positions import sinusoidal


def attention(x):
    scores = x @ x.T / math.sqrt(x.shape[-1])
    return scores.softmax(dim=-1) @ x


def demo():
    torch.manual_seed(7)
    x = torch.randn(3, 4, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    original = attention(x)
    rearranged = attention(x[permutation])
    print("无位置：Attn(PX) == P Attn(X)：", torch.allclose(rearranged, original[permutation]))
    pe = sinusoidal(torch.arange(3), 4)
    absolute_original = attention(x + pe)
    absolute_rearranged = attention(x[permutation] + pe)
    print("重排 token、保留当前位置索引后仍等变吗：",
          torch.allclose(absolute_rearranged, absolute_original[permutation]))
    print("此实验不加因果 mask；不能把该等变结论直接套到带 mask 的场景。")


if __name__ == "__main__":
    demo()
