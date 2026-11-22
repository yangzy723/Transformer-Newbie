"""第三课：同一组小矩阵，逐步计算单头因果 Attention。"""
# %% 1. 输入与三个投影矩阵（行向量约定）
import math
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'class2/code'))
from positions import apply_rope

DTYPE = torch.float64
TOKENS = ['我', '喜欢', '喝']
X = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=DTYPE)
W_Q = torch.tensor([[1, 0], [0, 2]], dtype=DTYPE)
W_K = torch.tensor([[1, 1], [0, 1]], dtype=DTYPE)
W_V = torch.tensor([[1, -1], [2, 0]], dtype=DTYPE)


def project(x, w_q=W_Q, w_k=W_K, w_v=W_V):
    return x @ w_q, x @ w_k, x @ w_v


# %% 2. 分数 → 屏蔽未来 → 每行归一化 → 混合 V
def attention(q, k, v, causal=True):
    """方阵自注意力；支持 [...,T,d_k]，最后一维沿 key 做 Softmax。"""
    raw_scores = q @ k.transpose(-2, -1)
    scores = raw_scores / math.sqrt(q.shape[-1])
    forbidden = torch.zeros(scores.shape[-2:], dtype=torch.bool, device=q.device)
    if causal:
        forbidden = torch.ones_like(forbidden).triu(1)
    masked = scores.masked_fill(forbidden, -torch.inf)
    weights = masked.softmax(dim=-1)
    output = weights @ v
    return dict(raw_scores=raw_scores, scores=scores, forbidden=forbidden,
                masked=masked, weights=weights, output=output)


class AttentionProjection(nn.Module):
    """三张可训练的投影矩阵；本课固定初值以便逐项计算。"""
    def __init__(self):
        super().__init__()
        self.w_q = nn.Parameter(W_Q.clone())
        self.w_k = nn.Parameter(W_K.clone())
        self.w_v = nn.Parameter(W_V.clone())

    def forward(self, x, causal=True):
        q, k, v = project(x, self.w_q, self.w_k, self.w_v)
        return attention(q, k, v, causal)


# %% 3. 运行并观察每一步
def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    q, k, v = project(X)
    print('token 顺序：', TOKENS)
    for name, value in {'X': X, 'Q': q, 'K': k, 'V': v, **attention(q,k,v)}.items():
        print(f'{name}:\n{value}')


if __name__ == '__main__':
    main()
