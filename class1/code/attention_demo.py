"""用 Python 标准库计算一个可手算的 Attention；无需安装依赖。

这些 Q/K/V 是人为指定的数字，不是训练好的语言向量。
矩阵的行是 query 位置，列是被关注的 key 位置。
"""

import argparse
import html
import math
from pathlib import Path


TOKENS = ["我", "喜欢", "猫"]
Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
V = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]


def softmax(row):
    """减去最大值以避免 exp 溢出；每行变成和为 1 的正数。"""
    maximum = max(row)
    exps = [math.exp(value - maximum) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def attention(q, k, v, causal=False):
    """本演示只支持长度相同的 self-attention，无 batch/head 维度。"""
    if not q or len(q) != len(k) or len(k) != len(v):
        raise ValueError("Q/K/V 必须非空且有相同的位置数")
    d_k = len(q[0])
    if d_k == 0 or any(len(row) != d_k for row in q + k):
        raise ValueError("Q 和 K 的向量维度必须一致且大于 0")
    d_v = len(v[0])
    if d_v == 0 or any(len(row) != d_v for row in v):
        raise ValueError("V 的向量维度必须一致且大于 0")
    scores = []
    for i, query in enumerate(q):
        row = []
        for j, key in enumerate(k):
            dot = sum(a * b for a, b in zip(query, key))
            row.append(-math.inf if causal and j > i else dot / math.sqrt(d_k))
        scores.append(row)
    weights = [softmax(row) for row in scores]
    output = [
        [sum(weight * value[col] for weight, value in zip(row, v))
         for col in range(d_v)]
        for row in weights
    ]
    return scores, weights, output


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print("  [" + ", ".join(f"{value:7.3f}" for value in row) + "]")


def save_heatmap(weights, path, causal):
    """生成离线可打开的 SVG；横轴 key，纵轴 query。"""
    cell, left, top = 95, 150, 85
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="490" height="455">',
             '<rect width="100%" height="100%" fill="#F4F7FB"/>',
             '<g font-family="sans-serif" text-anchor="middle" fill="#17263C">',
             f'<text x="245" y="30" font-size="20">Attention / causal={causal}</text>',
             '<text x="292" y="58" font-size="15">Key（被关注的位置）→</text>']
    for i, token in enumerate(TOKENS):
        escaped = html.escape(token)
        parts.append(f'<text x="{left+i*cell+cell/2}" y="78">{escaped}</text>')
        parts.append(f'<text x="110" y="{top+i*cell+cell/2+6}">{escaped}</text>')
        for j, weight in enumerate(weights[i]):
            x, y = left + j * cell, top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" '
                         f'fill="#247C91" fill-opacity="{0.08+0.92*weight:.3f}"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+6}">{weight:.3f}</text>')
    parts += ['<text x="245" y="403" font-size="15">每行和为 1；人工数字仅用于观察计算</text>',
              '<text x="245" y="430" font-size="15">左侧：Query（发出查询的位置）↓</text>',
              '</g></svg>']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal", action="store_true", help="只看当前位置及之前的位置")
    parser.add_argument("--svg", type=Path, help="可选：保存 Attention 热力图")
    args = parser.parse_args(argv)
    scores, weights, output = attention(Q, K, V, causal=args.causal)
    print("tokens =", TOKENS, "| causal =", args.causal)
    print("Q/K/V: (3, 2)；scores/weights: (3, 3)；output: (3, 2)")
    print_matrix("1. QK^T / sqrt(d_k)：匹配分数", scores)
    print_matrix("2. softmax：每行权重", weights)
    print_matrix("3. weights @ V：混合后的向量", output)
    if args.svg:
        save_heatmap(weights, args.svg, args.causal)
        print("热力图已保存：", args.svg)
    return weights, output


if __name__ == "__main__":
    main()
