"""向量化正弦编码与 RoPE；RoPE 接受 [..., T, D]，D 为正偶数。"""
import torch


def sinusoidal(positions, width):
    if width <= 0 or width % 2:
        raise ValueError("width 必须为正偶数")
    positions = torch.as_tensor(positions, dtype=torch.float64)
    frequencies = 10000.0 ** (-torch.arange(0, width, 2, dtype=torch.float64) / width)
    angles = positions[:, None] * frequencies[None, :]
    result = torch.empty(len(positions), width, dtype=torch.float64)
    result[:, 0::2], result[:, 1::2] = angles.sin(), angles.cos()
    return result


def apply_rope(x, positions):
    width = x.shape[-1]
    positions = torch.as_tensor(positions, device=x.device, dtype=x.dtype)
    if width <= 0 or width % 2 or positions.ndim != 1 or len(positions) != x.shape[-2]:
        raise ValueError("要求 x=[...,T,D]、positions=[T]，D为正偶数")
    frequencies = 10000.0 ** (-torch.arange(0, width, 2, device=x.device, dtype=x.dtype) / width)
    angles = positions[:, None] * frequencies[None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    result = torch.empty_like(x)
    result[..., 0::2] = even * angles.cos() - odd * angles.sin()
    result[..., 1::2] = even * angles.sin() + odd * angles.cos()
    return result


def demo():
    pe = sinusoidal(torch.arange(3), 4)
    print("四维正弦位置编码：", pe.round(decimals=4).tolist())
    torch.manual_seed(7)
    q, k = torch.randn(2, 3, 4, dtype=torch.float64)
    positions = torch.arange(3)
    score = apply_rope(q, positions) @ apply_rope(k, positions).T
    shifted = apply_rope(q, positions + 5) @ apply_rope(k, positions + 5).T
    print("保持内容、整体平移位置后点积最大差异：", float((score-shifted).abs().max()))
    print("RoPE 是否保持各向量范数：", torch.allclose(q.norm(dim=-1), apply_rope(q, positions).norm(dim=-1)))
    print("公式可计算远处位置，不等于完整模型在任意长度上都能泛化。")


if __name__ == "__main__":
    demo()
