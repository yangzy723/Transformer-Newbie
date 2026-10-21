"""把交叉熵的单位置梯度手算结果与 autograd 对齐。"""
import torch
from torch.nn import functional as F


def demo():
    torch.manual_seed(7)
    embedding_vector = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
    output_weight = torch.randn(6, 4, dtype=torch.float64)
    target = torch.tensor([2])
    logits = embedding_vector @ output_weight.T
    loss = F.cross_entropy(logits, target)
    loss.backward()
    probabilities = logits.detach().softmax(dim=-1)
    error = probabilities - F.one_hot(target, 6)
    manual_gradient = error @ output_weight
    torch.testing.assert_close(embedding_vector.grad, manual_gradient)
    print("autograd：", embedding_vector.grad.tolist())
    print("手算 (p-y) @ W：", manual_gradient.tolist())
    print("两者相同：", torch.allclose(embedding_vector.grad, manual_gradient))
    return embedding_vector.grad, manual_gradient


if __name__ == "__main__":
    demo()
