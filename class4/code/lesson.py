"""第四课的可复算数值：行向量 FFN、输出概率与解码，全部仅用标准库。

矩阵约定为 ``行向量 x @ W``，W 的形状是 [输入维度, 输出维度]。
各段玩具参数独立设置；FFN 的输出是向量更新量，绝不是词表概率。
"""

import argparse
import json
import math
from pathlib import Path
import random


FFN_INPUT = [1.0, -2.0]
FFN_BATCH = [[1.0, -2.0], [-1.0, 2.0], [0.5, 0.25]]
FFN_W1 = [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]]
FFN_W2 = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
MULTIHEAD_Z1 = [1.0, 2.0]
MULTIHEAD_Z2 = [-1.0, 3.0]
MULTIHEAD_WO = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
]
LM_G = [1.0, 2.0]
LM_U = [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
BASE_PROBABILITIES = [0.5, 0.3, 0.2]
TEMPERATURES = [0.5, 1.0, 2.0]
TOKENS = ["牛奶", "茶", "水"]
IDENTITY_2 = [[1.0, 0.0], [0.0, 1.0]]
SWIGLU_W_VALUE = [[1.0, 1.0], [0.0, 1.0]]


def matrix_shape(matrix):
    """返回矩形非空矩阵的 (行数, 列数)，拒绝不一致的行长度。"""
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal length")
    return len(matrix), width


def matmul(left, right):
    """矩阵相乘；维度不匹配时明确报错，不用 zip 静默截断。"""
    rows, inner = matrix_shape(left)
    other_inner, columns = matrix_shape(right)
    if inner != other_inner:
        raise ValueError("matrix dimensions do not match")
    return [
        [math.fsum(left[i][k] * right[k][j] for k in range(inner))
         for j in range(columns)]
        for i in range(rows)
    ]


def linear(x, weights, bias=None):
    """计算一个行向量的 x @ weights + bias。"""
    result = matmul([x], weights)[0]
    if bias is None:
        return result
    if len(bias) != len(result):
        raise ValueError("bias length must equal output dimension")
    return [value + shift for value, shift in zip(result, bias)]


def relu(value):
    """ReLU(x) = max(0, x)，便于手算的非线性。"""
    return max(0.0, value)


def gelu(value):
    """精确 GELU(x) = x Φ(x)，用 erf 实现，不用 tanh 近似。"""
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def silu(value):
    """SiLU(x) = x sigmoid(x)，按符号分支避免 exp 溢出。"""
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


def ffn_step(x, w1=FFN_W1, w2=FFN_W2, activation=relu):
    """返回 FFN 的输入、升维、激活、投影；本课示例偏置均为 0。"""
    expanded = linear(x, w1)
    activated = [activation(value) for value in expanded]
    return {
        "input": list(x), "expanded": expanded, "activated": activated,
        "output": linear(activated, w2),
    }


def layer_norm(x, eps=1e-5, gamma=None, beta=None):
    """沿单个 token 的特征维归一化，方差除以 D，非无偏样本方差。"""
    if not x or eps <= 0.0:
        raise ValueError("LayerNorm requires a non-empty vector and eps > 0")
    dimension = len(x)
    gamma = [1.0] * dimension if gamma is None else gamma
    beta = [0.0] * dimension if beta is None else beta
    if len(gamma) != dimension or len(beta) != dimension:
        raise ValueError("gamma and beta must match the input dimension")
    mean = math.fsum(x) / dimension
    variance = math.fsum((value - mean) ** 2 for value in x) / dimension
    normalized = [(value - mean) / math.sqrt(variance + eps) for value in x]
    return {
        "input": list(x), "mean": mean, "variance": variance,
        "epsilon": eps, "gamma": list(gamma), "beta": list(beta),
        "normalized": normalized,
        "output": [scale * value + shift
                   for scale, value, shift in zip(gamma, normalized, beta)],
    }


def swiglu_step(x=FFN_INPUT):
    """独立的双分支 SwiGLU 玩具例：SiLU(xW_gate) ⊙ (xW_value)，再投影。"""
    gate_projection = linear(x, IDENTITY_2)
    gate = [silu(value) for value in gate_projection]
    value_projection = linear(x, SWIGLU_W_VALUE)
    product = [a * b for a, b in zip(gate, value_projection)]
    return {
        "input": list(x), "w_gate": IDENTITY_2, "w_value": SWIGLU_W_VALUE,
        "w_down": IDENTITY_2, "gate_projection": gate_projection,
        "gate": gate, "value_projection": value_projection,
        "product": product, "output": linear(product, IDENTITY_2),
    }


def softmax(logits, temperature=1.0):
    """先减最大值再指数化；temperature 必须为正，0 请走 greedy 分支。"""
    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("logits must be finite and non-empty")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    maximum = max(logits)
    weights = [math.exp((value - maximum) / temperature) for value in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def validate_probabilities(probabilities):
    """检查非负、有限且和近似为 1 的概率向量。"""
    if (not probabilities or
            any(not math.isfinite(p) or p < 0.0 for p in probabilities) or
            not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12)):
        raise ValueError("probabilities must be non-negative, finite and sum to 1")


def greedy(probabilities):
    """选择最大值的下标，若并列则保留词表中靠前的一项。"""
    validate_probabilities(probabilities)
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def inverse_cdf(probabilities, uniform):
    """把 [0,1) 中的数落到累积概率区间，边界属于右侧区间。"""
    validate_probabilities(probabilities)
    if not 0.0 <= uniform < 1.0:
        raise ValueError("uniform value must be in [0, 1)")
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if uniform < cumulative:
            return index
    return len(probabilities) - 1  # 仅吸收浮点加法的尾部舍入误差。


def sample_index(probabilities, rng):
    """从固定概率分布抽一个词表下标；rng 由调用者控制。"""
    return inverse_cdf(probabilities, rng.random())


def sample_counts(probabilities, count=1000, seed=7):
    """重复抽样并记录频数；复现用局部 RNG，不改变程序全局随机状态。"""
    validate_probabilities(probabilities)
    if not isinstance(count, int) or count <= 0:
        raise ValueError("sample count must be a positive integer")
    rng = random.Random(seed)
    counts = [0] * len(probabilities)
    for _ in range(count):
        counts[sample_index(probabilities, rng)] += 1
    return {
        "seed": seed, "count": count, "counts": counts,
        "frequencies": [value / count for value in counts],
    }


def retained_distribution(probabilities, indices):
    """按原词表下标保留候选，其他概率置零，并对保留部分重新归一化。"""
    mass = math.fsum(probabilities[index] for index in indices)
    if mass <= 0.0:
        raise ValueError("retained candidates must have positive probability mass")
    retained = [0.0] * len(probabilities)
    for index in indices:
        retained[index] = probabilities[index] / mass
    return {"indices": list(indices), "retained_mass": mass, "probabilities": retained}


def top_k(probabilities, k):
    """按概率降序保留固定 k 项，再归一化；并列时按原词表顺序。"""
    validate_probabilities(probabilities)
    if not isinstance(k, int) or not 1 <= k <= len(probabilities):
        raise ValueError("k must be between 1 and the vocabulary size")
    ordered = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    return {"k": k, **retained_distribution(probabilities, ordered[:k])}


def top_p(probabilities, p):
    """保留概率降序中累计质量 >= p 的最短前缀，并归一化。

    使用极小容差识别本应相等的边界，避免 0.5+0.3 的浮点舍入多留一项。
    """
    validate_probabilities(probabilities)
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    ordered = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    indices = []
    for index in ordered:
        indices.append(index)
        cumulative = math.fsum(probabilities[item] for item in indices)
        if cumulative >= p or math.isclose(cumulative, p, rel_tol=1e-12, abs_tol=1e-15):
            break
    return {"p": p, **retained_distribution(probabilities, indices)}


def decode(logits, temperature=1.0, top_p_threshold=1.0, rng=None):
    """输出端一次解码：温度为 0 时 greedy；否则温度→top-p→抽样。"""
    if temperature == 0:
        return greedy(softmax(logits))
    probabilities = softmax(logits, temperature)
    retained = top_p(probabilities, top_p_threshold)["probabilities"]
    return sample_index(retained, random.Random() if rng is None else rng)


def build_report(output_path=None):
    """计算课程所有示例；可选写成 UTF-8 JSON，返回同一报告字典。"""
    ffn = ffn_step(FFN_INPUT)
    ffn.update({"w1": FFN_W1, "w2": FFN_W2,
                "bias1": [0.0] * 4, "bias2": [0.0] * 2,
                "without_activation": linear(linear(FFN_INPUT, FFN_W1), FFN_W2),
                "combined_weights": matmul(FFN_W1, FFN_W2)})
    norm = layer_norm([1.3, 1.8])
    norm_ffn = ffn_step(norm["output"])
    norm.update({"ffn": norm_ffn, "expanded": norm_ffn["expanded"],
                 "activated": norm_ffn["activated"], "ffn_output": norm_ffn["output"],
                 "residual_output": [a + b for a, b in zip(norm["input"], norm_ffn["output"])]})
    concatenated = MULTIHEAD_Z1 + MULTIHEAD_Z2
    logits = linear(LM_G, LM_U)
    sampling_logits = [math.log(p) for p in BASE_PROBABILITIES]
    report = {
        "convention": "row vector x @ W; independent pedagogical toy parameters",
        "ffn": ffn,
        "positions": {"input": FFN_BATCH,
                      "output": [ffn_step(row)["output"] for row in FFN_BATCH],
                      "steps": [ffn_step(row) for row in FFN_BATCH]},
        "norm": norm, "layer_norm": norm,
        "multihead": {"z1": MULTIHEAD_Z1, "z2": MULTIHEAD_Z2,
                      "concatenated": concatenated, "w_o": MULTIHEAD_WO,
                      "output": linear(concatenated, MULTIHEAD_WO)},
        "swiglu": swiglu_step(),
        "lm_head": {"input": LM_G, "weights": LM_U, "bias": [0.0] * 3,
                    "logits": logits, "probabilities": softmax(logits)},
        "activations": [{"x": x, "relu": relu(x), "gelu": gelu(x), "silu": silu(x)}
                        for x in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]],
        "sampling": {
            "tokens": TOKENS, "base_probabilities": BASE_PROBABILITIES,
            "logits": sampling_logits,
            "temperatures": [{"temperature": t, "probabilities": softmax(sampling_logits, t)}
                             for t in TEMPERATURES],
            "greedy_index": greedy(BASE_PROBABILITIES),
            "inverse_cdf": [{"uniform": r, "index": inverse_cdf(BASE_PROBABILITIES, r)}
                            for r in [0.12, 0.64, 0.93]],
            "frequency_1000": sample_counts(BASE_PROBABILITIES),
            "top_k": top_k(BASE_PROBABILITIES, 2),
            "top_p": top_p(BASE_PROBABILITIES, 0.7),
            "top_p_boundary": top_p(BASE_PROBABILITIES, 0.5),
            "top_p_boundary_08": top_p(BASE_PROBABILITIES, 0.8),
        },
        "parameter_counts": {
            "toy": {"d_model": 2, "d_ff": 4, "without_bias": 16, "with_bias": 22},
            "original_transformer": {"d_model": 512, "d_ff": 2048,
                                     "without_bias": 2 * 512 * 2048,
                                     "with_bias": 2 * 512 * 2048 + 2048 + 512},
            "swiglu_budget_example": {"d_model": 512, "hidden": 1365,
                                      "without_bias": 3 * 512 * 1365},
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets" / "lesson_report.json")
    arguments = parser.parse_args()
    result = build_report(arguments.output)
    print("FFN update:", result["ffn"]["output"])
    print("LMHead probabilities:", result["lm_head"]["probabilities"])
    print("Fixed-distribution 1000 draws:", result["sampling"]["frequency_1000"]["counts"])
    print("Wrote", arguments.output)
