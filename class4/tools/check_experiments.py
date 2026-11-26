"""用标准库验证第四课的数值、形状、边界和独立性；不依赖随机频数恰好固定。"""

import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("class4_lesson", ROOT / "code" / "lesson.py")
lesson = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lesson)


def close(actual, expected, tolerance=1e-10):
    """比较任意层数列表与浮点数，保留严格的形状检查。"""
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), (actual, expected)
        for a, b in zip(actual, expected):
            close(a, b, tolerance)
    else:
        assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def raises_value_error(call):
    """要求非法维度或解码输入明确抛出 ValueError。"""
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main():
    """检查示例并打印简明结果；断言失败时直接以非零状态退出。"""
    report = lesson.build_report()
    ffn = report["ffn"]
    close(ffn["expanded"], [1, -1, -2, 2])
    close(ffn["activated"], [1, 0, 0, 2])
    close(ffn["output"], [1, 2])
    close(ffn["combined_weights"], [[0, 0], [0, 0]])
    close(ffn["without_activation"], [0, 0])
    close(report["positions"]["output"], [[1, 2], [1, 2], [0.5, 0.25]])
    changed = [list(row) for row in lesson.FFN_BATCH]
    changed[1] = [-3, 4]
    changed_outputs = [lesson.ffn_step(row)["output"] for row in changed]
    close(changed_outputs[0], report["positions"]["output"][0])
    close(changed_outputs[1], [3, 4])
    close(changed_outputs[2], report["positions"]["output"][2])
    close(lesson.ffn_step([0, 0])["output"], [0, 0])
    # 非线性反例：f(a+b) != f(a)+f(b)，但不能据此称整个 Transformer 为线性。
    left = lesson.ffn_step([0, 0])["output"]
    right = [a + b for a, b in zip(lesson.ffn_step([1, -2])["output"],
                                  lesson.ffn_step([-1, 2])["output"])]
    assert left != right
    norm = report["norm"]
    close(norm["mean"], 1.55)
    close(norm["variance"], 0.0625)
    expected_norm = 0.25 / math.sqrt(0.0625 + 1e-5)
    close(norm["normalized"], [-expected_norm, expected_norm])
    close(norm["ffn_output"], [expected_norm, expected_norm])
    close(norm["residual_output"], [1.3 + expected_norm, 1.8 + expected_norm])
    close(lesson.layer_norm([2, 2])["output"], [0, 0])
    close(lesson.layer_norm([1, 3], gamma=[2, 3], beta=[4, 5])["output"],
          [4 - 2 / math.sqrt(1 + 1e-5), 5 + 3 / math.sqrt(1 + 1e-5)])
    close(report["multihead"]["concatenated"], [1, 2, -1, 3])
    close(report["multihead"]["output"], [0, 5, -1, 3])
    close(report["lm_head"]["logits"], [1, 2, -1])
    close(report["lm_head"]["probabilities"],
          [0.25949646034241913, 0.7053845126982412, 0.03511902695933972])
    assert len(ffn["output"]) == 2 and len(report["lm_head"]["probabilities"]) == 3
    close(lesson.softmax([1001, 1002, 999]), lesson.softmax([1, 2, -1]))
    close(lesson.softmax([10000, 10000]), [0.5, 0.5])
    sampling = report["sampling"]
    close(lesson.softmax(sampling["logits"]), lesson.BASE_PROBABILITIES)
    close(sampling["temperatures"][0]["probabilities"], [25 / 38, 9 / 38, 4 / 38])
    for item in sampling["temperatures"]:
        probabilities = item["probabilities"]
        close(sum(probabilities), 1)
        assert probabilities[0] > probabilities[1] > probabilities[2]
    assert sampling["temperatures"][0]["probabilities"][0] > 0.5
    assert sampling["temperatures"][2]["probabilities"][0] < 0.5
    assert sampling["greedy_index"] == 0
    assert [item["index"] for item in sampling["inverse_cdf"]] == [0, 1, 2]
    assert lesson.inverse_cdf([0.5, 0.3, 0.2], 0) == 0
    assert lesson.inverse_cdf([0.5, 0.3, 0.2], 0.5) == 1
    assert lesson.inverse_cdf([0, 0.5, 0.5], 0) == 1
    assert lesson.inverse_cdf([0.5, 0.3, 0.2], 0.9999999999) == 2
    frequency = sampling["frequency_1000"]
    assert sum(frequency["counts"]) == frequency["count"] == 1000
    assert all(count > 0 for count in frequency["counts"])
    assert frequency == lesson.sample_counts(lesson.BASE_PROBABILITIES, count=1000, seed=7)
    assert all(abs(p - observed) < 0.08
               for p, observed in zip(lesson.BASE_PROBABILITIES, frequency["frequencies"]))
    for key in ["top_k", "top_p", "top_p_boundary_08"]:
        assert sampling[key]["indices"] == [0, 1]
        close(sampling[key]["probabilities"], [0.625, 0.375, 0])
    assert sampling["top_p_boundary"]["indices"] == [0]
    close(sampling["top_p_boundary"]["probabilities"], [1, 0, 0])
    assert lesson.top_p(lesson.softmax(sampling["logits"]), 0.8)["indices"] == [0, 1]
    assert lesson.top_p([0.2, 0.5, 0.3], 0.7)["indices"] == [1, 2]
    assert lesson.top_p([0.2, 0.5, 0.3], 1)["indices"] == [1, 2, 0]
    assert lesson.top_k([0.5, 0.5], 1)["indices"] == [0]
    close(lesson.silu(0), 0)
    close(lesson.gelu(0), 0)
    close(lesson.silu(-1000), 0)
    close(lesson.silu(1000), 1000)
    swiglu = report["swiglu"]
    close(swiglu["gate_projection"], [1, -2])
    close(swiglu["value_projection"], [1, -1])
    close(swiglu["output"], [0.7310585786300049, 0.2384058440442351])
    assert swiglu["gate"][1] < 0  # SiLU gate 不是 [0,1] 内的概率。
    assert lesson.decode([1, 2, -1], temperature=0) == 1
    assert lesson.decode([1, 2, -1], temperature=1, top_p_threshold=0.1, rng=random.Random(7)) == 1
    assert lesson.matrix_shape(lesson.FFN_W1) == (2, 4)
    assert lesson.matrix_shape(lesson.FFN_W2) == (4, 2)
    for invalid in [lambda: lesson.matmul([[1, 2]], [[1, 2]]),
                    lambda: lesson.matmul([[1], [1, 2]], [[1]]),
                    lambda: lesson.linear([1, 2], [[1], [2]], bias=[1, 2]),
                    lambda: lesson.softmax([1, 2], 0),
                    lambda: lesson.softmax([1, float("nan")]),
                    lambda: lesson.top_p([0.5, 0.5], 0),
                    lambda: lesson.top_k([0.5, 0.5], 3),
                    lambda: lesson.inverse_cdf([0.5, 0.5], 1),
                    lambda: lesson.sample_counts([0.5, 0.5], 0),
                    lambda: lesson.layer_norm([])]:
        raises_value_error(invalid)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "report.json"
        lesson.build_report(destination)
        assert json.loads(destination.read_text(encoding="utf-8")) == report
    print("PASS: FFN steps/shape/nonlinearity, per-position independence, LayerNorm/residual,")
    print("      concat/W_O, LMHead/softmax invariance, temperature/ranking, inverse-CDF,")
    print("      1000 samples, top-k/top-p boundaries, activations/SwiGLU, JSON round-trip")
    print("Observed 1000-draw counts (not an exact-count assertion):", frequency["counts"])


if __name__ == "__main__":
    main()
