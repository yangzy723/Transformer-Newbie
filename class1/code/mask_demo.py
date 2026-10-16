# 可选的 3 分钟演示：未来分数变成负无穷，未来权重就是 0。
import math

scores = [0.707, -math.inf, -math.inf]  # 第一个位置只能读自己
exp_scores = []
for score in scores:
    exp_scores.append(math.exp(score - max(scores)))
weights = [value / sum(exp_scores) for value in exp_scores]

print("未来位置被挡住的分数：", scores)
print("对应权重：", weights)
