"""四个 token 的二分类：只训练 embedding，输出层固定为恒等映射。"""
import math

tokens = ["酒", "牛奶", "苹果", "香蕉"]
targets = [0, 0, 1, 1]  # 标签 0：配“喝”；标签 1：配“吃”。
table = [[0.1, -0.1], [-0.1, 0.1], [0.2, 0.0], [0.0, 0.2]]

def probabilities(row):
    values = [math.exp(v - max(row)) for v in row]
    return [v / sum(values) for v in values]

def loss():
    return sum(-math.log(probabilities(row)[target])
               for row, target in zip(table, targets)) / len(tokens)

initial_loss = loss()
print("更新前：", table)
for step in range(40):
    for row, target in zip(table, targets):
        probs = probabilities(row)
        for dimension in range(2):
            gradient = probs[dimension] - int(dimension == target)
            row[dimension] -= 0.2 * gradient
print("更新后：")
for token, row in zip(tokens, table):
    print(token, [round(v, 3) for v in row])
print("平均交叉熵：", round(initial_loss, 4), "→", round(loss(), 4))
print("只说明这四个 token 的分类拟合改善，不是完整语言模型预训练。")
