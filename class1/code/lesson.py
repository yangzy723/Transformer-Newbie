# %% 0. 准备：把一组分数变成总和为 1 的权重
import math


def softmax(scores):
    exp_scores = []
    for score in scores:
        exp_scores.append(math.exp(score - max(scores)))
    total = sum(exp_scores)
    return [value / total for value in exp_scores]


# %% 1. 文字 → 编号 → 数字列表（这里的数字是人工设置的）
text = "我 喜欢 喝"
tokens = text.split()
vocab = {"我": 0, "喜欢": 1, "喝": 2, "酒": 3, "牛奶": 4}
token_ids = [vocab[token] for token in tokens]
embedding_table = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 1.0], [0.8, 0.2]]
vectors = [embedding_table[token_id] for token_id in token_ids]
print("1. token：", tokens)
print("   ID：", token_ids)
print("   向量：", vectors)

# %% 2. 从“喝”这个位置发出查询（本例 Q/K/V 用恒等投影）
query = vectors[-1]
keys = vectors
values = vectors
print("2. 当前查询 Q：", query)

# %% 3. 匹配分数 → 权重，只算当前 query 这一行
scores = []
for key in keys:
    dot = query[0] * key[0] + query[1] * key[1]
    scores.append(dot / math.sqrt(2))
weights = softmax(scores)
print("3. 匹配分数：", [round(score, 3) for score in scores])
print("   混合权重：", [round(weight, 3) for weight in weights])

# %% 4. 按权重混合 V，得到一个新的向量
mixed = [0.0, 0.0]
for i in range(len(tokens)):
    mixed[0] += weights[i] * values[i][0]
    mixed[1] += weights[i] * values[i][1]
print("4. Attention 输出：", [round(value, 3) for value in mixed])

# %% 5. 向量 → 候选分数 → 概率 → 下一 token
candidates = ["酒", "牛奶"]
logits = [2 * mixed[0], mixed[1]]  # 人工输出投影，未训练；只用两个候选演示
probabilities = softmax(logits)
for i in range(len(candidates)):
    print("5. 候选：", candidates[i], round(probabilities[i], 3))
next_token = candidates[probabilities.index(max(probabilities))]
print("   接上一个 token：", text + " " + next_token)
