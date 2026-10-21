# %% 5. token ID 是查表地址；每一行是一组可训练的参数
import random

vocab = ["你", "喝", "喜", "奶", "我", "欢", "牛", "酒",
         "喜欢", "喜欢喝", "喜欢喝牛", "喜欢喝牛奶"]
rng = random.Random(7)
table = [[round(rng.uniform(-0.5, 0.5), 3) for _ in range(4)] for _ in vocab]
token_ids = [4, 11]
vectors = [table[token_id] for token_id in token_ids]
for token_id, vector in zip(token_ids, vectors):
    print("ID", token_id, vocab[token_id], "→", vector)
print("词表大小 / 向量维度：", len(table), len(table[0]))

# %% 6. 重复查同一行，得到相同的输入 embedding
print("再次查‘我’：", table[4])
print("两次查询相同：", vectors[0] == table[4])
print("这是随机初始化，不是训练好的词义表示。")
