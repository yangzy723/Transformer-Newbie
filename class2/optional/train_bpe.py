"""统计相邻片段频次，学习四条 BPE 合并规则。仅处理本例字符集合。"""
from collections import Counter

def merge_pair(parts, pair):
    result = []
    i = 0
    while i < len(parts):
        if tuple(parts[i:i + 2]) == pair:
            result.append(parts[i] + parts[i + 1])
            i += 2
        else:
            result.append(parts[i])
            i += 1
    return result

corpus = {"我喜欢喝牛奶": 5, "你喜欢喝牛奶": 3, "我喜欢喝酒": 2}
pieces = {word: list(word) for word in corpus}
learned_merges = []
for step in range(4):
    counts = Counter()
    for word, parts in pieces.items():
        for pair in zip(parts, parts[1:]):
            counts[pair] += corpus[word]
    best = max(counts, key=counts.get)  # 同频时使用首次出现的 pair，固定本例结果。
    learned_merges.append(best)
    print("第", step + 1, "轮：", best, "出现", counts[best], "次")
    pieces = {word: merge_pair(parts, best) for word, parts in pieces.items()}
print("学到的规则：", learned_merges)
print("切分结果：", pieces)
