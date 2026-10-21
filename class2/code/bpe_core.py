"""可读的字符级 BPE：加权训练与按 rank 编码。不是字节级库的替代品。"""
from collections import Counter


def merge_pair(parts, pair):
    result, i = [], 0
    while i < len(parts):
        if tuple(parts[i:i + 2]) == pair:
            result.append(parts[i] + parts[i + 1])
            i += 2
        else:
            result.append(parts[i])
            i += 1
    return result


def train_bpe(corpus, num_merges):
    """corpus: {文本:出现次数}；同频时按首次出现顺序选择。"""
    pieces = {text: list(text) for text in corpus}
    merges, history = [], []
    for step in range(num_merges):
        counts = Counter()
        for text, parts in pieces.items():
            for pair in zip(parts, parts[1:]):
                counts[pair] += corpus[text]
        if not counts:
            break
        pair = max(counts, key=counts.get)
        merges.append(pair)
        history.append({"rank": step, "pair": pair, "count": counts[pair],
                        "top_pairs": counts.most_common(3)})
        pieces = {text: merge_pair(parts, pair) for text, parts in pieces.items()}
    return merges, history


def encode_bpe(text, merges):
    """每轮在当前相邻 pair 中选最小 rank，再合并；不按 token ID 或长度排序。"""
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    parts, trace = list(text), []
    while len(parts) > 1:
        candidates = [(ranks[pair], pair) for pair in zip(parts, parts[1:])
                      if pair in ranks]
        if not candidates:
            break
        rank, pair = min(candidates, key=lambda item: item[0])
        before = parts[:]
        parts = merge_pair(parts, pair)
        trace.append({"rank": rank, "pair": pair, "before": before, "after": parts[:]})
    return parts, trace


def demo():
    corpus = {"我喜欢喝牛奶": 5, "你喜欢喝牛奶": 3, "我喜欢喝酒": 2}
    merges, history = train_bpe(corpus, 4)
    print("训练记录：", history)
    print("固定规则编码：", encode_bpe("我喜欢喝酒", merges))
    conflict_merges, _ = train_bpe({"bc": 6, "ab": 5, "abc": 1}, 2)
    print("重叠 pair 的规则：", conflict_merges)
    print("abc 按 rank 编码：", encode_bpe("abc", conflict_merges)[0])
    print("此例最长前缀会选 ab/c，但按 rank 应为 a/bc。")


if __name__ == "__main__":
    demo()
