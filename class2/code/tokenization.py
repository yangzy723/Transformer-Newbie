# %% 1. 同一句话，可以有不同的切分单位
text = "我喜欢喝牛奶"
print("按字：", list(text))
print("一种词切分：", ["我", "喜欢", "喝", "牛奶"])

# %% 2. 按一条规则，从左到右合并相邻片段
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

# %% 3. 使用已经学好的规则，再把片段转换成 ID
merges = [("喜", "欢"), ("喜欢", "喝"), ("喜欢喝", "牛"), ("喜欢喝牛", "奶")]
parts = list(text)
for pair in merges:
    parts = merge_pair(parts, pair)
    print("应用规则", pair, "→", parts)
vocab = ["你", "喝", "喜", "奶", "我", "欢", "牛", "酒"]
vocab += [left + right for left, right in merges]
token_ids = [vocab.index(token) for token in parts]
print("最终 token：", parts)
print("token ID：", token_ids)

# %% 4. UTF-8 字节不是最终 token：两个汉字在这里占六个字节
word = "牛奶"
print("字符数：", len(word), "UTF-8 字节数：", len(word.encode("utf-8")))
print("UTF-8 字节：", list(word.encode("utf-8")))
