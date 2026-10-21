# %% 7. 为 token 位置 0、1、2 构造四维正弦位置编码
import math

def position_vector(position, width=4):
    result = []
    for i in range(width // 2):
        angle = position / (10000 ** (2 * i / width))
        result += [math.sin(angle), math.cos(angle)]
    return result

for position in range(3):
    print("位置", position, "→", [round(v, 4) for v in position_vector(position)])

# %% 8. 同一个 token 换一个位置，组合后的输入会改变
word_vector = [0.127, 0.448, 0.077, -0.103]  # embeddings.py 中 ID=4 的一行
combined = []
for position in (0, 2):
    pe = position_vector(position)
    vector = [word_vector[i] + pe[i] for i in range(4)]
    combined.append(vector)
    print("‘我’在位置", position, "→", [round(v, 4) for v in vector])
print("token 向量相同，组合后的输入相同吗？", combined[0] == combined[1])
