"""二维旋转示例：Q/K 的旋转点积只依赖相对位移。"""
import math

def rotate(vector, position, frequency=0.2):
    angle = position * frequency
    x, y = vector
    return [x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle)]

def score(query_position, key_position):
    query = rotate([1.0, 2.0], query_position)
    key = rotate([3.0, 4.0], key_position)
    return sum(q * k for q, k in zip(query, key))

a, b = score(1, 3), score(6, 8)
print("Q 在 1、K 在 3：", round(a, 6))
print("Q 在 6、K 在 8：", round(b, 6))
print("相对位移同为 2，分数相同吗？", math.isclose(a, b, abs_tol=1e-12))
print("这是旋转点积的局部性质，不代表完整模型可无限外推。")
