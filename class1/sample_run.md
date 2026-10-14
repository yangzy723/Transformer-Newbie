# Class 1 实际运行示例

这是准备课程时真实执行的结果；课堂上请先运行，再对照。不要把以下缓存输出当作现场执行。

环境：Python 3.13.5 / PyTorch 2.14.1 / macOS Apple Silicon / CPU；seed=42，200 步。

## 演示 1：Attention

```text
无遮罩的第一行：
scores  = [0.707, 0.000, 0.707]
weights = [0.401, 0.198, 0.401]
output  = [1.604, 0.797]

因果遮罩的第一行：
scores  = [0.707, -inf, -inf]
weights = [1.000, 0.000, 0.000]
output  = [1.000, 0.000]
```

## 演示 2：训练与生成

参数量 **14,060**；输入 `[8,6]`，logits `[8,6,12]`，Attention `[8,2,6,6]`。

```text
step   0 | train loss 2.6010
step   1 | train loss 1.9257
step  25 | train loss 0.3536
step  50 | train loss 0.3482
step  75 | train loss 0.3477
step 100 | train loss 0.3475
step 125 | train loss 0.3473
step 150 | train loss 0.3472
step 175 | train loss 0.3471
step 200 | train loss 0.3471

训练前：我 喜欢 牛奶 香蕉 你 茶 <eos>
训练后：我 喜欢 喝 茶 。 <eos>
```

本次记录的训练耗时为 0.127 秒，不含安装及首次导入。其他机器和版本的耗时与近似并列候选的排序可能不同。

## 怎样理解

loss 下降表明更会拟合八句训练语料。多个合法后续使 loss 不必到 0；这组数据的理想平均条件交叉熵约为 0.3466。没有验证集，不能判断泛化。

例如“喜欢”后约有 0.50 的“吃”和 0.50 的“喝”；“喝”后约有 0.50 的“茶”和 0.50 的“牛奶”。完整逐步候选见已执行 Notebook。

## 已验证

5 项关键检查通过：权重归一化与手算例子、因果遮罩及未来 V 隔离、形状与多头权重、改变未来 token 不影响更早 logits、训练下降与生成终止。

重跑：`python class1/code/check_demo.py`。

原始报告见 [sample_training_report.json](assets/sample_training_report.json)。
