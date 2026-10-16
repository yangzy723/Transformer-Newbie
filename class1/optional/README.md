# 拓展：真正训练一个小型 Transformer

完成 [第一节主线](../README.md) 后，再看本目录。它保留上一版的完整实现，涉及 PyTorch、类、多头、四维张量和梯度训练。第一节课无需逐行讲解这些代码。

拓展训练使用“茶、牛奶”等原始语料；简明版人工示例使用“酒、牛奶”。这是两套独立实验，其候选概率和参数不可混用。

在仓库根目录运行：

```bash
python -m pip install -r class1/optional/requirements.txt
python class1/optional/tiny_transformer.py --steps 200
python class1/optional/attention_demo.py --causal
python class1/optional/check_demo.py
```

已有本次项目 `.venv` 的同学可用 `.venv/bin/python` 直接执行，无需再次安装。

- [tiny_transformer.py](tiny_transformer.py)：1 层、2 头、D=32 的完整模型，包含可学习的 token/位置向量、因果 Attention、FFN、残差、LayerNorm。
- [advanced_demo.ipynb](advanced_demo.ipynb)：上一版高级 Notebook，保留真实训练输出。
- [训练结果示例](training_sample_run.md)：训练 loss 从约 2.601 到 0.347，仅说明对八句训练语料的拟合改善，无验证集。

若课堂同学理解顺畅，老师可在第 51–54 分钟运行训练脚本，只展示 loss 和生成结果；实现细节放到下一节。本目录中的模型和主线 `lesson.py` 的人工数字示例是两个实验，不要把主线中的数字称为训练结果。
