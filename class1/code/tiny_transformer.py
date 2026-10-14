"""一个可在 CPU 上训练的、单层双头的 decoder-only Transformer。

教学模型：空格分词、学习式位置向量、Pre-LN、无 dropout、无 KV cache。
只拟合八句人工句子，不代表真实大模型能力；无下载、无 API、无 GPU 要求。
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SENTENCES = [
    "我 喜欢 吃 苹果 。",
    "我 喜欢 吃 香蕉 。",
    "我 喜欢 喝 茶 。",
    "我 喜欢 喝 牛奶 。",
    "你 喜欢 吃 苹果 。",
    "你 喜欢 吃 香蕉 。",
    "你 喜欢 喝 茶 。",
    "你 喜欢 喝 牛奶 。",
]
BOS, EOS = "<bos>", "<eos>"
VOCAB = [BOS, EOS] + sorted(set(" ".join(SENTENCES).split()))
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCAB)}
MAX_LEN = 16


def encode(text):
    tokens = text.split()
    unknown = [token for token in tokens if token not in TOKEN_TO_ID]
    if unknown:
        raise ValueError(f"不在教学词表中的 token：{unknown}；请用空格分词")
    return [TOKEN_TO_ID[token] for token in tokens]


def decode(ids):
    return " ".join(VOCAB[int(i)] for i in ids)


def make_batch():
    # 7 个 token：<bos> 我 喜欢 吃 苹果 。 <eos>
    sequences = [[TOKEN_TO_ID[BOS]] + encode(s) + [TOKEN_TO_ID[EOS]]
                 for s in SENTENCES]
    tokens = torch.tensor(sequences, dtype=torch.long)
    # x/y 都是 [8, 6]；每个位置学习预测右边的一个 token。
    return tokens[:, :-1], tokens[:, 1:]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=2):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, length, width = x.shape                 # [B, T, D]
        q, k, v = self.qkv(x).chunk(3, dim=-1)         # 各 [B, T, D]
        q, k, v = [t.view(batch, length, self.n_heads, self.d_head)
                   .transpose(1, 2) for t in (q, k, v)]  # [B, H, T, d]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        future = torch.ones(length, length, dtype=torch.bool,
                            device=x.device).triu(diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = F.softmax(scores, dim=-1)           # [B, H, T, T]
        mixed = weights @ v                          # [B, H, T, d]
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(mixed), weights              # [B, T, D]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=32, n_heads=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        # Pre-LN：先归一化再进入子层；与原论文的 Post-LN 排列不同。
        context, weights = self.attention(self.ln1(x))
        x = x + context
        x = x + self.ffn(self.ln2(x))
        return x, weights


class TinyTransformer(nn.Module):
    def __init__(self, d_model=32, n_heads=2):
        super().__init__()
        self.token_embedding = nn.Embedding(len(VOCAB), d_model)
        self.position_embedding = nn.Embedding(MAX_LEN, d_model)
        self.block = TransformerBlock(d_model, n_heads)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, len(VOCAB))

    def forward(self, token_ids):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= MAX_LEN:
            raise ValueError(f"输入应为 [B, T]，且 1 <= T <= {MAX_LEN}")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x, weights = self.block(x)
        return self.lm_head(self.final_norm(x)), weights


@torch.no_grad()
def next_token_options(model, ids, top_k=4):
    logits, _ = model(torch.tensor([ids], dtype=torch.long))
    probabilities = F.softmax(logits[0, -1], dim=-1)
    values, indices = probabilities.topk(min(top_k, len(VOCAB)))
    return [{"token": VOCAB[i], "probability": round(float(p), 4)}
            for p, i in zip(values, indices.tolist())]


@torch.no_grad()
def generate(model, prompt="我 喜欢", max_new_tokens=6, trace=False):
    model.eval()
    ids = [TOKEN_TO_ID[BOS]] + encode(prompt)
    if len(ids) > MAX_LEN:
        raise ValueError(f"prompt 加 <bos> 后最多 {MAX_LEN} 个 token")
    for step in range(max_new_tokens):
        if len(ids) >= MAX_LEN:
            break
        options = next_token_options(model, ids)
        if trace:
            print(f"第 {step+1} 步，上下文：{decode(ids)}")
            print("  候选：", options)
        # 用贪心选择方便课堂观察；真实应用也可按概率采样。
        next_id = TOKEN_TO_ID[options[0]["token"]]
        ids.append(next_id)
        if next_id == TOKEN_TO_ID[EOS]:
            break
    return decode(ids[1:])


def train_model(steps=200, seed=42, verbose=True):
    if steps < 1:
        raise ValueError("steps 必须大于 0")
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # 小矩阵用单线程，避免线程调度比计算还慢。
    model = TinyTransformer()
    x, y = make_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    def loss_value():
        logits, _ = model(x)
        # 输入原始 logits；cross_entropy 内部处理 log-softmax。
        return F.cross_entropy(logits.reshape(-1, len(VOCAB)), y.reshape(-1))

    with torch.no_grad():
        initial_loss = float(loss_value())
    before = generate(model)
    history = [{"step": 0, "loss": initial_loss}]
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        loss = loss_value()
        optimizer.zero_grad()
        loss.backward()  # 自动计算：参数如何改变会让误差更小。
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            with torch.no_grad():
                measured = float(loss_value())  # 记录更新后的同一训练集 loss。
            history.append({"step": step, "loss": measured})
            if verbose:
                print(f"step {step:3d} | train loss {measured:.4f}")
    model.eval()
    report = {
        "seed": seed, "steps": steps, "device": "cpu",
        "torch_version": torch.__version__, "vocab": VOCAB,
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": list(x.shape), "logits_shape": [8, 6, len(VOCAB)],
        "attention_shape": [8, 2, 6, 6], "history": history,
        "initial_loss": initial_loss, "final_loss": history[-1]["loss"],
        "before_training": before, "after_training": generate(model),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scope": "同一训练集的拟合演示，无验证集，不能据此判断泛化能力",
    }
    return model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="我 喜欢")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args(argv)
    print("人工句子：", SENTENCES)
    print("词表：", TOKEN_TO_ID)
    x, y = make_batch()
    print("第一句 x：", decode(x[0]), "\n第一句 y：", decode(y[0]))
    model, report = train_model(args.steps, args.seed)
    print("输入 / logits / attention：", report["input_shape"],
          report["logits_shape"], report["attention_shape"])
    print("参数量：", report["parameters"])
    print("训练前：", report["before_training"])
    print("训练后：", generate(model, args.prompt, trace=True))
    print(f"loss: {report['initial_loss']:.4f} -> {report['final_loss']:.4f}")
    print("耗时：", report["elapsed_seconds"], "秒；", report["scope"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "tiny_transformer.pt")
    return report


if __name__ == "__main__":
    main()
