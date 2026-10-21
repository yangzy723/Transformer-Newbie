"""Embedding → Linear → 下一 token 交叉熵；真实训练的 bigram 基线。"""
import torch
from torch import nn
from torch.nn import functional as F

from runtime import load_tokenizer


def bigram_batch(tokenizer):
    texts = ["我喜欢喝牛奶。", "我喜欢喝酒。", "我喜欢喝水。", "你喜欢喝牛奶吗？",
             "我喜欢吃苹果。", "你喜欢吃香蕉。", "你喜欢吃苹果吗？"]
    pairs = []
    for text in texts:
        sequence = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(text).ids
        sequence += [tokenizer.token_to_id("<eos>")]
        pairs += list(zip(sequence, sequence[1:]))
    return torch.tensor(pairs, dtype=torch.long).T


def train(steps=120, verbose=True):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    tokenizer = load_tokenizer()
    x, y = bigram_batch(tokenizer)
    embedding = nn.Embedding(tokenizer.get_vocab_size(), 16)
    output = nn.Linear(16, tokenizer.get_vocab_size())
    optimizer = torch.optim.Adam(list(embedding.parameters()) + list(output.parameters()), lr=0.05)
    before = embedding.weight.detach().clone()
    history = []
    for step in range(steps + 1):
        logits = output(embedding(x))
        loss = F.cross_entropy(logits, y)
        if step % 30 == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach())})
            if verbose:
                print("step", step, "loss", round(float(loss.detach()), 4))
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    changed = (embedding.weight.detach() - before).norm(dim=1)
    report = {"history": history, "vocab_size": tokenizer.get_vocab_size(),
              "embedding_shape": list(embedding.weight.shape), "pairs": len(x),
              "changed_rows": int((changed > 0).sum()), "observed_rows": int(x.unique().numel())}
    if verbose:
        print("表形状 / 训练 pair 数：", report["embedding_shape"], report["pairs"])
        print("更新行数 / 实际输入过的行数：", report["changed_rows"], report["observed_rows"])
        print("只用当前 token 预测下一项，不能替代利用完整上下文的 Transformer。")
    return embedding, output, report


if __name__ == "__main__":
    train()
