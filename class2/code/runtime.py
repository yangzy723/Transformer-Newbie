"""课程共用的本地配置与 batch 构造；不联网下载模型。"""
from pathlib import Path

from tokenizers import Tokenizer
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_tokenizer(size=320):
    return Tokenizer.from_file(str(ROOT / f"assets/tokenizer_{size}.json"))


def make_batch(tokenizer, texts):
    bos, eos, pad = [tokenizer.token_to_id(t) for t in ("<bos>", "<eos>", "<pad>")]
    sequences = [[bos] + tokenizer.encode(text).ids + [eos] for text in texts]
    ids = torch.full((len(sequences), max(map(len, sequences))), pad, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence)
    return ids, ids.ne(pad)
