"""使用真实 ByteLevel BPE 实现，在本地语料上训练并保存配置。"""
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train(vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=SPECIALS)
    corpus = (ROOT / "data/corpus.txt").read_text(encoding="utf-8").splitlines()
    tokenizer.train_from_iterator(corpus * 10, trainer=trainer)
    return tokenizer


def main():
    (ROOT / "assets").mkdir(exist_ok=True)
    samples = ["我喜欢喝牛奶。", " 我喜欢喝牛奶。", "Hello world!", "Hello, world!", "未见过的字🦄"]
    report = {"tokenizers_version": tokenizers.__version__, "models": {}}
    for size in (320, 384):
        tokenizer = train(size)
        tokenizer.save(str(ROOT / f"assets/tokenizer_{size}.json"))
        entries = []
        for text in samples:
            encoded = tokenizer.encode(text)
            entries.append({"text": text, "tokens": encoded.tokens, "ids": encoded.ids,
                            "offsets": encoded.offsets, "length": len(encoded.ids),
                            "roundtrip": tokenizer.decode(encoded.ids) == text})
        report["models"][str(size)] = {"vocab_size": tokenizer.get_vocab_size(), "examples": entries}
    (ROOT / "assets/tokenizer_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
