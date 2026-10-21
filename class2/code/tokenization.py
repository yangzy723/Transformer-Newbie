"""真实 ByteLevel BPE：读取已训练配置，观察预切分、token、ID 与回解。"""
from runtime import load_tokenizer


def demo():
    samples = ["我喜欢喝牛奶。", " 我喜欢喝牛奶。", "Hello world!",
               "Hello, world!", "未见过的字🦄"]
    for size in (320, 384):
        tokenizer = load_tokenizer(size)
        print("\n词表大小：", tokenizer.get_vocab_size())
        for text in samples:
            encoded = tokenizer.encode(text)
            print("输入：", repr(text))
            print("tokens：", encoded.tokens)
            print("ids：", encoded.ids, "长度：", len(encoded.ids))
            print("offsets：", encoded.offsets)
            print("完整回解相同：", tokenizer.decode(encoded.ids) == text)
    tokenizer = load_tokenizer()
    print("预切分 Hello, world!：", tokenizer.pre_tokenizer.pre_tokenize_str("Hello, world!"))
    print("注意：ByteLevel 的 token 字符串是字节的可显示编码，不一定是完整汉字。")


if __name__ == "__main__":
    demo()
