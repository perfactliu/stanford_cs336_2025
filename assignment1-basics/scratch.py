# Problem1
# chr(0)
# print(chr(0))
# print(repr(chr(0)))
# print("this is a test" + chr(0) + "string")

# Problem2
# test_string = "hello! こんにちは!"
# utf8_encoded = test_string.encode("utf-8")
# utf16_encoded = test_string.encode("utf-16")
# utf32_encoded = test_string.encode("utf-32")
#
# print(utf8_encoded)
# print(utf16_encoded)
# print(utf32_encoded)

# Problem3
# from tests.adapters import run_train_bpe, get_tokenizer
# import pickle
# input_path = 'tests/fixtures/tinystories_sample_5M.txt'
# vocab, merges = run_train_bpe(
#     input_path=input_path,
#     vocab_size=5000,
#     special_tokens=["<|endoftext|>"],
# )
# # print(vocab)
# # print(merges)
# # 创建包含两个数据结构的字典
# data = {
#     'merges': merges,
#     'vocab': vocab
# }
#
# # 存储到文件
# with open('tokenizer_data.pkl', 'wb') as f:
#     pickle.dump(data, f)

# 从文件加载
# with open('tokenizer_data.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
#
# merges_loaded = loaded_data['merges']
# vocab_loaded = loaded_data['vocab']
# tokenizer = get_tokenizer(
#     vocab=vocab_loaded,
#     merges=merges_loaded,
#     special_tokens=["<|endoftext|>"]
# )
# print('loaded tokenzier')
# test_string = "Lily and Tom were twins who liked to decorate things."
# encoded_ids = tokenizer.encode(test_string)
# print(encoded_ids)
# decoded_string = tokenizer.decode(encoded_ids)
# print(decoded_string)

# Problem4
from tests.adapters import get_tokenizer
import pickle
import torch

with open('tokenizer_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

merges_loaded = loaded_data['merges']
vocab_loaded = loaded_data['vocab']
tokenizer = get_tokenizer(
    vocab=vocab_loaded,
    merges=merges_loaded,
    special_tokens=["<|endoftext|>"]
)

input_path = 'tests/fixtures/tinystories_sample_5M.txt'
output_path = 'tokenized_text.pt'
with open(input_path, "rb") as f:
    text = f.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip() != ""]
    # 确保所有行是 str 类型（防御性写法）
    non_blank_lines = [line.decode("utf-8") if isinstance(line, bytes) else line for line in non_blank_lines]
    clean_text = "\n".join(non_blank_lines)
# Tokenize to IDs
token_ids = tokenizer.encode(text)  # list[int]

# Convert to tensor
token_tensor = torch.tensor(token_ids, dtype=torch.long)
print(token_tensor)

# Save to file
torch.save(token_tensor, output_path)
print(f"Saved tokenized text to {output_path}, total {len(token_ids)} tokens.")
