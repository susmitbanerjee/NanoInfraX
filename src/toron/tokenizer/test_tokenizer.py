from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "../models/tokenizer/tokenizer.json"
)

encoded = tokenizer.encode(
    "HikariPool connection exhausted"
)

print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)
print("Mask:", encoded.attention_mask)
print("Decoded:", tokenizer.decode(encoded.ids))
