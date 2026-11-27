from pathlib import Path
import polars as pl
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train():
    df= pl.read_parquet("../datasets/processed/train.parquet")
    texts = df["text"].to_list()
    tokenizer = Tokenizer(BPE(unk_tok=["UNK"]))
    tokenizer.pre_tokenizer = ByteLevel( add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=8192,
        min_frequency=2,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=["[PAD]", "[UNK]"]
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.enable_truncation(max_length=256)
    path = Path("../models/tokenizer")
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path/"tokenizer.json"))
    print("Vocab size:", tokenizer.get_vocab_size())

if __name__ == "__main__":
    train()