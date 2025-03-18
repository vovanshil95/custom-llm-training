from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex
)
from datasets import load_dataset
import pickle

dataset = load_dataset("danasone/wikipedia_ru", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


tokenizer = Tokenizer(models.BPE())

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC(),
    normalizers.StripAccents(),
    normalizers.Replace(Regex(r"\s+"), " ")
])

tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])


trainer = trainers.BpeTrainer(
    limit_alphabet=2048,
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<|eos|>", "<|pad|>", "<|bos|>", "<|unk|>", "<|mask|>", "<|user|>", "<|bot|>", "<|end|>"],
    show_progress=True
)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer, length=len(dataset))

tokenizer.save("custom_ru_tokenizer.json")
