from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset

dataset = load_dataset("danasone/wikipedia_ru", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


tokenizer = Tokenizer(models.BPE())

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC()
])

tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()


trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["<|endoftext|>"], show_progress=True)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer, length=len(dataset))

tokenizer.save("custom_ru_tokenizer.json")
