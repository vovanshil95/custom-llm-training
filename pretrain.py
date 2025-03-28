from transformers import LlamaConfig
from huggingface_hub import notebook_login
from transformers import LlamaForCausalLM
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk, load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import wandb
import numpy as np
import torch
import evaluate
from sacrebleu import corpus_bleu

from tqdm.auto import tqdm



for device in range(torch.cuda.device_count()):
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()



dataset = load_dataset("danasone/wikipedia_ru", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000 * 100):
        yield dataset[i : i + 1000]["text"]


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file='custom_ru_tokenizer.json'
)
special_tokens = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "unk_token": "<|unk|>",
    "pad_token": "<|pad|>",
    "mask_token": "<|mask|>",
    "additional_special_tokens": ["<|user|>", "<|bot|>", "<|end|>"]
}
tokenizer.add_special_tokens(special_tokens)



SMALL_PART_SIZE = 512
CONTEXT_SIZE = 4096
MAX_TRAIN_LENGTH = 2048



custom_config = LlamaConfig(
    vocab_size=32000,
    hidden_size=896,
    intermediate_size=3584,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=8,
    max_position_embeddings=CONTEXT_SIZE,
    rope_theta=10000.0,
    attention_bias=False,
    pad_token_id=tokenizer.pad_token_id,
    tie_word_embeddings=True,
    initializer_range=1.5e-4
)


model = LlamaForCausalLM(custom_config)


tokenized_dataset = load_from_disk('tokenized_dataset')



test_dataset = tokenized_dataset["test"].map(
    lambda example: {"num_tokens": len(example["input_ids"])},
    batched=False,
    num_proc=16
)
sorted_test_dataset = test_dataset.sort("num_tokens", reverse=True)


class DiffSizeDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, padding_side='left', pad_to_multiple_of=8, **kwargs):
        super().__init__(tokenizer, pad_to_multiple_of=pad_to_multiple_of, **kwargs)
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_length = max(len(f['input_ids']) for f in features)
        
        if self.pad_to_multiple_of is not None:
            padded_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        else:
            padded_length = max_length

        batch = self.tokenizer.pad(
            features,
            padding='longest',
            pad_to_multiple_of=padded_length,
            return_tensors='pt',
            padding_side=self.padding_side
        )
        
        labels = batch['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch['labels'] = labels

        return batch


class BatchEvalTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def evaluate(self, ignore_keys):

        dataloader = self.get_eval_dataloader()

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        model.eval()

        progress_bar = tqdm(
            total=len(dataloader),
            desc="Evaluation",
            unit="batch",
            dynamic_ncols=True,
        )

        losses = []
        
        accs = {
            'accuracy': [],
            'sum_tokens': []
        }

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                outputs = model(batch["input_ids"], labels=batch["input_ids"])
    
                batch_preds = []
                batch_labels = []
    
                for seq_ids, seq_logits in zip(batch["input_ids"], outputs.logits):
                    
                    unpadded_seq_logits = seq_logits[seq_ids != self.processing_class.pad_token_id]
                    unpadded_seq_ids = seq_ids[seq_ids != self.processing_class.pad_token_id]
                    unpadded_preds = torch.argmax(unpadded_seq_logits, dim=-1)
    
                    batch_preds.append(unpadded_preds[:-1])
                    batch_labels.append(unpadded_seq_ids[1:])
    
                batch_preds = torch.cat(batch_preds)
                batch_labels = torch.cat(batch_labels)
                
                accuracy = (batch_preds == batch_labels).float().mean()
                    
                accs['accuracy'].append(accuracy)
                accs['sum_tokens'].append(len(batch_labels))
                
                losses.append(outputs.loss.cpu())
    
                
                progress_bar.set_postfix({
                    "step": step+1
                })
                
                progress_bar.update(1)
    
            progress_bar.close()
                
            loss = torch.mean(torch.cat(losses))
            accuracy = torch.sum(torch.Tensor(accs['accuracy']) * torch.Tensor(accs['sum_tokens'])) / sum(accs['sum_tokens'])
            
            try:
                perplexity = torch.exp(loss)
            except OverflowError:
                perplexity = float("inf")
    
            metrics = {'eval_loss': loss.item(), 'eval_perplexity': perplexity.item(), 'eval_accuracy': accuracy.item()} 
            
        self.log(metrics) 
        self.optimizer.zero_grad()
        
        return metrics


data_collator = DiffSizeDataCollator(tokenizer, mlm=False)


GPU_COUNT = 3
BATCH_PER_GPU_TRAIN = 1
BATCH_PER_GPU_TEST = 2
STEPS_TO_UPDATE = 512
EVALS_PER_EPOCH = 25

args = TrainingArguments(
    output_dir="Llama-ru-220M",
    hub_model_id="NLPVladimir/Llama-ru-220M",
    per_device_train_batch_size=BATCH_PER_GPU_TRAIN,
    per_device_eval_batch_size=BATCH_PER_GPU_TEST,
    eval_strategy="steps",
    eval_steps=int(len(tokenized_dataset['train']) / STEPS_TO_UPDATE / EVALS_PER_EPOCH),
    logging_steps=1,
    gradient_accumulation_steps=int(STEPS_TO_UPDATE / BATCH_PER_GPU_TRAIN / GPU_COUNT),
    num_train_epochs=1,
    weight_decay=0.001,
    warmup_steps=100,
    lr_scheduler_type="constant_with_warmup",
    learning_rate=1e-3,
    save_steps=int(len(tokenized_dataset['train']) / STEPS_TO_UPDATE / EVALS_PER_EPOCH),
    fp16=True,
    fp16_full_eval=True,
    push_to_hub=True,
    run_name='Llama-ru-220M_pretraining',
    report_to="wandb",
    optim="sgd",
    resume_from_checkpoint=True
)

trainer = BatchEvalTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=sorted_test_dataset,
)


trainer.train()


trainer.push_to_hub()

