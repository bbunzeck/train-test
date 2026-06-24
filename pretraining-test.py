# Training llama models from scratch
#
# Import all needed libraries:

from pathlib import Path

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)

from tokenizers.normalizers import Lowercase, Strip, StripAccents, NFD

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    LlamaConfig)

from datasets import load_dataset

import wandb
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Llama model from scratch')
    parser.add_argument('--wandb-token', type=str, required=True, help='Weights & Biases API token')
    parser.add_argument('--hf-token', type=str, required=True, help='HuggingFace API token')
    parser.add_argument('--model-path', type=str, required=True, help='Local path to save model checkpoints')
    parser.add_argument('--hf-repo', type=str, required=True, help='HuggingFace repo ID to push model to')
    parser.add_argument('--dataset', type=str, required=True, help='HuggingFace dataset ID to train on')
    parser.add_argument('--wandb-project', type=str, required=True, help='Weights & Biases project name')
    return parser.parse_args()

args = parse_args()

wandb.login(key=args.wandb_token)
hf_token = args.hf_token
model_path = Path(args.model_path)
hf_repo = args.hf_repo

model_path.mkdir(parents=True, exist_ok=True)

os.environ['WANDB_PROJECT'] = args.wandb_project

# ### Paths
#
# Set paths to training data, eval data, and model directory:

synthetic_data = load_dataset(args.dataset)

# ### Tokenizer

# Initialize with BPE:

tokenizer = Tokenizer(models.BPE())

# Normalizer that sets everything to normal unicode, lowercase, and strips white spaces and accents
#
# (explanations here: https://huggingface.co/docs/tokenizers/components)

tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), StripAccents()])

# Pre-tokenization (division of text into tokens on which BPE can be performed):

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Set vocab size, add special tokens:

trainer = trainers.BpeTrainer(vocab_size=16000, special_tokens=["<|endoftext|>", "<pad>",])

tokenizer.train_from_iterator(iterator = synthetic_data['train']['text'], trainer=trainer)

# By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't want the offsets to include these whitespaces, then this PostProcessor must be used:
#
# (https://huggingface.co/docs/tokenizers/api/post-processors)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

tokenizer.decoder = decoders.ByteLevel()

# Save it:

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    unk_token="<|endoftext|>",
    pad_token="<|endoftext|>",
)

wrapped_tokenizer.save_pretrained(model_path / 'tokenizer')

# ### Training

# Load tokenizer:

tokenizer = AutoTokenizer.from_pretrained(model_path / 'tokenizer')

# Creates batches (https://huggingface.co/docs/transformers/pad_truncation)

context_length = 64

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        padding=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True)

    input_batch = []

    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = synthetic_data.map(tokenize,
                                      batched=True,
                                      remove_columns=synthetic_data["train"].column_names)

# Initiate new Llama with config as wished:

config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    num_hidden_layers=8,
    intermediate_size=256,
    num_attention_heads=8,
    bos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    unk_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    max_position_embeddings=context_length
)

# Set seed for weight initialization:

set_seed(42)

# New model object:

model = LlamaForCausalLM(config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set training parameters:

training_args = TrainingArguments(
    output_dir=model_path,
    #overwrite_output_dir=True,
    #save_strategy = "epoch", # saves after every epoch
    save_strategy = "steps",
    save_steps = 0.1, # if below zero, then saves after every (n*100)% of training steps
    #save_total_limit=100,  # set to zero to avoid saving
    eval_strategy = "epoch",
    #eval_steps = 0.1,
    num_train_epochs= 2,
    #max_steps = 1,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=16,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    learning_rate=3e-4, # normal: 5e-4
    logging_steps=10,
    #fp16=True, ## only on CUDA
    #load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    #use_mps_device=True, ## only on apple silicon
    #use_cpu = True
    ### WANDB stuff
    report_to="wandb",  # enable logging to W&B
    ### HF stuff
    push_to_hub = True,
    hub_strategy = 'all_checkpoints',
    hub_model_id = hf_repo,
    hub_token = hf_token
)

# Custom saving logic (just edit the `steps` list):
class CustomCheckpoints(TrainerCallback):
    def __init__(self, save_steps):
        self.save_steps = set(save_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            control.should_save = True
        return control

steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Initialize trainer object:

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    callbacks=[LogSpacedCheckpoints(steps)]
)

# Train:

trainer.train()

# Save final model

trainer.save_model(model_path / 'final')