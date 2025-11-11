import sys
sys.path.append("/nlp_group/suqifeng/minimind")
import argparse
from typing import Optional, Union, List
from dataclasses import dataclass, field

import datasets
from transformers import AutoTokenizer, TrainerCallback
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import swanlab

from datasets import load_dataset
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


################
# Model kwargs
################
@dataclass
class SuqfModelConfig(ModelConfig):
    model_name_or_path: Optional[str] = field(
        default="/nlp_group/suqifeng/minimind/result/20250930_pretrain/checkpoint-15939",
        metadata={
            "help": "Model checkpoint for weights initialization. default used glm4"
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

################
# Datasets kwargs
################
@dataclass
class DataTrainingArguments:
    data_files: Optional[str] = field(
        default="/nlp_group/suqifeng/minimind/data/sft_mini_512.jsonl",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

################
# Train kwargs
################
@dataclass
class MySFTConfig(SFTConfig):
    output_dir: Optional[str] = field(
        default="/nlp_group/suqifeng/minimind/result/sft_mini_512_20250930",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'lora-glm4-9b-toolcall' if not provided."
        },
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    per_device_train_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=5e-4, metadata={"help": "The initial learning rate for AdamW."}
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    bf16_full_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_seq_length` are truncated "
            "from the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    eval_strategy: Union[str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(
        default=0.02,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_steps: float = field(
        default=0.1,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

################
# Print prediction text callback
################
class SavePredictCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def on_save(self, args, state, control, model, processing_class, **kwargs):
        if state.is_world_process_zero:
            tokenizer = processing_class
            batch_test_message = [
                [{"role": "user", "content": "你好，告诉我你的名字。"}],
                [{"role": "user", "content": "告诉我1+2等于多少？"}],
            ]
            batch_inputs_text = tokenizer.apply_chat_template(
                batch_test_message,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                padding_side="left",
                add_generation_prompt=True,
            ).to(model.device)
            batch_inputs_text.pop("token_type_ids", None)  # 关键：删掉多余键
            # print(batch_inputs_text)
            outputs = model.generate(**batch_inputs_text, max_new_tokens=512)
            batch_reponse = tokenizer.batch_decode(
                outputs, skip_special_tokens=False
            )
            log_text_list = [swanlab.Text(response) for response in batch_reponse]
            swanlab.log({"Prediction": log_text_list}, step=state.global_step)

# 设置参数
dataclass_types = (SuqfModelConfig, DataTrainingArguments, MySFTConfig)
parser = TrlParser(dataclass_types)
model_args, data_args, training_args = parser.parse_args_and_config()

train_parser = SuqfModelConfig
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

# 加载数据和处理数据
raw_datasets = load_dataset("json", data_files="/nlp_group/suqifeng/minimind/data/sft_mini_512.jsonl")
raw_datasets = raw_datasets["train"].train_test_split(test_size=0.05, seed=42)

def formatting_func(examples):
    # 处理SFT数据，拼接指令和输入
    conversations = examples["conversations"]
    output_texts = tokenizer.apply_chat_template(conversations, tokenize=False)
    return output_texts

model = MiniMindForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["test"],
    data_collator=None,
    processing_class=tokenizer,
    formatting_func=formatting_func,
    callbacks=[SavePredictCallback()],
)

trainer.train()

# Save
trainer.save_model(training_args.output_dir)

