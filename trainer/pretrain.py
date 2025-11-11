# /nlp_group/suqifeng/minimind/data/pretrain_hq.jsonl
import sys
from typing import Optional
sys.path.append("/nlp_group/suqifeng/minimind")
from datasets import load_dataset, load_from_disk
import os
import argparse
import time
import math
import warnings
from itertools import chain
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import transformers
import swanlab
# swanlab.login(api_key="B60zLHywIb4tDYO65FN4s", save=True)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# from dataset.lm_dataset import PretrainDataset


class PretrainTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        self.n_gpu = kwargs.pop("n_gpu", 1)
        super().__init__(*args, **kwargs)
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[torch.Tensor] = None, **kwargs):
        # 1) 取出 labels，避免模型内部再算一遍
        labels = inputs.pop("labels", None)

        # 2) 前向
        outputs = model(**inputs)
        logits = outputs["logits"]

        if labels is None:
            # 若模型本身就返回了 loss，可直接用；否则抛错
            loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else None
            return (loss, outputs) if return_outputs else loss

        # 3) 手动 shift 并计算 CE
        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :].contiguous()    # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()        # [B, T-1]

        # 确保未训练位置已是 -100；若不确定，可在这里再做一次保护：
        # shift_labels = shift_labels.masked_fill(shift_labels == self.tokenizer.pad_token_id, -100)
        # print("shift_logits shape:", shift_logits.shape)
        # print("shift_labels shape:", shift_labels.shape)
        # reduction = "sum" if num_items_in_batch is not None else "mean"
        reduction = "mean"
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction=reduction)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
            loss = loss / shift_logits.size(-2)

        # 4) （可选）跨进程取均值：通常不必手动放大/缩放
        # loss = self.accelerator.reduce(loss, reduction="mean")

        # 5) 调试打印（仅主进程，且无梯度、搬到 CPU）
        if getattr(self.accelerator, "is_main_process", True):
            if getattr(self, "state", None) is None or self.state.global_step % 200 == 0:
                with torch.no_grad():
                    pred_ids = shift_logits.argmax(dim=-1)
                    self.accelerator.print("pred_ids first100:", pred_ids[0, :100].detach().cpu().tolist())
                    self.accelerator.print("input_ids first100:", inputs["input_ids"][0, :100].detach().cpu().tolist())
                    self.accelerator.print("true_labels first100:", shift_labels[0, :100].detach().cpu().tolist())

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="./result/20250928_pretrain")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=float, default=200)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--n_gpu", type=int, default=torch.cuda.device_count())
    parser.add_argument("--processed_cache_dir", type=str, default="./data/processed_cache_pretrain")
    
    parser.add_argument("--data_path", type=str, default="./data/pretrain_hq.jsonl")
    args = parser.parse_args()

    if args.local_rank in [-1, 0]:
        swanlab.init()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)


    raw_datasets = load_dataset("json", data_files=args.data_path, cache_dir=getattr(args, "cache_dir", None))
    raw_datasets["train"] = raw_datasets["train"]

    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.05, seed=2333)
    # raw_datasets = raw_datasets.train_test_split(test_size=0.1, seed=2333)
    print("dataset info")
    print(raw_datasets)

    print("dataset eval:")
    print(raw_datasets["test"][0])

    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    print("Model Config:")
    print(config)
    tokenizer = AutoTokenizer.from_pretrained('./our_model')
    # tokenizer = AutoTokenizer.from_pretrained('/mmu_nlp_hdd/zhanghongzhi/models/Qwen2.5-0.5B-Instruct')


    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], add_special_tokens=False)
        return outputs

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= args.max_seq_len:
            total_length = (total_length // args.max_seq_len) * args.max_seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.max_seq_len] for i in range(0, total_length, args.max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy()
        return result
    
    # 如果缓存目录已存在，直接加载；否则构建并保存
    if os.path.isdir(args.processed_cache_dir):
        tokenized_datasets = load_from_disk(args.processed_cache_dir)  # 直接复用
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=args.num_workers,
            # 注意：不要再写 load_from_cache_file=False
        )
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.num_workers,
            desc=f"Grouping texts in chunks of {args.max_seq_len}",
        )
        tokenized_datasets.save_to_disk(args.processed_cache_dir)
    
    print("Tokenized Datasets:")
    print(tokenized_datasets)
    print(tokenized_datasets['train'][0])
    print("block size:", len(tokenized_datasets['train'][0]['input_ids']))




    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = MiniMindForCausalLM(config)
    # model = AutoModelForCausalLM.from_pretrained('/mmu_nlp_hdd/zhanghongzhi/models/Qwen2.5-0.5B-Instruct')

    # 如果你自己写了 lm_head，确保它的 out_features 也和新的 num_embeddings 一致
    print("Vocab size:", len(tokenizer))
    # print("Embedding size:", model.model.embed_token.weight.size(0))
    # assert model.model.embed_token.weight.size(0) == len(tokenizer)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model Size: {model_size/1000**2:.1f}M parameters")

    train_args = transformers.TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.log_interval,
        gradient_accumulation_steps=args.accumulation_steps,  # 梯度累计总数
        num_train_epochs=args.epochs, # 训练epoch数
        weight_decay=0.1,
        warmup_steps=args.warmup_iters,
        optim="adamw_torch",  # 优化器使用adamw
        lr_scheduler_type="cosine",  # 学习率衰减策略
        learning_rate=args.learning_rate,  # 基础学习率，
        save_steps=args.save_steps,
        save_total_limit=10,
        bf16=True,  # 开启bf16训练, 对于Amper架构以下的显卡建议替换为fp16=True
    )

    print("Training Arguments:")
    print(train_args)

    from swanlab.integration.transformers import SwanLabCallback
    trainer = PretrainTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        # data_collator=transformers.default_data_collator,
        train_dataset=tokenized_datasets["train"],  # .select(range(10000))
        eval_dataset=tokenized_datasets["test"],
        callbacks=[SwanLabCallback()],
        n_gpu=args.n_gpu if args.n_gpu > 0 else 1,
    )
    trainer.train()
    model.save_pretrained(args.out_dir)  # 保存模型的路径


    # 如果是主进程
    if args.local_rank in [-1, 0]:
        model.eval()
        # 评测：让模型根据提示词生成文本
        # generate
        prompts = ["牛顿", "北京市", "亚洲历史"]
        examples = []
        for i in range(3):
            # 根据提示词生成数据
            text = model.generate(
                input_ids=tokenizer(prompts[i], return_tensors="pt").input_ids.cuda(),
                max_length=100,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.8,
                num_return_sequences=1,
            )
            text = tokenizer.decode(text[0], skip_special_tokens=True)
            print(f"Prompt: {prompts[i]}\nGenerated: {text}\n")
            # 记录日志
            text = swanlab.Text(text)
            examples.append(text)
        swanlab.log({"Generate": examples})