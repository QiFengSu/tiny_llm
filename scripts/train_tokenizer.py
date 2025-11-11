import random
import json
import os
from pathlib import Path
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

random.seed(42)

# ========= 直接在这里改路径即可 =========
DATA_PATH = Path("../data/pretrain_hq.jsonl")   # 训练用 JSONL 数据
TOKENIZER_DIR = Path("../our_model/")                  # tokenizer 输出目录
# =====================================

def read_texts_from_jsonl(file_path: Path):
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

def train_tokenizer(data_path: Path, tokenizer_dir: Path):
    # 基础校验
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path.resolve()}")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 特殊 token（顺序决定 id：0,1,2）
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 训练
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 校验特殊 token 的索引（与 special_tokens 顺序一致）
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 保存 tokenizer
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.model.save(str(tokenizer_dir))

    # 手动创建并保存配置
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
    }
    with (tokenizer_dir / "tokenizer_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print(f"Tokenizer training completed and saved to: {tokenizer_dir.resolve()}")

def eval_tokenizer(tokenizer_dir: Path):
    from transformers import AutoTokenizer

    # 加载预训练的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"}
    ]
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(new_prompt)

    # 词表长度（含特殊符号）
    actual_vocab_size = len(tokenizer)
    print("tokenizer实际词表长度：", actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print("encoder长度：", len(model_inputs["input_ids"]))

    input_ids = model_inputs["input_ids"]
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("decoder和原始文本是否一致：", response == new_prompt)

def main():
    # 只需在文件顶部修改 DATA_PATH / TOKENIZER_DIR 即可
    train_tokenizer(DATA_PATH, TOKENIZER_DIR)
    eval_tokenizer(TOKENIZER_DIR)

if __name__ == "__main__":
    main()
