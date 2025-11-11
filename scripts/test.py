
import sys
sys.path.append("/nlp_group/suqifeng/minimind")

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

# 1. 模型路径
model_name = "/nlp_group/suqifeng/minimind/result/20250929_pretrain/checkpoint-4554"

# 2. 分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. 输入处理函数
def prepare_inputs(tokenizer, text, model):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)
    return inputs

# 5. 提示词
prompts = ["牛顿", "北京市", "亚洲历史", "根据下面的文本，回答以下问题：谁是主人公？\n《红楼梦》是"]

text = tokenizer.decode([265, 11, 262, 319, 319, 201, 201, 201, 201, 201, 248, 201, 201, 201, 201, 201, 263, 201, 263, 201, 201, 201, 201, 265, 201, 1475, 263, 1670, 312, 201, 1416, 201, 287, 263, 1259, 265, 201, 201, 1791, 1083, 1923, 1683, 3666, 2322, 263, 201, 201, 1, 263, 263, 263, 263, 263, 201, 263, 263, 263, 263, 263, 263, 5673, 263, 201, 300, 2215, 263, 265, 263, 263, 223, 265, 263, 263, 263, 602, 2717, 505, 263, 263, 263, 263, 265, 263, 262, 263, 265, 263, 263, 262, 263, 263, 263, 263, 263, 201, 223, 1, 3681, 265, 1256], skip_special_tokens=True)
print(f"Generated: {text}\n")
