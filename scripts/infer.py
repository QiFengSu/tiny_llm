import sys
sys.path.append("/nlp_group/suqifeng/minimind")

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

# 1. 模型路径
model_name = "/nlp_group/suqifeng/minimind/result/20250930_pretrain/checkpoint-15939"

# 2. 分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 模型（bfloat16）
model = MiniMindForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# 4. 输入处理函数
def prepare_inputs(tokenizer, text, model):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)
    return inputs

# 5. 提示词
prompts = ["牛顿", "北京市", "亚洲历史", "根据下面的文本，回答以下问题：谁是主人公？\n《红楼梦》是", "<|im_start|>鉴别一组中文文章的风格和特点，例如官"]

# 6. 生成
for prompt in prompts:
    inputs = prepare_inputs(tokenizer, prompt, model)

    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_return_sequences=1,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}\nGenerated: {text}\n")
