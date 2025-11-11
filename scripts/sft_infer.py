import sys
sys.path.append("/nlp_group/suqifeng/minimind")

from model.model_minimind import MiniMindForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载 tokenizer 和模型
model_path = "/nlp_group/suqifeng/minimind/result/sft/sft_mini_512_20250930/checkpoint-27048"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = MiniMindForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

# 确保 pad_token 已设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id


# 3. 推理函数
def infer(text: str, max_new_tokens=128):
    # 返回字符串，而不是 token id
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], 
        tokenize=False,   # 关键！强制返回 string
        add_generation_prompt=True
    )
    print("Prompt after applying chat template:", prompt)

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


# 4. 测试推理
examples = ["牛顿是谁？", "介绍一下北京市", "亚洲历史的开端"]
for text in examples:
    print("Prompt:", text)
    print("Output:", infer(text))
    print("="*50)
