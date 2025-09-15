from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from accelerate import Accelerator
from peft import PeftModel
from vec_compression import CustomModel

# 确保使用 GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 Accelerator
accelerator = Accelerator()

# 加载模型和分词器
model_name = "./Qwen/Qwen2_0_5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# # 使用 accelerator 准备模型
# adr = "./Qwen/Qwen2-7B-Instruct"
# model = CustomModel.from_pretrained(adr, 896)
# 加载训练好的Lora模型
model = PeftModel.from_pretrained(model, model_id="./output(wuli)/Qwen2_0_5_7B/checkpoint-7400")
# 加载全量微调的模型
# model = AutoModelForCausalLM.from_pretrained("./output(wuli)/Qwen2_0_5_7B/checkpoint-7400", device_map="auto", torch_dtype="auto")

model = accelerator.prepare(model)

def get_final_hidden_states(prompt):
    # 分词
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 获取输入的 token ID 序列
    input_ids = inputs["input_ids"]

    # 获取输入的 attention mask（指示哪些位置有效）
    attention_mask = inputs["attention_mask"]

    # 获取嵌入表示
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # # 对每个 token 的隐藏状态应用线性变换
    # linear_output = model.linear(hidden_states).to(torch.float32)

    # 获取最后一个隐藏状态
    final_hidden_state = hidden_states[-1].to(torch.float32)

    # # 打印线性变换后的输出
    # print(f"Linear output:")
    # print(linear_output.cpu().numpy())
    # print(linear_output.cpu().shape)
    return final_hidden_state

    # # 打印每一层的隐藏状态
    # print(f"Final hidden state:")
    # print(final_hidden_state.cpu().numpy())
    # print(final_hidden_state.cpu().shape)
    # print(f"hidden_states{hidden_states}")
    # print(f"size{len(hidden_states)}")    # 25层
    # return final_hidden_state


if __name__ == "__main__":
    # 示例输入
    prompt = "Hello, how are you?"
    get_final_hidden_states(prompt)
