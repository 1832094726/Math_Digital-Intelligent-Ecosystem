import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # transformer>=4.37.2

"""================Qwen-7B-15GB-推理运行大小-17GB-微调训练32GB--================="""

device = "cuda"
model_id = "./Model/qwen/Qwen1___5-14B-Chat"
# 这里设置torch_dtype=torch.bfloat16 ，否则模型会按照全精度加载，GPU推理运存会从17G翻倍到34G
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16,
                                             trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

def qwen_chat(prompt, chat_history):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. "}
    ]
    messages += chat_history  # 添加之前的对话到messages中
    messages += [{"role": "user", "content": prompt}]  # 添加当前用户的输入

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("=== tokenizer is finished ===")
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f'response: {response}')
    # 将当前的用户输入和模型回复添加到聊天历史中
    chat_history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return response, chat_history


if __name__ == '__main__':
    chat_history = []  # 初始化对话历史
    while True:
        user_input = input("User: ")
        if(user_input.lower() == 'exit'):
            break
        output, chat_history = qwen_chat(prompt=user_input, chat_history=chat_history)  # 更新chat_history
