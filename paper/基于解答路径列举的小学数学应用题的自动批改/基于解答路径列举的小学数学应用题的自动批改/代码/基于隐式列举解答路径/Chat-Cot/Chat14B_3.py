import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./Model/qwen/Qwen1___5-14B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

# 原题：李老师买来故事书42本，连环画28本，分给同学54本，李老师还有多少本
# 情景1:{李老师买来故事书42本，连环画28本，李老师买书的总数为[A]本}
# 情景2: {李老师买书的总数为[A]本，分给同学54本，李老师还有[B]本}

cot_prompts = [
    {
        "original_text": "一个梯形的上是2分米，下底是6分米，高是3.5分米，它的面积=多少平方分米．",
        "scene_1": "{一个梯形的上是2分米，下底是6分米，高是3.5分米}->{它的面积=[A]平方分米．}"
    },
    {
        "original_text": "一个梯形的面积是9.6dm^2，上底是5dm，高是3dm，下底=．",
        "scene_1": "{一个梯形的面积是9.6dm^2，上底是5dm，高是3dm}->{下底=[A]dm．}"
    },
    {
        "original_text": "一个梯形的面积是25dm^2，上底是4.8dm，下底是5.2dm，高=．",
        "scene_1": "{一个梯形的面积是25dm^2，上底是4.8dm，下底是5.2dm}->{高=[A]dm．}"
    },
    {
        "original_text": "一个梯形的上、下底之和是36dm，是高的4倍，这个梯形的面积=多少dm^2．",
        "scene_1": "{一个梯形的上、下底之和是36dm，是高的4倍}->{高是[A]dm}",
        "scene_2": "{一个梯形的上、下底之和是36dm，高是[A]dm}->{这个梯形的面积=[B]dm^2．}"
    },
    {
        "original_text": "一个梯形的上底是12米，下底是24米，它是高的2倍，求该梯形的面积=多少平方米？",
        "scene_1": "{一个梯形的上底是12米，下底是24米，它是高的2倍}->{高是[A]米}",
        "scene_2": "{一个梯形的上底是12米，下底是24米，高是[A]米}->{求该梯形的面积=[B]平方米？}"
    },
    {
        "original_text": "一个梯形的上底是6厘米，下底是12厘米，高比下底少4厘米，求该梯形的面积=多少平方厘米？",
        "scene_1": "{一个梯形的上底是6厘米，下底是12厘米，高比下底少4厘米}->{高是[A]厘米}",
        "scene_2": "{一个梯形的上底是6厘米，下底是12厘米，高是[A]厘米}->{求该梯形的面积=[B]平方厘米？}"
    }
]

def qwen_chat(user_input):
    messages = [
        {"role": "system", "content": "You are a math teacher."},
        {"role": "user", "content": f"根据已知示例{cot_prompts}\n，请将{user_input}\n分解为情景链，请先分析题目中的信息，然后逐步构建情景，每一个情景应该包含题目中的一部分已知量和根据这部分已知量可以推出的未知量。不用计算未知量的值，将其用代数符号（A,B,C,...）表示即可。请确保每个情景都是封闭的，即每个情景推理出的未知量可以作为后续情景的已知量。"}
        # {"role": "user",
        #  "content": f"根据已知示例{cot_prompts}\n，识别出{user_input}\n对应情景，按照示例格式标明归属情景中的句子。题目中含有信息的句子必须要被归纳到特定的情景中去，且含有信息的句子可能在两个情景中出现。只用归纳情境，不用输出多余的任何文字，并且不用计算出答案,scene_1需要给出的答案用[A]代替，并将这个情境应用到情境scene_2中，算出的答案用[B]代替。"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("=== tokenizer is finished ===")
    with torch.no_grad():
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

    print(f'response: \n{response}')

    return response


if __name__ == '__main__':
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        output = qwen_chat(user_input=user_input)
