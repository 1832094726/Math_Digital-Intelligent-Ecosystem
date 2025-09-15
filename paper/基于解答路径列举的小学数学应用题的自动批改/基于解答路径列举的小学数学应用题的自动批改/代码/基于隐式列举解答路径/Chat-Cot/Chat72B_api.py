import json
from http import HTTPStatus
import dashscope
import ast

dashscope.api_key="sk-5527debcbee940c682bdbcb459579a06"

cot_prompts = [
    {
        "original_text": "在长12cm、宽9cm的长方形中，剪出一个最大的正方形，正方形的面积=多少cm^2．",
        "scene_1": "在长12cm、宽9cm的长方形中，剪出一个最大的正方形，正方形的面积=[A]cm^2．"
    },
    {
        "original_text": "一张长方形硬纸板长12厘米，宽9厘米．在这个长方形中，剪去一个最大的正方形，剩下的小长方形的面积=？",
        "scene_1": "一张长方形硬纸板长12厘米，宽9厘米．在这个长方形中，剪去一个最大的正方形，正方形的面积为[A]平方厘米．",
        "scene_2": "一张长方形硬纸板长12厘米，宽9厘米，长方形的面积为[B]平方厘米．",
        "scene_3": "一张长方形硬纸板长12厘米，宽9厘米．在这个长方形中，剪去一个最大的正方形，正方形的面积为[A]平方厘米，长方形的面积为[B]平方厘米，剩下的小长方形的面积=[C]cm^2．"
    },
    {
        "original_text": "如果在一个长6分米、宽2分米的长方形纸上剪下一个最大的半圆，那么半圆的面积是多少平方分米？",
        "scene_1": "如果在一个长6分米、宽2分米的长方形纸上剪下一个最大的半圆，那么半圆的面积是[A]平方分米？"
    },
    {
        "original_text": "从一个边长是4cm的正方形里剪下一个最大的圆，剩下的面积=多少cm^2．",
        "scene_1": "从一个边长是4cm的正方形里剪下一个最大的圆，圆的面积为[A]cm^2.",
        "scene_2": "从一个边长是4cm的正方形里剪下一个最大的圆，圆的面积为[A]cm^2，正方形的面积为[B]cm^2．",
        "scene_3": "从一个边长是4cm的正方形里剪下一个最大的圆，圆的面积为[A]cm^2，正方形的面积为[B]cm^2，剩下的铁皮面积=[C]平方厘米．"
    }
]

def call_with_messages(user_input):
    prompt = f"根据已知示例{cot_prompts}\n，识别出{user_input}\n对应情景，按照示例格式标明归属情景。题目中含有信息的句子必须要被归纳到特定的情景中去，且含有信息的句子可能在两个情景中出现。只用归纳情境，不用输出多余的任何文字，并且不用计算出答案，将其用代数符号（A,B,C,...）表示即可。"
    # prompt = f"根据已知示例{cot_prompts}\n，请将{user_input}\n分解为情景链，请先分析题目中的信息，然后逐步构建情景，每一个情景应该包含题目中的一部分已知量和根据这部分已知量可以推出的未知量。不用计算未知量的值，将其用代数符号（A,B,C,...）表示即可。请确保每个情景都是封闭的，即每个情景推理出的未知量可以作为后续情景的已知量。"
    messages = [{'role': 'user', 'content': prompt}]
    response = dashscope.Generation.call(
        'qwen1.5-72b-chat',
        messages=messages,
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        try:
            # 修正可能的字符串字典格式问题
            response_content = response.output.choices[0].message.content.replace("'", "\"")
            # 尝试将修正后的字符串解析为JSON
            response_dict = json.loads(response_content)
            output_data = {
                "original_text": user_input,
                **response_dict  # 展开这个字典，直接并入output_data
            }
            json_output = json.dumps(output_data, indent=4, ensure_ascii=False)
            print("\n输出:\n", json_output)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print("Error processing content in response:", e)
            print("Full response:", response)
    else:
        print('Status code:', response.status_code)
        print('Error message:', response.message)

    #  原始
    # if response.status_code == HTTPStatus.OK:
    #     try:
    #         # 将字符串形式的字典转换为真正的字典
    #         response_content = ast.literal_eval(response.output.choices[0].message.content)
    #         output_data = {
    #             "original_text": user_input,
    #             **response_content  # 展开这个字典，直接并入output_data
    #         }
    #         json_output = json.dumps(output_data, indent=4, ensure_ascii=False)
    #         print("\n输出:\n", json_output)
    #     except (KeyError, IndexError, ValueError) as e:
    #         print("Error processing content in response:", e)
    #         print("Full response:", response)
    # else:
    #     print('Status code:', response.status_code)
    #     print('Error message:', response.message)

if __name__ == '__main__':
    while True:
        user_input = input("输入: ")
        if user_input.lower() == 'exit':
            break
        call_with_messages(user_input)

