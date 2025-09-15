#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学题目转方程训练数据集生成器
用于生成符合硅基流动微调要求的JSONL格式数据集
"""

import json
import re
from typing import List, Dict, Any

class MathEquationDatasetGenerator:
    def __init__(self):
        self.system_prompt = "你是一个专业的数学题目解析助手，能够将自然语言描述的数学应用题转换为标准的数学方程式。请根据题目描述，提取关键信息并构建相应的数学方程。"
        
        # 题目类型到方程模板的映射
        self.equation_templates = {
            "加法应用题": {
                "pattern": r"(.+)有(\d+).+(.+)有(\d+).+一共|总共",
                "template": "设总数为x，则方程为：\nx = {num1} + {num2}\n因此方程式为：x = {num1} + {num2}"
            },
            "乘法应用题": {
                "pattern": r"(.+)(\d+).+每.+(\d+).+一共|总共",
                "template": "设总数为x，则方程为：\nx = {num1} × {num2}\n因此方程式为：x = {num1} × {num2}"
            },
            "面积计算": {
                "pattern": r"长方形.+长.+(\d+).+宽.+(\d+).+面积",
                "template": "根据长方形面积公式，面积等于长乘以宽：\n设面积为S，则方程为：\nS = {length} × {width}\n因此方程式为：S = {length} × {width}"
            },
            "梯形面积": {
                "pattern": r"梯形.+上底.+(\d+).+下底.+(\d+).+高.+(\d+).+面积",
                "template": "根据梯形面积公式，面积等于上下底之和乘以高再除以2：\n设面积为S，则方程为：\nS = ({upper} + {lower}) × {height} ÷ 2\n因此方程式为：S = ({upper} + {lower}) × {height} ÷ 2"
            },
            "一元一次方程": {
                "pattern": r"方程.+(\w+).+([+\-*/=]).+(\d+)",
                "template": "这是一个一元一次方程：\n{equation}\n因此方程式为：{equation}"
            },
            "一元二次方程": {
                "pattern": r"方程.+(\w+)².+([+\-]).+(\d+)\w+([+\-]).+(\d+).+=.+0",
                "template": "这是一个一元二次方程，已经是标准形式：\n{equation}\n因此方程式为：{equation}"
            }
        }
    
    def load_homework_data(self, file_path: str) -> List[Dict]:
        """加载作业数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载数据失败: {e}")
            return []
    
    def extract_numbers(self, text: str) -> List[str]:
        """提取文本中的数字"""
        return re.findall(r'\d+(?:\.\d+)?', text)
    
    def classify_question_type(self, question_text: str) -> str:
        """根据题目内容分类"""
        if re.search(r'一共|总共', question_text) and re.search(r'有.*\d+', question_text):
            if re.search(r'每.*\d+', question_text):
                return "乘法应用题"
            else:
                return "加法应用题"
        elif re.search(r'长方形.*面积', question_text):
            return "面积计算"
        elif re.search(r'梯形.*面积', question_text):
            return "梯形面积"
        elif re.search(r'方程.*x²', question_text):
            return "一元二次方程"
        elif re.search(r'方程.*x', question_text):
            return "一元一次方程"
        else:
            return "其他"
    
    def generate_equation_response(self, question_text: str, question_type: str) -> str:
        """根据题目类型生成方程解析"""
        numbers = self.extract_numbers(question_text)
        
        if question_type == "加法应用题" and len(numbers) >= 2:
            return f"根据题目描述，我需要将两个数相加：\n设总数为x，则方程为：\nx = {numbers[0]} + {numbers[1]}\n因此方程式为：x = {numbers[0]} + {numbers[1]}"
        
        elif question_type == "乘法应用题" and len(numbers) >= 2:
            return f"根据题目描述，总数等于数量乘以单价：\n设总数为x，则方程为：\nx = {numbers[0]} × {numbers[1]}\n因此方程式为：x = {numbers[0]} × {numbers[1]}"
        
        elif question_type == "面积计算" and len(numbers) >= 2:
            return f"根据长方形面积公式，面积等于长乘以宽：\n设面积为S，则方程为：\nS = {numbers[0]} × {numbers[1]}\n因此方程式为：S = {numbers[0]} × {numbers[1]}"
        
        elif question_type == "梯形面积" and len(numbers) >= 3:
            return f"根据梯形面积公式，面积等于上下底之和乘以高再除以2：\n设面积为S，则方程为：\nS = ({numbers[0]} + {numbers[1]}) × {numbers[2]} ÷ 2\n因此方程式为：S = ({numbers[0]} + {numbers[1]}) × {numbers[2]} ÷ 2"
        
        elif question_type in ["一元一次方程", "一元二次方程"]:
            # 提取方程部分
            equation_match = re.search(r'[x²\w\s+\-*/=0-9]+=[x²\w\s+\-*/=0-9]+', question_text)
            if equation_match:
                equation = equation_match.group().strip()
                return f"这是一个{question_type}，已经是标准形式：\n{equation}\n因此方程式为：{equation}"
        
        # 默认响应
        return f"根据题目描述，建立相应的数学方程：\n请根据具体情况分析题目中的数量关系。"
    
    def create_training_sample(self, question_text: str, answer: str = None) -> Dict:
        """创建单个训练样本"""
        question_type = self.classify_question_type(question_text)
        response = self.generate_equation_response(question_text, question_type)
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": question_text
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        }
    
    def generate_from_homework_data(self, homework_file: str, output_file: str, max_samples: int = 1000):
        """从作业数据生成训练集"""
        homework_data = self.load_homework_data(homework_file)
        training_samples = []
        
        for homework in homework_data:
            if len(training_samples) >= max_samples:
                break
                
            questions = homework.get('questions', [])
            for question in questions:
                if len(training_samples) >= max_samples:
                    break
                    
                question_text = question.get('content', '')
                answer = question.get('answer', '')
                
                if question_text:
                    sample = self.create_training_sample(question_text, answer)
                    training_samples.append(sample)
        
        # 保存为JSONL格式
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"生成了 {len(training_samples)} 个训练样本，保存到 {output_file}")
        return training_samples
    
    def add_manual_samples(self, output_file: str):
        """添加手工设计的高质量样本"""
        manual_samples = [
            {
                "question": "一个梯形的上、下底之和是36dm，是高的4倍，这个梯形的面积=多少dm²",
                "response": "根据题目描述，我需要分步建立方程：\n设高为h，则上下底之和为4h\n已知上下底之和为36dm，所以：4h = 36\n解得：h = 9dm\n设面积为S，根据梯形面积公式：\nS = (上底 + 下底) × 高 ÷ 2 = 36 × 9 ÷ 2\n因此方程式为：S = 36 × 9 ÷ 2"
            },
            {
                "question": "小明从家到学校，前一半路程用时20分钟，后一半路程用时30分钟，全程平均速度是多少？",
                "response": "根据平均速度公式，平均速度等于总路程除以总时间：\n设全程为2s，总时间为20 + 30 = 50分钟\n设平均速度为v，则方程为：\nv = 2s / 50\n因此方程式为：v = 2s / 50 = s / 25"
            },
            {
                "question": "甲乙两人同时从A地出发到B地，甲的速度是60km/h，乙的速度是80km/h，乙比甲早到30分钟，求AB两地的距离。",
                "response": "根据题目描述，利用时间差建立方程：\n设AB距离为s公里\n甲用时：s/60小时，乙用时：s/80小时\n乙比甲早到30分钟(0.5小时)，则方程为：\ns/60 - s/80 = 0.5\n因此方程式为：s/60 - s/80 = 0.5"
            }
        ]
        
        # 追加到文件
        with open(output_file, 'a', encoding='utf-8') as f:
            for sample_data in manual_samples:
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": sample_data["question"]
                        },
                        {
                            "role": "assistant", 
                            "content": sample_data["response"]
                        }
                    ]
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"添加了 {len(manual_samples)} 个手工样本")

def main():
    """主函数"""
    generator = MathEquationDatasetGenerator()
    
    # 从作业数据生成训练集
    homework_file = "data/homework.json"
    output_file = "data/math_equation_fine_tuning_dataset.jsonl"
    
    print("开始生成数学题目转方程训练数据集...")
    
    # 生成基础样本
    generator.generate_from_homework_data(homework_file, output_file, max_samples=800)
    
    # 添加高质量手工样本
    generator.add_manual_samples(output_file)
    
    print("数据集生成完成！")
    print(f"输出文件: {output_file}")
    print("数据集格式符合硅基流动微调要求：")
    print("- 每行一个独立的JSON对象")
    print("- 包含messages数组")
    print("- system、user、assistant角色完整")
    print("- 数据量控制在5000行以内")

if __name__ == "__main__":
    main()
