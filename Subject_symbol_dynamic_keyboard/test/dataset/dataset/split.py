import json
import re

def extract_symbols(text, symbols):
    """从 text 中提取在 symbols 列表中的符号"""
    return [symbol for symbol in symbols if symbol in text]

def split_subitems_for_dict(data):
    """处理原始字典格式的数据"""
    result = {}
    successfully_processed = []  # 用于存储成功处理的条目
    for key, value in data.items():
        # 如果 id 中已经包含子题目标志，则跳过
        if re.search(r"\(\d+\)", key):
            result[key] = value
            continue

        # 确保 'question' 和 'answer' 都是字符串，如果是 None 则设为空字符串
        question = value.get("question") or ""
        answer = value.get("answer") or ""
        pdf_path = value.get("pdf_path") or ""
        
        # 确保 'question_symbols' 和 'answer_symbols' 为列表，如果是 None 则设为空列表
        question_symbols = value.get("question_symbols") or []
        answer_symbols = value.get("answer_symbols") or []

        # 提取题目条件和答案条件
        question_condition_match = re.match(r"^(.*?)(?=\(\d+\))", question)
        question_condition = question_condition_match.group(0).strip() if question_condition_match else ""
        
        answer_condition_match = re.match(r"^(.*?)(?=\(\d+\))", answer)
        answer_condition = answer_condition_match.group(0).strip() if answer_condition_match else ""
        
        # 分割子题目和答案
        sub_questions = re.split(r"(?=\(\d+\))", question)
        sub_answers = re.split(r"(?=\(\d+\))", answer)

        if len(sub_questions) > 1 and len(sub_answers) == len(sub_questions):
            successfully_processed.append(key)  # 记录成功处理的条目
            for i, (q_part, a_part) in enumerate(zip(sub_questions[1:], sub_answers[1:]), start=1):
                q_text = f"{question_condition} {q_part.strip()}"
                a_text = f"{answer_condition} {a_part.strip()}"
                q_symbols = extract_symbols(q_text, question_symbols)
                a_symbols = extract_symbols(a_text, answer_symbols)

                new_id = f"{key}.{i}"
                result[new_id] = {
                    "question": q_text,
                    "question_symbols": q_symbols,
                    "answer": a_text,
                    "answer_symbols": a_symbols,
                    "pdf_path": pdf_path
                }
        else:
            result[key] = value
    return result, successfully_processed

def split_subitems_for_list(data):
    """处理新列表格式的数据"""
    result = []
    successfully_processed = []  # 用于存储成功处理的条目
    for item in data:
        key = item.get("id", "")
        
        # 如果 id 中已经包含子题目标志，则跳过
        if re.search(r"\(\d+\)", key):
            result.append(item)
            continue

        # 确保 'question' 和 'answer' 都是字符串，如果是 None 则设为空字符串
        question = item.get("question") or ""
        answer = item.get("answer") or ""
        
        # 确保 'question_symbols' 和 'answer_symbols' 为列表，如果是 None 则设为空列表
        question_symbols = item.get("question_symbols") or []
        answer_symbols = item.get("answer_symbols") or []
        
        pdf_path = item.get("pdf_path", "")
        
        # 提取题目条件和答案条件
        question_condition_match = re.match(r"^(.*?)(?=\(\d+\))", question)
        question_condition = question_condition_match.group(0).strip() if question_condition_match else ""
        
        answer_condition_match = re.match(r"^(.*?)(?=\(\d+\))", answer)
        answer_condition = answer_condition_match.group(0).strip() if answer_condition_match else ""
        
        # 分割子题目和答案
        sub_questions = re.split(r"(?=\(\d+\))", question)
        sub_answers = re.split(r"(?=\(\d+\))", answer)

        if len(sub_questions) > 1 and len(sub_answers) == len(sub_questions):
            successfully_processed.append(key)  # 记录成功处理的条目
            for i, (q_part, a_part) in enumerate(zip(sub_questions[1:], sub_answers[1:]), start=1):
                q_text = f"{question_condition} {q_part.strip()}"
                a_text = f"{answer_condition} {a_part.strip()}"
                q_symbols = extract_symbols(q_text, question_symbols)
                a_symbols = extract_symbols(a_text, answer_symbols)

                result.append({
                    "id": f"{key}.{i}",
                    "question": q_text,
                    "question_symbols": q_symbols,
                    "answer": a_text,
                    "answer_symbols": a_symbols,
                    "pdf_path": pdf_path
                })
        else:
            result.append(item)
    return result, successfully_processed

# 加载数据并自动选择处理函数
file_path = "output2.json"
output_path = "output2_split.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 判断数据格式并调用对应的处理函数
if isinstance(data, dict):
    processed_data, successfully_processed = split_subitems_for_dict(data)
elif isinstance(data, list):
    processed_data, successfully_processed = split_subitems_for_list(data)
else:
    raise ValueError("数据格式不支持。")

# 保存处理后的数据
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

# 输出成功处理的条目列表
print("Successfully processed items:", successfully_processed)
