import json
import os
import re
import math
from datetime import datetime

def ensure_directory_exists(directory_path):
    """确保目录存在，如不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def ensure_file_exists(file_path, default_content=None):
    """确保文件存在，如不存在则创建并写入默认内容"""
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)
    
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            if default_content is not None:
                if isinstance(default_content, (dict, list)):
                    json.dump(default_content, f, ensure_ascii=False, indent=2)
                else:
                    f.write(str(default_content))
            else:
                # 默认创建空列表
                json.dump([], f, ensure_ascii=False, indent=2)

def load_json_data(file_path, default=None):
    """加载JSON数据，如文件不存在或出错则返回默认值"""
    if not os.path.exists(file_path):
        return default if default is not None else []
    
    try:
        with open(file_path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON数据错误: {e}")
        return default if default is not None else []

def save_json_data(file_path, data):
    """保存JSON数据到文件"""
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)
    
    try:
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存JSON数据错误: {e}")
        return False

def extract_math_keywords(text):
    """从文本中提取数学关键词"""
    # 数学领域关键词
    geometry_keywords = ['三角形', '圆', '正方形', '矩形', '平行', '垂直', '角度', 
                        '面积', '体积', '周长', '直角', '等腰', '等边', '四边形']
    
    algebra_keywords = ['方程', '不等式', '函数', '多项式', '因式', '系数', '一次',
                       '二次', '三次', '解', '根', '分式', '有理式', '无理式']
    
    set_theory_keywords = ['集合', '元素', '子集', '交集', '并集', '补集', '差集',
                          '全集', '空集', '属于', '包含']
    
    statistics_keywords = ['平均值', '中位数', '众数', '方差', '标准差', '概率',
                          '分布', '抽样', '频率', '百分比']
    
    # 提取关键词
    found_keywords = {
        'geometry': [],
        'algebra': [],
        'set_theory': [],
        'statistics': []
    }
    
    for keyword in geometry_keywords:
        if keyword in text:
            found_keywords['geometry'].append(keyword)
    
    for keyword in algebra_keywords:
        if keyword in text:
            found_keywords['algebra'].append(keyword)
    
    for keyword in set_theory_keywords:
        if keyword in text:
            found_keywords['set_theory'].append(keyword)
    
    for keyword in statistics_keywords:
        if keyword in text:
            found_keywords['statistics'].append(keyword)
    
    return found_keywords

def calculate_similarity(text1, text2):
    """计算两段文本的相似度（简单实现）"""
    # 将文本分词（简单按字符分）
    chars1 = set(text1)
    chars2 = set(text2)
    
    # 计算交集和并集
    intersection = chars1.intersection(chars2)
    union = chars1.union(chars2)
    
    # 计算Jaccard相似度
    if len(union) == 0:
        return 0
    
    similarity = len(intersection) / len(union)
    return similarity

def calculate_difficulty(question_text, user_knowledge):
    """根据问题文本和用户知识模型计算题目难度"""
    # 提取问题中的关键词
    keywords = extract_math_keywords(question_text)
    all_keywords = []
    for category in keywords.values():
        all_keywords.extend(category)
    
    if not all_keywords:
        return 0.5  # 默认中等难度
    
    # 计算用户对这些关键词的平均掌握度
    total_mastery = 0
    count = 0
    
    for keyword in all_keywords:
        if keyword in user_knowledge:
            total_mastery += user_knowledge[keyword]
            count += 1
    
    # 如果用户对所有关键词都没有掌握度记录，返回默认难度
    if count == 0:
        return 0.5
    
    avg_mastery = total_mastery / count
    
    # 难度与掌握度成反比：掌握度高，难度低
    difficulty = 1 - avg_mastery
    
    return difficulty

def analyze_answer_pattern(answers, correct_answers):
    """分析答题模式，找出错误类型和模式"""
    if not answers or not correct_answers:
        return {
            "accuracy": 0,
            "error_types": [],
            "suggestions": ["提供更多的答题数据以进行分析"]
        }
    
    # 计算正确率
    correct_count = sum(1 for a, c in zip(answers, correct_answers) if a == c)
    accuracy = correct_count / len(answers) if answers else 0
    
    # 分析错误类型
    error_types = []
    suggestions = []
    
    # 示例：检测计算错误
    calculation_errors = 0
    concept_errors = 0
    
    for i, (answer, correct) in enumerate(zip(answers, correct_answers)):
        try:
            # 尝试将答案转换为数字进行比较
            ans_num = float(answer) if answer else 0
            correct_num = float(correct) if correct else 0
            
            # 计算误差
            error = abs(ans_num - correct_num)
            
            if 0 < error < 0.1 * abs(correct_num):
                # 小计算错误
                calculation_errors += 1
            elif error > 0:
                # 概念性错误
                concept_errors += 1
        except:
            # 非数值比较，可能是概念性错误
            if answer != correct:
                concept_errors += 1
    
    if calculation_errors > len(answers) * 0.3:
        error_types.append("计算错误")
        suggestions.append("注意计算过程，检查计算步骤")
    
    if concept_errors > len(answers) * 0.3:
        error_types.append("概念理解错误")
        suggestions.append("复习相关概念和解题方法")
    
    if accuracy < 0.6:
        suggestions.append("需要更多练习来提高掌握度")
    elif accuracy > 0.9:
        suggestions.append("表现优秀，可以尝试更有挑战性的题目")
    
    return {
        "accuracy": accuracy,
        "error_types": error_types,
        "suggestions": suggestions
    }

def generate_timestamp():
    """生成当前时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") 