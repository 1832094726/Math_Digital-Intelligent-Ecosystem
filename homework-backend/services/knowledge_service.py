import json
import os
import re

# 加载知识点数据
DATA_ROOT = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "data")
KNOWLEDGE_DATA_PATH = os.path.join(DATA_ROOT, "knowledge.json")

def load_knowledge_data():
    """加载知识点数据"""
    if not os.path.exists(KNOWLEDGE_DATA_PATH):
        return []
    
    try:
        with open(KNOWLEDGE_DATA_PATH, 'r', encoding="utf8") as f:
            return json.load(f)
    except:
        return []

def extract_knowledge_points(question_text):
    """根据题目内容提取相关知识点
    
    Args:
        question_text: 题目内容文本
        
    Returns:
        相关知识点列表，每个知识点包含id、name等信息
    """
    # 加载知识点数据
    all_knowledge = load_knowledge_data()
    if not all_knowledge:
        return []
    
    # 匹配的知识点
    matched_points = []
    
    # 简单的关键词匹配方法
    # 1. 几何相关知识点
    if re.search(r'三角形|等腰|等边|直角|锐角|钝角', question_text):
        triangle_points = [k for k in all_knowledge if '三角形' in k.get('name', '')]
        matched_points.extend(triangle_points)
    
    if re.search(r'圆|圆周|直径|半径|弧|弦|扇形', question_text):
        circle_points = [k for k in all_knowledge if '圆' in k.get('name', '')]
        matched_points.extend(circle_points)
    
    if re.search(r'正方形|长方形|矩形|平行四边形|梯形|菱形', question_text):
        quadrilateral_points = [k for k in all_knowledge if any(shape in k.get('name', '') 
                                                              for shape in ['方形', '矩形', '平行四边形', '梯形', '菱形'])]
        matched_points.extend(quadrilateral_points)
    
    # 2. 代数相关知识点
    if re.search(r'方程|等式|不等式|解方程', question_text):
        equation_points = [k for k in all_knowledge if any(term in k.get('name', '') 
                                                        for term in ['方程', '等式', '不等式'])]
        matched_points.extend(equation_points)
    
    if re.search(r'一次|二次|多项式|因式分解', question_text):
        polynomial_points = [k for k in all_knowledge if any(term in k.get('name', '') 
                                                          for term in ['一次', '二次', '多项式', '因式分解'])]
        matched_points.extend(polynomial_points)
    
    # 3. 集合与统计相关知识点
    if re.search(r'集合|交集|并集|补集|元素', question_text):
        set_points = [k for k in all_knowledge if '集合' in k.get('name', '')]
        matched_points.extend(set_points)
    
    if re.search(r'统计|平均|中位数|众数|方差|标准差|概率', question_text):
        stat_points = [k for k in all_knowledge if any(term in k.get('name', '') 
                                                    for term in ['统计', '平均', '中位数', '众数', '方差', '标准差', '概率'])]
        matched_points.extend(stat_points)
    
    # 去重
    seen_ids = set()
    unique_points = []
    for point in matched_points:
        if point.get('id') not in seen_ids:
            seen_ids.add(point.get('id'))
            unique_points.append(point)
    
    # 如果没有匹配到知识点，返回空列表
    if not unique_points:
        return []
    
    return unique_points

def get_question_knowledge_points(question_id=None, question_text=None):
    """获取题目相关的知识点
    
    Args:
        question_id: 题目ID（可选）
        question_text: 题目内容（可选）
        
    Returns:
        相关知识点列表
    """
    # 如果提供了题目ID，尝试从作业数据中查找题目
    if question_id:
        homework_data_path = os.path.join(DATA_ROOT, "homework.json")
        try:
            with open(homework_data_path, 'r', encoding="utf8") as f:
                homeworks = json.load(f)
                for hw in homeworks:
                    for question in hw.get("questions", []):
                        if question.get("id") == question_id:
                            # 如果题目中已有知识点标注，直接返回
                            if "knowledge_points" in question and question["knowledge_points"]:
                                # 获取完整的知识点信息
                                knowledge_points = question["knowledge_points"]
                                all_knowledge = load_knowledge_data()
                                detailed_points = []
                                for kp_name in knowledge_points:
                                    for k in all_knowledge:
                                        if k.get('name') == kp_name:
                                            detailed_points.append(k)
                                            break
                                return detailed_points
                            # 否则根据题目内容提取
                            return extract_knowledge_points(question.get("content", ""))
        except:
            pass
    
    # 如果提供了题目内容，直接提取
    if question_text:
        return extract_knowledge_points(question_text)
    
    return [] 