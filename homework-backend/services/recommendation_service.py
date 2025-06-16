import json
import os
import re
import random

# 加载推荐数据
DATA_ROOT = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "data")
SYMBOLS_DATA_PATH = os.path.join(DATA_ROOT, "symbols.json")
KNOWLEDGE_DATA_PATH = os.path.join(DATA_ROOT, "knowledge.json")

def load_symbols_data():
    """加载符号数据"""
    if not os.path.exists(SYMBOLS_DATA_PATH):
        return []
    
    try:
        with open(SYMBOLS_DATA_PATH, 'r', encoding="utf8") as f:
            return json.load(f)
    except:
        return []

def load_knowledge_data():
    """加载知识点数据"""
    if not os.path.exists(KNOWLEDGE_DATA_PATH):
        return []
    
    try:
        with open(KNOWLEDGE_DATA_PATH, 'r', encoding="utf8") as f:
            return json.load(f)
    except:
        return []

def get_recommended_symbols(data):
    """获取推荐符号
    
    Args:
        data: 包含问题上下文的数据
        
    Returns:
        推荐的符号列表
    """
    question_text = data.get('question_text', '')
    user_id = data.get('user_id', 1)
    
    # 加载符号数据
    all_symbols = load_symbols_data()
    if not all_symbols:
        return {"symbols": [], "by_category": {}}
    
    # 基于问题文本的关键词匹配
    recommended_symbols = []
    
    # 几何相关内容
    if re.search(r'三角形|角度|圆|正方形|矩形|平行四边形|梯形|面积|周长', question_text):
        geometry_symbols = [s for s in all_symbols if s.get('category') == '几何符号']
        recommended_symbols.extend(geometry_symbols)
    
    # 代数相关内容
    if re.search(r'方程|函数|求解|计算|代数|多项式|因式分解', question_text):
        algebra_symbols = [s for s in all_symbols if s.get('category') in ['基本运算', '关系符号']]
        recommended_symbols.extend(algebra_symbols)
    
    # 集合相关内容
    if re.search(r'集合|元素|属于|包含|交集|并集', question_text):
        set_symbols = [s for s in all_symbols if s.get('category') == '集合符号']
        recommended_symbols.extend(set_symbols)
    
    # 如果没有匹配到任何符号，返回一些通用符号
    if not recommended_symbols:
        basic_symbols = [s for s in all_symbols if s.get('category') == '基本运算' or s.get('difficulty', 1) < 0.3]
        recommended_symbols = basic_symbols[:8]  # 取前8个基础符号
    
    # 去重
    seen_ids = set()
    unique_symbols = []
    for symbol in recommended_symbols:
        if symbol.get('id') not in seen_ids:
            seen_ids.add(symbol.get('id'))
            unique_symbols.append(symbol)
    
    # 按类别分组
    symbols_by_category = {}
    for symbol in unique_symbols:
        category = symbol.get("category", "其他")
        if category not in symbols_by_category:
            symbols_by_category[category] = []
        symbols_by_category[category].append(symbol)
    
    return {
        "symbols": unique_symbols[:15],  # 限制返回数量
        "by_category": symbols_by_category
    }

def get_recommended_knowledge(data):
    """获取推荐知识点
    
    Args:
        data: 包含问题上下文的数据
        
    Returns:
        推荐的知识点列表
    """
    question_text = data.get('question_text', '')
    user_id = data.get('user_id', 1)
    
    # 加载知识点数据
    all_knowledge = load_knowledge_data()
    if not all_knowledge:
        return {"knowledge_points": []}
    
    # 基于问题文本的关键词匹配
    recommended_points = []
    
    # 几何相关知识点
    if re.search(r'三角形', question_text):
        triangle_points = [k for k in all_knowledge if '三角形' in k.get('name', '')]
        recommended_points.extend(triangle_points)
    
    # 圆相关知识点
    if re.search(r'圆', question_text):
        circle_points = [k for k in all_knowledge if '圆' in k.get('name', '')]
        recommended_points.extend(circle_points)
    
    # 方程相关知识点
    if re.search(r'方程', question_text):
        equation_points = [k for k in all_knowledge if '方程' in k.get('name', '')]
        recommended_points.extend(equation_points)
    
    # 集合相关知识点
    if re.search(r'集合', question_text):
        set_points = [k for k in all_knowledge if '集合' in k.get('name', '')]
        recommended_points.extend(set_points)
    
    # 统计相关知识点
    if re.search(r'统计|平均|概率', question_text):
        stat_points = [k for k in all_knowledge if '统计' in k.get('name', '') or '概率' in k.get('name', '')]
        recommended_points.extend(stat_points)
    
    # 如果没有匹配到任何知识点，返回一些基础知识点
    if not recommended_points:
        basic_points = [k for k in all_knowledge if k.get('difficulty', 1) < 0.6]
        recommended_points = basic_points[:2]  # 取前2个基础知识点
    
    # 去重
    seen_ids = set()
    unique_points = []
    for point in recommended_points:
        if point.get('id') not in seen_ids:
            seen_ids.add(point.get('id'))
            unique_points.append(point)
    
    return {
        "knowledge_points": unique_points[:5]  # 限制返回数量
    }

def get_recommended_exercises(data):
    """获取推荐练习
    
    Args:
        data: 包含问题上下文和用户信息的数据
        
    Returns:
        推荐的练习列表
    """
    question_text = data.get('question_text', '')
    user_id = data.get('user_id', 1)
    knowledge_points = data.get('knowledge_points', [])
    
    # 加载作业数据，从中提取练习题
    homework_data_path = os.path.join(DATA_ROOT, "homework.json")
    all_exercises = []
    
    try:
        with open(homework_data_path, 'r', encoding="utf8") as f:
            homeworks = json.load(f)
            # 从所有作业中提取题目
            for hw in homeworks:
                for question in hw.get("questions", []):
                    exercise = {
                        "id": question.get("id"),
                        "title": question.get("content", "")[:20] + "...",  # 截取前20个字符作为标题
                        "content": question.get("content", ""),
                        "difficulty": 0.5,  # 默认难度
                        "knowledge_points": question.get("knowledge_points", []),
                        "homework_id": hw.get("id"),
                        "type": question.get("type", "")
                    }
                    all_exercises.append(exercise)
    except:
        all_exercises = []
    
    # 如果没有练习题，返回空列表
    if not all_exercises:
        return {"exercises": []}
    
    # 基于知识点和问题文本推荐练习
    recommended_exercises = []
    
    # 根据知识点匹配
    if knowledge_points:
        for kp in knowledge_points:
            for exercise in all_exercises:
                if kp in exercise.get("knowledge_points", []) and exercise not in recommended_exercises:
                    recommended_exercises.append(exercise)
    
    # 如果基于知识点没有找到足够的练习，基于问题文本匹配
    if len(recommended_exercises) < 2:
        # 几何相关练习
        if re.search(r'三角形|圆|几何', question_text):
            for exercise in all_exercises:
                if any(kp in ["三角形", "圆", "几何"] for kp in exercise.get("knowledge_points", [])) and exercise not in recommended_exercises:
                    recommended_exercises.append(exercise)
        
        # 代数相关练习
        if re.search(r'方程|函数|代数', question_text):
            for exercise in all_exercises:
                if any(kp in ["方程", "函数", "代数"] for kp in exercise.get("knowledge_points", [])) and exercise not in recommended_exercises:
                    recommended_exercises.append(exercise)
    
    # 如果仍然没有找到足够的练习，随机选择一些
    if len(recommended_exercises) < 2:
        remaining = [e for e in all_exercises if e not in recommended_exercises]
        if remaining:
            recommended_exercises.extend(random.sample(remaining, min(2, len(remaining))))
    
    # 按难度排序
    recommended_exercises.sort(key=lambda x: x.get("difficulty", 0.5))
    
    return {
        "exercises": recommended_exercises[:5]  # 限制返回数量
    } 