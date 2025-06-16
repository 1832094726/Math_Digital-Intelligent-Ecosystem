import re
import random
from .data_utils import extract_math_keywords, calculate_similarity

def analyze_question(question_text):
    """分析数学题目，提取关键信息
    
    Args:
        question_text: 题目文本
        
    Returns:
        分析结果，包含题目类型、难度估计、关键词等
    """
    # 提取关键词
    keywords = extract_math_keywords(question_text)
    
    # 确定题目主要类型
    question_type = "未知"
    difficulty = 0.5  # 默认中等难度
    
    # 根据关键词判断题目类型
    if keywords['geometry']:
        question_type = "几何"
        # 根据关键词数量和复杂度估计难度
        if len(keywords['geometry']) > 3 or '体积' in keywords['geometry']:
            difficulty = 0.7
        elif '等边' in keywords['geometry'] or '等腰' in keywords['geometry']:
            difficulty = 0.6
        else:
            difficulty = 0.5
    elif keywords['algebra']:
        question_type = "代数"
        # 根据关键词判断难度
        if '三次' in keywords['algebra'] or '无理式' in keywords['algebra']:
            difficulty = 0.8
        elif '二次' in keywords['algebra'] or '因式' in keywords['algebra']:
            difficulty = 0.6
        else:
            difficulty = 0.5
    elif keywords['set_theory']:
        question_type = "集合论"
        difficulty = 0.6
    elif keywords['statistics']:
        question_type = "统计"
        if '标准差' in keywords['statistics'] or '分布' in keywords['statistics']:
            difficulty = 0.7
        else:
            difficulty = 0.5
    
    # 检测题目是否包含计算要求
    calculation_required = bool(re.search(r'计算|求|算出|多少|值是', question_text))
    
    # 检测题目是否要求证明
    proof_required = bool(re.search(r'证明|证实|论证|推导|推理', question_text))
    
    # 如果是证明题，难度通常更高
    if proof_required:
        difficulty = min(1.0, difficulty + 0.2)
    
    return {
        "question_type": question_type,
        "difficulty": difficulty,
        "keywords": keywords,
        "calculation_required": calculation_required,
        "proof_required": proof_required
    }

def recommend_learning_resources(question_analysis, user_model=None):
    """根据题目分析和用户模型推荐学习资源
    
    Args:
        question_analysis: 题目分析结果
        user_model: 用户模型数据
        
    Returns:
        推荐的学习资源列表
    """
    resources = []
    
    # 根据题目类型推荐基础资源
    question_type = question_analysis["question_type"]
    
    if question_type == "几何":
        resources.append({
            "title": "几何基础概念",
            "type": "article",
            "difficulty": 0.3,
            "url": "/resources/geometry/basics"
        })
        
        # 根据关键词推荐特定资源
        keywords = question_analysis["keywords"]["geometry"]
        if "三角形" in keywords:
            resources.append({
                "title": "三角形性质详解",
                "type": "video",
                "difficulty": 0.5,
                "url": "/resources/geometry/triangle"
            })
        if "圆" in keywords:
            resources.append({
                "title": "圆的性质与应用",
                "type": "interactive",
                "difficulty": 0.5,
                "url": "/resources/geometry/circle"
            })
            
    elif question_type == "代数":
        resources.append({
            "title": "代数基础",
            "type": "article",
            "difficulty": 0.3,
            "url": "/resources/algebra/basics"
        })
        
        keywords = question_analysis["keywords"]["algebra"]
        if any(k in keywords for k in ["二次", "一元二次"]):
            resources.append({
                "title": "一元二次方程解法",
                "type": "video",
                "difficulty": 0.5,
                "url": "/resources/algebra/quadratic_equations"
            })
        if "函数" in keywords:
            resources.append({
                "title": "函数图像与性质",
                "type": "interactive",
                "difficulty": 0.6,
                "url": "/resources/algebra/functions"
            })
    
    elif question_type == "集合论":
        resources.append({
            "title": "集合论基础",
            "type": "article",
            "difficulty": 0.4,
            "url": "/resources/set_theory/basics"
        })
        
    elif question_type == "统计":
        resources.append({
            "title": "统计学基础",
            "type": "article",
            "difficulty": 0.4,
            "url": "/resources/statistics/basics"
        })
        
        keywords = question_analysis["keywords"]["statistics"]
        if "概率" in keywords:
            resources.append({
                "title": "概率论入门",
                "type": "video",
                "difficulty": 0.5,
                "url": "/resources/statistics/probability"
            })
    
    # 如果有用户模型，根据用户偏好调整推荐
    if user_model:
        # 根据用户学习风格推荐适合的资源类型
        learning_style = user_model.get("learning_style", "")
        
        if learning_style == "visual":
            # 增加视频和图像资源的权重
            video_resources = [r for r in resources if r["type"] in ["video", "interactive"]]
            if not video_resources:
                # 添加一个通用视频资源
                resources.append({
                    "title": f"{question_type}知识点视频讲解",
                    "type": "video",
                    "difficulty": question_analysis["difficulty"],
                    "url": f"/resources/{question_type.lower()}/video_overview"
                })
        
        elif learning_style == "reading":
            # 增加文章资源的权重
            article_resources = [r for r in resources if r["type"] == "article"]
            if not article_resources:
                # 添加一个通用文章资源
                resources.append({
                    "title": f"{question_type}知识点详解",
                    "type": "article",
                    "difficulty": question_analysis["difficulty"],
                    "url": f"/resources/{question_type.lower()}/detailed_explanation"
                })
        
        # 根据用户难度偏好调整资源
        difficulty_preference = user_model.get("difficulty_preference", "medium")
        
        if difficulty_preference == "easy" and question_analysis["difficulty"] > 0.5:
            # 添加更基础的资源
            resources.append({
                "title": f"{question_type}入门指南",
                "type": "tutorial",
                "difficulty": 0.3,
                "url": f"/resources/{question_type.lower()}/beginner_guide"
            })
        
        elif difficulty_preference == "hard" and question_analysis["difficulty"] < 0.7:
            # 添加更有挑战性的资源
            resources.append({
                "title": f"{question_type}高级技巧",
                "type": "challenge",
                "difficulty": 0.8,
                "url": f"/resources/{question_type.lower()}/advanced_techniques"
            })
    
    # 确保至少有一个推荐资源
    if not resources:
        resources.append({
            "title": "数学学习资源",
            "type": "general",
            "difficulty": 0.5,
            "url": "/resources/general/math"
        })
    
    return resources

def generate_hint(question_text, user_model=None):
    """根据题目生成解题提示
    
    Args:
        question_text: 题目文本
        user_model: 用户模型数据
        
    Returns:
        解题提示
    """
    # 分析题目
    analysis = analyze_question(question_text)
    
    # 基于题目类型生成提示
    hints = []
    
    if analysis["question_type"] == "几何":
        geometry_keywords = analysis["keywords"]["geometry"]
        
        if "三角形" in geometry_keywords:
            if "等边" in geometry_keywords:
                hints.append("等边三角形的三条边相等，三个角也相等，均为60°")
            elif "等腰" in geometry_keywords:
                hints.append("等腰三角形有两条边相等，底边上的高平分底边")
            elif "直角" in geometry_keywords:
                hints.append("可以尝试使用勾股定理：a²+b²=c²")
            else:
                hints.append("考虑三角形的基本性质：三角形内角和为180°")
                
        if "圆" in geometry_keywords:
            hints.append("圆的面积公式：S=πr²，周长公式：C=2πr")
            
        if "平行" in geometry_keywords or "垂直" in geometry_keywords:
            hints.append("考虑平行线或垂直线的性质，可能涉及到相似三角形")
            
    elif analysis["question_type"] == "代数":
        algebra_keywords = analysis["keywords"]["algebra"]
        
        if "方程" in algebra_keywords:
            if "一次" in algebra_keywords:
                hints.append("解一次方程：将未知数项移到等式一边，常数项移到另一边")
            elif "二次" in algebra_keywords:
                hints.append("二次方程可以使用公式法求解：x = (-b ± √(b² - 4ac)) / 2a")
                
        if "函数" in algebra_keywords:
            hints.append("分析函数的定义域、值域和单调性")
            
    elif analysis["question_type"] == "集合论":
        hints.append("使用文氏图可以帮助理解集合间的关系")
        hints.append("注意交集、并集、补集的定义和性质")
        
    elif analysis["question_type"] == "统计":
        statistics_keywords = analysis["keywords"]["statistics"]
        
        if "平均值" in statistics_keywords:
            hints.append("平均值 = 总和 / 数量")
            
        if "概率" in statistics_keywords:
            hints.append("概率 = 特定事件的数量 / 所有可能事件的数量")
    
    # 如果是证明题，添加证明相关提示
    if analysis["proof_required"]:
        hints.append("证明题可以考虑直接证明、反证法或数学归纳法")
        
    # 如果是计算题，添加计算相关提示
    if analysis["calculation_required"]:
        hints.append("注意计算过程中的单位换算和数值精度")
    
    # 根据用户模型调整提示
    if user_model:
        knowledge_mastery = user_model.get("knowledge_mastery", {})
        
        # 找出用户掌握度较低的相关知识点
        weak_points = []
        for category in analysis["keywords"]:
            for keyword in analysis["keywords"][category]:
                if keyword in knowledge_mastery and knowledge_mastery[keyword] < 0.5:
                    weak_points.append(keyword)
        
        # 为薄弱知识点提供额外提示
        if weak_points:
            hints.append(f"这道题涉及到你可能不太熟悉的概念：{', '.join(weak_points)}，建议先复习这些概念")
    
    # 如果没有生成任何提示，提供通用提示
    if not hints:
        hints.append("仔细分析题目条件，明确题目要求")
        hints.append("尝试画图或列出方程来解决问题")
    
    # 随机选择1-3条提示
    if len(hints) > 3:
        return random.sample(hints, 3)
    return hints

def predict_user_performance(question_analysis, user_model):
    """预测用户在特定题目上的表现
    
    Args:
        question_analysis: 题目分析结果
        user_model: 用户模型数据
        
    Returns:
        预测结果，包括成功概率、预计用时等
    """
    if not user_model:
        # 没有用户模型，返回默认预测
        return {
            "success_probability": 0.5,
            "estimated_time": 300,  # 秒
            "confidence": 0.3
        }
    
    # 获取题目相关知识点
    all_keywords = []
    for category in question_analysis["keywords"]:
        all_keywords.extend(question_analysis["keywords"][category])
    
    # 计算用户对相关知识点的平均掌握度
    knowledge_mastery = user_model.get("knowledge_mastery", {})
    total_mastery = 0
    count = 0
    
    for keyword in all_keywords:
        if keyword in knowledge_mastery:
            total_mastery += knowledge_mastery[keyword]
            count += 1
    
    # 如果没有找到相关知识点的掌握度，使用默认值
    avg_mastery = total_mastery / count if count > 0 else 0.5
    
    # 计算题目难度与用户掌握度之间的差距
    difficulty_gap = question_analysis["difficulty"] - avg_mastery
    
    # 预测成功概率
    # 掌握度高于难度，成功概率高；掌握度低于难度，成功概率低
    success_probability = max(0.1, min(0.95, 0.8 - difficulty_gap))
    
    # 预测完成时间（秒）
    # 基础时间 + 难度因素 + 掌握度因素
    base_time = 180  # 3分钟基础时间
    difficulty_factor = question_analysis["difficulty"] * 300  # 难度越大，时间越长
    mastery_factor = (1 - avg_mastery) * 200  # 掌握度越低，时间越长
    
    estimated_time = base_time + difficulty_factor + mastery_factor
    
    # 如果是证明题，时间通常更长
    if question_analysis["proof_required"]:
        estimated_time *= 1.5
    
    # 预测的置信度
    # 如果有足够的相关知识点掌握度数据，置信度高
    confidence = min(0.9, 0.3 + count * 0.1)
    
    return {
        "success_probability": round(success_probability, 2),
        "estimated_time": round(estimated_time),
        "confidence": round(confidence, 2)
    } 