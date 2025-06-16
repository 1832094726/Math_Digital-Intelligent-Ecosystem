import json
import os
from datetime import datetime

# 加载作业数据
DATA_ROOT = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "data")
HOMEWORK_DATA_PATH = os.path.join(DATA_ROOT, "homework.json")

def load_homework_data():
    """加载作业数据"""
    if not os.path.exists(HOMEWORK_DATA_PATH):
        return []
    
    try:
        with open(HOMEWORK_DATA_PATH, 'r', encoding="utf8") as f:
            return json.load(f)
    except:
        return []

def save_homework_data(data):
    """保存作业数据"""
    with open(HOMEWORK_DATA_PATH, 'w', encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_homework_list(user_id):
    """获取作业列表
    
    Args:
        user_id: 用户ID
        
    Returns:
        作业列表
    """
    # 从JSON文件加载真实数据
    homeworks = load_homework_data()
    
    # 转换为前端所需格式
    result = []
    for hw in homeworks:
        result.append({
            "id": hw["id"],
            "title": hw["title"],
            "subject": hw["subject"],
            "grade": hw["grade"],
            "deadline": hw["deadline"].split()[0] if "deadline" in hw else "",
            "status": hw["status"],
            "progress": 0,  # 默认进度，实际应该从用户作业进度记录中获取
            "questions_count": len(hw["questions"]) if "questions" in hw else 0
        })
    
    return result

def get_homework_detail(homework_id):
    """获取作业详情
    
    Args:
        homework_id: 作业ID
        
    Returns:
        作业详情
    """
    # 从JSON文件加载真实数据
    homeworks = load_homework_data()
    
    # 查找指定ID的作业
    for hw in homeworks:
        if hw["id"] == homework_id:
            # 清除答案，避免前端直接显示答案
            result = hw.copy()
            if "questions" in result:
                for q in result["questions"]:
                    if "answer" in q:
                        q["answer"] = ""  # 清除答案
            return result
    
    return {"error": "作业不存在"}

def submit_homework(data):
    """提交作业
    
    Args:
        data: 提交的作业数据
        
    Returns:
        提交结果
    """
    homework_id = data.get("homework_id")
    user_id = data.get("user_id")
    answers = data.get("answers", [])
    
    if not homework_id or not user_id or not answers:
        return {"success": False, "message": "提交数据不完整"}
    
    # 从JSON文件加载真实数据
    homeworks = load_homework_data()
    
    # 查找指定ID的作业
    homework = None
    for hw in homeworks:
        if hw["id"] == homework_id:
            homework = hw
            break
    
    if not homework:
        return {"success": False, "message": "作业不存在"}
    
    # 计算得分
    total_score = 0
    max_score = 0
    correct_count = 0
    feedback = []
    
    for answer in answers:
        question_id = answer.get("question_id")
        user_answer = answer.get("answer", "")
        
        # 查找对应的题目
        for question in homework.get("questions", []):
            if question["id"] == question_id:
                max_score += question.get("score", 0)
                
                # 简单比较答案是否正确
                if user_answer == question.get("answer", ""):
                    total_score += question.get("score", 0)
                    correct_count += 1
                    feedback.append(f"题目 {question_id} 回答正确")
                else:
                    feedback.append(f"题目 {question_id} 回答错误，正确答案是：{question.get('answer', '')}")
                break
    
    # 计算百分比得分
    percentage_score = round((total_score / max_score * 100) if max_score > 0 else 0)
    
    # 生成反馈
    general_feedback = ""
    if percentage_score >= 90:
        general_feedback = "优秀！你对这些知识点掌握得非常好。"
    elif percentage_score >= 70:
        general_feedback = "良好。大部分题目都答对了，但还有提升空间。"
    elif percentage_score >= 60:
        general_feedback = "及格。建议复习错题，巩固知识点。"
    else:
        general_feedback = "需要更多练习。建议重新学习相关知识点。"
    
    return {
        "success": True,
        "message": "作业提交成功",
        "score": percentage_score,
        "correct_count": correct_count,
        "total_count": len(answers),
        "feedback": general_feedback,
        "detail_feedback": feedback
    }

def save_progress(data):
    """保存作业进度
    
    Args:
        data: 作业进度数据
        
    Returns:
        保存结果
    """
    homework_id = data.get("homework_id")
    user_id = data.get("user_id")
    answers = data.get("answers", [])
    
    if not homework_id or not user_id:
        return {"success": False, "message": "数据不完整"}
    
    # 这里应该保存到用户进度数据库中
    # 由于是演示，我们只返回成功信息
    
    return {
        "success": True,
        "message": "进度保存成功",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    } 