import json
import os
from datetime import datetime

# 用户数据路径
DATA_ROOT = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "data")
USERS_DATA_PATH = os.path.join(DATA_ROOT, "users.json")
USER_MODELS_PATH = os.path.join(DATA_ROOT, "user_models.json")

def load_users_data():
    """加载用户数据"""
    if not os.path.exists(USERS_DATA_PATH):
        # 创建默认用户数据
        default_users = [
            {
                "id": 1,
                "username": "student1",
                "name": "张三",
                "grade": "初中二年级",
                "school": "示例中学",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
        with open(USERS_DATA_PATH, 'w', encoding="utf8") as f:
            json.dump(default_users, f, ensure_ascii=False, indent=2)
        return default_users
    
    try:
        with open(USERS_DATA_PATH, 'r', encoding="utf8") as f:
            return json.load(f)
    except:
        return []

def load_user_models():
    """加载用户模型数据"""
    if not os.path.exists(USER_MODELS_PATH):
        # 创建默认用户模型数据
        default_models = [
            {
                "user_id": 1,
                "knowledge_mastery": {
                    "三角形": 0.7,
                    "等腰三角形": 0.5,
                    "直角三角形": 0.6,
                    "方程": 0.8,
                    "一次方程": 0.9,
                    "二次方程": 0.4
                },
                "learning_style": "visual",
                "difficulty_preference": "medium",
                "symbol_usage_frequency": {
                    "+": 120,
                    "-": 100,
                    "×": 80,
                    "÷": 60,
                    "=": 150,
                    "△": 30,
                    "∠": 25
                },
                "recent_activities": [
                    {
                        "type": "homework",
                        "id": 3,
                        "timestamp": "2023-06-20 15:30:00",
                        "score": 85
                    },
                    {
                        "type": "practice",
                        "id": 102,
                        "timestamp": "2023-06-19 16:45:00",
                        "completed": True
                    }
                ],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
        with open(USER_MODELS_PATH, 'w', encoding="utf8") as f:
            json.dump(default_models, f, ensure_ascii=False, indent=2)
        return default_models
    
    try:
        with open(USER_MODELS_PATH, 'r', encoding="utf8") as f:
            return json.load(f)
    except:
        return []

def save_user_model(model):
    """保存用户模型数据"""
    models = load_user_models()
    
    # 更新或添加用户模型
    updated = False
    for i, m in enumerate(models):
        if m["user_id"] == model["user_id"]:
            models[i] = model
            updated = True
            break
    
    if not updated:
        models.append(model)
    
    with open(USER_MODELS_PATH, 'w', encoding="utf8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)

def get_user_info(user_id):
    """获取用户信息
    
    Args:
        user_id: 用户ID
        
    Returns:
        用户信息
    """
    users = load_users_data()
    user_models = load_user_models()
    
    # 查找用户基本信息
    user_info = None
    for user in users:
        if user["id"] == user_id:
            user_info = user
            break
    
    if not user_info:
        return {"error": "用户不存在"}
    
    # 查找用户模型
    user_model = None
    for model in user_models:
        if model["user_id"] == user_id:
            user_model = model
            break
    
    # 组合用户信息和模型
    result = {
        "basic_info": user_info,
        "model": user_model
    }
    
    return result

def update_user_model(data):
    """更新用户模型
    
    Args:
        data: 用户模型更新数据
        
    Returns:
        更新结果
    """
    user_id = data.get('user_id')
    if not user_id:
        return {"error": "缺少用户ID"}
    
    # 加载当前用户模型
    user_models = load_user_models()
    current_model = None
    for model in user_models:
        if model["user_id"] == user_id:
            current_model = model
            break
    
    if not current_model:
        return {"error": "用户模型不存在"}
    
    # 更新知识点掌握度
    if 'knowledge_updates' in data:
        for k, v in data['knowledge_updates'].items():
            if k in current_model["knowledge_mastery"]:
                # 根据新的学习数据调整掌握度
                current_model["knowledge_mastery"][k] = min(1.0, current_model["knowledge_mastery"][k] + v)
            else:
                # 新知识点
                current_model["knowledge_mastery"][k] = v
    
    # 更新符号使用频率
    if 'symbol_usage' in data:
        for symbol, count in data['symbol_usage'].items():
            if symbol in current_model["symbol_usage_frequency"]:
                current_model["symbol_usage_frequency"][symbol] += count
            else:
                current_model["symbol_usage_frequency"][symbol] = count
    
    # 添加新的活动记录
    if 'activity' in data:
        current_model["recent_activities"].insert(0, data['activity'])
        # 保留最近10条活动记录
        current_model["recent_activities"] = current_model["recent_activities"][:10]
    
    # 更新时间戳
    current_model["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存更新后的模型
    save_user_model(current_model)
    
    return {
        "success": True,
        "message": "用户模型更新成功",
        "model": current_model
    } 