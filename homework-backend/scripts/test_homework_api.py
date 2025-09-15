# -*- coding: utf-8 -*-
"""
测试作业管理API
"""
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:5000"

def test_homework_api():
    """测试作业管理API"""
    print("开始作业管理API测试...\n")
    
    # 1. 首先登录获取token（使用teacher用户）
    print("=== 1. 教师登录 ===")
    login_data = {
        "username": "teacher001",
        "password": "Teacher123!"  # 使用教师的密码
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        print(f"登录状态码: {response.status_code}")
        
        if response.status_code == 200:
            login_result = response.json()
            teacher_token = login_result["data"]["access_token"]
            print("✅ 教师登录成功")
        else:
            print(f"❌ 教师登录失败: {response.text}")
            return
            
    except Exception as e:
        print(f"❌ 登录请求失败: {e}")
        return
    
    # 2. 创建作业
    print("\n=== 2. 创建作业 ===")
    homework_data = {
        "title": "七年级数学第一单元测试",
        "description": "本测试包含整数运算、分数运算和基础代数内容",
        "subject": "数学",
        "grade": 7,
        "difficulty_level": 3,
        "max_score": 100,
        "time_limit": 60,
        "due_date": (datetime.now() + timedelta(days=7)).isoformat(),
        "start_date": datetime.now().isoformat(),
        "category": "单元测试",
        "tags": ["代数", "运算", "基础"],
        "instructions": "请仔细阅读题目，按要求作答。",
        "auto_grade": True,
        "max_attempts": 2,
        "show_answers": False
    }
    
    headers = {"Authorization": f"Bearer {teacher_token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/api/homework/create", 
                               json=homework_data, headers=headers)
        print(f"创建作业状态码: {response.status_code}")
        
        if response.status_code == 201:
            create_result = response.json()
            homework_id = create_result["data"]["id"]
            print(f"✅ 作业创建成功，ID: {homework_id}")
            print(f"作业标题: {create_result['data']['title']}")
        else:
            print(f"❌ 创建作业失败: {response.text}")
            return
            
    except Exception as e:
        print(f"❌ 创建作业请求失败: {e}")
        return
    
    # 3. 获取作业详情
    print(f"\n=== 3. 获取作业详情 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/homework/{homework_id}", headers=headers)
        print(f"获取详情状态码: {response.status_code}")
        
        if response.status_code == 200:
            detail_result = response.json()
            print("✅ 获取作业详情成功")
            print(f"作业标题: {detail_result['data']['title']}")
            print(f"发布状态: {detail_result['data']['is_published']}")
        else:
            print(f"❌ 获取作业详情失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 获取详情请求失败: {e}")
    
    # 4. 更新作业
    print(f"\n=== 4. 更新作业 ===")
    update_data = {
        "title": "七年级数学第一单元测试（更新版）",
        "description": "更新：增加了应用题部分",
        "question_count": 15
    }
    
    try:
        response = requests.put(f"{BASE_URL}/api/homework/{homework_id}", 
                              json=update_data, headers=headers)
        print(f"更新作业状态码: {response.status_code}")
        
        if response.status_code == 200:
            update_result = response.json()
            print("✅ 作业更新成功")
            print(f"新标题: {update_result['data']['title']}")
        else:
            print(f"❌ 更新作业失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 更新作业请求失败: {e}")
    
    # 5. 发布作业
    print(f"\n=== 5. 发布作业 ===")
    try:
        response = requests.post(f"{BASE_URL}/api/homework/{homework_id}/publish", 
                               headers=headers)
        print(f"发布作业状态码: {response.status_code}")
        
        if response.status_code == 200:
            publish_result = response.json()
            print("✅ 作业发布成功")
            print(f"发布状态: {publish_result['data']['is_published']}")
        else:
            print(f"❌ 发布作业失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 发布作业请求失败: {e}")
    
    # 6. 获取作业列表
    print(f"\n=== 6. 获取作业列表 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/homework/list?page=1&limit=5", 
                              headers=headers)
        print(f"获取列表状态码: {response.status_code}")
        
        if response.status_code == 200:
            list_result = response.json()
            print("✅ 获取作业列表成功")
            print(f"作业数量: {len(list_result['data']['homeworks'])}")
            for hw in list_result['data']['homeworks']:
                print(f"  - {hw['title']} (发布状态: {hw['is_published']})")
        else:
            print(f"❌ 获取作业列表失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 获取列表请求失败: {e}")
    
    # 7. 搜索作业
    print(f"\n=== 7. 搜索作业 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/homework/search?keyword=数学&grade=7", 
                              headers=headers)
        print(f"搜索作业状态码: {response.status_code}")
        
        if response.status_code == 200:
            search_result = response.json()
            print("✅ 搜索作业成功")
            print(f"搜索结果: {len(search_result['data']['homeworks'])} 个作业")
        else:
            print(f"❌ 搜索作业失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 搜索作业请求失败: {e}")
    
    # 8. 获取统计信息
    print(f"\n=== 8. 获取统计信息 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/homework/statistics", headers=headers)
        print(f"获取统计状态码: {response.status_code}")
        
        if response.status_code == 200:
            stats_result = response.json()
            print("✅ 获取统计信息成功")
            stats = stats_result['data']
            print(f"总作业数: {stats.get('total_homeworks', 0)}")
            print(f"已发布: {stats.get('published_homeworks', 0)}")
            print(f"草稿: {stats.get('draft_homeworks', 0)}")
        else:
            print(f"❌ 获取统计信息失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 获取统计请求失败: {e}")
    
    # 9. 学生视角测试（如果有学生用户）
    print(f"\n=== 9. 学生登录测试 ===")
    student_login_data = {
        "username": "test_student_001",
        "password": "TestPassword123!"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=student_login_data)
        
        if response.status_code == 200:
            student_result = response.json()
            student_token = student_result["data"]["access_token"]
            student_headers = {"Authorization": f"Bearer {student_token}"}
            
            print("✅ 学生登录成功")
            
            # 学生查看已发布作业
            response = requests.get(f"{BASE_URL}/api/homework/list", headers=student_headers)
            if response.status_code == 200:
                student_list = response.json()
                print(f"学生可见作业数: {len(student_list['data']['homeworks'])}")
            else:
                print(f"❌ 学生获取作业列表失败: {response.text}")
        else:
            print(f"学生登录失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 学生登录请求失败: {e}")
    
    print(f"\n=== 作业管理API测试完成 ===")

if __name__ == "__main__":
    test_homework_api()
