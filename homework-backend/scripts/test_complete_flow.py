#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流程测试
"""
import requests
import json

def test_complete_flow():
    """测试完整流程"""
    print("🚀 开始完整流程测试...")
    
    base_url = "http://localhost:5000"
    
    try:
        # 1. 登录
        print("\n🔐 步骤1: 用户登录")
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        if login_response.status_code != 200:
            print(f"❌ 登录失败: {login_response.status_code}")
            print(f"响应: {login_response.text}")
            return False
        
        login_data = login_response.json()
        if not login_data.get('success'):
            print(f"❌ 登录失败: {login_data.get('message')}")
            return False
        
        token = login_data['data']['access_token']
        user_info = login_data['data']['user']
        print(f"✅ 登录成功: {user_info['real_name']} ({user_info['role']})")
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # 2. 获取作业列表
        print("\n📚 步骤2: 获取作业列表")
        homework_response = requests.get(f"{base_url}/api/homework/list", headers=headers)
        
        if homework_response.status_code != 200:
            print(f"❌ 获取作业列表失败: {homework_response.status_code}")
            print(f"响应: {homework_response.text}")
            return False
        
        homework_data = homework_response.json()
        if not homework_data.get('success'):
            print(f"❌ 获取作业列表失败: {homework_data.get('message')}")
            return False
        
        homeworks = homework_data['data']['homeworks']
        print(f"✅ 获取作业列表成功，共 {len(homeworks)} 个作业")
        
        if len(homeworks) == 0:
            print("⚠️ 没有作业数据，请先运行数据修复脚本")
            return False
        
        # 显示作业信息
        for i, hw in enumerate(homeworks, 1):
            print(f"  作业{i}: {hw['title']} | 题目数: {hw['question_count']} | 分数: {hw['max_score']}")
        
        # 3. 获取第一个作业的详情
        first_homework = homeworks[0]
        homework_id = first_homework['id']
        
        print(f"\n📖 步骤3: 获取作业详情 (ID: {homework_id})")
        detail_response = requests.get(f"{base_url}/api/homework/{homework_id}", headers=headers)
        
        if detail_response.status_code != 200:
            print(f"❌ 获取作业详情失败: {detail_response.status_code}")
            print(f"响应: {detail_response.text}")
            return False
        
        detail_data = detail_response.json()
        if not detail_data.get('success'):
            print(f"❌ 获取作业详情失败: {detail_data.get('message')}")
            return False
        
        homework_detail = detail_data['data']
        print(f"✅ 获取作业详情成功: {homework_detail['title']}")
        print(f"  描述: {homework_detail['description']}")
        print(f"  题目数: {homework_detail['question_count']}")
        print(f"  时间限制: {homework_detail['time_limit']}分钟")
        
        # 4. 获取作业题目
        print(f"\n📝 步骤4: 获取作业题目 (ID: {homework_id})")
        questions_response = requests.get(f"{base_url}/api/homework/{homework_id}/questions", headers=headers)
        
        if questions_response.status_code != 200:
            print(f"❌ 获取作业题目失败: {questions_response.status_code}")
            print(f"响应: {questions_response.text}")
            return False
        
        questions_data = questions_response.json()
        if not questions_data.get('success'):
            print(f"❌ 获取作业题目失败: {questions_data.get('message')}")
            return False
        
        questions = questions_data['data']['questions']
        print(f"✅ 获取作业题目成功，共 {len(questions)} 道题目")
        
        # 显示题目信息
        for i, q in enumerate(questions, 1):
            print(f"  题目{i}: {q['content'][:50]}... ({q['question_type']}) - {q['score']}分")
        
        # 5. 测试学生作业API
        print("\n👨‍🎓 步骤5: 获取学生作业")
        student_homework_response = requests.get(f"{base_url}/api/student-homework", headers=headers)
        
        if student_homework_response.status_code == 200:
            student_data = student_homework_response.json()
            if student_data.get('success'):
                student_homeworks = student_data['data']
                print(f"✅ 获取学生作业成功，共 {len(student_homeworks)} 个作业")
            else:
                print(f"⚠️ 学生作业API返回: {student_data.get('message')}")
        else:
            print(f"⚠️ 学生作业API状态码: {student_homework_response.status_code}")
        
        # 6. 总结
        print("\n📊 测试总结:")
        print(f"  ✅ 用户登录: 成功")
        print(f"  ✅ 作业列表: {len(homeworks)} 个作业")
        print(f"  ✅ 作业详情: 成功获取")
        print(f"  ✅ 作业题目: {len(questions)} 道题目")
        print(f"  ✅ 前端应该能够正常显示作业和题目")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """测试所有相关API端点"""
    print("\n🔍 API端点测试...")
    
    base_url = "http://localhost:5000"
    
    # 测试端点列表
    endpoints = [
        ("POST", "/api/auth/login", "登录API"),
        ("GET", "/api/homework/list", "作业列表API"),
        ("GET", "/api/homework/1", "作业详情API"),
        ("GET", "/api/homework/1/questions", "作业题目API"),
        ("GET", "/api/student-homework", "学生作业API"),
    ]
    
    # 先登录获取token
    try:
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        if login_response.status_code == 200:
            login_data = login_response.json()
            if login_data.get('success'):
                token = login_data['data']['access_token']
                headers = {'Authorization': f'Bearer {token}'}
            else:
                headers = {}
        else:
            headers = {}
    except:
        headers = {}
    
    print("\nAPI端点状态:")
    for method, endpoint, name in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", headers=headers, timeout=5)
            elif method == "POST" and "login" in endpoint:
                response = requests.post(f"{base_url}{endpoint}", json={
                    "username": "test_student_001",
                    "password": "password"
                }, timeout=5)
            else:
                continue
            
            status = "✅" if response.status_code in [200, 201] else "❌"
            print(f"  {status} {name}: {response.status_code}")
            
        except Exception as e:
            print(f"  ❌ {name}: 连接失败 ({e})")

if __name__ == "__main__":
    print("🧪 开始完整流程测试...")
    
    # 基本API测试
    test_api_endpoints()
    
    # 完整流程测试
    success = test_complete_flow()
    
    if success:
        print("\n🎉 所有测试通过！前端应该能够正常显示作业和题目了。")
        print("\n📋 建议操作:")
        print("1. 刷新前端页面 (http://localhost:8080/homework)")
        print("2. 检查作业列表是否正常显示")
        print("3. 点击作业查看题目是否正常加载")
    else:
        print("\n⚠️ 测试发现问题，请检查:")
        print("1. 后端服务是否正常运行")
        print("2. 数据库连接是否正常")
        print("3. 是否需要运行数据修复脚本")
