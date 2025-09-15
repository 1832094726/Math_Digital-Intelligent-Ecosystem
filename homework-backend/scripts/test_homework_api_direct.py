#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试作业API
"""
import requests
import json

def test_homework_api():
    """测试作业API"""
    print("🧪 直接测试作业API...")
    
    base_url = "http://localhost:5000"
    
    try:
        # 1. 先登录获取token
        print("\n🔐 执行登录...")
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        print(f"登录响应状态: {login_response.status_code}")
        print(f"登录响应内容: {login_response.text}")
        
        if login_response.status_code != 200:
            print("❌ 登录失败，无法继续测试")
            return False
        
        login_data = login_response.json()
        if not login_data.get('success'):
            print(f"❌ 登录失败: {login_data.get('message')}")
            return False
        
        token = login_data['data']['access_token']
        print(f"✅ 登录成功，获取token: {token[:20]}...")
        
        # 2. 测试作业列表API
        print("\n📚 测试作业列表API...")
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        homework_response = requests.get(f"{base_url}/api/homework/list", headers=headers)
        
        print(f"作业列表响应状态: {homework_response.status_code}")
        print(f"作业列表响应内容: {homework_response.text}")
        
        if homework_response.status_code == 200:
            homework_data = homework_response.json()
            if homework_data.get('success'):
                homeworks = homework_data['data']['homeworks']
                print(f"✅ 获取作业列表成功，共 {len(homeworks)} 个作业")
                
                for i, hw in enumerate(homeworks, 1):
                    print(f"  作业{i}: {hw.get('title', 'N/A')} | 科目: {hw.get('subject', 'N/A')} | 年级: {hw.get('grade', 'N/A')}")
                
                return len(homeworks) > 0
            else:
                print(f"❌ API返回失败: {homework_data.get('message')}")
        else:
            print(f"❌ API请求失败: {homework_response.status_code}")
        
        # 3. 测试学生作业API
        print("\n👨‍🎓 测试学生作业API...")
        student_homework_response = requests.get(f"{base_url}/api/student-homework", headers=headers)
        
        print(f"学生作业响应状态: {student_homework_response.status_code}")
        print(f"学生作业响应内容: {student_homework_response.text}")
        
        if student_homework_response.status_code == 200:
            student_data = student_homework_response.json()
            if student_data.get('success'):
                student_homeworks = student_data['data']
                print(f"✅ 获取学生作业成功，共 {len(student_homeworks)} 个作业")
                
                for i, hw in enumerate(student_homeworks, 1):
                    print(f"  学生作业{i}: {hw.get('title', 'N/A')} | 状态: {hw.get('status', 'N/A')}")
                
                return len(student_homeworks) > 0
            else:
                print(f"❌ 学生作业API返回失败: {student_data.get('message')}")
        else:
            print(f"❌ 学生作业API请求失败: {student_homework_response.status_code}")
        
        return False
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_users():
    """使用不同用户测试"""
    print("\n👥 使用不同用户测试...")
    
    base_url = "http://localhost:5000"
    
    # 测试用户列表
    test_users = [
        {"username": "test_student_001", "password": "password", "role": "学生"},
        {"username": "teacher_wang", "password": "password", "role": "教师"},
        {"username": "student_001", "password": "password", "role": "学生"},
    ]
    
    for user in test_users:
        print(f"\n🧪 测试用户: {user['username']} ({user['role']})")
        
        try:
            # 登录
            login_response = requests.post(f"{base_url}/api/auth/login", json={
                "username": user['username'],
                "password": user['password']
            })
            
            if login_response.status_code == 200:
                login_data = login_response.json()
                if login_data.get('success'):
                    token = login_data['data']['access_token']
                    print(f"  ✅ 登录成功")
                    
                    # 获取作业列表
                    headers = {'Authorization': f'Bearer {token}'}
                    homework_response = requests.get(f"{base_url}/api/homework/list", headers=headers)
                    
                    if homework_response.status_code == 200:
                        homework_data = homework_response.json()
                        if homework_data.get('success'):
                            homeworks = homework_data['data']['homeworks']
                            print(f"  📚 作业数量: {len(homeworks)}")
                        else:
                            print(f"  ❌ 作业API失败: {homework_data.get('message')}")
                    else:
                        print(f"  ❌ 作业API状态码: {homework_response.status_code}")
                else:
                    print(f"  ❌ 登录失败: {login_data.get('message')}")
            else:
                print(f"  ❌ 登录状态码: {login_response.status_code}")
                
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")

if __name__ == "__main__":
    print("🚀 开始API测试...")
    
    # 基本测试
    success = test_homework_api()
    
    # 多用户测试
    test_with_different_users()
    
    if success:
        print("\n✅ API测试完成，发现作业数据")
    else:
        print("\n⚠️ API测试完成，但没有发现作业数据")
        print("建议运行 check_homework_data.py 检查数据库中的作业数据")
