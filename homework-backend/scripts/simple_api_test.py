#!/usr/bin/env python3
"""
简单的API测试脚本
"""

import requests
import json



def test_feedback_api(token, homework_id=1):
    """测试反馈API"""
    url = f'http://localhost:5000/api/feedback/homework/{homework_id}'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    print(f"\n测试反馈API: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text[:500]}...")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 反馈API测试成功")
                return True
            else:
                print(f"❌ 反馈API失败: {result.get('message')}")
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
    
    return False

def test_analytics_api(token, homework_id=1):
    """测试分析API"""
    url = f'http://localhost:5000/api/analytics/homework/{homework_id}'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    print(f"\n测试分析API: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text[:500]}...")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 分析API测试成功")
                return True
            else:
                print(f"❌ 分析API失败: {result.get('message')}")
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
    
    return False

def test_student_feedback():
    """测试学生反馈功能"""
    print("🎓 测试学生反馈功能")
    print("-" * 30)

    # 学生登录
    url = 'http://localhost:5000/api/auth/login'
    data = {'username': 'test_student_001', 'password': 'password'}

    response = requests.post(url, json=data)
    if response.status_code != 200:
        print("❌ 学生登录失败")
        return False

    token = response.json()['data']['access_token']
    print("✅ 学生登录成功")

    # 测试反馈API
    return test_feedback_api(token)

def test_teacher_analytics():
    """测试教师分析功能"""
    print("\n👨‍🏫 测试教师分析功能")
    print("-" * 30)

    # 教师登录
    url = 'http://localhost:5000/api/auth/login'
    data = {'username': 'teacher001', 'password': 'password'}

    response = requests.post(url, json=data)
    if response.status_code != 200:
        print("❌ 教师登录失败")
        return False

    token = response.json()['data']['access_token']
    print("✅ 教师登录成功")

    # 测试分析API
    return test_analytics_api(token)

def main():
    print("🚀 开始完整API测试")
    print("=" * 50)

    # 测试学生反馈功能
    feedback_success = test_student_feedback()

    # 测试教师分析功能
    analytics_success = test_teacher_analytics()

    print("\n" + "=" * 50)
    print("📋 测试结果汇总:")
    print(f"  学生反馈API: {'✅' if feedback_success else '❌'}")
    print(f"  教师分析API: {'✅' if analytics_success else '❌'}")

    if feedback_success and analytics_success:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败")

if __name__ == '__main__':
    main()
