#!/usr/bin/env python3
"""
测试简化版反馈API
"""

import requests
import json

def test_simple_feedback():
    # 1. 登录获取token
    login_response = requests.post('http://localhost:5000/api/auth/login', json={
        'username': 'test_student_001',
        'password': 'password'
    })
    
    if login_response.status_code != 200:
        print(f"❌ 登录失败: {login_response.status_code}")
        print(login_response.text)
        return
    
    token = login_response.json()['data']['access_token']
    print("✅ 登录成功")
    
    # 2. 测试简化版反馈API
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    feedback_response = requests.get('http://localhost:5000/api/simple-feedback/homework/1', headers=headers)
    
    print(f"简化版反馈API响应状态: {feedback_response.status_code}")
    
    if feedback_response.status_code == 200:
        print("✅ 简化版反馈API测试成功！")
        response_data = feedback_response.json()
        print("响应数据结构:")
        print(f"  - 作业信息: {response_data['data']['homework_info']['title']}")
        print(f"  - 个人成绩: {response_data['data']['personal_performance']['total_score']}/{response_data['data']['personal_performance']['max_score']}")
        print(f"  - 班级排名: {response_data['data']['class_statistics']['student_rank']}")
        print(f"  - 题目数量: {len(response_data['data']['question_feedback'])}")
    else:
        print("❌ 简化版反馈API测试失败")
        try:
            response_data = feedback_response.json()
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        except:
            print(feedback_response.text)

if __name__ == '__main__':
    test_simple_feedback()
