#!/usr/bin/env python3
"""
直接测试反馈API
"""

import requests
import json

def test_feedback_direct():
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
    
    # 2. 测试反馈API
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    feedback_response = requests.get('http://localhost:5000/api/feedback/homework/1', headers=headers)
    
    print(f"反馈API响应状态: {feedback_response.status_code}")
    print("响应内容:")
    
    try:
        response_data = feedback_response.json()
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
    except:
        print(feedback_response.text)

if __name__ == '__main__':
    test_feedback_direct()
