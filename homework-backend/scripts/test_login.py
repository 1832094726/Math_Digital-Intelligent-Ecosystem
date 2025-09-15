#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试登录API脚本
"""
import requests
import json

def test_login():
    """测试登录API"""
    url = 'http://localhost:5000/api/auth/login'
    
    # 测试数据
    test_cases = [
        {'username': 'test_student_001', 'password': 'password'},
        {'username': 'teacher001', 'password': 'password'},
        {'username': 'admin', 'password': 'password'}
    ]
    
    for i, credentials in enumerate(test_cases, 1):
        print(f'\n=== 测试用例 {i}: {credentials["username"]} ===')
        try:
            response = requests.post(url, json=credentials, timeout=10)
            print(f'状态码: {response.status_code}')
            print(f'响应: {response.text}')
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    token = data.get('data', {}).get('access_token')
                    if token:
                        print(f'登录成功，Token: {token[:50]}...')
                        
                        # 测试使用token访问受保护的API
                        headers = {'Authorization': f'Bearer {token}'}
                        homework_response = requests.get(
                            'http://localhost:5000/api/homework/list',
                            headers=headers,
                            timeout=10
                        )
                        print(f'作业列表API状态码: {homework_response.status_code}')
                        if homework_response.status_code == 200:
                            print('✅ Token验证成功')
                        else:
                            print(f'❌ Token验证失败: {homework_response.text}')
                    else:
                        print('❌ 登录响应中没有token')
                else:
                    print(f'❌ 登录失败: {data.get("message")}')
            else:
                print(f'❌ HTTP错误: {response.status_code}')
                
        except Exception as e:
            print(f'❌ 请求异常: {e}')

if __name__ == "__main__":
    test_login()
