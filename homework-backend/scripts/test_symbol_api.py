#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试符号推荐API
"""

import requests
import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import get_db_connection

def check_users():
    """检查数据库中的用户"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询用户表
        cursor.execute('SELECT id, username, password_hash, role FROM users LIMIT 5')
        users = cursor.fetchall()
        
        print('现有用户:')
        for user in users:
            print(f'ID: {user["id"]}, 用户名: {user["username"]}, 角色: {user["role"]}')
        
        cursor.close()
        conn.close()
        
        return users
        
    except Exception as e:
        print(f'查询用户失败: {e}')
        return []

def test_login_and_symbol_api():
    """测试登录和符号推荐API"""
    
    # 先检查用户
    users = check_users()
    if not users:
        print("没有找到用户，无法测试")
        return
    
    # 使用第一个用户进行测试
    test_user = users[0]
    print(f"\n使用用户 {test_user['username']} 进行测试...")
    
    # 尝试登录 - 先尝试已知的测试用户
    login_data = {
        'username': 'test_student',
        'password': 'Test123!@#'
    }
    
    try:
        print("1. 测试登录...")
        response = requests.post('http://localhost:5000/api/auth/login', json=login_data)
        print(f'登录响应状态: {response.status_code}')
        
        if response.status_code == 200:
            response_data = response.json()
            token = response_data.get('token') or response_data.get('data', {}).get('access_token')
            print(f'获取到token: {token[:50]}...')
            
            # 测试符号推荐API
            print("\n2. 测试符号推荐API...")
            headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
            symbol_data = {
                'context': '解一元二次方程 x² + 5x + 6 = 0',
                'limit': 5
            }
            
            symbol_response = requests.post('http://localhost:5000/api/recommend/symbols', 
                                          json=symbol_data, headers=headers)
            print(f'符号推荐API响应状态: {symbol_response.status_code}')
            print('符号推荐API响应内容:')
            print(json.dumps(symbol_response.json(), indent=2, ensure_ascii=False))
            
            # 测试符号使用统计API
            print("\n3. 测试符号使用统计API...")
            usage_data = {
                'symbol_text': 'x',
                'question_id': 'test_q1',
                'context': '一元二次方程'
            }
            
            usage_response = requests.post('http://localhost:5000/api/recommend/symbols/usage',
                                         json=usage_data, headers=headers)
            print(f'符号使用统计API响应状态: {usage_response.status_code}')
            print('符号使用统计API响应内容:')
            print(json.dumps(usage_response.json(), indent=2, ensure_ascii=False))
            
        else:
            print(f'登录失败: {response.text}')
            
            # 尝试创建测试用户
            print("\n尝试创建测试用户...")
            register_data = {
                'username': 'test_student',
                'password': 'Test123!@#',
                'email': 'test@example.com',
                'real_name': '测试学生',
                'role': 'student'
            }
            
            register_response = requests.post('http://localhost:5000/api/auth/register', json=register_data)
            print(f'注册响应状态: {register_response.status_code}')
            print(f'注册响应内容: {register_response.text}')
            
            if register_response.status_code == 201:
                # 用新用户重新登录
                login_data['username'] = 'test_student'
                login_data['password'] = 'Test123!@#'
                
                login_response = requests.post('http://localhost:5000/api/auth/login', json=login_data)
                if login_response.status_code == 200:
                    login_data_response = login_response.json()
                    token = login_data_response.get('token') or login_data_response.get('data', {}).get('access_token')
                    print(f'新用户登录成功，token: {token[:50]}...')
                    
                    # 重新测试符号推荐API
                    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
                    symbol_data = {
                        'context': '解一元二次方程 x² + 5x + 6 = 0',
                        'limit': 5
                    }
                    
                    symbol_response = requests.post('http://localhost:5000/api/recommend/symbols', 
                                                  json=symbol_data, headers=headers)
                    print(f'\n新用户符号推荐API响应状态: {symbol_response.status_code}')
                    print('新用户符号推荐API响应内容:')
                    print(json.dumps(symbol_response.json(), indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f'测试失败: {e}')

if __name__ == "__main__":
    print("🧪 开始测试符号推荐API...")
    print("=" * 50)
    test_login_and_symbol_api()
    print("\n✅ 测试完成!")
