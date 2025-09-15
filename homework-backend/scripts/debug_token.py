#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试JWT token问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
from services.auth_service import AuthService
from models.user import User
from models.database import db

def test_login_and_token():
    """测试登录和token验证"""
    print("=== JWT Token 调试 ===")
    
    # 1. 测试登录
    print("\n1. 测试登录...")
    login_url = "http://localhost:5000/api/auth/login"
    login_data = {
        "username": "test_student_001",
        "password": "password"
    }
    
    try:
        response = requests.post(login_url, json=login_data)
        print(f"登录响应状态码: {response.status_code}")
        print(f"登录响应内容: {response.text}")
        
        if response.status_code == 200:
            login_result = response.json()
            if login_result.get('success'):
                token = login_result['data']['access_token']
                print(f"✅ 登录成功，获得token: {token[:50]}...")
                
                # 2. 验证token
                print("\n2. 验证token...")
                try:
                    payload = AuthService.verify_token(token)
                    print(f"✅ Token验证成功: {payload}")
                except Exception as e:
                    print(f"❌ Token验证失败: {e}")
                
                # 3. 测试API调用
                print("\n3. 测试API调用...")
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                }
                
                api_url = "http://localhost:5000/api/homework/list"
                try:
                    api_response = requests.get(api_url, headers=headers)
                    print(f"API响应状态码: {api_response.status_code}")
                    print(f"API响应内容: {api_response.text}")
                except Exception as e:
                    print(f"❌ API调用失败: {e}")
                    
            else:
                print(f"❌ 登录失败: {login_result.get('message')}")
        else:
            print(f"❌ 登录请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 登录请求异常: {e}")

def test_user_data():
    """测试用户数据"""
    print("\n=== 用户数据检查 ===")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # 检查用户
                cursor.execute("SELECT id, username, role FROM users WHERE username = %s", ('test_student_001',))
                user = cursor.fetchone()
                
                if user:
                    print(f"✅ 用户存在: {user}")
                    
                    # 检查用户权限
                    user_obj = User.get_by_id(user['id'])
                    if user_obj:
                        print(f"✅ 用户对象: {user_obj.to_dict()}")
                    else:
                        print("❌ 无法获取用户对象")
                else:
                    print("❌ 用户不存在")
                    
    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")

if __name__ == "__main__":
    test_user_data()
    test_login_and_token()
