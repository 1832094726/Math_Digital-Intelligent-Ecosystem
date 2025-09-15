# -*- coding: utf-8 -*-
"""
认证功能测试脚本
"""
import requests
import json

BASE_URL = "http://localhost:5000/api/auth"

def test_user_registration():
    """测试用户注册功能"""
    print("=== 测试用户注册功能 ===")
    
    # 测试数据
    user_data = {
        "username": "test_student_001",
        "email": "test001@example.com",
        "password": "Password123!",
        "real_name": "测试学生",
        "role": "student",
        "grade": 7,
        "school": "测试中学",
        "class_name": "七年级1班",
        "device_type": "web"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/register", json=user_data)
        result = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get('success'):
            print("✅ 用户注册成功")
            return result['data']['access_token'], result['data']['user']
        else:
            print(f"❌ 用户注册失败: {result.get('message')}")
            return None, None
            
    except Exception as e:
        print(f"❌ 注册请求失败: {e}")
        return None, None

def test_user_login():
    """测试用户登录功能"""
    print("\n=== 测试用户登录功能 ===")
    
    login_data = {
        "username": "test_student_001",
        "password": "Password123!",
        "device_type": "web"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/login", json=login_data)
        result = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get('success'):
            print("✅ 用户登录成功")
            return result['data']['access_token']
        else:
            print(f"❌ 用户登录失败: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"❌ 登录请求失败: {e}")
        return None

def test_get_profile(token):
    """测试获取用户信息"""
    print("\n=== 测试获取用户信息 ===")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.get(f"{BASE_URL}/profile", headers=headers)
        result = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get('success'):
            print("✅ 获取用户信息成功")
        else:
            print(f"❌ 获取用户信息失败: {result.get('message')}")
            
    except Exception as e:
        print(f"❌ 获取用户信息请求失败: {e}")

def test_update_profile(token):
    """测试更新用户信息"""
    print("\n=== 测试更新用户信息 ===")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    update_data = {
        "phone": "13800138000",
        "learning_preferences": {
            "difficulty_preference": 0.7,
            "subject_interests": ["algebra", "geometry"]
        }
    }
    
    try:
        response = requests.put(f"{BASE_URL}/profile", headers=headers, json=update_data)
        result = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get('success'):
            print("✅ 更新用户信息成功")
        else:
            print(f"❌ 更新用户信息失败: {result.get('message')}")
            
    except Exception as e:
        print(f"❌ 更新用户信息请求失败: {e}")

def test_logout(token):
    """测试用户登出"""
    print("\n=== 测试用户登出 ===")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/logout", headers=headers)
        result = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get('success'):
            print("✅ 用户登出成功")
        else:
            print(f"❌ 用户登出失败: {result.get('message')}")
            
    except Exception as e:
        print(f"❌ 登出请求失败: {e}")

def test_invalid_login():
    """测试无效登录"""
    print("\n=== 测试无效登录 ===")
    
    login_data = {
        "username": "nonexistent_user",
        "password": "wrong_password"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/login", json=login_data)
        result = response.json()
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if not result.get('success') and response.status_code == 401:
            print("✅ 无效登录正确被拒绝")
        else:
            print("❌ 无效登录处理不正确")
            
    except Exception as e:
        print(f"❌ 无效登录测试失败: {e}")

def main():
    """主测试函数"""
    print("开始认证功能测试...\n")
    
    # 1. 测试用户注册
    token, user = test_user_registration()
    
    if not token:
        print("❌ 注册失败，跳过后续测试")
        return
    
    # 2. 测试用户登录
    login_token = test_user_login()
    
    if login_token:
        # 3. 测试获取用户信息
        test_get_profile(login_token)
        
        # 4. 测试更新用户信息
        test_update_profile(login_token)
        
        # 5. 测试用户登出
        test_logout(login_token)
    
    # 6. 测试无效登录
    test_invalid_login()
    
    print("\n=== 认证功能测试完成 ===")

if __name__ == "__main__":
    main()


