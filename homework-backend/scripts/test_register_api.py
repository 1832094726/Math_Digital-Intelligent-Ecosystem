#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用户注册API
"""
import requests
import json
import random
import string

def generate_random_string(length=8):
    """生成随机字符串"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def test_student_register():
    """测试学生注册"""
    print("🎓 测试学生注册...")
    
    base_url = "http://localhost:5000"
    
    # 生成随机用户数据
    random_suffix = generate_random_string(6)
    
    student_data = {
        "role": "student",
        "username": f"test_student_{random_suffix}",
        "email": f"student_{random_suffix}@test.com",
        "real_name": f"测试学生{random_suffix}",
        "password": "password123",
        "grade": 7,
        "school": "测试中学",
        "class_name": "1班",
        "student_id": f"2024{random_suffix}"
    }
    
    try:
        response = requests.post(f"{base_url}/api/auth/register", json=student_data)
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        
        if response.status_code == 201:
            print("✅ 学生注册成功")
            return True
        else:
            print("❌ 学生注册失败")
            return False
            
    except Exception as e:
        print(f"❌ 注册请求异常: {e}")
        return False

def test_teacher_register():
    """测试教师注册"""
    print("\n👨‍🏫 测试教师注册...")
    
    base_url = "http://localhost:5000"
    
    # 生成随机用户数据
    random_suffix = generate_random_string(6)
    
    teacher_data = {
        "role": "teacher",
        "username": f"test_teacher_{random_suffix}",
        "email": f"teacher_{random_suffix}@test.com",
        "real_name": f"测试教师{random_suffix}",
        "password": "password123",
        "school": "测试中学",
        "teaching_grades": [6, 7, 8],
        "phone": "13800138000"
    }
    
    try:
        response = requests.post(f"{base_url}/api/auth/register", json=teacher_data)
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        
        if response.status_code == 201:
            print("✅ 教师注册成功")
            return True
        else:
            print("❌ 教师注册失败")
            return False
            
    except Exception as e:
        print(f"❌ 注册请求异常: {e}")
        return False

def test_parent_register():
    """测试家长注册"""
    print("\n👨‍👩‍👧‍👦 测试家长注册...")
    
    base_url = "http://localhost:5000"
    
    # 生成随机用户数据
    random_suffix = generate_random_string(6)
    
    parent_data = {
        "role": "parent",
        "username": f"test_parent_{random_suffix}",
        "email": f"parent_{random_suffix}@test.com",
        "real_name": f"测试家长{random_suffix}",
        "password": "password123",
        "phone": "13900139000",
        "child_name": f"测试孩子{random_suffix}",
        "child_school": "测试小学",
        "child_grade": 5
    }
    
    try:
        response = requests.post(f"{base_url}/api/auth/register", json=parent_data)
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        
        if response.status_code == 201:
            print("✅ 家长注册成功")
            return True
        else:
            print("❌ 家长注册失败")
            return False
            
    except Exception as e:
        print(f"❌ 注册请求异常: {e}")
        return False

def test_duplicate_register():
    """测试重复注册"""
    print("\n🔄 测试重复注册...")
    
    base_url = "http://localhost:5000"
    
    # 使用已存在的用户名
    duplicate_data = {
        "role": "student",
        "username": "test_student_001",  # 已存在的用户名
        "email": "duplicate@test.com",
        "real_name": "重复测试",
        "password": "password123",
        "grade": 7,
        "school": "测试中学",
        "class_name": "1班",
        "student_id": "202400001"
    }
    
    try:
        response = requests.post(f"{base_url}/api/auth/register", json=duplicate_data)
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        
        if response.status_code == 400:
            print("✅ 重复注册正确被拒绝")
            return True
        else:
            print("❌ 重复注册处理有问题")
            return False
            
    except Exception as e:
        print(f"❌ 注册请求异常: {e}")
        return False

def test_invalid_data():
    """测试无效数据"""
    print("\n❌ 测试无效数据...")
    
    base_url = "http://localhost:5000"
    
    # 缺少必填字段
    invalid_data = {
        "role": "student",
        "username": "test_invalid",
        # 缺少email、password等必填字段
    }
    
    try:
        response = requests.post(f"{base_url}/api/auth/register", json=invalid_data)
        
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        
        if response.status_code == 400:
            print("✅ 无效数据正确被拒绝")
            return True
        else:
            print("❌ 无效数据处理有问题")
            return False
            
    except Exception as e:
        print(f"❌ 注册请求异常: {e}")
        return False

def test_login_after_register():
    """测试注册后登录"""
    print("\n🔐 测试注册后登录...")
    
    base_url = "http://localhost:5000"
    
    # 先注册一个新用户
    random_suffix = generate_random_string(6)
    username = f"test_login_{random_suffix}"
    password = "password123"
    
    register_data = {
        "role": "student",
        "username": username,
        "email": f"login_{random_suffix}@test.com",
        "real_name": f"登录测试{random_suffix}",
        "password": password,
        "grade": 8,
        "school": "测试中学",
        "class_name": "2班",
        "student_id": f"2024{random_suffix}"
    }
    
    try:
        # 注册
        register_response = requests.post(f"{base_url}/api/auth/register", json=register_data)
        
        if register_response.status_code != 201:
            print("❌ 注册失败，无法测试登录")
            return False
        
        print("✅ 注册成功，开始测试登录...")
        
        # 登录
        login_data = {
            "username": username,
            "password": password
        }
        
        login_response = requests.post(f"{base_url}/api/auth/login", json=login_data)
        
        print(f"登录状态码: {login_response.status_code}")
        
        if login_response.status_code == 200:
            login_result = login_response.json()
            if login_result.get('success'):
                print("✅ 注册后登录成功")
                return True
            else:
                print(f"❌ 登录失败: {login_result.get('message')}")
                return False
        else:
            print("❌ 登录请求失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始用户注册API测试...")
    
    tests = [
        ("学生注册", test_student_register),
        ("教师注册", test_teacher_register),
        ("家长注册", test_parent_register),
        ("重复注册", test_duplicate_register),
        ("无效数据", test_invalid_data),
        ("注册后登录", test_login_after_register)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试通过！注册功能正常工作。")
    else:
        print("⚠️ 部分测试失败，需要检查注册功能。")
