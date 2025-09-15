#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试前端兼容性
"""
import requests
import json

def test_frontend_compatibility():
    """测试前端兼容性"""
    print("🔧 测试前端数据兼容性...")
    
    base_url = "http://localhost:5000"
    
    try:
        # 1. 登录
        print("\n🔐 登录测试...")
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        if login_response.status_code != 200:
            print(f"❌ 登录失败: {login_response.status_code}")
            return False
        
        login_data = login_response.json()
        if not login_data.get('success'):
            print(f"❌ 登录失败: {login_data.get('message')}")
            return False
        
        token = login_data['data']['access_token']
        print("✅ 登录成功")
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # 2. 获取作业列表并检查数据结构
        print("\n📚 检查作业列表数据结构...")
        homework_response = requests.get(f"{base_url}/api/homework/list", headers=headers)
        
        if homework_response.status_code != 200:
            print(f"❌ 获取作业列表失败: {homework_response.status_code}")
            return False
        
        homework_data = homework_response.json()
        if not homework_data.get('success'):
            print(f"❌ 获取作业列表失败: {homework_data.get('message')}")
            return False
        
        homeworks = homework_data['data']['homeworks']
        print(f"✅ 获取到 {len(homeworks)} 个作业")
        
        if len(homeworks) == 0:
            print("⚠️ 没有作业数据")
            return False
        
        # 3. 检查第一个作业的数据结构
        first_homework = homeworks[0]
        print(f"\n🔍 检查作业数据结构: {first_homework['title']}")
        
        # 前端期望的字段
        expected_fields = [
            'id', 'title', 'description', 'subject', 'grade',
            'deadline', 'difficulty', 'status', 'progress',
            'question_count', 'max_score', 'problems', 'savedAnswers'
        ]
        
        missing_fields = []
        present_fields = []
        
        for field in expected_fields:
            if field in first_homework:
                present_fields.append(field)
                print(f"  ✅ {field}: {first_homework[field]}")
            else:
                missing_fields.append(field)
                print(f"  ❌ {field}: 缺失")
        
        print(f"\n📊 字段统计:")
        print(f"  存在字段: {len(present_fields)}/{len(expected_fields)}")
        print(f"  缺失字段: {missing_fields}")
        
        # 4. 检查数据类型
        print(f"\n🔍 检查数据类型...")
        type_checks = [
            ('id', int),
            ('title', str),
            ('question_count', int),
            ('max_score', (int, float)),
            ('status', str),
            ('progress', (int, float)),
            ('problems', list),
            ('savedAnswers', dict)
        ]
        
        for field, expected_type in type_checks:
            if field in first_homework:
                actual_value = first_homework[field]
                if isinstance(actual_value, expected_type):
                    print(f"  ✅ {field}: {type(actual_value).__name__} (正确)")
                else:
                    print(f"  ❌ {field}: {type(actual_value).__name__} (期望: {expected_type})")
            else:
                print(f"  ⚠️ {field}: 字段不存在")
        
        # 5. 生成前端测试数据
        print(f"\n📋 生成前端测试数据...")
        frontend_data = {
            "homeworks": homeworks,
            "currentHomework": first_homework if homeworks else None,
            "user": login_data['data']['user']
        }
        
        print("前端可用的数据结构:")
        print(json.dumps(frontend_data, indent=2, ensure_ascii=False)[:500] + "...")
        
        # 6. 验证前端组件期望
        print(f"\n🎯 验证前端组件期望...")
        
        # HomeworkManagement组件期望
        homework_management_checks = [
            ('homeworks数组', isinstance(homeworks, list)),
            ('作业有title', 'title' in first_homework),
            ('作业有status', 'status' in first_homework),
            ('作业有deadline', 'deadline' in first_homework),
            ('作业有difficulty', 'difficulty' in first_homework),
            ('作业有progress', 'progress' in first_homework)
        ]
        
        all_passed = True
        for check_name, check_result in homework_management_checks:
            if check_result:
                print(f"  ✅ {check_name}")
            else:
                print(f"  ❌ {check_name}")
                all_passed = False
        
        if all_passed:
            print("\n🎉 前端兼容性测试通过！")
            print("HomeworkManagement组件应该能够正常显示作业列表")
        else:
            print("\n⚠️ 前端兼容性测试发现问题")
            print("需要进一步修复数据结构")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_mock_data():
    """生成模拟数据用于前端测试"""
    print("\n🎭 生成前端模拟数据...")
    
    mock_homework = {
        "id": 1,
        "title": "数学练习 - 有理数运算",
        "description": "练习有理数的基本运算",
        "subject": "数学",
        "grade": 7,
        "deadline": "2025-09-20 23:59:59",
        "difficulty": 2,
        "status": "not_started",
        "progress": 0,
        "question_count": 2,
        "max_score": 100,
        "problems": [
            {
                "id": 1,
                "content": "计算：(-3) + 5 = ?",
                "type": "single_choice",
                "options": ["2", "8", "-8", "0"],
                "score": 50
            },
            {
                "id": 2,
                "content": "计算：2 × (-4) = ?",
                "type": "fill_blank",
                "score": 50
            }
        ],
        "savedAnswers": {}
    }
    
    print("模拟作业数据:")
    print(json.dumps(mock_homework, indent=2, ensure_ascii=False))
    
    return mock_homework

if __name__ == "__main__":
    print("🚀 开始前端兼容性测试...")
    
    # 测试实际API
    success = test_frontend_compatibility()
    
    # 生成模拟数据
    mock_data = generate_mock_data()
    
    if success:
        print("\n✅ 测试完成：前端应该能够正常显示作业")
    else:
        print("\n⚠️ 测试完成：发现兼容性问题，需要进一步修复")
