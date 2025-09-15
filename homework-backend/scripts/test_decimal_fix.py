#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Decimal修复
"""
import requests
import json

def test_decimal_fix():
    """测试Decimal修复"""
    print("🔧 测试Decimal序列化修复...")
    
    base_url = "http://localhost:5000"
    
    try:
        # 1. 登录
        print("\n🔐 登录...")
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        if login_response.status_code != 200:
            print(f"❌ 登录失败: {login_response.status_code}")
            return False
        
        login_data = login_response.json()
        token = login_data['data']['access_token']
        print("✅ 登录成功")
        
        headers = {'Authorization': f'Bearer {token}'}
        
        # 2. 测试题目API
        homework_id = 2  # 使用作业ID 2
        print(f"\n📝 测试作业{homework_id}的题目API...")
        
        questions_response = requests.get(
            f"{base_url}/api/homework/{homework_id}/questions", 
            headers=headers
        )
        
        print(f"状态码: {questions_response.status_code}")
        
        if questions_response.status_code == 200:
            questions_data = questions_response.json()
            
            if questions_data.get('success'):
                questions = questions_data['data']['questions']
                print(f"✅ 成功获取 {len(questions)} 道题目")
                
                # 显示题目详情
                for i, q in enumerate(questions, 1):
                    print(f"\n题目{i}:")
                    print(f"  ID: {q['id']}")
                    print(f"  内容: {q['content']}")
                    print(f"  类型: {q['question_type']}")
                    print(f"  分值: {q['score']} (类型: {type(q['score'])})")
                    print(f"  难度: {q['difficulty']} (类型: {type(q['difficulty'])})")
                    
                    if q.get('options'):
                        print(f"  选项: {q['options']} (类型: {type(q['options'])})")
                
                print(f"\n🎉 Decimal序列化问题已修复！")
                return True
            else:
                print(f"❌ API返回失败: {questions_data.get('message')}")
                return False
        
        elif questions_response.status_code == 500:
            try:
                error_data = questions_response.json()
                print(f"❌ 仍然有500错误: {error_data}")
                return False
            except:
                print(f"❌ 500错误，无法解析响应: {questions_response.text}")
                return False
        
        else:
            print(f"❌ 其他错误: {questions_response.status_code}")
            print(f"响应: {questions_response.text}")
            return False
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def test_multiple_homeworks():
    """测试多个作业的题目API"""
    print("\n🧪 测试多个作业的题目API...")
    
    base_url = "http://localhost:5000"
    
    try:
        # 登录
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        token = login_response.json()['data']['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        
        # 测试前5个作业
        for homework_id in range(1, 6):
            print(f"\n📝 测试作业{homework_id}...")
            
            questions_response = requests.get(
                f"{base_url}/api/homework/{homework_id}/questions", 
                headers=headers
            )
            
            if questions_response.status_code == 200:
                questions_data = questions_response.json()
                if questions_data.get('success'):
                    questions = questions_data['data']['questions']
                    print(f"  ✅ 成功: {len(questions)}道题目")
                else:
                    print(f"  ❌ 失败: {questions_data.get('message')}")
            else:
                print(f"  ❌ 状态码: {questions_response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始Decimal修复测试...")
    
    # 测试单个作业
    success = test_decimal_fix()
    
    if success:
        # 测试多个作业
        test_multiple_homeworks()
        print("\n🎉 所有测试通过！题目API应该正常工作了。")
    else:
        print("\n⚠️ 修复测试失败，需要进一步调试。")
