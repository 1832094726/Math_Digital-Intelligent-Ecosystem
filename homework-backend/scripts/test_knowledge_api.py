#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试知识点推荐API
"""

import requests
import json
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API配置
BASE_URL = 'http://localhost:5000'
API_ENDPOINTS = {
    'login': f'{BASE_URL}/api/auth/login',
    'knowledge_recommend': f'{BASE_URL}/api/recommend/knowledge',
    'exercises_recommend': f'{BASE_URL}/api/recommend/exercises',
    'learning_path_recommend': f'{BASE_URL}/api/recommend/learning-path'
}

def test_login():
    """测试用户登录"""
    print("🔐 测试用户登录...")
    
    login_data = {
        'username': 'test_student_001',
        'password': 'student123'
    }
    
    try:
        response = requests.post(API_ENDPOINTS['login'], json=login_data)
        print(f"登录响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                token = result.get('access_token')
                print(f"✅ 登录成功，获取到token: {token[:20]}...")
                return token
            else:
                print(f"❌ 登录失败: {result.get('message', '未知错误')}")
                return None
        else:
            print(f"❌ 登录请求失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 登录异常: {e}")
        return None

def test_knowledge_recommendation(token):
    """测试知识点推荐API"""
    print("\n🧠 测试知识点推荐API...")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # 测试用例1: 基于上下文推荐
    test_cases = [
        {
            'name': '基于上下文推荐',
            'data': {
                'context': '解一元二次方程',
                'limit': 3
            }
        },
        {
            'name': '基于题目推荐',
            'data': {
                'question_id': 1,
                'limit': 5
            }
        },
        {
            'name': '基于用户状态推荐',
            'data': {
                'limit': 4
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        print(f"请求数据: {json.dumps(test_case['data'], ensure_ascii=False, indent=2)}")
        
        try:
            response = requests.post(API_ENDPOINTS['knowledge_recommend'], 
                                   json=test_case['data'], headers=headers)
            print(f"响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    recommendations = result.get('recommendations', [])
                    print(f"✅ 推荐成功，获得 {len(recommendations)} 个知识点:")
                    
                    for j, rec in enumerate(recommendations, 1):
                        print(f"  {j}. {rec.get('name', 'N/A')}")
                        print(f"     描述: {rec.get('description', 'N/A')}")
                        print(f"     年级: {rec.get('grade_level', 'N/A')}")
                        print(f"     难度: {rec.get('difficulty_level', 'N/A')}")
                        print(f"     相关度: {rec.get('relevance_score', 'N/A')}")
                        print(f"     推荐理由: {rec.get('recommendation_reason', 'N/A')}")
                        
                        related_points = rec.get('related_points', [])
                        if related_points:
                            print(f"     相关知识点: {', '.join([rp.get('name', 'N/A') for rp in related_points])}")
                        print()
                else:
                    print(f"❌ 推荐失败: {result.get('error', '未知错误')}")
            else:
                print(f"❌ 请求失败: {response.text}")
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")

def test_exercises_recommendation(token):
    """测试练习题推荐API"""
    print("\n📚 测试练习题推荐API...")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    test_data = {
        'knowledge_point_id': 1,
        'difficulty_level': 3,
        'limit': 5
    }
    
    print(f"请求数据: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(API_ENDPOINTS['exercises_recommend'], 
                               json=test_data, headers=headers)
        print(f"响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                recommendations = result.get('recommendations', [])
                print(f"✅ 推荐成功，获得 {len(recommendations)} 道练习题:")
                
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec.get('title', 'N/A')}")
                    print(f"     内容: {rec.get('content', 'N/A')}")
                    print(f"     难度: {rec.get('difficulty_level', 'N/A')}")
                    print(f"     预计时间: {rec.get('estimated_time', 'N/A')}分钟")
                    print(f"     推荐理由: {rec.get('recommendation_reason', 'N/A')}")
                    print()
            else:
                print(f"❌ 推荐失败: {result.get('error', '未知错误')}")
        else:
            print(f"❌ 请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")

def test_learning_path_recommendation(token):
    """测试学习路径推荐API"""
    print("\n🛤️ 测试学习路径推荐API...")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    test_data = {
        'target_knowledge_point': 2,
        'current_level': 2,
        'learning_style': 'visual'
    }
    
    print(f"请求数据: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(API_ENDPOINTS['learning_path_recommend'], 
                               json=test_data, headers=headers)
        print(f"响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                learning_path = result.get('learning_path', {})
                print(f"✅ 推荐成功，学习路径:")
                print(f"  路径ID: {learning_path.get('path_id', 'N/A')}")
                print(f"  目标知识点: {learning_path.get('target_knowledge_point', 'N/A')}")
                print(f"  预计时长: {learning_path.get('estimated_duration', 'N/A')}天")
                
                steps = learning_path.get('steps', [])
                print(f"  学习步骤 ({len(steps)}步):")
                for step in steps:
                    print(f"    步骤{step.get('step', 'N/A')}: {step.get('knowledge_point_name', 'N/A')}")
                    print(f"      预计时间: {step.get('estimated_time', 'N/A')}天")
                    print(f"      学习资源: {', '.join(step.get('resources', []))}")
                    print()
                    
                adaptations = learning_path.get('learning_style_adaptations', {})
                if adaptations:
                    print(f"  学习风格适配:")
                    for style, methods in adaptations.items():
                        print(f"    {style}: {', '.join(methods)}")
                    
            else:
                print(f"❌ 推荐失败: {result.get('error', '未知错误')}")
        else:
            print(f"❌ 请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试知识点推荐系统API...")
    print("=" * 60)
    
    # 1. 登录获取token
    token = test_login()
    if not token:
        print("❌ 无法获取访问令牌，测试终止")
        return
    
    # 2. 测试知识点推荐
    test_knowledge_recommendation(token)
    
    # 3. 测试练习题推荐
    test_exercises_recommendation(token)
    
    # 4. 测试学习路径推荐
    test_learning_path_recommendation(token)
    
    print("\n" + "=" * 60)
    print("🎉 知识点推荐系统API测试完成！")

if __name__ == '__main__':
    main()
