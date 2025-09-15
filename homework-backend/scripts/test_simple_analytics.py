#!/usr/bin/env python3
"""
测试简化版分析API
"""

import requests
import json

def test_simple_analytics():
    # 1. 登录获取token（使用学生账号测试）
    login_response = requests.post('http://localhost:5000/api/auth/login', json={
        'username': 'test_student_001',
        'password': 'password'
    })
    
    if login_response.status_code != 200:
        print(f"❌ 教师登录失败: {login_response.status_code}")
        print(login_response.text)
        return
    
    token = login_response.json()['data']['access_token']
    print("✅ 教师登录成功")
    
    # 2. 测试简化版分析API
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    analytics_response = requests.get('http://localhost:5000/api/simple-analytics/homework/1', headers=headers)
    
    print(f"简化版分析API响应状态: {analytics_response.status_code}")
    
    if analytics_response.status_code == 200:
        print("✅ 简化版分析API测试成功！")
        response_data = analytics_response.json()
        print("响应数据结构:")
        print(f"  - 作业信息: {response_data['data']['homework_info']['title']}")
        print(f"  - 提交统计: {response_data['data']['basic_statistics']['total_submissions']}人提交")
        print(f"  - 平均分: {response_data['data']['basic_statistics']['average_score']}")
        print(f"  - 分数分布: {len(response_data['data']['score_distribution'])}个区间")
        print(f"  - 题目数量: {len(response_data['data']['question_analysis'])}")
        print(f"  - 学生表现: {len(response_data['data']['student_performance'])}名学生")
    else:
        print("❌ 简化版分析API测试失败")
        try:
            response_data = analytics_response.json()
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        except:
            print(analytics_response.text)

if __name__ == '__main__':
    test_simple_analytics()
