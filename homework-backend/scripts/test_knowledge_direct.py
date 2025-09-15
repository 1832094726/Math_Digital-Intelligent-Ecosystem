#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试知识点推荐服务
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.knowledge_recommendation_service import knowledge_recommendation_service

def test_knowledge_recommendation_service():
    """测试知识点推荐服务"""
    print("🧠 测试知识点推荐服务...")
    print("=" * 50)
    
    # 测试用例1: 基于上下文推荐
    print("\n📝 测试用例1: 基于上下文推荐")
    result1 = knowledge_recommendation_service.recommend_knowledge_points(
        user_id=2,  # test_student_001的ID
        context="解一元二次方程",
        limit=3
    )
    print(f"结果: {result1}")
    
    # 测试用例2: 基于题目推荐
    print("\n📝 测试用例2: 基于题目推荐")
    result2 = knowledge_recommendation_service.recommend_knowledge_points(
        user_id=2,
        question_id=1,
        limit=5
    )
    print(f"结果: {result2}")
    
    # 测试用例3: 基于用户状态推荐
    print("\n📝 测试用例3: 基于用户状态推荐")
    result3 = knowledge_recommendation_service.recommend_knowledge_points(
        user_id=2,
        limit=4
    )
    print(f"结果: {result3}")
    
    print("\n" + "=" * 50)
    print("🎉 知识点推荐服务测试完成！")

if __name__ == '__main__':
    test_knowledge_recommendation_service()
