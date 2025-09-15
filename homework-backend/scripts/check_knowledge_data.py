#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查知识点数据
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import get_db_connection

def check_knowledge_data():
    """检查知识点和关系数据"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询知识点数据
        cursor.execute('SELECT id, name, description, grade_level, difficulty_level FROM knowledge_points')
        knowledge_points = cursor.fetchall()
        
        print('现有知识点:')
        for kp in knowledge_points:
            print(f'ID: {kp["id"]}, 名称: {kp["name"]}, 年级: {kp["grade_level"]}, 难度: {kp["difficulty_level"]}')
        
        # 查询知识点关系
        cursor.execute('SELECT id, source_point_id, target_point_id, relationship_type, strength FROM knowledge_relationships')
        relations = cursor.fetchall()
        
        print(f'\n知识点关系 ({len(relations)}个):')
        for rel in relations:
            print(f'关系ID: {rel["id"]}, 源: {rel["source_point_id"]} -> 目标: {rel["target_point_id"]}, 类型: {rel["relationship_type"]}, 强度: {rel["strength"]}')
        
        cursor.close()
        conn.close()
        
        return knowledge_points, relations
        
    except Exception as e:
        print(f'查询失败: {e}')
        return [], []

if __name__ == "__main__":
    print("📚 检查知识点数据...")
    print("=" * 50)
    check_knowledge_data()
    print("\n✅ 检查完成!")
