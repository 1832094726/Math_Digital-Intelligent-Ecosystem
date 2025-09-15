#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据库连接
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def test_connection():
    """测试数据库连接"""
    print("🔗 测试数据库连接...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 测试基本连接
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                print(f"✅ 数据库连接成功: {result}")
                
                # 检查现有表
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                print(f"📊 数据库中的表数量: {len(tables)}")
                
                # 检查用户表
                cursor.execute("SELECT COUNT(*) as count FROM users")
                user_count = cursor.fetchone()
                print(f"👥 用户数量: {user_count['count']}")
                
                # 检查作业表
                cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                homework_count = cursor.fetchone()
                print(f"📚 作业数量: {homework_count['count']}")
                
                # 检查是否有questions表
                cursor.execute("SHOW TABLES LIKE 'questions'")
                questions_table = cursor.fetchone()
                if questions_table:
                    cursor.execute("SELECT COUNT(*) as count FROM questions")
                    question_count = cursor.fetchone()
                    print(f"📝 题目数量: {question_count['count']}")
                else:
                    print("⚠️ questions表不存在")
                
                return True
                
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

if __name__ == "__main__":
    test_connection()
