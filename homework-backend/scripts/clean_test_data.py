#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理测试数据脚本
用于清理所有测试数据，恢复数据库到初始状态
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def clean_test_data():
    """清理所有测试数据"""
    print("=== 开始清理测试数据 ===")
    print("⚠️  警告：此操作将删除所有测试数据，请确认！")
    
    confirm = input("请输入 'YES' 确认清理: ")
    if confirm != 'YES':
        print("❌ 操作已取消")
        return
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 按照外键依赖关系的逆序删除数据
                tables_to_clean = [
                    'homework_submissions',
                    'homework_assignments', 
                    'questions',
                    'homeworks',
                    'class_students',
                    'classes',
                    'grades',
                    'schools',
                    'knowledge_points',
                    'user_sessions'
                ]
                
                for table in tables_to_clean:
                    cursor.execute(f"DELETE FROM {table}")
                    affected_rows = cursor.rowcount
                    print(f"✅ 清理表 {table}: {affected_rows} 行")
                
                # 清理测试用户（保留原有的test_student_001等）
                cursor.execute("""
                    DELETE FROM users 
                    WHERE username LIKE 'teacher_%' 
                       OR username LIKE 'student_%_%'
                       OR username IN ('teacher_wang', 'teacher_li', 'teacher_zhang')
                """)
                affected_rows = cursor.rowcount
                print(f"✅ 清理测试用户: {affected_rows} 行")
                
                # 重置自增ID
                reset_tables = [
                    'schools', 'grades', 'classes', 'homeworks', 
                    'questions', 'homework_assignments', 'homework_submissions',
                    'knowledge_points'
                ]
                
                for table in reset_tables:
                    cursor.execute(f"ALTER TABLE {table} AUTO_INCREMENT = 1")
                    print(f"✅ 重置 {table} 自增ID")
                
                conn.commit()
                print("\n✅ 测试数据清理完成！")
                
    except Exception as e:
        print(f"❌ 清理测试数据失败: {e}")
        raise

if __name__ == "__main__":
    clean_test_data()
