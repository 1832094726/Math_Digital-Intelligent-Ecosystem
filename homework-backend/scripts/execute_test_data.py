#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接执行测试数据创建
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from models.user import User
import json
from datetime import datetime, timedelta
import random

def execute_test_data():
    """直接执行测试数据创建"""
    print("=== 开始创建测试数据 ===")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 创建学校
                print("1. 创建学校...")
                cursor.execute("""
                    INSERT IGNORE INTO schools (id, school_name, school_code, school_type, address, phone, principal, established_year, description)
                    VALUES (1, '北京市第一中学', 'BJ001', 'public', '北京市朝阳区教育路123号', '010-12345678', '张校长', 1950, '北京市重点中学，数学教育特色学校')
                """)
                
                # 2. 创建年级
                print("2. 创建年级...")
                cursor.execute("""
                    INSERT IGNORE INTO grades (id, school_id, grade_name, grade_level, academic_year, grade_director)
                    VALUES (1, 1, '七年级', 7, '2024-2025', '李主任')
                """)
                
                # 3. 创建老师
                print("3. 创建老师...")
                teachers_sql = [
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, phone, profile, created_at, updated_at) 
                       VALUES (10, 'teacher_wang', 'wang@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '王老师', 'teacher', 7, '北京市第一中学', '13800001001', '{"subject": "数学", "teaching_years": 10, "specialty": "代数教学"}', NOW(), NOW())""",
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, phone, profile, created_at, updated_at) 
                       VALUES (11, 'teacher_li', 'li@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '李老师', 'teacher', 7, '北京市第一中学', '13800001002', '{"subject": "数学", "teaching_years": 8, "specialty": "几何教学"}', NOW(), NOW())""",
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, phone, profile, created_at, updated_at) 
                       VALUES (12, 'teacher_zhang', 'zhang@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '张老师', 'teacher', 7, '北京市第一中学', '13800001003', '{"subject": "数学", "teaching_years": 12, "specialty": "应用题教学"}', NOW(), NOW())"""
                ]
                for sql in teachers_sql:
                    cursor.execute(sql)
                
                # 4. 创建班级
                print("4. 创建班级...")
                cursor.execute("""
                    INSERT IGNORE INTO classes (id, school_id, grade_id, class_name, class_code, head_teacher_id, class_size, classroom)
                    VALUES (1, 1, 1, '七年级1班', 'G7C1', 10, 3, '教学楼A101')
                """)
                cursor.execute("""
                    INSERT IGNORE INTO classes (id, school_id, grade_id, class_name, class_code, head_teacher_id, class_size, classroom)
                    VALUES (2, 1, 1, '七年级2班', 'G7C2', 11, 3, '教学楼A102')
                """)
                
                # 5. 创建学生
                print("5. 创建学生...")
                students_sql = [
                    # 1班学生
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) 
                       VALUES (20, 'student_1_1', 'student11@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小明', 'student', 7, '北京市第一中学', '七年级1班', '20240101', '{"interests": ["数学", "科学"], "learning_style": "视觉型"}', NOW(), NOW())""",
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) 
                       VALUES (21, 'student_1_2', 'student12@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小红', 'student', 7, '北京市第一中学', '七年级1班', '20240102', '{"interests": ["数学", "科学"], "learning_style": "听觉型"}', NOW(), NOW())""",
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) 
                       VALUES (22, 'student_1_3', 'student13@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小刚', 'student', 7, '北京市第一中学', '七年级1班', '20240103', '{"interests": ["数学", "科学"], "learning_style": "动手型"}', NOW(), NOW())""",
                    # 2班学生
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) 
                       VALUES (23, 'student_2_1', 'student21@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小华', 'student', 7, '北京市第一中学', '七年级2班', '20240201', '{"interests": ["数学", "科学"], "learning_style": "视觉型"}', NOW(), NOW())""",
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) 
                       VALUES (24, 'student_2_2', 'student22@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小丽', 'student', 7, '北京市第一中学', '七年级2班', '20240202', '{"interests": ["数学", "科学"], "learning_style": "听觉型"}', NOW(), NOW())""",
                    """INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) 
                       VALUES (25, 'student_2_3', 'student23@school.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '小强', 'student', 7, '北京市第一中学', '七年级2班', '20240203', '{"interests": ["数学", "科学"], "learning_style": "动手型"}', NOW(), NOW())"""
                ]
                for sql in students_sql:
                    cursor.execute(sql)
                
                # 6. 添加学生到班级
                print("6. 添加学生到班级...")
                class_students_sql = [
                    "INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) VALUES (1, 20, CURDATE(), 1)",
                    "INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) VALUES (1, 21, CURDATE(), 1)",
                    "INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) VALUES (1, 22, CURDATE(), 1)",
                    "INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) VALUES (2, 23, CURDATE(), 1)",
                    "INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) VALUES (2, 24, CURDATE(), 1)",
                    "INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) VALUES (2, 25, CURDATE(), 1)"
                ]
                for sql in class_students_sql:
                    cursor.execute(sql)
                
                conn.commit()
                print("✅ 基础数据创建完成")
                
                return True
                
    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    execute_test_data()
