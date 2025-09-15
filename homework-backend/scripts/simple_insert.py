#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的数据插入脚本
"""
import mysql.connector
import json
from datetime import datetime, timedelta

def create_test_data():
    """创建测试数据"""
    print("🚀 开始创建测试数据...")
    
    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',
            database='homework_system',
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        
        # 1. 插入学校
        print("1. 插入学校...")
        cursor.execute("""
            INSERT IGNORE INTO schools (id, school_name, school_code, school_type, address, phone, principal, established_year, description, created_at, updated_at)
            VALUES (1, '北京市第一中学', 'BJ001', 'public', '北京市朝阳区教育路123号', '010-12345678', '张校长', 1950, '北京市重点中学，数学教育特色学校', NOW(), NOW())
        """)
        
        # 2. 插入年级
        print("2. 插入年级...")
        cursor.execute("""
            INSERT IGNORE INTO grades (id, school_id, grade_name, grade_level, academic_year, grade_director, created_at, updated_at)
            VALUES (1, 1, '七年级', 7, '2024-2025', '李主任', NOW(), NOW())
        """)
        
        # 3. 插入老师
        print("3. 插入老师...")
        teachers = [
            (10, 'teacher_wang', 'wang@school.com', '王老师'),
            (11, 'teacher_li', 'li@school.com', '李老师'),
            (12, 'teacher_zhang', 'zhang@school.com', '张老师')
        ]
        
        for teacher_id, username, email, real_name in teachers:
            cursor.execute("""
                INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, phone, profile, created_at, updated_at) 
                VALUES (%s, %s, %s, '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', %s, 'teacher', 7, '北京市第一中学', '13800001001', '{"subject": "数学"}', NOW(), NOW())
            """, (teacher_id, username, email, real_name))
        
        # 4. 插入班级
        print("4. 插入班级...")
        cursor.execute("""
            INSERT IGNORE INTO classes (id, school_id, grade_id, class_name, class_code, head_teacher_id, class_size, classroom, created_at, updated_at)
            VALUES (1, 1, 1, '七年级1班', 'G7C1', 10, 3, '教学楼A101', NOW(), NOW())
        """)
        cursor.execute("""
            INSERT IGNORE INTO classes (id, school_id, grade_id, class_name, class_code, head_teacher_id, class_size, classroom, created_at, updated_at)
            VALUES (2, 1, 1, '七年级2班', 'G7C2', 11, 3, '教学楼A102', NOW(), NOW())
        """)
        
        # 5. 插入学生
        print("5. 插入学生...")
        students = [
            (20, 'student_1_1', 'student11@school.com', '小明', '七年级1班', '20240101'),
            (21, 'student_1_2', 'student12@school.com', '小红', '七年级1班', '20240102'),
            (22, 'student_1_3', 'student13@school.com', '小刚', '七年级1班', '20240103'),
            (23, 'student_2_1', 'student21@school.com', '小华', '七年级2班', '20240201'),
            (24, 'student_2_2', 'student22@school.com', '小丽', '七年级2班', '20240202'),
            (25, 'student_2_3', 'student23@school.com', '小强', '七年级2班', '20240203')
        ]
        
        for student_id, username, email, real_name, class_name, student_no in students:
            cursor.execute("""
                INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, class_name, student_id, profile, created_at, updated_at) 
                VALUES (%s, %s, %s, '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', %s, 'student', 7, '北京市第一中学', %s, %s, '{"interests": ["数学"]}', NOW(), NOW())
            """, (student_id, username, email, real_name, class_name, student_no))
        
        # 6. 插入班级学生关系
        print("6. 插入班级学生关系...")
        class_students = [
            (1, 20), (1, 21), (1, 22),  # 1班学生
            (2, 23), (2, 24), (2, 25)   # 2班学生
        ]
        
        for class_id, student_id in class_students:
            cursor.execute("""
                INSERT IGNORE INTO class_students (class_id, student_id, enrollment_date, is_active) 
                VALUES (%s, %s, CURDATE(), 1)
            """, (class_id, student_id))
        
        # 7. 插入知识点
        print("7. 插入知识点...")
        knowledge_points = [
            (1, '有理数运算', '有理数的加减乘除运算'),
            (2, '代数式', '代数式的基本概念和运算'),
            (3, '几何图形', '平面几何基础图形')
        ]
        
        for kp_id, name, description in knowledge_points:
            cursor.execute("""
                INSERT IGNORE INTO knowledge_points (id, name, description, subject, grade, difficulty, parent_id, created_at, updated_at)
                VALUES (%s, %s, %s, '数学', 7, 2, NULL, NOW(), NOW())
            """, (kp_id, name, description))
        
        # 8. 插入作业
        print("8. 插入作业...")
        due_date = datetime.now() + timedelta(days=7)
        
        homeworks = [
            (1, '有理数运算练习 - 七年级1班', 10),
            (2, '有理数运算练习 - 七年级2班', 10),
            (3, '代数式化简 - 七年级1班', 11),
            (4, '代数式化简 - 七年级2班', 11),
            (5, '几何图形认识 - 七年级1班', 12),
            (6, '几何图形认识 - 七年级2班', 12)
        ]
        
        for hw_id, title, teacher_id in homeworks:
            cursor.execute("""
                INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at)
                VALUES (%s, %s, '数学', '练习题目', 7, 2, %s, %s, NOW(), 60, 50, 2, 1, 1, 0, '请仔细作答', '["练习"]', '课后练习', NOW(), NOW())
            """, (hw_id, title, teacher_id, due_date))
        
        conn.commit()
        print("✅ 基础数据创建完成！")
        
        # 显示统计
        cursor.execute("SELECT COUNT(*) FROM users WHERE role='teacher'")
        teacher_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM users WHERE role='student'")
        student_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM homeworks")
        homework_count = cursor.fetchone()[0]
        
        print(f"\n📊 数据统计:")
        print(f"   👨‍🏫 教师: {teacher_count}人")
        print(f"   👨‍🎓 学生: {student_count}人")
        print(f"   📚 作业: {homework_count}个")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        return False

if __name__ == "__main__":
    create_test_data()
