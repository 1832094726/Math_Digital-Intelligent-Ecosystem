#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示测试数据统计信息
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
import json

def show_data_stats():
    """显示数据统计信息"""
    print("=== 数据库统计信息 ===")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 用户统计
                print("\n📊 用户统计:")
                cursor.execute("""
                    SELECT role, COUNT(*) as count 
                    FROM users 
                    GROUP BY role
                """)
                users = cursor.fetchall()
                for user in users:
                    print(f"  {user['role']}: {user['count']} 人")
                
                # 2. 学校班级统计
                print("\n🏫 学校班级统计:")
                cursor.execute("SELECT COUNT(*) as count FROM schools")
                school_count = cursor.fetchone()['count']
                print(f"  学校数量: {school_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM grades")
                grade_count = cursor.fetchone()['count']
                print(f"  年级数量: {grade_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM classes")
                class_count = cursor.fetchone()['count']
                print(f"  班级数量: {class_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM class_students")
                enrollment_count = cursor.fetchone()['count']
                print(f"  学生注册数: {enrollment_count}")
                
                # 3. 作业统计
                print("\n📝 作业统计:")
                cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                homework_count = cursor.fetchone()['count']
                print(f"  作业总数: {homework_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"  题目总数: {question_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_assignments")
                assignment_count = cursor.fetchone()['count']
                print(f"  作业分配数: {assignment_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_submissions")
                submission_count = cursor.fetchone()['count']
                print(f"  学生提交数: {submission_count}")
                
                # 4. 知识点统计
                print("\n🧠 知识点统计:")
                cursor.execute("SELECT COUNT(*) as count FROM knowledge_points")
                kp_count = cursor.fetchone()['count']
                print(f"  知识点数量: {kp_count}")
                
                # 5. 详细班级信息
                print("\n📋 班级详细信息:")
                cursor.execute("""
                    SELECT c.class_name, c.class_code, u.real_name as head_teacher,
                           COUNT(cs.student_id) as student_count
                    FROM classes c
                    LEFT JOIN users u ON c.head_teacher_id = u.id
                    LEFT JOIN class_students cs ON c.id = cs.class_id AND cs.is_active = 1
                    GROUP BY c.id, c.class_name, c.class_code, u.real_name
                """)
                classes = cursor.fetchall()
                for cls in classes:
                    print(f"  {cls['class_name']} ({cls['class_code']}) - 班主任: {cls['head_teacher']} - 学生: {cls['student_count']}人")
                
                # 6. 作业完成情况
                print("\n✅ 作业完成情况:")
                cursor.execute("""
                    SELECT h.title, COUNT(hs.id) as submissions,
                           AVG(hs.score) as avg_score,
                           AVG(hs.time_spent) as avg_time
                    FROM homeworks h
                    LEFT JOIN homework_assignments ha ON h.id = ha.homework_id
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    GROUP BY h.id, h.title
                """)
                homework_stats = cursor.fetchall()
                for hw in homework_stats:
                    avg_score = round(hw['avg_score'] or 0, 1)
                    avg_time = round(hw['avg_time'] or 0, 1)
                    print(f"  {hw['title']}: {hw['submissions']}份提交, 平均分: {avg_score}, 平均用时: {avg_time}分钟")
                
                # 7. 学生成绩统计
                print("\n🎯 学生成绩统计:")
                cursor.execute("""
                    SELECT u.real_name, u.class_name,
                           COUNT(hs.id) as completed_homeworks,
                           AVG(hs.score) as avg_score,
                           SUM(hs.time_spent) as total_time
                    FROM users u
                    LEFT JOIN homework_submissions hs ON u.id = hs.student_id
                    WHERE u.role = 'student'
                    GROUP BY u.id, u.real_name, u.class_name
                    HAVING completed_homeworks > 0
                    ORDER BY u.class_name, u.real_name
                """)
                student_stats = cursor.fetchall()
                for student in student_stats:
                    avg_score = round(student['avg_score'] or 0, 1)
                    total_time = student['total_time'] or 0
                    print(f"  {student['real_name']} ({student['class_name']}): {student['completed_homeworks']}份作业, 平均分: {avg_score}, 总用时: {total_time}分钟")
                
                print("\n✅ 统计信息显示完成！")
                
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
        raise

if __name__ == "__main__":
    show_data_stats()
