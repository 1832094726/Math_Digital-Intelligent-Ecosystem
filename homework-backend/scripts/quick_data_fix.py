#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速数据修复脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from datetime import datetime, timedelta
import json

def quick_data_fix():
    """快速修复数据"""
    print("🔧 快速数据修复...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 检查现有数据
                print("\n📊 检查现有数据...")
                
                cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                homework_count = cursor.fetchone()['count']
                print(f"现有作业数: {homework_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='student'")
                student_count = cursor.fetchone()['count']
                print(f"现有学生数: {student_count}")
                
                # 2. 如果没有作业，创建一些简单的作业
                if homework_count == 0:
                    print("\n📚 创建测试作业...")
                    
                    # 确保有教师用户
                    cursor.execute("SELECT id FROM users WHERE role='teacher' LIMIT 1")
                    teacher = cursor.fetchone()
                    
                    if not teacher:
                        print("创建测试教师...")
                        cursor.execute("""
                            INSERT INTO users (username, email, password_hash, real_name, role, grade, school)
                            VALUES ('teacher_test', 'teacher@test.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', '测试教师', 'teacher', 7, '测试学校')
                        """)
                        teacher_id = cursor.lastrowid
                    else:
                        teacher_id = teacher['id']
                    
                    # 创建作业
                    due_date = datetime.now() + timedelta(days=7)
                    
                    homeworks = [
                        ('数学练习1 - 有理数运算', '数学', '练习有理数的基本运算', 7, teacher_id),
                        ('数学练习2 - 代数式化简', '数学', '练习代数式的化简和求值', 7, teacher_id),
                        ('数学练习3 - 几何图形', '数学', '认识基本几何图形', 7, teacher_id),
                    ]
                    
                    for title, subject, description, grade, created_by in homeworks:
                        cursor.execute("""
                            INSERT INTO homeworks (title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category)
                            VALUES (%s, %s, %s, %s, 2, %s, %s, NOW(), 60, 100, 3, 1, 1, 0, '请仔细作答', '["练习"]', '课后练习')
                        """, (title, subject, description, grade, created_by, due_date))
                        
                        homework_id = cursor.lastrowid
                        print(f"  创建作业: {title} (ID: {homework_id})")
                        
                        # 为作业创建一些题目
                        if '有理数' in title:
                            questions = [
                                ('计算：(-3) + 5 = ?', 'single_choice', '["2", "8", "-8", "0"]', '2', 50),
                                ('计算：2 × (-4) = ?', 'fill_blank', None, '-8', 50)
                            ]
                        elif '代数式' in title:
                            questions = [
                                ('化简：3x + 2x = ?', 'fill_blank', None, '5x', 50),
                                ('当x=2时，2x+1的值是？', 'single_choice', '["3", "4", "5", "6"]', '5', 50)
                            ]
                        else:  # 几何
                            questions = [
                                ('三角形有几个内角？', 'single_choice', '["2", "3", "4", "5"]', '3', 50),
                                ('正方形有几条边？', 'fill_blank', None, '4', 50)
                            ]
                        
                        # 插入题目
                        for i, (content, q_type, options, answer, score) in enumerate(questions, 1):
                            cursor.execute("""
                                INSERT INTO questions (homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation)
                                VALUES (%s, %s, %s, %s, %s, %s, 2, %s, '[]', '请根据题目要求作答')
                            """, (homework_id, content, q_type, options, answer, score, i))
                
                # 3. 确保作业都是已发布状态
                print("\n✅ 确保作业发布状态...")
                cursor.execute("UPDATE homeworks SET is_published = 1 WHERE is_published = 0")
                updated_count = cursor.rowcount
                if updated_count > 0:
                    print(f"  发布了 {updated_count} 个作业")
                
                # 4. 检查学生用户
                if student_count == 0:
                    print("\n👨‍🎓 创建测试学生...")
                    students = [
                        ('student_001', 'student001@test.com', '张三'),
                        ('student_002', 'student002@test.com', '李四'),
                        ('student_003', 'student003@test.com', '王五'),
                    ]
                    
                    for username, email, real_name in students:
                        cursor.execute("""
                            INSERT INTO users (username, email, password_hash, real_name, role, grade, school)
                            VALUES (%s, %s, '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', %s, 'student', 7, '测试学校')
                        """, (username, email, real_name))
                        print(f"  创建学生: {real_name} ({username})")
                
                # 5. 创建作业分配
                print("\n📋 创建作业分配...")
                cursor.execute("SELECT COUNT(*) as count FROM homework_assignments")
                assignment_count = cursor.fetchone()['count']
                
                if assignment_count == 0:
                    cursor.execute("SELECT id FROM homeworks")
                    homeworks = cursor.fetchall()
                    
                    cursor.execute("SELECT id FROM users WHERE role='teacher' LIMIT 1")
                    teacher = cursor.fetchone()
                    
                    if homeworks and teacher:
                        for hw in homeworks:
                            cursor.execute("""
                                INSERT INTO homework_assignments (homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, is_active)
                                VALUES (%s, 'grade', 7, %s, NOW(), 1)
                            """, (hw['id'], teacher['id']))
                        print(f"  创建了 {len(homeworks)} 个作业分配")
                
                conn.commit()
                
                # 6. 最终检查
                print("\n📊 最终数据统计:")
                
                cursor.execute("SELECT COUNT(*) as count FROM homeworks WHERE is_published = 1")
                published_count = cursor.fetchone()['count']
                print(f"  已发布作业数: {published_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='student'")
                final_student_count = cursor.fetchone()['count']
                print(f"  学生用户数: {final_student_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_assignments WHERE is_active = 1")
                active_assignments = cursor.fetchone()['count']
                print(f"  激活的作业分配数: {active_assignments}")
                
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"  题目数: {question_count}")
                
                print("\n✅ 数据修复完成！")
                return True
                
    except Exception as e:
        print(f"❌ 数据修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_data_fix()
