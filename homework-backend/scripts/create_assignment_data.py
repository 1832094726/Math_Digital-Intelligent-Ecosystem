#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建作业分配和学生提交数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
import json
from datetime import datetime, timedelta
import random

def create_assignment_data():
    """创建作业分配和学生提交数据"""
    print("=== 创建作业分配和学生提交数据 ===")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 创建作业分配
                print("1. 创建作业分配...")
                assigned_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                due_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                
                assignments_sql = [
                    # 王老师的作业分配
                    f"""INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, due_date_override, is_active)
                        VALUES (1, 1, 'class', 1, 10, '{assigned_at}', '{due_date}', 1)""",
                    f"""INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, due_date_override, is_active)
                        VALUES (2, 2, 'class', 2, 10, '{assigned_at}', '{due_date}', 1)""",
                    # 李老师的作业分配
                    f"""INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, due_date_override, is_active)
                        VALUES (3, 3, 'class', 1, 11, '{assigned_at}', '{due_date}', 1)""",
                    f"""INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, due_date_override, is_active)
                        VALUES (4, 4, 'class', 2, 11, '{assigned_at}', '{due_date}', 1)""",
                    # 张老师的作业分配
                    f"""INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, due_date_override, is_active)
                        VALUES (5, 5, 'class', 1, 12, '{assigned_at}', '{due_date}', 1)""",
                    f"""INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, due_date_override, is_active)
                        VALUES (6, 6, 'class', 2, 12, '{assigned_at}', '{due_date}', 1)"""
                ]
                for sql in assignments_sql:
                    cursor.execute(sql)
                
                # 2. 创建学生提交数据
                print("2. 创建学生提交数据...")
                
                # 学生ID和班级对应关系
                students_class = {
                    20: 1, 21: 1, 22: 1,  # 1班学生
                    23: 2, 24: 2, 25: 2   # 2班学生
                }
                
                # 作业分配和班级对应关系
                assignments_class = {
                    1: 1, 3: 1, 5: 1,  # 1班的作业分配
                    2: 2, 4: 2, 6: 2   # 2班的作业分配
                }
                
                # 为每个学生创建每个作业的提交
                submission_id = 1
                for student_id, class_id in students_class.items():
                    # 找到该班级的所有作业分配
                    class_assignments = [aid for aid, cid in assignments_class.items() if cid == class_id]
                    
                    for assignment_id in class_assignments:
                        # 获取该作业的题目
                        cursor.execute("SELECT homework_id FROM homework_assignments WHERE id = %s", (assignment_id,))
                        homework_id = cursor.fetchone()['homework_id']
                        
                        cursor.execute("SELECT id, correct_answer, score FROM questions WHERE homework_id = %s", (homework_id,))
                        questions = cursor.fetchall()
                        
                        # 模拟学生答题
                        answers = {}
                        total_score = 0
                        
                        for question in questions:
                            # 80%概率答对
                            if random.random() < 0.8:
                                answers[str(question['id'])] = question['correct_answer']
                                total_score += question['score']
                            else:
                                # 错误答案
                                if question['correct_answer'] in ['0', '2', '4', '6']:
                                    answers[str(question['id'])] = str(random.randint(1, 9))
                                else:
                                    answers[str(question['id'])] = '错误答案'
                        
                        submitted_at = (datetime.now() - timedelta(days=random.randint(1, 5))).strftime('%Y-%m-%d %H:%M:%S')
                        graded_at = (datetime.now() - timedelta(days=random.randint(0, 3))).strftime('%Y-%m-%d %H:%M:%S')
                        
                        submission_sql = f"""
                            INSERT IGNORE INTO homework_submissions 
                            (id, assignment_id, student_id, answers, score, time_spent, status, submitted_at, graded_at, graded_by)
                            VALUES ({submission_id}, {assignment_id}, {student_id}, '{json.dumps(answers)}', {total_score}, 
                                   {random.randint(20, 50)}, 'graded', '{submitted_at}', '{graded_at}', 
                                   (SELECT assigned_by FROM homework_assignments WHERE id = {assignment_id}))
                        """
                        cursor.execute(submission_sql)
                        
                        print(f"✅ 学生{student_id}提交作业{assignment_id}，得分: {total_score}")
                        submission_id += 1
                
                # 3. 创建一些练习题
                print("3. 创建练习题...")
                exercises_sql = [
                    """INSERT IGNORE INTO exercises (id, title, content, question_type, options, correct_answer, difficulty, subject, grade, knowledge_points, explanation, created_at, updated_at)
                       VALUES (1, '有理数加法练习', '计算：(-5) + 3 = ?', 'single_choice', '["2", "-2", "8", "-8"]', '-2', 2, '数学', 7, '[1]', '负数加正数，绝对值大的数决定符号。', NOW(), NOW())""",
                    """INSERT IGNORE INTO exercises (id, title, content, question_type, options, correct_answer, difficulty, subject, grade, knowledge_points, explanation, created_at, updated_at)
                       VALUES (2, '代数式求值练习', '当a=3时，2a-1的值是？', 'fill_blank', NULL, '5', 3, '数学', 7, '[2]', '将a=3代入2a-1得到2×3-1=5。', NOW(), NOW())""",
                    """INSERT IGNORE INTO exercises (id, title, content, question_type, options, correct_answer, difficulty, subject, grade, knowledge_points, explanation, created_at, updated_at)
                       VALUES (3, '角度计算练习', '直角等于多少度？', 'single_choice', '["45°", "60°", "90°", "180°"]', '90°', 1, '数学', 7, '[3]', '直角等于90度。', NOW(), NOW())"""
                ]
                for sql in exercises_sql:
                    cursor.execute(sql)
                
                conn.commit()
                print("✅ 作业分配和学生提交数据创建完成")
                return True
                
    except Exception as e:
        print(f"❌ 创建分配数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_assignment_data()
