#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建丰富的测试数据
包括：3个老师、2个班级、每班3名学生、每个老师发布不同作业、学生完成作业等
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from models.user import User
import json
from datetime import datetime, timedelta
import random

def create_test_data():
    """创建测试数据"""
    print("=== 开始创建测试数据 ===")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 创建学校
                print("\n1. 创建学校...")
                school_data = {
                    'school_name': '北京市第一中学',
                    'school_code': 'BJ001',
                    'school_type': 'public',
                    'address': '北京市朝阳区教育路123号',
                    'phone': '010-12345678',
                    'principal': '张校长',
                    'established_year': 1950,
                    'description': '北京市重点中学，数学教育特色学校'
                }
                
                cursor.execute("""
                    INSERT INTO schools (school_name, school_code, school_type, address, phone, principal, established_year, description)
                    VALUES (%(school_name)s, %(school_code)s, %(school_type)s, %(address)s, %(phone)s, %(principal)s, %(established_year)s, %(description)s)
                    ON DUPLICATE KEY UPDATE school_name=VALUES(school_name)
                """, school_data)
                
                school_id = cursor.lastrowid or 1
                print(f"✅ 学校创建成功，ID: {school_id}")
                
                # 2. 创建年级
                print("\n2. 创建年级...")
                grade_data = {
                    'school_id': school_id,
                    'grade_name': '七年级',
                    'grade_level': 7,
                    'academic_year': '2024-2025',
                    'grade_director': '李主任'
                }
                
                cursor.execute("""
                    INSERT INTO grades (school_id, grade_name, grade_level, academic_year, grade_director)
                    VALUES (%(school_id)s, %(grade_name)s, %(grade_level)s, %(academic_year)s, %(grade_director)s)
                    ON DUPLICATE KEY UPDATE grade_name=VALUES(grade_name)
                """, grade_data)
                
                grade_id = cursor.lastrowid or 1
                print(f"✅ 年级创建成功，ID: {grade_id}")
                
                # 3. 创建3个老师
                print("\n3. 创建3个老师...")
                teachers = []
                teacher_data = [
                    {
                        'username': 'teacher_wang',
                        'email': 'wang@school.com',
                        'password': 'password',
                        'real_name': '王老师',
                        'role': 'teacher',
                        'grade': 7,
                        'school': '北京市第一中学',
                        'phone': '13800001001',
                        'profile': json.dumps({
                            'subject': '数学',
                            'teaching_years': 10,
                            'specialty': '代数教学'
                        })
                    },
                    {
                        'username': 'teacher_li',
                        'email': 'li@school.com', 
                        'password': 'password',
                        'real_name': '李老师',
                        'role': 'teacher',
                        'grade': 7,
                        'school': '北京市第一中学',
                        'phone': '13800001002',
                        'profile': json.dumps({
                            'subject': '数学',
                            'teaching_years': 8,
                            'specialty': '几何教学'
                        })
                    },
                    {
                        'username': 'teacher_zhang',
                        'email': 'zhang@school.com',
                        'password': 'password', 
                        'real_name': '张老师',
                        'role': 'teacher',
                        'grade': 7,
                        'school': '北京市第一中学',
                        'phone': '13800001003',
                        'profile': json.dumps({
                            'subject': '数学',
                            'teaching_years': 12,
                            'specialty': '应用题教学'
                        })
                    }
                ]
                
                for teacher in teacher_data:
                    try:
                        user = User.create(teacher)
                        teachers.append(user)
                        print(f"✅ 老师创建成功: {teacher['real_name']} (ID: {user.id})")
                    except Exception as e:
                        print(f"⚠️ 老师可能已存在: {teacher['real_name']}")
                        # 获取已存在的老师
                        existing_user = User.get_by_username(teacher['username'])
                        if existing_user:
                            teachers.append(existing_user)
                
                # 4. 创建2个班级
                print("\n4. 创建2个班级...")
                classes = []
                class_data = [
                    {
                        'school_id': school_id,
                        'grade_id': grade_id,
                        'class_name': '七年级1班',
                        'class_code': 'G7C1',
                        'head_teacher_id': teachers[0].id,
                        'class_size': 3,
                        'classroom': '教学楼A101'
                    },
                    {
                        'school_id': school_id,
                        'grade_id': grade_id,
                        'class_name': '七年级2班',
                        'class_code': 'G7C2', 
                        'head_teacher_id': teachers[1].id,
                        'class_size': 3,
                        'classroom': '教学楼A102'
                    }
                ]
                
                for class_info in class_data:
                    cursor.execute("""
                        INSERT INTO classes (school_id, grade_id, class_name, class_code, head_teacher_id, class_size, classroom)
                        VALUES (%(school_id)s, %(grade_id)s, %(class_name)s, %(class_code)s, %(head_teacher_id)s, %(class_size)s, %(classroom)s)
                        ON DUPLICATE KEY UPDATE class_name=VALUES(class_name)
                    """, class_info)
                    
                    class_id = cursor.lastrowid or len(classes) + 1
                    classes.append({'id': class_id, **class_info})
                    print(f"✅ 班级创建成功: {class_info['class_name']} (ID: {class_id})")
                
                # 5. 创建学生（每班3名）
                print("\n5. 创建学生...")
                students = []
                student_names = [
                    ['小明', '小红', '小刚'],  # 1班学生
                    ['小华', '小丽', '小强']   # 2班学生
                ]

                for class_idx, class_info in enumerate(classes):
                    for student_idx, name in enumerate(student_names[class_idx]):
                        student_data = {
                            'username': f'student_{class_idx+1}_{student_idx+1}',
                            'email': f'student{class_idx+1}{student_idx+1}@school.com',
                            'password': 'password',
                            'real_name': name,
                            'role': 'student',
                            'grade': 7,
                            'school': '北京市第一中学',
                            'class_name': class_info['class_name'],
                            'student_id': f'2024{class_idx+1:02d}{student_idx+1:02d}',
                            'profile': json.dumps({
                                'interests': ['数学', '科学'],
                                'learning_style': random.choice(['视觉型', '听觉型', '动手型'])
                            })
                        }

                        try:
                            student = User.create(student_data)
                            students.append({'user': student, 'class_id': class_info['id']})
                            print(f"✅ 学生创建成功: {name} (ID: {student.id})")

                            # 添加到班级学生表
                            cursor.execute("""
                                INSERT INTO class_students (class_id, student_id, enrollment_date, is_active)
                                VALUES (%s, %s, %s, 1)
                                ON DUPLICATE KEY UPDATE is_active=1
                            """, (class_info['id'], student.id, datetime.now().date()))

                        except Exception as e:
                            print(f"⚠️ 学生可能已存在: {name}")
                            existing_student = User.get_by_username(student_data['username'])
                            if existing_student:
                                students.append({'user': existing_student, 'class_id': class_info['id']})

                conn.commit()
                print("✅ 学生数据创建完成")

                return {
                    'school_id': school_id,
                    'grade_id': grade_id,
                    'teachers': teachers,
                    'classes': classes,
                    'students': students
                }

    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        raise

def create_homework_data(base_data):
    """创建作业相关数据"""
    print("\n=== 创建作业数据 ===")

    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:

                # 6. 创建知识点
                print("\n6. 创建知识点...")
                knowledge_points = [
                    {
                        'name': '有理数运算',
                        'description': '有理数的加减乘除运算',
                        'subject': '数学',
                        'grade': 7,
                        'difficulty': 2,
                        'parent_id': None
                    },
                    {
                        'name': '代数式',
                        'description': '代数式的基本概念和运算',
                        'subject': '数学',
                        'grade': 7,
                        'difficulty': 3,
                        'parent_id': None
                    },
                    {
                        'name': '几何图形',
                        'description': '平面几何基础图形',
                        'subject': '数学',
                        'grade': 7,
                        'difficulty': 2,
                        'parent_id': None
                    }
                ]

                kp_ids = []
                for kp in knowledge_points:
                    cursor.execute("""
                        INSERT INTO knowledge_points (name, description, subject, grade, difficulty, parent_id)
                        VALUES (%(name)s, %(description)s, %(subject)s, %(grade)s, %(difficulty)s, %(parent_id)s)
                        ON DUPLICATE KEY UPDATE description=VALUES(description)
                    """, kp)
                    kp_id = cursor.lastrowid or len(kp_ids) + 1
                    kp_ids.append(kp_id)
                    print(f"✅ 知识点创建: {kp['name']} (ID: {kp_id})")

                # 7. 创建作业（每个老师针对不同班级发布作业）
                print("\n7. 创建作业...")
                homeworks = []
                homework_templates = [
                    {
                        'title': '有理数运算练习',
                        'subject': '数学',
                        'description': '练习有理数的加减乘除运算',
                        'difficulty_level': 2,
                        'knowledge_point_id': kp_ids[0]
                    },
                    {
                        'title': '代数式化简',
                        'subject': '数学',
                        'description': '学习代数式的化简方法',
                        'difficulty_level': 3,
                        'knowledge_point_id': kp_ids[1]
                    },
                    {
                        'title': '几何图形认识',
                        'subject': '数学',
                        'description': '认识基本的几何图形',
                        'difficulty_level': 2,
                        'knowledge_point_id': kp_ids[2]
                    }
                ]

                # 每个老师为每个班级创建一个作业
                for teacher_idx, teacher in enumerate(base_data['teachers']):
                    for class_idx, class_info in enumerate(base_data['classes']):
                        hw_template = homework_templates[teacher_idx]

                        homework_data = {
                            'title': f"{hw_template['title']} - {class_info['class_name']}",
                            'subject': hw_template['subject'],
                            'description': hw_template['description'],
                            'grade': 7,
                            'difficulty_level': hw_template['difficulty_level'],
                            'created_by': teacher.id,
                            'due_date': datetime.now() + timedelta(days=7),
                            'start_date': datetime.now(),
                            'time_limit': 60,
                            'max_score': 100,
                            'max_attempts': 2,
                            'is_published': True,
                            'auto_grade': True,
                            'show_answers': False,
                            'instructions': '请仔细阅读题目，认真作答。',
                            'tags': json.dumps(['基础', '练习']),
                            'category': '课后练习'
                        }

                        cursor.execute("""
                            INSERT INTO homeworks (title, subject, description, grade, difficulty_level, created_by,
                                                 due_date, start_date, time_limit, max_score, max_attempts,
                                                 is_published, auto_grade, show_answers, instructions, tags, category)
                            VALUES (%(title)s, %(subject)s, %(description)s, %(grade)s, %(difficulty_level)s, %(created_by)s,
                                   %(due_date)s, %(start_date)s, %(time_limit)s, %(max_score)s, %(max_attempts)s,
                                   %(is_published)s, %(auto_grade)s, %(show_answers)s, %(instructions)s, %(tags)s, %(category)s)
                        """, homework_data)

                        homework_id = cursor.lastrowid
                        homeworks.append({
                            'id': homework_id,
                            'teacher_id': teacher.id,
                            'class_id': class_info['id'],
                            'knowledge_point_id': hw_template['knowledge_point_id'],
                            **homework_data
                        })
                        print(f"✅ 作业创建: {homework_data['title']} (ID: {homework_id})")

                # 8. 创建题目
                print("\n8. 创建题目...")
                question_templates = [
                    # 有理数运算题目
                    {
                        'content': '计算：(-3) + 5 - 2 = ?',
                        'question_type': 'single_choice',
                        'options': json.dumps(['0', '2', '4', '-10']),
                        'correct_answer': '0',
                        'score': 10,
                        'difficulty': 2,
                        'knowledge_point_idx': 0
                    },
                    {
                        'content': '计算：(-2) × 3 ÷ (-1) = ?',
                        'question_type': 'fill_blank',
                        'options': None,
                        'correct_answer': '6',
                        'score': 15,
                        'difficulty': 2,
                        'knowledge_point_idx': 0
                    },
                    # 代数式题目
                    {
                        'content': '化简：3x + 2x - x = ?',
                        'question_type': 'fill_blank',
                        'options': None,
                        'correct_answer': '4x',
                        'score': 15,
                        'difficulty': 3,
                        'knowledge_point_idx': 1
                    },
                    {
                        'content': '当x=2时，代数式2x+1的值是多少？',
                        'question_type': 'single_choice',
                        'options': json.dumps(['3', '4', '5', '6']),
                        'correct_answer': '5',
                        'score': 10,
                        'difficulty': 3,
                        'knowledge_point_idx': 1
                    },
                    # 几何题目
                    {
                        'content': '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？',
                        'question_type': 'single_choice',
                        'options': json.dumps(['直角三角形', '等腰三角形', '等边三角形', '钝角三角形']),
                        'correct_answer': '等边三角形',
                        'score': 10,
                        'difficulty': 2,
                        'knowledge_point_idx': 2
                    }
                ]

                # 为每个作业创建题目
                for homework in homeworks:
                    # 根据作业的知识点选择相应的题目
                    relevant_questions = [q for q in question_templates
                                        if q['knowledge_point_idx'] == homework['knowledge_point_id'] - kp_ids[0]]

                    for q_idx, question_template in enumerate(relevant_questions):
                        question_data = {
                            'homework_id': homework['id'],
                            'content': question_template['content'],
                            'question_type': question_template['question_type'],
                            'options': question_template['options'],
                            'correct_answer': question_template['correct_answer'],
                            'score': question_template['score'],
                            'difficulty': question_template['difficulty'],
                            'order_index': q_idx + 1,
                            'knowledge_points': json.dumps([kp_ids[question_template['knowledge_point_idx']]]),
                            'explanation': '请根据所学知识仔细计算。'
                        }

                        cursor.execute("""
                            INSERT INTO questions (homework_id, content, question_type, options, correct_answer,
                                                 score, difficulty, order_index, knowledge_points, explanation)
                            VALUES (%(homework_id)s, %(content)s, %(question_type)s, %(options)s, %(correct_answer)s,
                                   %(score)s, %(difficulty)s, %(order_index)s, %(knowledge_points)s, %(explanation)s)
                        """, question_data)

                        question_id = cursor.lastrowid
                        print(f"✅ 题目创建: 作业{homework['id']} - 题目{q_idx+1} (ID: {question_id})")

                # 9. 创建作业分配
                print("\n9. 创建作业分配...")
                assignments = []
                for homework in homeworks:
                    assignment_data = {
                        'homework_id': homework['id'],
                        'assigned_to_type': 'class',
                        'assigned_to_id': homework['class_id'],
                        'assigned_by': homework['teacher_id'],
                        'assigned_at': datetime.now(),
                        'due_date_override': homework['due_date'],
                        'is_active': True
                    }

                    cursor.execute("""
                        INSERT INTO homework_assignments (homework_id, assigned_to_type, assigned_to_id, assigned_by,
                                                         assigned_at, due_date_override, is_active)
                        VALUES (%(homework_id)s, %(assigned_to_type)s, %(assigned_to_id)s, %(assigned_by)s,
                               %(assigned_at)s, %(due_date_override)s, %(is_active)s)
                    """, assignment_data)

                    assignment_id = cursor.lastrowid
                    assignments.append({'id': assignment_id, **assignment_data})
                    print(f"✅ 作业分配: 作业{homework['id']} -> 班级{homework['class_id']} (ID: {assignment_id})")

                # 10. 创建学生作业提交
                print("\n10. 创建学生作业提交...")
                for assignment in assignments:
                    # 获取该班级的学生
                    class_students = [s for s in base_data['students'] if s['class_id'] == assignment['assigned_to_id']]

                    for student_info in class_students:
                        student = student_info['user']

                        # 模拟学生答题（随机生成答案）
                        answers = {}
                        cursor.execute("SELECT id, correct_answer FROM questions WHERE homework_id = %s",
                                     (assignment['homework_id'],))
                        questions = cursor.fetchall()

                        total_score = 0
                        for question in questions:
                            # 80%概率答对
                            if random.random() < 0.8:
                                answers[str(question['id'])] = question['correct_answer']
                                cursor.execute("SELECT score FROM questions WHERE id = %s", (question['id'],))
                                score = cursor.fetchone()['score']
                                total_score += score
                            else:
                                answers[str(question['id'])] = '错误答案'

                        submission_data = {
                            'assignment_id': assignment['id'],
                            'student_id': student.id,
                            'answers': json.dumps(answers),
                            'score': total_score,
                            'time_spent': random.randint(20, 50),  # 20-50分钟
                            'status': 'graded',
                            'submitted_at': datetime.now() - timedelta(days=random.randint(1, 5)),
                            'graded_at': datetime.now() - timedelta(days=random.randint(0, 3)),
                            'graded_by': assignment['assigned_by']
                        }

                        cursor.execute("""
                            INSERT INTO homework_submissions (assignment_id, student_id, answers, score, time_spent,
                                                             status, submitted_at, graded_at, graded_by)
                            VALUES (%(assignment_id)s, %(student_id)s, %(answers)s, %(score)s, %(time_spent)s,
                                   %(status)s, %(submitted_at)s, %(graded_at)s, %(graded_by)s)
                        """, submission_data)

                        submission_id = cursor.lastrowid
                        print(f"✅ 学生提交: {student.real_name} -> 作业{assignment['homework_id']} (分数: {total_score})")

                conn.commit()
                print("\n✅ 所有测试数据创建完成！")

    except Exception as e:
        print(f"❌ 创建作业数据失败: {e}")
        raise

if __name__ == "__main__":
    base_data = create_test_data()
    create_homework_data(base_data)
