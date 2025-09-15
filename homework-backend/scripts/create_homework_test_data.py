#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建作业测试数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
import json
from datetime import datetime, timedelta

def create_homework_test_data():
    """创建作业测试数据"""
    print("=== 创建作业测试数据 ===")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 创建知识点
                print("1. 创建知识点...")
                knowledge_points_sql = [
                    """INSERT IGNORE INTO knowledge_points (id, name, description, subject, grade, difficulty, parent_id, created_at, updated_at)
                       VALUES (1, '有理数运算', '有理数的加减乘除运算', '数学', 7, 2, NULL, NOW(), NOW())""",
                    """INSERT IGNORE INTO knowledge_points (id, name, description, subject, grade, difficulty, parent_id, created_at, updated_at)
                       VALUES (2, '代数式', '代数式的基本概念和运算', '数学', 7, 3, NULL, NOW(), NOW())""",
                    """INSERT IGNORE INTO knowledge_points (id, name, description, subject, grade, difficulty, parent_id, created_at, updated_at)
                       VALUES (3, '几何图形', '平面几何基础图形', '数学', 7, 2, NULL, NOW(), NOW())"""
                ]
                for sql in knowledge_points_sql:
                    cursor.execute(sql)
                
                # 2. 创建作业
                print("2. 创建作业...")
                due_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                start_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                homeworks_sql = [
                    # 王老师给1班的作业
                    f"""INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at)
                        VALUES (1, '有理数运算练习 - 七年级1班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, '{due_date}', '{start_date}', 60, 100, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW())""",
                    # 王老师给2班的作业
                    f"""INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at)
                        VALUES (2, '有理数运算练习 - 七年级2班', '数学', '练习有理数的加减乘除运算', 7, 2, 10, '{due_date}', '{start_date}', 60, 100, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW())""",
                    # 李老师给1班的作业
                    f"""INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at)
                        VALUES (3, '代数式化简 - 七年级1班', '数学', '学习代数式的化简方法', 7, 3, 11, '{due_date}', '{start_date}', 60, 100, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW())""",
                    # 李老师给2班的作业
                    f"""INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at)
                        VALUES (4, '代数式化简 - 七年级2班', '数学', '学习代数式的化简方法', 7, 3, 11, '{due_date}', '{start_date}', 60, 100, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW())""",
                    # 张老师给1班的作业
                    f"""INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at)
                        VALUES (5, '几何图形认识 - 七年级1班', '数学', '认识基本的几何图形', 7, 2, 12, '{due_date}', '{start_date}', 60, 100, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW())""",
                    # 张老师给2班的作业
                    f"""INSERT IGNORE INTO homeworks (id, title, subject, description, grade, difficulty_level, created_by, due_date, start_date, time_limit, max_score, max_attempts, is_published, auto_grade, show_answers, instructions, tags, category, created_at, updated_at)
                        VALUES (6, '几何图形认识 - 七年级2班', '数学', '认识基本的几何图形', 7, 2, 12, '{due_date}', '{start_date}', 60, 100, 2, 1, 1, 0, '请仔细阅读题目，认真作答。', '["基础", "练习"]', '课后练习', NOW(), NOW())"""
                ]
                for sql in homeworks_sql:
                    cursor.execute(sql)
                
                # 3. 创建题目
                print("3. 创建题目...")
                questions_sql = [
                    # 有理数运算题目 (作业1,2)
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (1, 1, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '请根据有理数运算法则计算。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (2, 1, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', NULL, '6', 25, 2, 2, '[1]', '请根据有理数运算法则计算。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (3, 2, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '请根据有理数运算法则计算。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (4, 2, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', NULL, '6', 25, 2, 2, '[1]', '请根据有理数运算法则计算。', NOW(), NOW())""",
                    
                    # 代数式题目 (作业3,4)
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (5, 3, '化简：3x + 2x - x = ?', 'fill_blank', NULL, '4x', 30, 3, 1, '[2]', '请根据代数式化简法则计算。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (6, 3, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '请将x=2代入代数式计算。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (7, 4, '化简：3x + 2x - x = ?', 'fill_blank', NULL, '4x', 30, 3, 1, '[2]', '请根据代数式化简法则计算。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (8, 4, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '请将x=2代入代数式计算。', NOW(), NOW())""",
                    
                    # 几何题目 (作业5,6)
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (9, 5, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (10, 5, '正方形有几条对称轴？', 'fill_blank', NULL, '4', 25, 2, 2, '[3]', '正方形有4条对称轴。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (11, 6, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形。', NOW(), NOW())""",
                    """INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation, created_at, updated_at)
                       VALUES (12, 6, '正方形有几条对称轴？', 'fill_blank', NULL, '4', 25, 2, 2, '[3]', '正方形有4条对称轴。', NOW(), NOW())"""
                ]
                for sql in questions_sql:
                    cursor.execute(sql)
                
                conn.commit()
                print("✅ 作业和题目数据创建完成")
                return True
                
    except Exception as e:
        print(f"❌ 创建作业数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_homework_test_data()
