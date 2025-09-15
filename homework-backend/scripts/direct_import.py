#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接导入数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
from datetime import datetime, timedelta

def direct_import():
    """直接导入数据"""
    print("🚀 开始直接导入数据...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 删除不需要的表
                print("🗑️ 清理不需要的表...")
                cursor.execute("DROP TABLE IF EXISTS simple_questions")
                cursor.execute("DROP TABLE IF EXISTS test_questions")
                
                # 2. 创建questions表
                print("📝 创建questions表...")
                cursor.execute("DROP TABLE IF EXISTS questions")
                cursor.execute("""
                    CREATE TABLE questions (
                      id bigint(20) NOT NULL AUTO_INCREMENT COMMENT '题目ID',
                      homework_id bigint(20) NOT NULL COMMENT '作业ID',
                      content text NOT NULL COMMENT '题目内容',
                      question_type enum('single_choice','multiple_choice','fill_blank','calculation','proof','application') NOT NULL DEFAULT 'single_choice' COMMENT '题目类型',
                      options json DEFAULT NULL COMMENT '选择题选项(JSON格式)',
                      correct_answer text NOT NULL COMMENT '正确答案',
                      score decimal(5,2) NOT NULL DEFAULT '0.00' COMMENT '题目分值',
                      difficulty int(11) NOT NULL DEFAULT '1' COMMENT '难度等级(1-5)',
                      order_index int(11) NOT NULL DEFAULT '1' COMMENT '题目顺序',
                      knowledge_points json DEFAULT NULL COMMENT '关联知识点ID列表',
                      explanation text DEFAULT NULL COMMENT '题目解析',
                      created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                      updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                      PRIMARY KEY (id),
                      KEY idx_homework_id (homework_id),
                      KEY idx_question_type (question_type),
                      KEY idx_difficulty (difficulty)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目表'
                """)
                
                # 3. 插入基础数据
                print("👥 插入用户数据...")
                
                # 插入老师
                teachers = [
                    (10, 'teacher_wang', 'wang@school.com', '王老师'),
                    (11, 'teacher_li', 'li@school.com', '李老师'),
                    (12, 'teacher_zhang', 'zhang@school.com', '张老师')
                ]
                
                for teacher_id, username, email, real_name in teachers:
                    cursor.execute("""
                        INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, created_at, updated_at) 
                        VALUES (%s, %s, %s, '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', %s, 'teacher', 7, '北京市第一中学', NOW(), NOW())
                    """, (teacher_id, username, email, real_name))
                
                # 4. 插入作业数据
                print("📚 插入作业数据...")
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
                
                # 5. 插入题目数据
                print("📝 插入题目数据...")
                questions = [
                    # 有理数运算题目
                    (1, 1, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '请根据有理数运算法则计算。'),
                    (2, 1, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', None, '6', 25, 2, 2, '[1]', '请根据有理数运算法则计算。'),
                    (3, 2, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '请根据有理数运算法则计算。'),
                    (4, 2, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', None, '6', 25, 2, 2, '[1]', '请根据有理数运算法则计算。'),
                    # 代数式题目
                    (5, 3, '化简：3x + 2x - x = ?', 'fill_blank', None, '4x', 30, 3, 1, '[2]', '请根据代数式化简法则计算。'),
                    (6, 3, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '请将x=2代入代数式计算。'),
                    (7, 4, '化简：3x + 2x - x = ?', 'fill_blank', None, '4x', 30, 3, 1, '[2]', '请根据代数式化简法则计算。'),
                    (8, 4, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '请将x=2代入代数式计算。'),
                    # 几何题目
                    (9, 5, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形。'),
                    (10, 5, '正方形有几条对称轴？', 'fill_blank', None, '4', 25, 2, 2, '[3]', '正方形有4条对称轴。'),
                    (11, 6, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形。'),
                    (12, 6, '正方形有几条对称轴？', 'fill_blank', None, '4', 25, 2, 2, '[3]', '正方形有4条对称轴。')
                ]
                
                for q_data in questions:
                    cursor.execute("""
                        INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, q_data)
                
                conn.commit()
                
                # 6. 显示统计信息
                print("\n📊 数据导入统计:")
                
                cursor.execute("SELECT COUNT(*) FROM users WHERE role='teacher'")
                teacher_count = cursor.fetchone()['count']
                print(f"  👨‍🏫 教师: {teacher_count} 人")
                
                cursor.execute("SELECT COUNT(*) FROM homeworks")
                homework_count = cursor.fetchone()['count']
                print(f"  📚 作业: {homework_count} 个")
                
                cursor.execute("SELECT COUNT(*) FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"  📝 题目: {question_count} 道")
                
                # 检查questions表结构
                print("\n🔍 questions表结构:")
                cursor.execute("DESCRIBE questions")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  {col['Field']:<20} {col['Type']:<30}")
                
                print("\n✅ 数据导入完成！")
                return True
                
    except Exception as e:
        print(f"❌ 数据导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    direct_import()
