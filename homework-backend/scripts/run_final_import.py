#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行最终数据导入
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from datetime import datetime, timedelta

def run_final_import():
    """运行最终数据导入"""
    print("🚀 开始最终数据导入...")
    
    try:
        # 连接数据库
        connection = pymysql.connect(
            host='obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud',
            port=3306,
            user='hcj',
            password='Xv0Mu8_:',
            database='testccnu',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            
            # 1. 清理并创建questions表
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
            
            # 2. 插入教师数据
            print("👨‍🏫 插入教师数据...")
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
            
            # 3. 插入学生数据
            print("👨‍🎓 插入学生数据...")
            students = [
                (20, 'student_001', 'student001@school.com', '张三'),
                (21, 'student_002', 'student002@school.com', '李四'),
                (22, 'student_003', 'student003@school.com', '王五'),
                (23, 'student_004', 'student004@school.com', '赵六'),
                (24, 'student_005', 'student005@school.com', '钱七'),
                (25, 'student_006', 'student006@school.com', '孙八')
            ]
            
            for student_id, username, email, real_name in students:
                cursor.execute("""
                    INSERT IGNORE INTO users (id, username, email, password_hash, real_name, role, grade, school, created_at, updated_at) 
                    VALUES (%s, %s, %s, '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VcSAg9S6O', %s, 'student', 7, '北京市第一中学', NOW(), NOW())
                """, (student_id, username, email, real_name))
            
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
                (1, 1, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '根据有理数运算法则计算'),
                (2, 1, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', None, '6', 25, 2, 2, '[1]', '根据有理数运算法则计算'),
                (3, 2, '计算：(-3) + 5 - 2 = ?', 'single_choice', '["0", "2", "4", "-10"]', '0', 25, 2, 1, '[1]', '根据有理数运算法则计算'),
                (4, 2, '计算：(-2) × 3 ÷ (-1) = ?', 'fill_blank', None, '6', 25, 2, 2, '[1]', '根据有理数运算法则计算'),
                # 代数式题目
                (5, 3, '化简：3x + 2x - x = ?', 'fill_blank', None, '4x', 30, 3, 1, '[2]', '合并同类项计算'),
                (6, 3, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '代入计算'),
                (7, 4, '化简：3x + 2x - x = ?', 'fill_blank', None, '4x', 30, 3, 1, '[2]', '合并同类项计算'),
                (8, 4, '当x=2时，代数式2x+1的值是多少？', 'single_choice', '["3", "4", "5", "6"]', '5', 20, 3, 2, '[2]', '代入计算'),
                # 几何题目
                (9, 5, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形'),
                (10, 5, '正方形有几条对称轴？', 'fill_blank', None, '4', 25, 2, 2, '[3]', '正方形有4条对称轴'),
                (11, 6, '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？', 'single_choice', '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]', '等边三角形', 25, 2, 1, '[3]', '三个角都是60°的三角形是等边三角形'),
                (12, 6, '正方形有几条对称轴？', 'fill_blank', None, '4', 25, 2, 2, '[3]', '正方形有4条对称轴')
            ]
            
            for q_data in questions:
                cursor.execute("""
                    INSERT IGNORE INTO questions (id, homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, q_data)
            
            # 6. 插入作业分配
            print("📋 插入作业分配...")
            assignments = [
                (1, 1, 'class', 1, 10),  # 王老师给1班布置有理数作业
                (2, 2, 'class', 2, 10),  # 王老师给2班布置有理数作业
                (3, 3, 'class', 1, 11),  # 李老师给1班布置代数式作业
                (4, 4, 'class', 2, 11),  # 李老师给2班布置代数式作业
                (5, 5, 'class', 1, 12),  # 张老师给1班布置几何作业
                (6, 6, 'class', 2, 12)   # 张老师给2班布置几何作业
            ]
            
            for assign_id, hw_id, assign_type, assign_to, assign_by in assignments:
                cursor.execute("""
                    INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, is_active)
                    VALUES (%s, %s, %s, %s, %s, NOW(), 1)
                """, (assign_id, hw_id, assign_type, assign_to, assign_by))
            
            # 7. 插入学生提交记录
            print("📤 插入学生提交记录...")
            submissions = [
                # 学生20 (张三, 1班) - 优秀学生，大部分全对
                (1, 20, '{"1": "0", "2": "6"}', 50, 25),  # 有理数作业 - 全对
                (3, 20, '{"5": "4x", "6": "5"}', 50, 30),  # 代数式作业 - 全对
                (5, 20, '{"9": "等边三角形", "10": "4"}', 50, 20),  # 几何作业 - 全对
                # 学生21 (李四, 1班) - 中等学生
                (1, 21, '{"1": "2", "2": "6"}', 25, 28),  # 有理数作业 - 第一题错
                (3, 21, '{"5": "4x", "6": "4"}', 30, 35),  # 代数式作业 - 第二题错
                (5, 21, '{"9": "等边三角形", "10": "4"}', 50, 22),  # 几何作业 - 全对
                # 学生22 (王五, 1班) - 需要提高的学生
                (1, 22, '{"1": "0", "2": "错误"}', 25, 32),  # 有理数作业 - 第二题错
                (3, 22, '{"5": "5x", "6": "5"}', 20, 40),  # 代数式作业 - 第一题错
                (5, 22, '{"9": "直角三角形", "10": "4"}', 25, 25),  # 几何作业 - 第一题错
                # 学生23 (赵六, 2班) - 优秀学生
                (2, 23, '{"3": "0", "4": "6"}', 50, 26),  # 有理数作业 - 全对
                (4, 23, '{"7": "4x", "8": "5"}', 50, 32),  # 代数式作业 - 全对
                (6, 23, '{"11": "等边三角形", "12": "4"}', 50, 21),  # 几何作业 - 全对
                # 学生24 (钱七, 2班) - 中等学生
                (2, 24, '{"3": "4", "4": "6"}', 25, 30),  # 有理数作业 - 第一题错
                (4, 24, '{"7": "4x", "8": "3"}', 30, 38),  # 代数式作业 - 第二题错
                (6, 24, '{"11": "等腰三角形", "12": "4"}', 25, 27),  # 几何作业 - 第一题错
                # 学生25 (孙八, 2班) - 需要提高的学生
                (2, 25, '{"3": "0", "4": "错误"}', 25, 35),  # 有理数作业 - 第二题错
                (4, 25, '{"7": "3x", "8": "5"}', 20, 42),  # 代数式作业 - 第一题错
                (6, 25, '{"11": "等边三角形", "12": "3"}', 25, 28)   # 几何作业 - 第二题错
            ]
            
            for assign_id, student_id, answers, score, time_spent in submissions:
                cursor.execute("""
                    INSERT IGNORE INTO homework_submissions (assignment_id, student_id, answers, score, time_spent, status, submitted_at)
                    VALUES (%s, %s, %s, %s, %s, 'submitted', NOW())
                """, (assign_id, student_id, answers, score, time_spent))
            
            # 8. 插入知识点数据
            print("🧠 插入知识点数据...")
            knowledge_points = [
                (1, '有理数运算', '有理数的加减乘除运算法则', '数学', 7),
                (2, '代数式', '代数式的化简和求值', '数学', 7),
                (3, '几何图形', '基本几何图形的认识和性质', '数学', 7)
            ]
            
            for kp_id, name, description, subject, grade in knowledge_points:
                cursor.execute("""
                    INSERT IGNORE INTO knowledge_points (id, name, description, subject, grade, parent_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NULL, NOW(), NOW())
                """, (kp_id, name, description, subject, grade))
            
            connection.commit()
            
            # 9. 显示导入结果
            print("\n📊 数据导入统计:")
            
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='teacher'")
            teacher_count = cursor.fetchone()['count']
            print(f"  👨‍🏫 教师: {teacher_count} 人")
            
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='student'")
            student_count = cursor.fetchone()['count']
            print(f"  👨‍🎓 学生: {student_count} 人")
            
            cursor.execute("SELECT COUNT(*) as count FROM homeworks")
            homework_count = cursor.fetchone()['count']
            print(f"  📚 作业: {homework_count} 个")
            
            cursor.execute("SELECT COUNT(*) as count FROM questions")
            question_count = cursor.fetchone()['count']
            print(f"  📝 题目: {question_count} 道")
            
            cursor.execute("SELECT COUNT(*) as count FROM homework_submissions")
            submission_count = cursor.fetchone()['count']
            print(f"  📤 提交: {submission_count} 份")
            
            cursor.execute("SELECT COUNT(*) as count FROM knowledge_points")
            kp_count = cursor.fetchone()['count']
            print(f"  🧠 知识点: {kp_count} 个")
            
            print("\n✅ 最终数据导入完成！")
            return True
            
    except Exception as e:
        print(f"❌ 数据导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    run_final_import()
