#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复题目数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
import json

def fix_questions():
    """修复题目数据"""
    print("📝 修复题目数据...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 检查现有题目
                print("\n🔍 检查现有题目...")
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"现有题目数: {question_count}")
                
                # 2. 检查作业和题目的关联
                cursor.execute("""
                    SELECT h.id, h.title, COUNT(q.id) as question_count
                    FROM homeworks h
                    LEFT JOIN questions q ON h.id = q.homework_id
                    GROUP BY h.id, h.title
                    ORDER BY h.id
                """)
                homework_questions = cursor.fetchall()
                
                print("\n📚 作业题目统计:")
                for hw in homework_questions:
                    print(f"  作业{hw['id']}: {hw['title']} - {hw['question_count']}道题目")
                
                # 3. 为没有题目的作业创建题目
                print("\n➕ 为缺少题目的作业创建题目...")
                
                for hw in homework_questions:
                    if hw['question_count'] == 0:
                        homework_id = hw['id']
                        title = hw['title']
                        print(f"  为作业 '{title}' 创建题目...")
                        
                        # 根据作业标题确定题目类型
                        if '有理数' in title:
                            questions = [
                                {
                                    'content': '计算：(-3) + 5 - 2 = ?',
                                    'type': 'single_choice',
                                    'options': '["0", "2", "4", "-10"]',
                                    'answer': '0',
                                    'score': 25,
                                    'explanation': '根据有理数运算法则：(-3) + 5 - 2 = 2 - 2 = 0'
                                },
                                {
                                    'content': '计算：(-2) × 3 ÷ (-1) = ?',
                                    'type': 'fill_blank',
                                    'options': None,
                                    'answer': '6',
                                    'score': 25,
                                    'explanation': '根据有理数运算法则：(-2) × 3 ÷ (-1) = -6 ÷ (-1) = 6'
                                }
                            ]
                        elif '代数式' in title:
                            questions = [
                                {
                                    'content': '化简：3x + 2x - x = ?',
                                    'type': 'fill_blank',
                                    'options': None,
                                    'answer': '4x',
                                    'score': 30,
                                    'explanation': '合并同类项：3x + 2x - x = (3+2-1)x = 4x'
                                },
                                {
                                    'content': '当x=2时，代数式2x+1的值是多少？',
                                    'type': 'single_choice',
                                    'options': '["3", "4", "5", "6"]',
                                    'answer': '5',
                                    'score': 20,
                                    'explanation': '将x=2代入：2×2+1 = 4+1 = 5'
                                }
                            ]
                        elif '几何' in title:
                            questions = [
                                {
                                    'content': '一个三角形的三个内角分别是60°、60°、60°，这是什么三角形？',
                                    'type': 'single_choice',
                                    'options': '["直角三角形", "等腰三角形", "等边三角形", "钝角三角形"]',
                                    'answer': '等边三角形',
                                    'score': 25,
                                    'explanation': '三个角都是60°的三角形是等边三角形'
                                },
                                {
                                    'content': '正方形有几条对称轴？',
                                    'type': 'fill_blank',
                                    'options': None,
                                    'answer': '4',
                                    'score': 25,
                                    'explanation': '正方形有4条对称轴：2条对角线和2条中线'
                                }
                            ]
                        else:
                            # 默认数学题目
                            questions = [
                                {
                                    'content': '计算：2 + 3 = ?',
                                    'type': 'fill_blank',
                                    'options': None,
                                    'answer': '5',
                                    'score': 50,
                                    'explanation': '基本加法运算'
                                },
                                {
                                    'content': '下列哪个是偶数？',
                                    'type': 'single_choice',
                                    'options': '["1", "2", "3", "5"]',
                                    'answer': '2',
                                    'score': 50,
                                    'explanation': '偶数是能被2整除的数'
                                }
                            ]
                        
                        # 插入题目
                        for i, q in enumerate(questions, 1):
                            cursor.execute("""
                                INSERT INTO questions (homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation)
                                VALUES (%s, %s, %s, %s, %s, %s, 2, %s, '[]', %s)
                            """, (
                                homework_id,
                                q['content'],
                                q['type'],
                                q['options'],
                                q['answer'],
                                q['score'],
                                i,
                                q['explanation']
                            ))
                        
                        print(f"    ✅ 创建了 {len(questions)} 道题目")
                
                # 4. 检查questions表是否存在，如果不存在则创建
                cursor.execute("SHOW TABLES LIKE 'questions'")
                questions_table = cursor.fetchone()
                
                if not questions_table:
                    print("\n🏗️ 创建questions表...")
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
                    print("✅ questions表创建完成")
                
                conn.commit()
                
                # 5. 最终检查
                print("\n📊 最终题目统计:")
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                final_question_count = cursor.fetchone()['count']
                print(f"总题目数: {final_question_count}")
                
                cursor.execute("""
                    SELECT h.id, h.title, COUNT(q.id) as question_count
                    FROM homeworks h
                    LEFT JOIN questions q ON h.id = q.homework_id
                    GROUP BY h.id, h.title
                    ORDER BY h.id
                """)
                final_homework_questions = cursor.fetchall()
                
                print("\n📚 最终作业题目统计:")
                for hw in final_homework_questions:
                    print(f"  作业{hw['id']}: {hw['title']} - {hw['question_count']}道题目")
                
                # 6. 测试题目API
                print("\n🧪 测试题目数据...")
                cursor.execute("""
                    SELECT q.*, h.title as homework_title
                    FROM questions q
                    JOIN homeworks h ON q.homework_id = h.id
                    LIMIT 3
                """)
                sample_questions = cursor.fetchall()
                
                print("示例题目:")
                for q in sample_questions:
                    print(f"  {q['homework_title']} - {q['content']} ({q['question_type']})")
                
                print("\n✅ 题目数据修复完成！")
                return True
                
    except Exception as e:
        print(f"❌ 题目修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_questions()
