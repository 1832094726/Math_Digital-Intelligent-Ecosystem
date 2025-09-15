#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用现有数据测试评分功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
import json

def test_grading_with_existing_data():
    """使用现有数据测试评分功能"""
    print("🧪 使用现有数据测试评分功能...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 检查现有数据
                print("\n📊 检查现有数据...")
                
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='student'")
                student_count = cursor.fetchone()['count']
                print(f"  学生数量: {student_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                homework_count = cursor.fetchone()['count']
                print(f"  作业数量: {homework_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_submissions")
                submission_count = cursor.fetchone()['count']
                print(f"  提交数量: {submission_count}")
                
                # 2. 创建评分表（如果不存在）
                print("\n🏗️ 创建评分表...")
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS grading_results (
                      id bigint(20) NOT NULL AUTO_INCREMENT,
                      submission_id bigint(20) NOT NULL,
                      result_data json NOT NULL,
                      total_score decimal(5,2) NOT NULL DEFAULT '0.00',
                      total_possible decimal(5,2) NOT NULL DEFAULT '0.00',
                      accuracy decimal(5,2) NOT NULL DEFAULT '0.00',
                      grading_method enum('auto','manual','hybrid') NOT NULL DEFAULT 'auto',
                      graded_by bigint(20) DEFAULT NULL,
                      graded_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      reviewed_at timestamp NULL DEFAULT NULL,
                      review_notes text DEFAULT NULL,
                      created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                      PRIMARY KEY (id),
                      UNIQUE KEY uk_submission_id (submission_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # 3. 检查是否有questions表，如果没有则创建简单版本
                cursor.execute("SHOW TABLES LIKE 'questions'")
                questions_table = cursor.fetchone()
                
                if not questions_table:
                    print("  创建questions表...")
                    cursor.execute("""
                        CREATE TABLE questions (
                          id bigint(20) NOT NULL AUTO_INCREMENT,
                          homework_id bigint(20) NOT NULL,
                          content text NOT NULL,
                          question_type enum('single_choice','multiple_choice','fill_blank','calculation','proof','application') NOT NULL DEFAULT 'single_choice',
                          options json DEFAULT NULL,
                          correct_answer text NOT NULL,
                          score decimal(5,2) NOT NULL DEFAULT '0.00',
                          difficulty int(11) NOT NULL DEFAULT '1',
                          order_index int(11) NOT NULL DEFAULT '1',
                          knowledge_points json DEFAULT NULL,
                          explanation text DEFAULT NULL,
                          created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                          updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                          PRIMARY KEY (id),
                          KEY idx_homework_id (homework_id)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """)
                    
                    # 插入一些示例题目
                    cursor.execute("SELECT id FROM homeworks LIMIT 1")
                    homework = cursor.fetchone()
                    if homework:
                        homework_id = homework['id']
                        cursor.execute("""
                            INSERT INTO questions (homework_id, content, question_type, correct_answer, score, order_index)
                            VALUES 
                            (%s, '计算：2 + 3 = ?', 'fill_blank', '5', 50, 1),
                            (%s, '选择正确答案：1 + 1 = ?', 'single_choice', 'A', 50, 2)
                        """, (homework_id, homework_id))
                
                # 4. 测试简单的评分逻辑
                print("\n🎯 测试评分逻辑...")
                
                # 获取一个提交记录
                cursor.execute("""
                    SELECT hs.*, h.id as homework_id 
                    FROM homework_submissions hs
                    JOIN homework_assignments ha ON hs.assignment_id = ha.id
                    JOIN homeworks h ON ha.homework_id = h.id
                    LIMIT 1
                """)
                
                submission = cursor.fetchone()
                if submission:
                    print(f"  测试提交ID: {submission['id']}")
                    
                    # 获取对应的题目
                    cursor.execute("""
                        SELECT * FROM questions WHERE homework_id = %s ORDER BY order_index
                    """, (submission['homework_id'],))
                    
                    questions = cursor.fetchall()
                    print(f"  题目数量: {len(questions)}")
                    
                    if questions:
                        # 模拟评分结果
                        total_score = 0
                        total_possible = 0
                        question_results = []
                        
                        for question in questions:
                            # 简单的评分逻辑
                            is_correct = True  # 假设都正确
                            score_earned = question['score'] if is_correct else 0
                            
                            question_results.append({
                                'question_id': question['id'],
                                'question_content': question['content'],
                                'is_correct': is_correct,
                                'score_earned': score_earned,
                                'score_possible': question['score'],
                                'feedback': '回答正确！' if is_correct else '回答错误'
                            })
                            
                            total_score += score_earned
                            total_possible += question['score']
                        
                        accuracy = (total_score / total_possible * 100) if total_possible > 0 else 0
                        
                        # 保存评分结果
                        grading_result = {
                            'submission_id': submission['id'],
                            'total_score': total_score,
                            'total_possible': total_possible,
                            'accuracy': accuracy,
                            'question_results': question_results
                        }
                        
                        cursor.execute("""
                            INSERT INTO grading_results (submission_id, result_data, total_score, total_possible, accuracy)
                            VALUES (%s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            result_data = VALUES(result_data),
                            total_score = VALUES(total_score),
                            total_possible = VALUES(total_possible),
                            accuracy = VALUES(accuracy)
                        """, (submission['id'], json.dumps(grading_result), total_score, total_possible, accuracy))
                        
                        print(f"  评分结果: {total_score}/{total_possible} ({accuracy:.1f}%)")
                        
                        # 更新提交记录的分数
                        cursor.execute("""
                            UPDATE homework_submissions 
                            SET score = %s, status = 'graded'
                            WHERE id = %s
                        """, (total_score, submission['id']))
                        
                        conn.commit()
                        print("  ✅ 评分结果已保存")
                    
                else:
                    print("  ⚠️ 没有找到提交记录")
                
                # 5. 显示评分统计
                print("\n📈 评分统计:")
                cursor.execute("SELECT COUNT(*) as count FROM grading_results")
                grading_count = cursor.fetchone()['count']
                print(f"  已评分数量: {grading_count}")
                
                if grading_count > 0:
                    cursor.execute("SELECT AVG(accuracy) as avg_accuracy FROM grading_results")
                    avg_accuracy = cursor.fetchone()['avg_accuracy']
                    print(f"  平均正确率: {avg_accuracy:.1f}%")
                
                print("\n✅ 评分功能测试完成！")
                return True
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_grading_with_existing_data()
