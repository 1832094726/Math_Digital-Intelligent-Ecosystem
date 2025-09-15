#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单修复题目数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def simple_fix_questions():
    """简单修复题目数据"""
    print("🔧 简单修复题目数据...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 检查现有题目
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"现有题目数: {question_count}")
                
                # 2. 获取所有作业
                cursor.execute("SELECT id, title FROM homeworks ORDER BY id")
                homeworks = cursor.fetchall()
                print(f"现有作业数: {len(homeworks)}")
                
                if not homeworks:
                    print("❌ 没有作业数据")
                    return False
                
                # 3. 为每个作业检查并创建题目
                for homework in homeworks:
                    homework_id = homework['id']
                    title = homework['title']
                    
                    # 检查这个作业是否有题目
                    cursor.execute("SELECT COUNT(*) as count FROM questions WHERE homework_id = %s", (homework_id,))
                    hw_question_count = cursor.fetchone()['count']
                    
                    if hw_question_count == 0:
                        print(f"为作业 '{title}' (ID: {homework_id}) 创建题目...")
                        
                        # 创建2道简单题目
                        questions = [
                            {
                                'content': f'这是作业"{title}"的第一道题目，请选择正确答案。',
                                'type': 'single_choice',
                                'options': '["选项A", "选项B", "选项C", "选项D"]',
                                'answer': '选项A',
                                'score': 50
                            },
                            {
                                'content': f'这是作业"{title}"的第二道题目，请填写答案。',
                                'type': 'fill_blank',
                                'options': None,
                                'answer': '正确答案',
                                'score': 50
                            }
                        ]
                        
                        for i, q in enumerate(questions, 1):
                            cursor.execute("""
                                INSERT INTO questions (homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation)
                                VALUES (%s, %s, %s, %s, %s, %s, 2, %s, '[]', '这是一道测试题目')
                            """, (
                                homework_id,
                                q['content'],
                                q['type'],
                                q['options'],
                                q['answer'],
                                q['score'],
                                i
                            ))
                        
                        print(f"  ✅ 创建了 {len(questions)} 道题目")
                    else:
                        print(f"作业 '{title}' 已有 {hw_question_count} 道题目")
                
                conn.commit()
                
                # 4. 最终统计
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                final_question_count = cursor.fetchone()['count']
                print(f"\n📊 最终统计:")
                print(f"总题目数: {final_question_count}")
                
                cursor.execute("""
                    SELECT h.id, h.title, COUNT(q.id) as question_count
                    FROM homeworks h
                    LEFT JOIN questions q ON h.id = q.homework_id
                    GROUP BY h.id, h.title
                    ORDER BY h.id
                """)
                homework_questions = cursor.fetchall()
                
                print("作业题目分布:")
                for hw in homework_questions:
                    print(f"  作业{hw['id']}: {hw['title']} - {hw['question_count']}道题目")
                
                print("\n✅ 题目数据修复完成！")
                return True
                
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_fix_questions()
