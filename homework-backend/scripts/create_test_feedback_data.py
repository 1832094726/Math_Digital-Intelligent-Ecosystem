#!/usr/bin/env python3
"""
创建测试反馈数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import DatabaseManager
import json
from datetime import datetime, timedelta

def main():
    db = DatabaseManager()
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 创建作业分配记录（将作业1分配给学生2）
                print("创建作业分配记录...")
                cursor.execute("""
                    INSERT INTO homework_assignments 
                    (homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE is_active = VALUES(is_active)
                """, (1, 'student', 2, 3, datetime.now(), 1))
                
                assignment_id = cursor.lastrowid or 1
                print(f"✅ 作业分配记录创建成功，ID: {assignment_id}")
                
                # 2. 创建作业提交记录
                print("创建作业提交记录...")
                submission_data = {
                    "answers": [
                        {"question_id": 1, "answer": "15", "is_correct": True},
                        {"question_id": 2, "answer": "错误答案", "is_correct": False},
                        {"question_id": 3, "answer": "正确答案", "is_correct": True}
                    ],
                    "total_time": 1800,  # 30分钟
                    "completion_rate": 100
                }
                
                cursor.execute("""
                    INSERT INTO homework_submissions 
                    (assignment_id, student_id, homework_id, answers, submission_data, 
                     submitted_at, score, max_score, status, time_spent, attempt_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    score = VALUES(score), status = VALUES(status)
                """, (
                    assignment_id, 2, 1, 
                    json.dumps(submission_data["answers"]),
                    json.dumps(submission_data),
                    datetime.now() - timedelta(hours=1),
                    85.5, 100.0, 'graded', 1800, 1
                ))
                
                submission_id = cursor.lastrowid or 1
                print(f"✅ 作业提交记录创建成功，ID: {submission_id}")
                
                # 3. 跳过题目答案记录（表不存在）
                print("⚠️  跳过题目答案记录（question_answers表不存在）")
                
                # 4. 更新作业1的创建者为teacher001（用户ID 3）
                print("更新作业创建者...")
                cursor.execute("""
                    UPDATE homeworks 
                    SET created_by = %s 
                    WHERE id = %s
                """, (3, 1))
                
                print("✅ 作业创建者更新成功")
                
                # 5. 创建一些班级统计数据（模拟其他学生的提交）
                print("创建班级统计数据...")
                other_students = [
                    {"student_id": 10, "score": 92.0},
                    {"student_id": 11, "score": 78.5},
                    {"student_id": 12, "score": 88.0},
                    {"student_id": 13, "score": 76.0}
                ]
                
                for i, student_data in enumerate(other_students):
                    # 创建分配记录
                    cursor.execute("""
                        INSERT INTO homework_assignments 
                        (homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, is_active)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE is_active = VALUES(is_active)
                    """, (1, 'student', student_data["student_id"], 3, datetime.now(), 1))
                    
                    other_assignment_id = cursor.lastrowid or (assignment_id + i + 1)
                    
                    # 创建提交记录
                    cursor.execute("""
                        INSERT INTO homework_submissions 
                        (assignment_id, student_id, homework_id, submitted_at, 
                         score, max_score, status, time_spent, attempt_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        score = VALUES(score), status = VALUES(status)
                    """, (
                        other_assignment_id, student_data["student_id"], 1,
                        datetime.now() - timedelta(hours=2+i),
                        student_data["score"], 100.0, 'graded', 1500+i*100, 1
                    ))
                
                print(f"✅ 创建了 {len(other_students)} 个其他学生的提交记录")
                
                conn.commit()
                print("\n🎉 所有测试数据创建成功！")
                
                # 验证数据
                print("\n验证数据:")
                cursor.execute("""
                    SELECT hs.student_id, hs.score, u.username 
                    FROM homework_submissions hs
                    JOIN users u ON hs.student_id = u.id
                    WHERE hs.homework_id = 1
                """)
                submissions = cursor.fetchall()
                
                print("作业1的提交记录:")
                for sub in submissions:
                    print(f"  学生: {sub['username']}, 分数: {sub['score']}")
                    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
