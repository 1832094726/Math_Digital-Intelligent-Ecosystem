#!/usr/bin/env python3
"""
调试反馈查询
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import DatabaseManager

def main():
    db = DatabaseManager()
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                homework_id = 1
                user_id = 2  # test_student_001
                
                print(f"查询作业 {homework_id} 学生 {user_id} 的提交记录...")
                
                # 测试查询
                cursor.execute("""
                    SELECT hs.*, h.title, h.max_score, h.subject, h.grade as grade_level
                    FROM homework_submissions hs
                    JOIN homeworks h ON hs.homework_id = h.id
                    WHERE hs.homework_id = %s AND hs.student_id = %s
                    ORDER BY hs.submitted_at DESC
                    LIMIT 1
                """, (homework_id, user_id))
                
                submission = cursor.fetchone()
                
                if submission:
                    print("✅ 找到提交记录:")
                    for key, value in submission.items():
                        print(f"  {key}: {value}")
                else:
                    print("❌ 未找到提交记录")
                    
                    # 检查是否有该学生的任何提交
                    print("\n检查该学生的所有提交:")
                    cursor.execute("""
                        SELECT * FROM homework_submissions 
                        WHERE student_id = %s
                    """, (user_id,))
                    
                    all_submissions = cursor.fetchall()
                    print(f"学生 {user_id} 的所有提交记录数量: {len(all_submissions)}")
                    
                    for sub in all_submissions:
                        print(f"  提交ID: {sub['id']}, 作业ID: {sub['homework_id']}, 状态: {sub['status']}")
                    
                    # 检查是否有该作业的任何提交
                    print(f"\n检查作业 {homework_id} 的所有提交:")
                    cursor.execute("""
                        SELECT * FROM homework_submissions 
                        WHERE homework_id = %s
                    """, (homework_id,))
                    
                    hw_submissions = cursor.fetchall()
                    print(f"作业 {homework_id} 的所有提交记录数量: {len(hw_submissions)}")
                    
                    for sub in hw_submissions:
                        print(f"  提交ID: {sub['id']}, 学生ID: {sub['student_id']}, 状态: {sub['status']}")
                    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
