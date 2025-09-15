#!/usr/bin/env python3
"""
检查questions表结构
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
                # 检查questions表结构
                print("questions表结构:")
                cursor.execute("DESCRIBE questions")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  {col['Field']}: {col['Type']}")
                    
                # 检查是否有数据
                print("\nquestions表数据样例:")
                cursor.execute("SELECT * FROM questions WHERE homework_id = 1 LIMIT 3")
                questions = cursor.fetchall()
                for q in questions:
                    print(f"  ID: {q['id']}, Content: {q.get('content', q.get('question_content', 'N/A'))[:50]}...")
                    
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()
