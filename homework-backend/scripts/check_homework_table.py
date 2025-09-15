#!/usr/bin/env python3
"""
检查homeworks表结构
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
                # 检查homeworks表结构
                print("homeworks表结构:")
                cursor.execute("DESCRIBE homeworks")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  {col['Field']}: {col['Type']} {col['Null']} {col['Key']} {col['Default']}")
                
                print("\n检查是否有total_score字段:")
                has_total_score = any(col['Field'] == 'total_score' for col in columns)
                print(f"  has total_score: {has_total_score}")
                
                # 检查homeworks表的数据
                print("\nhomeworks表数据样例:")
                cursor.execute("SELECT * FROM homeworks LIMIT 3")
                homeworks = cursor.fetchall()
                for hw in homeworks:
                    print(f"  ID: {hw['id']}, Title: {hw.get('title', 'N/A')}")
                    
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()
