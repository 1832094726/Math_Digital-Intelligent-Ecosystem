#!/usr/bin/env python3
"""
检查作业分配表结构
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
                # 检查homework_assignments表结构
                print("homework_assignments表结构:")
                cursor.execute("DESCRIBE homework_assignments")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  {col['Field']}: {col['Type']}")
                
                # 检查homework_submissions表结构
                print("\nhomework_submissions表结构:")
                cursor.execute("DESCRIBE homework_submissions")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  {col['Field']}: {col['Type']}")
                    
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()
