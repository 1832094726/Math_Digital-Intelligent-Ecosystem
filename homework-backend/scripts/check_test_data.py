#!/usr/bin/env python3
"""
检查测试数据
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
                # 检查作业数据
                print("作业数据:")
                cursor.execute("SELECT id, title, created_by FROM homeworks LIMIT 5")
                homeworks = cursor.fetchall()
                for hw in homeworks:
                    print(f"  ID: {hw['id']}, Title: {hw['title']}, Created by: {hw['created_by']}")
                
                # 检查作业分配数据
                print("\n作业分配数据:")
                cursor.execute("SELECT * FROM homework_assignments LIMIT 5")
                assignments = cursor.fetchall()
                for assign in assignments:
                    print(f"  ID: {assign['id']}, Homework: {assign['homework_id']}, Student: {assign['student_id']}")
                
                # 检查作业提交数据
                print("\n作业提交数据:")
                cursor.execute("SELECT * FROM homework_submissions LIMIT 5")
                submissions = cursor.fetchall()
                for sub in submissions:
                    print(f"  ID: {sub['id']}, Assignment: {sub['assignment_id']}, Status: {sub['submission_status']}")
                
                # 检查用户数据
                print("\n用户数据:")
                cursor.execute("SELECT id, username, role FROM users WHERE role IN ('student', 'teacher')")
                users = cursor.fetchall()
                for user in users:
                    print(f"  ID: {user['id']}, Username: {user['username']}, Role: {user['role']}")
                    
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()
