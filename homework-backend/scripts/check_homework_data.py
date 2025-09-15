#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查作业数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def check_homework_data():
    """检查作业数据"""
    print("📚 检查作业数据...")
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 检查作业表结构
                print("\n🏗️ 检查作业表结构...")
                cursor.execute("DESCRIBE homeworks")
                columns = cursor.fetchall()
                print("作业表字段:")
                for col in columns:
                    print(f"  {col['Field']:<20} {col['Type']:<30} {col['Null']:<5} {col['Default']}")
                
                # 2. 检查作业数据
                print("\n📊 检查作业数据...")
                cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                total_count = cursor.fetchone()['count']
                print(f"总作业数: {total_count}")
                
                if total_count > 0:
                    cursor.execute("""
                        SELECT id, title, subject, grade, is_published, created_by, created_at
                        FROM homeworks 
                        ORDER BY created_at DESC
                        LIMIT 10
                    """)
                    homeworks = cursor.fetchall()
                    
                    print("\n作业列表:")
                    for hw in homeworks:
                        status = "已发布" if hw['is_published'] else "未发布"
                        print(f"  ID:{hw['id']} | {hw['title']} | {hw['subject']} | 年级:{hw['grade']} | {status} | 创建者:{hw['created_by']}")
                
                # 3. 检查发布状态
                cursor.execute("SELECT COUNT(*) as count FROM homeworks WHERE is_published = 1")
                published_count = cursor.fetchone()['count']
                print(f"\n已发布作业数: {published_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM homeworks WHERE is_published = 0")
                unpublished_count = cursor.fetchone()['count']
                print(f"未发布作业数: {unpublished_count}")
                
                # 4. 检查作业分配
                print("\n📋 检查作业分配...")
                cursor.execute("SELECT COUNT(*) as count FROM homework_assignments")
                assignment_count = cursor.fetchone()['count']
                print(f"作业分配数: {assignment_count}")
                
                if assignment_count > 0:
                    cursor.execute("""
                        SELECT ha.id, h.title, ha.assigned_to_type, ha.assigned_to_id, ha.is_active
                        FROM homework_assignments ha
                        JOIN homeworks h ON ha.homework_id = h.id
                        ORDER BY ha.assigned_at DESC
                        LIMIT 10
                    """)
                    assignments = cursor.fetchall()
                    
                    print("\n作业分配列表:")
                    for assign in assignments:
                        status = "激活" if assign['is_active'] else "未激活"
                        print(f"  ID:{assign['id']} | {assign['title']} | 分配给:{assign['assigned_to_type']}:{assign['assigned_to_id']} | {status}")
                
                # 5. 检查学生提交
                print("\n📤 检查学生提交...")
                cursor.execute("SELECT COUNT(*) as count FROM homework_submissions")
                submission_count = cursor.fetchone()['count']
                print(f"学生提交数: {submission_count}")
                
                # 6. 检查用户数据
                print("\n👥 检查用户数据...")
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'student'")
                student_count = cursor.fetchone()['count']
                print(f"学生用户数: {student_count}")
                
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'teacher'")
                teacher_count = cursor.fetchone()['count']
                print(f"教师用户数: {teacher_count}")
                
                if student_count > 0:
                    cursor.execute("""
                        SELECT id, username, real_name, grade, school
                        FROM users 
                        WHERE role = 'student'
                        LIMIT 5
                    """)
                    students = cursor.fetchall()
                    
                    print("\n学生用户列表:")
                    for student in students:
                        print(f"  ID:{student['id']} | {student['username']} | {student['real_name']} | 年级:{student['grade']}")
                
                # 7. 检查API可能的问题
                print("\n🔍 诊断API问题...")
                
                # 检查是否有适合学生的作业
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM homeworks 
                    WHERE is_published = 1 AND grade = 7
                """)
                grade7_published = cursor.fetchone()['count']
                print(f"7年级已发布作业数: {grade7_published}")
                
                # 检查作业分配是否正确
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM homework_assignments ha
                    JOIN homeworks h ON ha.homework_id = h.id
                    WHERE h.is_published = 1 AND ha.is_active = 1
                """)
                active_published_assignments = cursor.fetchone()['count']
                print(f"已发布且激活的作业分配数: {active_published_assignments}")
                
                # 8. 修复数据问题（如果需要）
                print("\n🔧 检查是否需要修复数据...")
                
                if published_count == 0 and total_count > 0:
                    print("⚠️ 发现问题：所有作业都未发布")
                    print("正在修复：将所有作业设置为已发布...")
                    cursor.execute("UPDATE homeworks SET is_published = 1")
                    conn.commit()
                    print("✅ 修复完成：所有作业已设置为发布状态")
                
                if assignment_count > 0:
                    cursor.execute("SELECT COUNT(*) as count FROM homework_assignments WHERE is_active = 0")
                    inactive_assignments = cursor.fetchone()['count']
                    if inactive_assignments > 0:
                        print(f"⚠️ 发现问题：{inactive_assignments} 个作业分配未激活")
                        print("正在修复：激活所有作业分配...")
                        cursor.execute("UPDATE homework_assignments SET is_active = 1")
                        conn.commit()
                        print("✅ 修复完成：所有作业分配已激活")
                
                print("\n✅ 作业数据检查完成！")
                return True
                
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_homework_data()
