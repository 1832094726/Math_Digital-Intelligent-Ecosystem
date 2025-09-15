#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一执行所有测试数据创建
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def setup_all_test_data():
    """统一执行所有测试数据创建"""
    print("🚀 开始创建完整的测试数据集...")
    
    try:
        # 执行基础数据创建
        print("\n📋 步骤1: 创建基础数据...")
        from scripts.execute_test_data import execute_test_data
        if not execute_test_data():
            print("❌ 基础数据创建失败")
            return False
        
        # 执行作业数据创建
        print("\n📝 步骤2: 创建作业数据...")
        from scripts.create_homework_test_data import create_homework_test_data
        if not create_homework_test_data():
            print("❌ 作业数据创建失败")
            return False
        
        # 执行分配数据创建
        print("\n📊 步骤3: 创建分配和提交数据...")
        from scripts.create_assignment_data import create_assignment_data
        if not create_assignment_data():
            print("❌ 分配数据创建失败")
            return False
        
        # 显示统计信息
        print("\n📈 步骤4: 显示数据统计...")
        show_final_stats()
        
        print("\n🎉 所有测试数据创建完成！")
        print("\n📌 测试数据概览:")
        print("   👥 用户: 3名老师 + 6名学生")
        print("   🏫 学校: 1所学校，1个年级，2个班级")
        print("   📚 作业: 6个作业（每个老师给每个班级1个）")
        print("   📝 题目: 12道题目（每个作业2道题）")
        print("   ✅ 提交: 18份学生提交（每个学生3份作业）")
        print("   🧠 知识点: 3个知识点（有理数、代数式、几何）")
        print("   💪 练习题: 3道练习题")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_final_stats():
    """显示最终统计信息"""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 统计各类数据
                stats = {}
                
                # 用户统计
                cursor.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
                users = cursor.fetchall()
                stats['users'] = {user['role']: user['count'] for user in users}
                
                # 基础数据统计
                cursor.execute("SELECT COUNT(*) as count FROM schools")
                stats['schools'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM classes")
                stats['classes'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                stats['homeworks'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                stats['questions'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_assignments")
                stats['assignments'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_submissions")
                stats['submissions'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM knowledge_points")
                stats['knowledge_points'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM exercises")
                stats['exercises'] = cursor.fetchone()['count']
                
                # 显示统计信息
                print("📊 数据统计:")
                print(f"   👨‍🏫 教师: {stats['users'].get('teacher', 0)}人")
                print(f"   👨‍🎓 学生: {stats['users'].get('student', 0)}人")
                print(f"   🏫 学校: {stats['schools']}所")
                print(f"   🏛️ 班级: {stats['classes']}个")
                print(f"   📚 作业: {stats['homeworks']}个")
                print(f"   📝 题目: {stats['questions']}道")
                print(f"   📋 分配: {stats['assignments']}个")
                print(f"   ✅ 提交: {stats['submissions']}份")
                print(f"   🧠 知识点: {stats['knowledge_points']}个")
                print(f"   💪 练习题: {stats['exercises']}道")
                
                # 显示班级详情
                cursor.execute("""
                    SELECT c.class_name, u.real_name as teacher_name, COUNT(cs.student_id) as student_count
                    FROM classes c
                    LEFT JOIN users u ON c.head_teacher_id = u.id
                    LEFT JOIN class_students cs ON c.id = cs.class_id AND cs.is_active = 1
                    GROUP BY c.id, c.class_name, u.real_name
                """)
                classes = cursor.fetchall()
                
                print("\n🏛️ 班级详情:")
                for cls in classes:
                    print(f"   {cls['class_name']}: {cls['teacher_name']}老师，{cls['student_count']}名学生")
                
                # 显示作业完成情况
                cursor.execute("""
                    SELECT h.title, COUNT(hs.id) as submission_count, AVG(hs.score) as avg_score
                    FROM homeworks h
                    LEFT JOIN homework_assignments ha ON h.id = ha.homework_id
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    GROUP BY h.id, h.title
                """)
                homework_stats = cursor.fetchall()
                
                print("\n📊 作业完成情况:")
                for hw in homework_stats:
                    avg_score = round(hw['avg_score'] or 0, 1)
                    print(f"   {hw['title']}: {hw['submission_count']}份提交，平均分 {avg_score}")
                
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")

if __name__ == "__main__":
    setup_all_test_data()
