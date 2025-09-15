#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行SQL脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db

def run_sql_script():
    """执行SQL脚本插入测试数据"""
    print("🚀 开始执行SQL脚本插入测试数据...")
    
    try:
        # 读取SQL文件
        script_path = os.path.join(os.path.dirname(__file__), 'insert_test_data.sql')
        with open(script_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # 分割SQL语句
        sql_statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip() and not stmt.strip().startswith('--')]
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                for i, statement in enumerate(sql_statements):
                    if statement:
                        try:
                            cursor.execute(statement)
                            print(f"✅ 执行语句 {i+1}/{len(sql_statements)}")
                        except Exception as e:
                            if "Duplicate entry" in str(e):
                                print(f"⚠️ 语句 {i+1} 数据已存在，跳过")
                            else:
                                print(f"❌ 语句 {i+1} 执行失败: {e}")
                                print(f"SQL: {statement[:100]}...")
                
                conn.commit()
                print("\n🎉 SQL脚本执行完成！")
                
                # 显示统计信息
                print("\n📊 数据统计:")
                
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='teacher'")
                teacher_count = cursor.fetchone()['count']
                print(f"   👨‍🏫 教师: {teacher_count}人")
                
                cursor.execute("SELECT COUNT(*) as count FROM users WHERE role='student'")
                student_count = cursor.fetchone()['count']
                print(f"   👨‍🎓 学生: {student_count}人")
                
                cursor.execute("SELECT COUNT(*) as count FROM schools")
                school_count = cursor.fetchone()['count']
                print(f"   🏫 学校: {school_count}所")
                
                cursor.execute("SELECT COUNT(*) as count FROM classes")
                class_count = cursor.fetchone()['count']
                print(f"   🏛️ 班级: {class_count}个")
                
                cursor.execute("SELECT COUNT(*) as count FROM homeworks")
                homework_count = cursor.fetchone()['count']
                print(f"   📚 作业: {homework_count}个")
                
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"   📝 题目: {question_count}道")
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_assignments")
                assignment_count = cursor.fetchone()['count']
                print(f"   📋 分配: {assignment_count}个")
                
                cursor.execute("SELECT COUNT(*) as count FROM homework_submissions")
                submission_count = cursor.fetchone()['count']
                print(f"   ✅ 提交: {submission_count}份")
                
                cursor.execute("SELECT COUNT(*) as count FROM knowledge_points")
                kp_count = cursor.fetchone()['count']
                print(f"   🧠 知识点: {kp_count}个")
                
                cursor.execute("SELECT COUNT(*) as count FROM exercises")
                exercise_count = cursor.fetchone()['count']
                print(f"   💪 练习题: {exercise_count}道")
                
                # 显示班级详情
                print("\n🏛️ 班级详情:")
                cursor.execute("""
                    SELECT c.class_name, u.real_name as teacher_name, COUNT(cs.student_id) as student_count
                    FROM classes c
                    LEFT JOIN users u ON c.head_teacher_id = u.id
                    LEFT JOIN class_students cs ON c.id = cs.class_id AND cs.is_active = 1
                    GROUP BY c.id, c.class_name, u.real_name
                """)
                classes = cursor.fetchall()
                for cls in classes:
                    print(f"   {cls['class_name']}: {cls['teacher_name']}老师，{cls['student_count']}名学生")
                
                # 显示作业完成情况
                print("\n📊 作业完成情况:")
                cursor.execute("""
                    SELECT h.title, COUNT(hs.id) as submission_count, AVG(hs.score) as avg_score
                    FROM homeworks h
                    LEFT JOIN homework_assignments ha ON h.id = ha.homework_id
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    GROUP BY h.id, h.title
                """)
                homework_stats = cursor.fetchall()
                for hw in homework_stats:
                    avg_score = round(hw['avg_score'] or 0, 1)
                    print(f"   {hw['title']}: {hw['submission_count']}份提交，平均分 {avg_score}")
                
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
        print(f"❌ 执行SQL脚本失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_sql_script()
