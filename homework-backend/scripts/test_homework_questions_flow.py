#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试作业题目完整流程
"""
import requests
import json

def test_homework_questions_flow():
    """测试作业题目完整流程"""
    print("🧪 测试作业题目完整流程...")
    
    base_url = "http://localhost:5000"
    
    try:
        # 1. 登录
        print("\n🔐 步骤1: 用户登录")
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        if login_response.status_code != 200:
            print(f"❌ 登录失败: {login_response.status_code}")
            return False
        
        login_data = login_response.json()
        if not login_data.get('success'):
            print(f"❌ 登录失败: {login_data.get('message')}")
            return False
        
        token = login_data['data']['access_token']
        print("✅ 登录成功")
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # 2. 获取作业列表
        print("\n📚 步骤2: 获取作业列表")
        homework_response = requests.get(f"{base_url}/api/homework/list", headers=headers)
        
        if homework_response.status_code != 200:
            print(f"❌ 获取作业列表失败: {homework_response.status_code}")
            return False
        
        homework_data = homework_response.json()
        homeworks = homework_data['data']['homeworks']
        print(f"✅ 获取到 {len(homeworks)} 个作业")
        
        if len(homeworks) == 0:
            print("⚠️ 没有作业数据")
            return False
        
        # 3. 选择第一个作业
        first_homework = homeworks[0]
        homework_id = first_homework['id']
        print(f"\n🎯 步骤3: 选择作业 '{first_homework['title']}' (ID: {homework_id})")
        
        # 4. 获取作业详情
        print(f"\n📖 步骤4: 获取作业详情")
        detail_response = requests.get(f"{base_url}/api/homework/{homework_id}", headers=headers)
        
        if detail_response.status_code != 200:
            print(f"❌ 获取作业详情失败: {detail_response.status_code}")
            return False
        
        detail_data = detail_response.json()
        homework_detail = detail_data['data']
        print(f"✅ 获取作业详情成功")
        print(f"  标题: {homework_detail['title']}")
        print(f"  题目数: {homework_detail['question_count']}")
        
        # 5. 获取作业题目
        print(f"\n📝 步骤5: 获取作业题目")
        questions_response = requests.get(f"{base_url}/api/homework/{homework_id}/questions", headers=headers)
        
        if questions_response.status_code != 200:
            print(f"❌ 获取作业题目失败: {questions_response.status_code}")
            print(f"响应: {questions_response.text}")
            return False
        
        questions_data = questions_response.json()
        if not questions_data.get('success'):
            print(f"❌ 获取作业题目失败: {questions_data.get('message')}")
            return False
        
        questions = questions_data['data']['questions']
        print(f"✅ 获取到 {len(questions)} 道题目")
        
        # 6. 显示题目详情
        print(f"\n📋 步骤6: 题目详情")
        for i, question in enumerate(questions, 1):
            print(f"  题目{i}:")
            print(f"    ID: {question['id']}")
            print(f"    内容: {question['content']}")
            print(f"    类型: {question['question_type']}")
            print(f"    分值: {question['score']}")
            
            if question.get('options'):
                print(f"    选项: {question['options']}")
            
            print()
        
        # 7. 模拟前端数据结构
        print(f"\n🎭 步骤7: 模拟前端数据结构")
        
        # 合并作业详情和题目（模拟前端fetchHomeworkDetail的结果）
        homework_with_questions = {
            **homework_detail,
            'questions': questions,
            'problems': questions  # 前端期望的字段名
        }
        
        print("前端获得的完整作业数据:")
        print(f"  作业ID: {homework_with_questions['id']}")
        print(f"  作业标题: {homework_with_questions['title']}")
        print(f"  题目数量: {len(homework_with_questions['questions'])}")
        print(f"  是否有problems字段: {'problems' in homework_with_questions}")
        print(f"  是否有questions字段: {'questions' in homework_with_questions}")
        
        # 8. 验证前端期望的数据结构
        print(f"\n✅ 步骤8: 验证前端期望")
        
        frontend_checks = [
            ('作业有questions数组', isinstance(homework_with_questions.get('questions'), list)),
            ('作业有problems数组', isinstance(homework_with_questions.get('problems'), list)),
            ('题目有id字段', all('id' in q for q in questions)),
            ('题目有content字段', all('content' in q for q in questions)),
            ('题目有question_type字段', all('question_type' in q for q in questions)),
            ('题目有score字段', all('score' in q for q in questions)),
        ]
        
        all_passed = True
        for check_name, check_result in frontend_checks:
            if check_result:
                print(f"  ✅ {check_name}")
            else:
                print(f"  ❌ {check_name}")
                all_passed = False
        
        # 9. 生成前端可用的JSON
        print(f"\n📄 步骤9: 生成前端测试数据")
        
        frontend_data = {
            "currentHomework": homework_with_questions,
            "answers": {},
            "activeProblemIds": [q['id'] for q in questions],
            "selectedQuestionId": questions[0]['id'] if questions else None
        }
        
        # 保存到文件供前端测试
        with open('homework-backend/scripts/frontend_test_data.json', 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, indent=2, ensure_ascii=False)
        
        print("✅ 前端测试数据已保存到 frontend_test_data.json")
        
        if all_passed and len(questions) > 0:
            print("\n🎉 完整流程测试通过！")
            print("前端应该能够:")
            print("  ✅ 显示作业列表")
            print("  ✅ 点击作业获取详情")
            print("  ✅ 显示题目列表")
            print("  ✅ 进行答题操作")
            return True
        else:
            print("\n⚠️ 测试发现问题")
            return False
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_questions_exist():
    """检查数据库中是否有题目数据"""
    print("\n🔍 检查数据库题目数据...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from models.database import db
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # 检查题目总数
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                total_questions = cursor.fetchone()['count']
                print(f"数据库中总题目数: {total_questions}")
                
                if total_questions == 0:
                    print("⚠️ 数据库中没有题目，需要运行数据修复脚本")
                    return False
                
                # 检查每个作业的题目数
                cursor.execute("""
                    SELECT h.id, h.title, COUNT(q.id) as question_count
                    FROM homeworks h
                    LEFT JOIN questions q ON h.id = q.homework_id
                    GROUP BY h.id, h.title
                    ORDER BY h.id
                """)
                
                homework_questions = cursor.fetchall()
                print("\n作业题目统计:")
                for hw in homework_questions:
                    print(f"  作业{hw['id']}: {hw['title']} - {hw['question_count']}道题目")
                
                return total_questions > 0
                
    except Exception as e:
        print(f"❌ 检查数据库失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始作业题目完整流程测试...")
    
    # 先检查数据库
    has_questions = check_questions_exist()
    
    if not has_questions:
        print("\n⚠️ 建议先运行数据修复脚本:")
        print("python homework-backend/scripts/fix_questions.py")
        print("python homework-backend/scripts/quick_data_fix.py")
    
    # 测试完整流程
    success = test_homework_questions_flow()
    
    if success:
        print("\n🎉 所有测试通过！前端应该能够正常显示题目了。")
    else:
        print("\n⚠️ 测试发现问题，需要进一步调试。")
