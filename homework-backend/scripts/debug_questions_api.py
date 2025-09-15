#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试题目API
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import db
import requests

def debug_questions_api():
    """调试题目API"""
    print("🔍 调试题目API...")
    
    try:
        # 1. 检查数据库连接
        print("\n📊 检查数据库...")
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # 检查作业
                cursor.execute("SELECT id, title FROM homeworks LIMIT 5")
                homeworks = cursor.fetchall()
                print(f"作业数量: {len(homeworks)}")
                for hw in homeworks:
                    print(f"  作业{hw['id']}: {hw['title']}")
                
                if not homeworks:
                    print("❌ 没有作业数据")
                    return False
                
                # 检查题目
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"题目总数: {question_count}")
                
                if question_count == 0:
                    print("⚠️ 没有题目数据，正在创建...")
                    # 为第一个作业创建题目
                    first_homework_id = homeworks[0]['id']
                    
                    questions = [
                        {
                            'content': '计算：2 + 3 = ?',
                            'type': 'fill_blank',
                            'answer': '5',
                            'score': 50
                        },
                        {
                            'content': '下列哪个是偶数？',
                            'type': 'single_choice',
                            'options': '["1", "2", "3", "5"]',
                            'answer': '2',
                            'score': 50
                        }
                    ]
                    
                    for i, q in enumerate(questions, 1):
                        cursor.execute("""
                            INSERT INTO questions (homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation)
                            VALUES (%s, %s, %s, %s, %s, %s, 2, %s, '[]', '基础题目')
                        """, (
                            first_homework_id,
                            q['content'],
                            q['type'],
                            q.get('options'),
                            q['answer'],
                            q['score'],
                            i
                        ))
                    
                    conn.commit()
                    print(f"✅ 为作业{first_homework_id}创建了{len(questions)}道题目")
                
                # 再次检查题目
                cursor.execute("""
                    SELECT h.id as homework_id, h.title, q.id as question_id, q.content, q.question_type
                    FROM homeworks h
                    LEFT JOIN questions q ON h.id = q.homework_id
                    ORDER BY h.id, q.order_index
                    LIMIT 10
                """)
                homework_questions = cursor.fetchall()
                
                print("\n作业题目关联:")
                for hq in homework_questions:
                    if hq['question_id']:
                        print(f"  作业{hq['homework_id']}: 题目{hq['question_id']} - {hq['content'][:30]}...")
                    else:
                        print(f"  作业{hq['homework_id']}: 无题目")
        
        # 2. 测试API
        print("\n🧪 测试API...")
        
        # 登录获取token
        base_url = "http://localhost:5000"
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
        headers = {'Authorization': f'Bearer {token}'}
        
        # 测试作业详情API
        homework_id = homeworks[0]['id']
        print(f"\n📖 测试作业详情API (ID: {homework_id})")
        
        detail_response = requests.get(f"{base_url}/api/homework/{homework_id}", headers=headers)
        print(f"作业详情API状态: {detail_response.status_code}")
        
        if detail_response.status_code == 200:
            detail_data = detail_response.json()
            print(f"作业详情成功: {detail_data['data']['title']}")
        else:
            print(f"作业详情失败: {detail_response.text}")
        
        # 测试题目API
        print(f"\n📝 测试题目API (ID: {homework_id})")
        
        questions_response = requests.get(f"{base_url}/api/homework/{homework_id}/questions", headers=headers)
        print(f"题目API状态: {questions_response.status_code}")
        
        if questions_response.status_code == 200:
            questions_data = questions_response.json()
            if questions_data.get('success'):
                questions = questions_data['data']['questions']
                print(f"✅ 题目API成功: {len(questions)}道题目")
                for q in questions:
                    print(f"  题目{q['id']}: {q['content']}")
            else:
                print(f"❌ 题目API失败: {questions_data.get('message')}")
        else:
            print(f"❌ 题目API失败: {questions_response.text}")
            
            # 尝试解析错误
            try:
                error_data = questions_response.json()
                print(f"错误详情: {error_data}")
            except:
                print("无法解析错误响应")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_questions_api()
