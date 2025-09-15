#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试题目API
"""
import requests
import json

def test_questions_api_direct():
    """直接测试题目API"""
    print("🧪 直接测试题目API...")
    
    base_url = "http://localhost:5000"
    
    try:
        # 1. 登录获取token
        print("\n🔐 登录...")
        login_response = requests.post(f"{base_url}/api/auth/login", json={
            "username": "test_student_001",
            "password": "password"
        })
        
        if login_response.status_code != 200:
            print(f"❌ 登录失败: {login_response.status_code}")
            print(f"响应: {login_response.text}")
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
        print("\n📚 获取作业列表...")
        homework_response = requests.get(f"{base_url}/api/homework/list", headers=headers)
        
        if homework_response.status_code != 200:
            print(f"❌ 获取作业列表失败: {homework_response.status_code}")
            return False
        
        homework_data = homework_response.json()
        homeworks = homework_data['data']['homeworks']
        
        if not homeworks:
            print("❌ 没有作业数据")
            return False
        
        print(f"✅ 获取到 {len(homeworks)} 个作业")
        
        # 3. 测试每个作业的题目API
        for homework in homeworks[:3]:  # 只测试前3个
            homework_id = homework['id']
            title = homework['title']
            
            print(f"\n📝 测试作业 '{title}' (ID: {homework_id}) 的题目API...")
            
            try:
                questions_response = requests.get(
                    f"{base_url}/api/homework/{homework_id}/questions", 
                    headers=headers,
                    timeout=10
                )
                
                print(f"  状态码: {questions_response.status_code}")
                print(f"  响应头: {dict(questions_response.headers)}")
                
                if questions_response.status_code == 200:
                    try:
                        questions_data = questions_response.json()
                        if questions_data.get('success'):
                            questions = questions_data['data']['questions']
                            print(f"  ✅ 成功获取 {len(questions)} 道题目")
                            
                            for i, q in enumerate(questions[:2], 1):  # 只显示前2道题
                                print(f"    题目{i}: {q.get('content', 'N/A')[:50]}...")
                        else:
                            print(f"  ❌ API返回失败: {questions_data.get('message')}")
                    except json.JSONDecodeError as e:
                        print(f"  ❌ JSON解析失败: {e}")
                        print(f"  响应内容: {questions_response.text[:200]}...")
                
                elif questions_response.status_code == 500:
                    print(f"  ❌ 500内部服务器错误")
                    try:
                        error_data = questions_response.json()
                        print(f"  错误信息: {error_data}")
                    except:
                        print(f"  原始错误: {questions_response.text[:200]}...")
                
                else:
                    print(f"  ❌ 其他错误: {questions_response.status_code}")
                    print(f"  响应: {questions_response.text[:200]}...")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 请求异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_directly():
    """直接测试数据库"""
    print("\n🗄️ 直接测试数据库...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from models.database import db
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 检查作业
                print("检查作业表...")
                cursor.execute("SELECT id, title, is_published FROM homeworks LIMIT 5")
                homeworks = cursor.fetchall()
                
                for hw in homeworks:
                    print(f"  作业{hw['id']}: {hw['title']} (发布: {hw['is_published']})")
                
                # 检查题目表是否存在
                print("\n检查题目表...")
                try:
                    cursor.execute("DESCRIBE questions")
                    columns = cursor.fetchall()
                    print("题目表结构:")
                    for col in columns:
                        print(f"  {col['Field']}: {col['Type']}")
                except Exception as e:
                    print(f"❌ 题目表不存在或有问题: {e}")
                    return False
                
                # 检查题目数据
                print("\n检查题目数据...")
                cursor.execute("SELECT COUNT(*) as count FROM questions")
                question_count = cursor.fetchone()['count']
                print(f"总题目数: {question_count}")
                
                if question_count == 0:
                    print("⚠️ 没有题目数据，正在创建测试数据...")
                    
                    # 为第一个作业创建题目
                    if homeworks:
                        first_homework_id = homeworks[0]['id']
                        
                        test_questions = [
                            {
                                'content': '这是一道测试题目，请选择正确答案。',
                                'type': 'single_choice',
                                'options': '["选项A", "选项B", "选项C", "选项D"]',
                                'answer': '选项A',
                                'score': 50
                            },
                            {
                                'content': '这是一道填空题，请填写答案。',
                                'type': 'fill_blank',
                                'options': None,
                                'answer': '测试答案',
                                'score': 50
                            }
                        ]
                        
                        for i, q in enumerate(test_questions, 1):
                            cursor.execute("""
                                INSERT INTO questions (homework_id, content, question_type, options, correct_answer, score, difficulty, order_index, knowledge_points, explanation)
                                VALUES (%s, %s, %s, %s, %s, %s, 2, %s, '[]', '这是一道测试题目')
                            """, (
                                first_homework_id,
                                q['content'],
                                q['type'],
                                q['options'],
                                q['answer'],
                                q['score'],
                                i
                            ))
                        
                        conn.commit()
                        print(f"✅ 为作业{first_homework_id}创建了{len(test_questions)}道题目")
                
                # 再次检查
                cursor.execute("""
                    SELECT h.id as homework_id, h.title, q.id as question_id, q.content, q.question_type
                    FROM homeworks h
                    LEFT JOIN questions q ON h.id = q.homework_id
                    ORDER BY h.id, q.order_index
                    LIMIT 10
                """)
                
                results = cursor.fetchall()
                print("\n作业题目关联:")
                for result in results:
                    if result['question_id']:
                        print(f"  作业{result['homework_id']}: 题目{result['question_id']} - {result['content'][:30]}...")
                    else:
                        print(f"  作业{result['homework_id']}: 无题目")
                
                return True
                
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始题目API直接测试...")
    
    # 先测试数据库
    db_success = test_database_directly()
    
    if db_success:
        # 再测试API
        api_success = test_questions_api_direct()
        
        if api_success:
            print("\n✅ 测试完成")
        else:
            print("\n⚠️ API测试发现问题")
    else:
        print("\n❌ 数据库测试失败")
