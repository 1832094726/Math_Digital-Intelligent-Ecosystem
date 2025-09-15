#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试评分功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.grading_service import GradingService
from models.database import db
import json

def test_grading():
    """测试评分功能"""
    print("🧪 开始测试评分功能...")
    
    try:
        # 初始化评分服务
        grading_service = GradingService()
        
        # 创建测试数据
        print("\n📝 创建测试数据...")
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 1. 确保有评分表
                print("1. 创建评分表...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS grading_results (
                      id bigint(20) NOT NULL AUTO_INCREMENT,
                      submission_id bigint(20) NOT NULL,
                      result_data json NOT NULL,
                      total_score decimal(5,2) NOT NULL DEFAULT '0.00',
                      total_possible decimal(5,2) NOT NULL DEFAULT '0.00',
                      accuracy decimal(5,2) NOT NULL DEFAULT '0.00',
                      grading_method enum('auto','manual','hybrid') NOT NULL DEFAULT 'auto',
                      graded_by bigint(20) DEFAULT NULL,
                      graded_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      reviewed_at timestamp NULL DEFAULT NULL,
                      review_notes text DEFAULT NULL,
                      created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                      PRIMARY KEY (id),
                      UNIQUE KEY uk_submission_id (submission_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                
                # 2. 创建测试提交记录
                print("2. 创建测试提交记录...")
                
                # 插入测试作业分配
                cursor.execute("""
                    INSERT IGNORE INTO homework_assignments (id, homework_id, assigned_to_type, assigned_to_id, assigned_by, assigned_at, is_active)
                    VALUES (100, 1, 'class', 1, 10, NOW(), 1)
                """)
                
                # 插入测试提交
                test_answers = {
                    "1": "0",      # 正确答案
                    "2": "6"       # 正确答案
                }
                
                cursor.execute("""
                    INSERT IGNORE INTO homework_submissions (id, assignment_id, student_id, answers, score, time_spent, status, submitted_at)
                    VALUES (100, 100, 20, %s, 0, 30, 'submitted', NOW())
                """, (json.dumps(test_answers),))
                
                conn.commit()
                
                # 3. 测试评分
                print("3. 测试自动评分...")
                result = grading_service.grade_submission(100)
                
                if result['success']:
                    print("✅ 评分成功！")
                    print(f"   总分: {result['total_score']}/{result['total_possible']}")
                    print(f"   正确率: {result['accuracy']}%")
                    
                    print("\n📊 题目评分详情:")
                    for i, q_result in enumerate(result['question_results'], 1):
                        status = "✅" if q_result['is_correct'] else "❌"
                        print(f"   题目{i}: {status} {q_result['score_earned']}/{q_result['score_possible']}分")
                        print(f"           学生答案: {q_result['student_answer']}")
                        print(f"           正确答案: {q_result['correct_answer']}")
                        print(f"           反馈: {q_result['feedback']}")
                    
                    print(f"\n📈 评分总结:")
                    summary = result['summary']
                    print(f"   总题数: {summary['total_questions']}")
                    print(f"   正确数: {summary['correct_count']}")
                    print(f"   正确率: {summary['accuracy_rate']}%")
                    
                    if summary['suggestions']:
                        print(f"   建议: {', '.join(summary['suggestions'])}")
                    
                else:
                    print(f"❌ 评分失败: {result['message']}")
                
                # 4. 测试不同答案的评分
                print("\n🔄 测试不同答案的评分...")
                
                test_cases = [
                    {"1": "2", "2": "6"},      # 第一题错误，第二题正确
                    {"1": "0", "2": "错误"},    # 第一题正确，第二题错误
                    {"1": "错误", "2": "错误"},  # 全部错误
                    {"1": "", "2": ""},        # 空答案
                ]
                
                for i, test_case in enumerate(test_cases, 1):
                    print(f"\n   测试案例 {i}: {test_case}")
                    
                    # 更新提交答案
                    cursor.execute("""
                        UPDATE homework_submissions 
                        SET answers = %s, status = 'submitted'
                        WHERE id = 100
                    """, (json.dumps(test_case),))
                    
                    # 删除之前的评分结果
                    cursor.execute("DELETE FROM grading_results WHERE submission_id = 100")
                    conn.commit()
                    
                    # 重新评分
                    result = grading_service.grade_submission(100)
                    
                    if result['success']:
                        print(f"     结果: {result['total_score']}/{result['total_possible']}分 ({result['accuracy']}%)")
                    else:
                        print(f"     失败: {result['message']}")
                
                # 5. 测试模糊匹配
                print("\n🔍 测试模糊匹配...")
                
                fuzzy_cases = [
                    {"1": "0", "2": "6.0"},     # 数字格式不同
                    {"1": "0", "2": " 6 "},     # 有空格
                    {"1": "0", "2": "六"},      # 中文数字
                ]
                
                for i, test_case in enumerate(fuzzy_cases, 1):
                    print(f"\n   模糊匹配案例 {i}: {test_case}")
                    
                    cursor.execute("""
                        UPDATE homework_submissions 
                        SET answers = %s, status = 'submitted'
                        WHERE id = 100
                    """, (json.dumps(test_case),))
                    
                    cursor.execute("DELETE FROM grading_results WHERE submission_id = 100")
                    conn.commit()
                    
                    result = grading_service.grade_submission(100)
                    
                    if result['success']:
                        print(f"     结果: {result['total_score']}/{result['total_possible']}分")
                        for q_result in result['question_results']:
                            if q_result['question_id'] == 2:  # 第二题
                                print(f"     第二题: {'✅' if q_result['is_correct'] else '❌'} {q_result['feedback']}")
                
                print("\n✅ 评分功能测试完成！")
                return True
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_grading()
