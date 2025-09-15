"""
作业提交模型 - 负责处理作业提交、评分和反馈相关的数据操作
"""
import json
from datetime import datetime
from models.database import db
from typing import Dict, Any

class HomeworkSubmission:
    """作业提交模型"""

    @staticmethod
    def submit_homework(student_id: int, assignment_id: int, answers: Dict[str, Any], 
                        time_spent: int, is_final: bool = True) -> Dict[str, Any]:
        """
        学生提交作业

        Args:
            student_id: 学生ID
            assignment_id: 作业分发ID
            answers: 学生答案
            time_spent: 答题用时（秒）
            is_final: 是否为最终提交

        Returns:
            Dict: 提交结果
        """
        try:
            # 1. 验证作业分发是否存在且有效
            assignment = db.execute_query_one(
                """SELECT ha.*, h.max_attempts, h.auto_grade, h.due_date as original_due_date
                   FROM homework_assignments ha
                   JOIN homeworks h ON ha.homework_id = h.id
                   JOIN class_students cs ON ha.assigned_to_id = cs.class_id
                   WHERE ha.id = %s AND cs.student_id = %s 
                   AND ha.assigned_to_type = 'class' AND ha.is_active = 1 AND cs.is_active = 1""",
                (assignment_id, student_id)
            )

            if not assignment:
                return {"success": False, "message": "作业不存在、已过期或无权限提交"}

            homework_id = assignment['homework_id']
            due_date = assignment['due_date_override'] or assignment['original_due_date']
            
            # 2. 检查是否超过截止日期
            if due_date and due_date < datetime.now():
                return {"success": False, "message": "已超过提交截止日期"}

            # 3. 检查是否已提交过
            existing_submission = db.execute_query_one(
                "SELECT * FROM homework_submissions WHERE student_id = %s AND assignment_id = %s",
                (student_id, assignment_id)
            )
            
            # 检查重做次数
            max_attempts = assignment['max_attempts'] or 1
            if existing_submission and existing_submission['attempt_count'] >= max_attempts:
                 return {"success": False, "message": f"已达到最大提交次数 ({max_attempts}次)"}

            # 4. 插入或更新提交记录
            submission_data = {
                "ip_address": "mock_ip", # 实际应从请求中获取
                "user_agent": "mock_agent"
            }
            
            if existing_submission:
                # 更新提交记录（适用于允许重做的场景）
                submission_id = existing_submission['id']
                db.execute_update(
                    """UPDATE homework_submissions 
                       SET answers = %s, submission_data = %s, submitted_at = NOW(),
                           time_spent = %s, status = 'submitted', attempt_count = attempt_count + 1
                       WHERE id = %s""",
                    (json.dumps(answers), json.dumps(submission_data), time_spent, submission_id)
                )
            else:
                # 创建新的提交记录
                submission_id = db.execute_insert(
                    """INSERT INTO homework_submissions
                       (assignment_id, student_id, homework_id, answers, submission_data,
                        submitted_at, time_spent, status, attempt_count)
                       VALUES (%s, %s, %s, %s, %s, NOW(), %s, 'submitted', 1)""",
                    (assignment_id, student_id, homework_id, json.dumps(answers), 
                     json.dumps(submission_data), time_spent)
                )

            # 5. （可选）触发自动评分
            auto_grade_result = None
            if assignment['auto_grade']:
                auto_grade_result = AutoGrading.grade_submission(submission_id)
                if 'error' not in auto_grade_result:
                    # 更新提交记录的分数和状态
                    db.execute_update(
                        "UPDATE homework_submissions SET score = %s, max_score = %s, status = 'graded', auto_grade_data = %s WHERE id = %s",
                        (auto_grade_result['score'], auto_grade_result['max_score'], 
                         json.dumps(auto_grade_result), submission_id)
                    )

            # 6. 删除作业进度记录
            db.execute_delete(
                "DELETE FROM homework_progress WHERE student_id = %s AND homework_id = %s",
                (student_id, homework_id)
            )

            return {
                "success": True, 
                "message": "作业提交成功",
                "submission_id": submission_id,
                "auto_grade_result": auto_grade_result
            }

        except Exception as e:
            return {"success": False, "message": f"提交失败: {str(e)}"}

    @staticmethod
    def get_submission_result(student_id: int, submission_id: int) -> Dict[str, Any]:
        """
        获取作业提交结果和反馈

        Args:
            student_id: 学生ID
            submission_id: 提交ID

        Returns:
            Dict: 提交结果
        """
        try:
            submission = db.execute_query_one(
                """SELECT hs.*, h.title, h.instructions
                   FROM homework_submissions hs
                   JOIN homeworks h ON hs.homework_id = h.id
                   WHERE hs.id = %s AND hs.student_id = %s""",
                (submission_id, student_id)
            )

            if not submission:
                return {"success": False, "message": "提交记录不存在"}
            
            # 解析JSON字段
            if submission.get('answers'):
                submission['answers'] = json.loads(submission['answers'])
            if submission.get('auto_grade_data'):
                submission['auto_grade_data'] = json.loads(submission['auto_grade_data'])

            return {"success": True, "data": submission}

        except Exception as e:
            return {"success": False, "message": f"获取提交结果失败: {str(e)}"}


class AutoGrading:
    """自动评分模型（模拟）"""

    @staticmethod
    def grade_submission(submission_id: int) -> Dict[str, Any]:
        """
        对提交的作业进行自动评分（模拟实现）

        Args:
            submission_id: 提交ID

        Returns:
            Dict: 评分结果
        """
        # 1. 获取提交的答案和作业的标准答案
        submission = db.execute_query_one(
            """SELECT hs.answers, h.instructions as questions_data, h.max_score
               FROM homework_submissions hs
               JOIN homeworks h ON hs.homework_id = h.id
               WHERE hs.id = %s""",
            (submission_id,)
        )
        
        if not submission:
            return {"error": "Submission not found"}

        try:
            student_answers = json.loads(submission.get('answers') or '{}')
            # 假设 questions_data 存储在 homeworks.instructions 字段中
            questions_data = json.loads(submission.get('questions_data') or '{}')
            questions = questions_data.get('questions', [])
            standard_answers = {q['id']: q['answer'] for q in questions if 'id' in q and 'answer' in q}
            
            if not standard_answers:
                 return {"score": 0, "max_score": 100.0, "details": "No standard answers found for grading."}

            correct_count = 0
            total_questions = len(standard_answers)
            grading_details = {}
            
            # 2. 对比答案
            for q_id, s_answer in standard_answers.items():
                is_correct = str(student_answers.get(str(q_id), "")).strip().lower() == str(s_answer).strip().lower()
                if is_correct:
                    correct_count += 1
                grading_details[str(q_id)] = {
                    "student_answer": student_answers.get(str(q_id), ""),
                    "standard_answer": s_answer,
                    "is_correct": is_correct
                }
            
            # 3. 计算分数
            score = 0
            max_score = float(submission.get('max_score') or 100.0)
            if total_questions > 0:
                score = (correct_count / total_questions) * max_score

            return {
                "score": round(score, 2),
                "max_score": max_score,
                "correct_count": correct_count,
                "total_questions": total_questions,
                "details": grading_details,
                "graded_at": datetime.now().isoformat()
            }
        except (json.JSONDecodeError, TypeError) as e:
            return {"error": f"Grading failed due to data format issue: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred during grading: {str(e)}"}

