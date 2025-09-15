#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动评分服务
"""
import re
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
from models.database import db

class GradingService:
    """自动评分服务类"""
    
    def __init__(self):
        self.grading_rules = {
            'single_choice': self._grade_single_choice,
            'multiple_choice': self._grade_multiple_choice,
            'fill_blank': self._grade_fill_blank,
            'calculation': self._grade_calculation,
            'proof': self._grade_proof,
            'application': self._grade_application
        }
    
    def grade_submission(self, submission_id: int) -> Dict[str, Any]:
        """
        评分学生提交的作业
        
        Args:
            submission_id: 提交记录ID
            
        Returns:
            评分结果字典
        """
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    
                    # 获取提交信息
                    cursor.execute("""
                        SELECT hs.*, ha.homework_id, h.title as homework_title
                        FROM homework_submissions hs
                        JOIN homework_assignments ha ON hs.assignment_id = ha.id
                        JOIN homeworks h ON ha.homework_id = h.id
                        WHERE hs.id = %s
                    """, (submission_id,))
                    
                    submission = cursor.fetchone()
                    if not submission:
                        return {'success': False, 'message': '提交记录不存在'}
                    
                    # 获取题目信息
                    cursor.execute("""
                        SELECT id, content, question_type, options, correct_answer, score, explanation
                        FROM questions
                        WHERE homework_id = %s
                        ORDER BY order_index
                    """, (submission['homework_id'],))
                    
                    questions = cursor.fetchall()
                    if not questions:
                        return {'success': False, 'message': '作业题目不存在'}
                    
                    # 解析学生答案
                    student_answers = json.loads(submission['answers']) if submission['answers'] else {}
                    
                    # 开始评分
                    grading_results = []
                    total_score = 0
                    total_possible = 0
                    
                    for question in questions:
                        question_id = str(question['id'])
                        student_answer = student_answers.get(question_id, '')
                        
                        # 评分单个题目
                        result = self._grade_question(
                            question=question,
                            student_answer=student_answer
                        )
                        
                        grading_results.append({
                            'question_id': question['id'],
                            'question_content': question['content'],
                            'question_type': question['question_type'],
                            'correct_answer': question['correct_answer'],
                            'student_answer': student_answer,
                            'score_earned': result['score_earned'],
                            'score_possible': question['score'],
                            'is_correct': result['is_correct'],
                            'feedback': result['feedback'],
                            'error_type': result.get('error_type', ''),
                            'suggestions': result.get('suggestions', [])
                        })
                        
                        total_score += result['score_earned']
                        total_possible += question['score']
                    
                    # 计算总体统计
                    accuracy = (total_score / total_possible * 100) if total_possible > 0 else 0
                    
                    # 更新提交记录
                    cursor.execute("""
                        UPDATE homework_submissions 
                        SET score = %s, status = 'graded', graded_at = NOW()
                        WHERE id = %s
                    """, (total_score, submission_id))
                    
                    # 保存详细评分结果
                    grading_result_data = {
                        'submission_id': submission_id,
                        'total_score': total_score,
                        'total_possible': total_possible,
                        'accuracy': accuracy,
                        'question_results': grading_results,
                        'graded_at': datetime.now().isoformat(),
                        'grading_method': 'auto'
                    }
                    
                    cursor.execute("""
                        INSERT INTO grading_results (submission_id, result_data, total_score, accuracy, graded_at, grading_method)
                        VALUES (%s, %s, %s, %s, NOW(), 'auto')
                        ON DUPLICATE KEY UPDATE
                        result_data = VALUES(result_data),
                        total_score = VALUES(total_score),
                        accuracy = VALUES(accuracy),
                        graded_at = NOW()
                    """, (submission_id, json.dumps(grading_result_data), total_score, accuracy))
                    
                    conn.commit()
                    
                    return {
                        'success': True,
                        'submission_id': submission_id,
                        'total_score': total_score,
                        'total_possible': total_possible,
                        'accuracy': round(accuracy, 2),
                        'question_results': grading_results,
                        'summary': self._generate_summary(grading_results)
                    }
                    
        except Exception as e:
            return {'success': False, 'message': f'评分失败: {str(e)}'}
    
    def _grade_question(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """
        评分单个题目
        
        Args:
            question: 题目信息
            student_answer: 学生答案
            
        Returns:
            评分结果
        """
        question_type = question['question_type']
        grading_func = self.grading_rules.get(question_type, self._grade_default)
        
        return grading_func(question, student_answer)
    
    def _grade_single_choice(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """评分单选题"""
        correct_answer = question['correct_answer'].strip()
        student_answer = student_answer.strip()
        
        is_correct = correct_answer.lower() == student_answer.lower()
        score_earned = question['score'] if is_correct else 0
        
        feedback = "回答正确！" if is_correct else f"回答错误。正确答案是：{correct_answer}"
        
        return {
            'is_correct': is_correct,
            'score_earned': score_earned,
            'feedback': feedback,
            'error_type': '' if is_correct else 'wrong_choice',
            'suggestions': [] if is_correct else [f"正确答案应该是：{correct_answer}"]
        }
    
    def _grade_multiple_choice(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """评分多选题"""
        # 多选题答案通常是逗号分隔的选项
        correct_answers = set(question['correct_answer'].split(','))
        student_answers = set(student_answer.split(',')) if student_answer else set()
        
        # 完全正确才得分
        is_correct = correct_answers == student_answers
        score_earned = question['score'] if is_correct else 0
        
        feedback = "回答正确！" if is_correct else f"回答错误。正确答案是：{question['correct_answer']}"
        
        return {
            'is_correct': is_correct,
            'score_earned': score_earned,
            'feedback': feedback,
            'error_type': '' if is_correct else 'wrong_selection',
            'suggestions': [] if is_correct else [f"正确答案应该是：{question['correct_answer']}"]
        }
    
    def _grade_fill_blank(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """评分填空题（支持模糊匹配）"""
        correct_answer = question['correct_answer'].strip()
        student_answer = student_answer.strip()
        
        # 精确匹配
        if correct_answer.lower() == student_answer.lower():
            return {
                'is_correct': True,
                'score_earned': question['score'],
                'feedback': "回答正确！",
                'error_type': '',
                'suggestions': []
            }
        
        # 数学表达式模糊匹配
        if self._is_math_equivalent(correct_answer, student_answer):
            return {
                'is_correct': True,
                'score_earned': question['score'],
                'feedback': "回答正确！（数学表达式等价）",
                'error_type': '',
                'suggestions': []
            }
        
        # 部分匹配（给部分分数）
        similarity = self._calculate_similarity(correct_answer, student_answer)
        if similarity > 0.8:  # 80%相似度
            partial_score = question['score'] * 0.5  # 给一半分数
            return {
                'is_correct': False,
                'score_earned': partial_score,
                'feedback': f"答案接近正确，给予部分分数。正确答案：{correct_answer}",
                'error_type': 'partial_correct',
                'suggestions': [f"正确答案应该是：{correct_answer}"]
            }
        
        return {
            'is_correct': False,
            'score_earned': 0,
            'feedback': f"回答错误。正确答案：{correct_answer}",
            'error_type': 'wrong_answer',
            'suggestions': [f"正确答案应该是：{correct_answer}"]
        }
    
    def _grade_calculation(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """评分计算题"""
        # 计算题主要看最终答案，也可以分析步骤
        return self._grade_fill_blank(question, student_answer)
    
    def _grade_proof(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """评分证明题（简化版）"""
        # 证明题评分较复杂，这里简化处理
        if not student_answer.strip():
            return {
                'is_correct': False,
                'score_earned': 0,
                'feedback': "未提供答案",
                'error_type': 'no_answer',
                'suggestions': ["请提供完整的证明过程"]
            }
        
        # 简单的关键词匹配
        keywords = ['因为', '所以', '证明', '由于', '根据', '得出']
        has_keywords = any(keyword in student_answer for keyword in keywords)
        
        if has_keywords and len(student_answer) > 20:
            return {
                'is_correct': True,
                'score_earned': question['score'] * 0.8,  # 给80%分数，需要人工复核
                'feedback': "证明过程基本完整，建议人工复核",
                'error_type': '',
                'suggestions': ["建议检查证明的逻辑严密性"]
            }
        
        return {
            'is_correct': False,
            'score_earned': question['score'] * 0.3,  # 给30%分数
            'feedback': "证明过程不够完整",
            'error_type': 'incomplete_proof',
            'suggestions': ["请提供更详细的证明步骤", "注意使用数学术语"]
        }
    
    def _grade_application(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """评分应用题"""
        # 应用题评分，主要看答案和过程
        return self._grade_fill_blank(question, student_answer)
    
    def _grade_default(self, question: Dict, student_answer: str) -> Dict[str, Any]:
        """默认评分方法"""
        return self._grade_fill_blank(question, student_answer)
    
    def _is_math_equivalent(self, answer1: str, answer2: str) -> bool:
        """判断两个数学表达式是否等价"""
        try:
            # 简单的数学表达式等价判断
            # 移除空格
            a1 = re.sub(r'\s+', '', answer1)
            a2 = re.sub(r'\s+', '', answer2)
            
            # 处理常见的等价形式
            equivalents = [
                (r'(\d+)x', r'\1*x'),  # 3x -> 3*x
                (r'x(\d+)', r'x*\1'),  # x3 -> x*3
                (r'\+\s*-', '-'),      # +- -> -
                (r'-\s*-', '+'),       # -- -> +
            ]
            
            for pattern, replacement in equivalents:
                a1 = re.sub(pattern, replacement, a1)
                a2 = re.sub(pattern, replacement, a2)
            
            return a1.lower() == a2.lower()
            
        except:
            return False
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        if not str1 or not str2:
            return 0.0
        
        # 简单的编辑距离相似度
        len1, len2 = len(str1), len(str2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0
        
        # 计算最长公共子序列
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i-1].lower() == str2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[len1][len2]
        return lcs_length / max(len1, len2)
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """生成评分总结"""
        total_questions = len(results)
        correct_count = sum(1 for r in results if r['is_correct'])
        
        error_types = {}
        for result in results:
            if not result['is_correct'] and result['error_type']:
                error_types[result['error_type']] = error_types.get(result['error_type'], 0) + 1
        
        return {
            'total_questions': total_questions,
            'correct_count': correct_count,
            'accuracy_rate': round(correct_count / total_questions * 100, 2) if total_questions > 0 else 0,
            'common_errors': error_types,
            'suggestions': self._generate_suggestions(error_types)
        }
    
    def _generate_suggestions(self, error_types: Dict[str, int]) -> List[str]:
        """根据错误类型生成学习建议"""
        suggestions = []
        
        if 'wrong_choice' in error_types:
            suggestions.append("建议复习选择题的解题技巧，仔细阅读题目要求")
        
        if 'wrong_answer' in error_types:
            suggestions.append("建议加强基础知识的练习，注意计算准确性")
        
        if 'incomplete_proof' in error_types:
            suggestions.append("建议学习证明题的标准格式和逻辑推理方法")
        
        if 'partial_correct' in error_types:
            suggestions.append("答案接近正确，建议注意细节和表达的准确性")
        
        if not suggestions:
            suggestions.append("继续保持良好的学习状态！")
        
        return suggestions
