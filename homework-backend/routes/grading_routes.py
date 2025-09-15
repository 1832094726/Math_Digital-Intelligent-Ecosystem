#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评分管理路由
"""
from flask import Blueprint, request, jsonify
from functools import wraps
import json
from datetime import datetime

from services.auth_service import AuthService
from services.grading_service import GradingService
from models.database import db

# 创建蓝图
grading_bp = Blueprint('grading', __name__, url_prefix='/api/grading')

# 初始化服务
auth_service = AuthService()
grading_service = GradingService()

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_result = auth_service.verify_token(request)
        if not auth_result['success']:
            return jsonify(auth_result), 401
        request.current_user = auth_result['user']
        return f(*args, **kwargs)
    return decorated_function

def require_teacher(f):
    """教师权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.current_user['role'] != 'teacher':
            return jsonify({'success': False, 'message': '需要教师权限'}), 403
        return f(*args, **kwargs)
    return decorated_function

@grading_bp.route('/grade/<int:submission_id>', methods=['POST'])
@require_auth
@require_teacher
def grade_submission(submission_id):
    """
    自动评分学生提交
    """
    try:
        # 执行自动评分
        result = grading_service.grade_submission(submission_id)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': '评分完成',
                'data': result
            })
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'评分失败: {str(e)}'
        }), 500

@grading_bp.route('/batch-grade', methods=['POST'])
@require_auth
@require_teacher
def batch_grade():
    """
    批量评分
    """
    try:
        data = request.get_json()
        submission_ids = data.get('submission_ids', [])
        
        if not submission_ids:
            return jsonify({
                'success': False,
                'message': '请提供要评分的提交ID列表'
            }), 400
        
        results = []
        success_count = 0
        
        for submission_id in submission_ids:
            result = grading_service.grade_submission(submission_id)
            results.append({
                'submission_id': submission_id,
                'success': result['success'],
                'score': result.get('total_score', 0) if result['success'] else 0,
                'message': result.get('message', '')
            })
            
            if result['success']:
                success_count += 1
        
        return jsonify({
            'success': True,
            'message': f'批量评分完成，成功评分 {success_count}/{len(submission_ids)} 份',
            'data': {
                'total_count': len(submission_ids),
                'success_count': success_count,
                'results': results
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'批量评分失败: {str(e)}'
        }), 500

@grading_bp.route('/result/<int:submission_id>', methods=['GET'])
@require_auth
def get_grading_result(submission_id):
    """
    获取评分结果
    """
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 检查权限
                if request.current_user['role'] == 'student':
                    # 学生只能查看自己的评分结果
                    cursor.execute("""
                        SELECT hs.student_id 
                        FROM homework_submissions hs
                        WHERE hs.id = %s
                    """, (submission_id,))
                    
                    submission = cursor.fetchone()
                    if not submission or submission['student_id'] != request.current_user['id']:
                        return jsonify({
                            'success': False,
                            'message': '无权查看此评分结果'
                        }), 403
                
                # 获取评分结果
                cursor.execute("""
                    SELECT gr.*, hs.student_id, u.real_name as student_name,
                           h.title as homework_title
                    FROM grading_results gr
                    JOIN homework_submissions hs ON gr.submission_id = hs.id
                    JOIN homework_assignments ha ON hs.assignment_id = ha.id
                    JOIN homeworks h ON ha.homework_id = h.id
                    JOIN users u ON hs.student_id = u.id
                    WHERE gr.submission_id = %s
                """, (submission_id,))
                
                result = cursor.fetchone()
                if not result:
                    return jsonify({
                        'success': False,
                        'message': '评分结果不存在'
                    }), 404
                
                # 解析详细结果
                result_data = json.loads(result['result_data'])
                
                return jsonify({
                    'success': True,
                    'data': {
                        'submission_id': submission_id,
                        'student_name': result['student_name'],
                        'homework_title': result['homework_title'],
                        'total_score': result['total_score'],
                        'total_possible': result['total_possible'],
                        'accuracy': result['accuracy'],
                        'grading_method': result['grading_method'],
                        'graded_at': result['graded_at'].isoformat() if result['graded_at'] else None,
                        'question_results': result_data.get('question_results', []),
                        'summary': result_data.get('summary', {}),
                        'review_notes': result['review_notes']
                    }
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取评分结果失败: {str(e)}'
        }), 500

@grading_bp.route('/homework/<int:homework_id>/statistics', methods=['GET'])
@require_auth
@require_teacher
def get_homework_statistics(homework_id):
    """
    获取作业评分统计
    """
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 检查作业权限
                cursor.execute("""
                    SELECT created_by FROM homeworks WHERE id = %s
                """, (homework_id,))
                
                homework = cursor.fetchone()
                if not homework:
                    return jsonify({
                        'success': False,
                        'message': '作业不存在'
                    }), 404
                
                if homework['created_by'] != request.current_user['id']:
                    return jsonify({
                        'success': False,
                        'message': '无权查看此作业统计'
                    }), 403
                
                # 获取整体统计
                cursor.execute("""
                    SELECT 
                        COUNT(hs.id) as total_submissions,
                        COUNT(gr.id) as graded_submissions,
                        AVG(gr.total_score) as avg_score,
                        AVG(gr.accuracy) as avg_accuracy,
                        MIN(gr.total_score) as min_score,
                        MAX(gr.total_score) as max_score
                    FROM homework_assignments ha
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    LEFT JOIN grading_results gr ON hs.id = gr.submission_id
                    WHERE ha.homework_id = %s
                """, (homework_id,))
                
                overall_stats = cursor.fetchone()
                
                # 获取题目统计
                cursor.execute("""
                    SELECT 
                        q.id as question_id,
                        q.content as question_content,
                        q.question_type,
                        q.score as max_score,
                        COUNT(hs.id) as submission_count,
                        AVG(CASE WHEN JSON_EXTRACT(gr.result_data, CONCAT('$.question_results[', q.order_index-1, '].is_correct')) = true THEN 1 ELSE 0 END) as accuracy_rate
                    FROM questions q
                    LEFT JOIN homework_assignments ha ON q.homework_id = ha.homework_id
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    LEFT JOIN grading_results gr ON hs.id = gr.submission_id
                    WHERE q.homework_id = %s
                    GROUP BY q.id, q.content, q.question_type, q.score
                    ORDER BY q.order_index
                """, (homework_id,))
                
                question_stats = cursor.fetchall()
                
                # 获取错误类型统计
                cursor.execute("""
                    SELECT 
                        error_type,
                        COUNT(*) as count,
                        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM error_analysis ea2 
                                           JOIN homework_submissions hs2 ON ea2.submission_id = hs2.id
                                           JOIN homework_assignments ha2 ON hs2.assignment_id = ha2.id
                                           WHERE ha2.homework_id = %s) as percentage
                    FROM error_analysis ea
                    JOIN homework_submissions hs ON ea.submission_id = hs.id
                    JOIN homework_assignments ha ON hs.assignment_id = ha.id
                    WHERE ha.homework_id = %s
                    GROUP BY error_type
                    ORDER BY count DESC
                """, (homework_id, homework_id))
                
                error_stats = cursor.fetchall()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'homework_id': homework_id,
                        'overall_statistics': {
                            'total_submissions': overall_stats['total_submissions'] or 0,
                            'graded_submissions': overall_stats['graded_submissions'] or 0,
                            'average_score': round(overall_stats['avg_score'] or 0, 2),
                            'average_accuracy': round(overall_stats['avg_accuracy'] or 0, 2),
                            'min_score': overall_stats['min_score'] or 0,
                            'max_score': overall_stats['max_score'] or 0,
                            'completion_rate': round((overall_stats['graded_submissions'] or 0) / max(overall_stats['total_submissions'] or 1, 1) * 100, 2)
                        },
                        'question_statistics': [
                            {
                                'question_id': q['question_id'],
                                'question_content': q['question_content'][:50] + '...' if len(q['question_content']) > 50 else q['question_content'],
                                'question_type': q['question_type'],
                                'max_score': q['max_score'],
                                'submission_count': q['submission_count'] or 0,
                                'accuracy_rate': round((q['accuracy_rate'] or 0) * 100, 2)
                            }
                            for q in question_stats
                        ],
                        'error_statistics': [
                            {
                                'error_type': e['error_type'],
                                'count': e['count'],
                                'percentage': round(e['percentage'], 2)
                            }
                            for e in error_stats
                        ]
                    }
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取统计信息失败: {str(e)}'
        }), 500

@grading_bp.route('/review/<int:submission_id>', methods=['POST'])
@require_auth
@require_teacher
def review_grading(submission_id):
    """
    人工复核评分
    """
    try:
        data = request.get_json()
        review_notes = data.get('review_notes', '')
        score_adjustments = data.get('score_adjustments', {})  # {question_id: new_score}
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 获取当前评分结果
                cursor.execute("""
                    SELECT result_data, total_score 
                    FROM grading_results 
                    WHERE submission_id = %s
                """, (submission_id,))
                
                result = cursor.fetchone()
                if not result:
                    return jsonify({
                        'success': False,
                        'message': '评分结果不存在'
                    }), 404
                
                result_data = json.loads(result['result_data'])
                
                # 应用分数调整
                new_total_score = 0
                for question_result in result_data.get('question_results', []):
                    question_id = str(question_result['question_id'])
                    if question_id in score_adjustments:
                        question_result['score_earned'] = float(score_adjustments[question_id])
                        question_result['manually_adjusted'] = True
                    new_total_score += question_result['score_earned']
                
                # 更新评分结果
                cursor.execute("""
                    UPDATE grading_results 
                    SET result_data = %s, total_score = %s, reviewed_at = NOW(), review_notes = %s
                    WHERE submission_id = %s
                """, (json.dumps(result_data), new_total_score, review_notes, submission_id))
                
                # 更新提交记录的分数
                cursor.execute("""
                    UPDATE homework_submissions 
                    SET score = %s 
                    WHERE id = %s
                """, (new_total_score, submission_id))
                
                conn.commit()
                
                return jsonify({
                    'success': True,
                    'message': '复核完成',
                    'data': {
                        'submission_id': submission_id,
                        'new_total_score': new_total_score,
                        'review_notes': review_notes
                    }
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'复核失败: {str(e)}'
        }), 500

@grading_bp.route('/rules/<int:homework_id>', methods=['GET'])
@require_auth
@require_teacher
def get_grading_rules(homework_id):
    """
    获取评分规则
    """
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                cursor.execute("""
                    SELECT * FROM grading_rules 
                    WHERE homework_id = %s AND is_active = 1
                    ORDER BY question_type
                """, (homework_id,))
                
                rules = cursor.fetchall()
                
                return jsonify({
                    'success': True,
                    'data': [
                        {
                            'id': rule['id'],
                            'question_type': rule['question_type'],
                            'rule_config': json.loads(rule['rule_config']),
                            'created_at': rule['created_at'].isoformat() if rule['created_at'] else None
                        }
                        for rule in rules
                    ]
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取评分规则失败: {str(e)}'
        }), 500

@grading_bp.route('/rules/<int:homework_id>', methods=['POST'])
@require_auth
@require_teacher
def update_grading_rules(homework_id):
    """
    更新评分规则
    """
    try:
        data = request.get_json()
        rules = data.get('rules', [])
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 检查作业权限
                cursor.execute("""
                    SELECT created_by FROM homeworks WHERE id = %s
                """, (homework_id,))
                
                homework = cursor.fetchone()
                if not homework or homework['created_by'] != request.current_user['id']:
                    return jsonify({
                        'success': False,
                        'message': '无权修改此作业的评分规则'
                    }), 403
                
                # 删除旧规则
                cursor.execute("""
                    DELETE FROM grading_rules WHERE homework_id = %s
                """, (homework_id,))
                
                # 插入新规则
                for rule in rules:
                    cursor.execute("""
                        INSERT INTO grading_rules (homework_id, question_type, rule_config, created_by)
                        VALUES (%s, %s, %s, %s)
                    """, (homework_id, rule['question_type'], json.dumps(rule['rule_config']), request.current_user['id']))
                
                conn.commit()
                
                return jsonify({
                    'success': True,
                    'message': '评分规则更新成功'
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'更新评分规则失败: {str(e)}'
        }), 500
