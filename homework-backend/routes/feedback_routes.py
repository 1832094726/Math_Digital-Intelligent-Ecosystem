"""
作业反馈路由
提供学生查看作业反馈的API接口
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import json
from datetime import datetime

from models.database import DatabaseManager

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')
db = DatabaseManager()

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # JWT认证逻辑
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'success': False, 'message': '缺少认证令牌'}), 401
            
            token = auth_header.split(' ')[1]
            # 这里应该验证JWT token，简化处理
            request.current_user = {'id': 2, 'role': 'student'}  # 模拟用户信息
            
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'success': False, 'message': '认证失败'}), 401
    return decorated_function

@feedback_bp.route('/homework/<int:homework_id>', methods=['GET'])
@require_auth
def get_homework_feedback(homework_id):
    """
    获取作业反馈详情
    包括个人得分、班级统计、错误分析、学习建议等
    """
    try:
        user_id = request.current_user['id']
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 获取学生的作业提交记录
                cursor.execute("""
                    SELECT hs.*, h.title, h.max_score as homework_max_score, h.subject, h.grade as grade_level
                    FROM homework_submissions hs
                    JOIN homeworks h ON hs.homework_id = h.id
                    WHERE hs.homework_id = %s AND hs.student_id = %s
                    ORDER BY hs.submitted_at DESC
                    LIMIT 1
                """, (homework_id, user_id))
                
                submission = cursor.fetchone()
                if not submission:
                    return jsonify({
                        'success': False,
                        'message': '未找到作业提交记录'
                    }), 404
                
                # 获取详细的评分结果（如果表存在）
                grading_result = None
                try:
                    cursor.execute("""
                        SELECT * FROM grading_results
                        WHERE submission_id = %s
                    """, (submission['id'],))
                    grading_result = cursor.fetchone()
                except:
                    # grading_results表不存在，使用默认值
                    grading_result = None
                
                # 获取班级统计信息
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_students,
                        AVG(hs.score) as avg_score,
                        MAX(hs.score) as max_score,
                        MIN(hs.score) as min_score,
                        STDDEV(hs.score) as std_dev
                    FROM homework_submissions hs
                    WHERE hs.homework_id = %s AND hs.status = 'graded'
                """, (homework_id,))
                
                class_stats = cursor.fetchone()
                
                # 计算学生排名
                cursor.execute("""
                    SELECT COUNT(*) + 1 as rank
                    FROM homework_submissions hs
                    WHERE hs.homework_id = %s AND hs.score > %s AND hs.status = 'graded'
                """, (homework_id, submission['score']))
                
                rank_result = cursor.fetchone()
                student_rank = rank_result['rank'] if rank_result else 1
                
                # 获取题目详细反馈（简化版，不依赖question_answers表）
                cursor.execute("""
                    SELECT q.*, NULL as student_answer, NULL as is_correct, NULL as score_earned
                    FROM questions q
                    WHERE q.homework_id = %s
                    ORDER BY q.order_index
                """, (homework_id,))
                
                questions = cursor.fetchall()
                
                # 生成学习建议
                learning_suggestions = generate_learning_suggestions(
                    submission, grading_result, questions, class_stats
                )
                
                # 构建反馈响应
                feedback_data = {
                    'homework_info': {
                        'id': homework_id,
                        'title': submission['title'],
                        'subject': submission['subject'],
                        'grade_level': submission['grade_level']
                    },
                    'personal_performance': {
                        'total_score': float(submission['score']) if submission['score'] else 0,
                        'max_score': float(submission['homework_max_score']) if submission['homework_max_score'] else 100,
                        'percentage': round((float(submission['score']) / float(submission['homework_max_score'])) * 100, 1) if submission['score'] and submission['homework_max_score'] else 0,
                        'completion_time': submission.get('time_spent'),
                        'submitted_at': submission['submitted_at'].isoformat() if submission['submitted_at'] else None
                    },
                    'class_statistics': {
                        'total_students': class_stats['total_students'] if class_stats else 0,
                        'class_average': round(class_stats['avg_score'], 1) if class_stats and class_stats['avg_score'] else 0,
                        'class_max': class_stats['max_score'] if class_stats else 0,
                        'class_min': class_stats['min_score'] if class_stats else 0,
                        'student_rank': student_rank,
                        'percentile': round((1 - (student_rank - 1) / max(class_stats['total_students'], 1)) * 100, 1) if class_stats else 100
                    },
                    'question_feedback': format_question_feedback(questions),
                    'learning_suggestions': learning_suggestions,
                    'error_analysis': analyze_errors(questions) if grading_result else {},
                    'generated_at': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'data': feedback_data
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取反馈失败: {str(e)}'
        }), 500

def format_question_feedback(questions):
    """格式化题目反馈"""
    feedback = []
    
    for question in questions:
        question_feedback = {
            'question_id': question['id'],
            'question_order': question['order_index'],
            'question_type': question['question_type'],
            'question_content': question['content'],
            'student_answer': question.get('student_answer', ''),
            'correct_answer': question.get('correct_answer', ''),
            'is_correct': question.get('is_correct', False),
            'score_earned': question.get('score_earned', 0),
            'max_score': float(question.get('score', 0)),
            'explanation': question.get('explanation', ''),
            'knowledge_points': json.loads(question.get('knowledge_points', '[]')) if question.get('knowledge_points') else []
        }
        
        # 添加错误分析
        if not question_feedback['is_correct']:
            question_feedback['error_analysis'] = analyze_question_error(question)
        
        feedback.append(question_feedback)
    
    return feedback

def analyze_question_error(question):
    """分析单题错误"""
    error_types = []
    suggestions = []
    
    question_type = question.get('question_type', '')
    student_answer = question.get('student_answer', '')
    correct_answer = question.get('correct_answer', '')
    
    if question_type == 'calculation':
        # 计算题错误分析
        try:
            if student_answer and correct_answer:
                student_val = float(student_answer)
                correct_val = float(correct_answer)
                error_rate = abs(student_val - correct_val) / abs(correct_val) if correct_val != 0 else 1
                
                if error_rate < 0.1:
                    error_types.append('计算精度错误')
                    suggestions.append('注意计算精度，检查小数点位置')
                elif error_rate < 0.5:
                    error_types.append('计算步骤错误')
                    suggestions.append('检查计算步骤，确保运算顺序正确')
                else:
                    error_types.append('方法错误')
                    suggestions.append('重新理解题目要求，选择正确的解题方法')
        except:
            error_types.append('答案格式错误')
            suggestions.append('检查答案格式，确保符合题目要求')
    
    elif question_type == 'choice':
        error_types.append('概念理解错误')
        suggestions.append('复习相关概念，理解选项之间的区别')
    
    elif question_type == 'fill_blank':
        if len(student_answer) != len(correct_answer):
            error_types.append('答案不完整')
            suggestions.append('确保回答了所有空白处')
        else:
            error_types.append('知识点掌握不牢')
            suggestions.append('加强相关知识点的练习')
    
    return {
        'error_types': error_types,
        'suggestions': suggestions
    }

def analyze_errors(questions):
    """分析整体错误模式"""
    total_questions = len(questions)
    wrong_questions = [q for q in questions if not q.get('is_correct', False)]
    
    if not wrong_questions:
        return {
            'overall_performance': 'excellent',
            'main_issues': [],
            'improvement_areas': []
        }
    
    # 按题型统计错误
    error_by_type = {}
    for question in wrong_questions:
        q_type = question.get('question_type', 'unknown')
        error_by_type[q_type] = error_by_type.get(q_type, 0) + 1
    
    # 识别主要问题
    main_issues = []
    improvement_areas = []
    
    error_rate = len(wrong_questions) / total_questions
    
    if error_rate > 0.7:
        main_issues.append('基础知识掌握不足')
        improvement_areas.append('建议重新学习本章节的基础概念')
    elif error_rate > 0.4:
        main_issues.append('部分知识点理解不够深入')
        improvement_areas.append('针对错题涉及的知识点进行专项练习')
    else:
        main_issues.append('个别题目出现失误')
        improvement_areas.append('注意审题和计算细节')
    
    # 根据错误题型给出建议
    if error_by_type.get('calculation', 0) > total_questions * 0.3:
        improvement_areas.append('加强计算能力训练，注意运算准确性')
    
    if error_by_type.get('application', 0) > 0:
        improvement_areas.append('提高解决实际问题的能力，多做应用题练习')
    
    return {
        'overall_performance': 'needs_improvement' if error_rate > 0.5 else 'good',
        'error_rate': round(error_rate * 100, 1),
        'main_issues': main_issues,
        'improvement_areas': improvement_areas,
        'error_distribution': error_by_type
    }

def generate_learning_suggestions(submission, grading_result, questions, class_stats):
    """生成个性化学习建议"""
    suggestions = []
    
    # 基于分数的建议
    score_percentage = (float(submission['score']) / float(submission['homework_max_score'])) * 100 if submission['score'] and submission['homework_max_score'] else 0
    
    if score_percentage >= 90:
        suggestions.append({
            'type': 'encouragement',
            'title': '优秀表现',
            'content': '你的表现非常出色！可以尝试更有挑战性的题目来进一步提升。',
            'priority': 'low'
        })
    elif score_percentage >= 70:
        suggestions.append({
            'type': 'improvement',
            'title': '继续努力',
            'content': '整体表现良好，重点关注错题涉及的知识点，争取更好的成绩。',
            'priority': 'medium'
        })
    else:
        suggestions.append({
            'type': 'remediation',
            'title': '需要加强',
            'content': '建议重新学习相关知识点，多做基础练习题巩固理解。',
            'priority': 'high'
        })
    
    # 基于班级排名的建议
    if class_stats and class_stats['total_students'] > 1:
        avg_score = class_stats['avg_score']
        if float(submission['score']) > avg_score:
            suggestions.append({
                'type': 'comparison',
                'title': '超越平均',
                'content': f'你的成绩超过了班级平均分({avg_score:.1f}分)，继续保持！',
                'priority': 'low'
            })
        else:
            suggestions.append({
                'type': 'comparison',
                'title': '追赶目标',
                'content': f'距离班级平均分({avg_score:.1f}分)还有提升空间，加油！',
                'priority': 'medium'
            })
    
    # 基于错题的具体建议
    wrong_questions = [q for q in questions if not q.get('is_correct', False)] if questions else []
    if wrong_questions:
        knowledge_points = set()
        for q in wrong_questions:
            try:
                points = q.get('knowledge_points', [])
                if isinstance(points, str):
                    points = json.loads(points)
                if isinstance(points, list):
                    knowledge_points.update([str(p) for p in points])
            except:
                continue

        if knowledge_points:
            suggestions.append({
                'type': 'knowledge_review',
                'title': '重点复习',
                'content': f'建议重点复习以下知识点：{", ".join(list(knowledge_points)[:3])}',
                'priority': 'high'
            })
    
    return suggestions

@feedback_bp.route('/homework/<int:homework_id>/share', methods=['POST'])
@require_auth
def share_feedback(homework_id):
    """分享作业反馈"""
    try:
        data = request.get_json()
        share_type = data.get('type', 'link')  # link, pdf, image
        
        # 生成分享链接或文件
        if share_type == 'link':
            share_url = f"/feedback/shared/{homework_id}?token=shared_token_here"
            return jsonify({
                'success': True,
                'share_url': share_url,
                'expires_at': (datetime.now().timestamp() + 7*24*3600)  # 7天后过期
            })
        
        return jsonify({
            'success': True,
            'message': '分享功能开发中'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'分享失败: {str(e)}'
        }), 500
