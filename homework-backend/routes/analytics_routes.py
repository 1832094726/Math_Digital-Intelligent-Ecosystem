"""
作业统计分析路由
为教师提供详细的作业分析报告和教学建议
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import json
from datetime import datetime, timedelta
from collections import defaultdict

from models.database import DatabaseManager

analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')
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
            request.current_user = {'id': 3, 'role': 'teacher'}  # 模拟用户信息
            
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'success': False, 'message': '认证失败'}), 401
    return decorated_function

def require_teacher(f):
    """教师权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.current_user['role'] != 'teacher':
            return jsonify({'success': False, 'message': '需要教师权限'}), 403
        return f(*args, **kwargs)
    return decorated_function

@analytics_bp.route('/homework/<int:homework_id>', methods=['GET'])
@require_auth
@require_teacher
def get_homework_analytics(homework_id):
    """
    获取单个作业的详细分析报告
    """
    try:
        teacher_id = request.current_user['id']
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 验证作业权限
                cursor.execute("""
                    SELECT * FROM homeworks 
                    WHERE id = %s AND created_by = %s
                """, (homework_id, teacher_id))
                
                homework = cursor.fetchone()
                if not homework:
                    return jsonify({
                        'success': False,
                        'message': '作业不存在或无权访问'
                    }), 404
                
                # 获取基础统计信息
                basic_stats = get_basic_statistics(cursor, homework_id)
                
                # 获取完成率分析
                completion_analysis = get_completion_analysis(cursor, homework_id)
                
                # 获取分数分布分析
                score_distribution = get_score_distribution(cursor, homework_id)
                
                # 获取题目分析
                question_analysis = get_question_analysis(cursor, homework_id)
                
                # 获取知识点掌握度分析
                knowledge_analysis = get_knowledge_analysis(cursor, homework_id)
                
                # 获取学生表现分析
                student_performance = get_student_performance(cursor, homework_id)
                
                # 生成教学建议
                teaching_suggestions = generate_teaching_suggestions(
                    basic_stats, score_distribution, question_analysis, knowledge_analysis
                )
                
                # 构建分析报告
                analytics_data = {
                    'homework_info': {
                        'id': homework_id,
                        'title': homework['title'],
                        'subject': homework['subject'],
                        'grade_level': homework['grade_level'],
                        'created_at': homework['created_at'].isoformat() if homework['created_at'] else None,
                        'due_date': homework['due_date'].isoformat() if homework['due_date'] else None
                    },
                    'basic_statistics': basic_stats,
                    'completion_analysis': completion_analysis,
                    'score_distribution': score_distribution,
                    'question_analysis': question_analysis,
                    'knowledge_analysis': knowledge_analysis,
                    'student_performance': student_performance,
                    'teaching_suggestions': teaching_suggestions,
                    'generated_at': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'data': analytics_data
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取分析报告失败: {str(e)}'
        }), 500

def get_basic_statistics(cursor, homework_id):
    """获取基础统计信息"""
    cursor.execute("""
        SELECT 
            COUNT(*) as total_assignments,
            COUNT(CASE WHEN hs.submission_status = 'completed' THEN 1 END) as completed_count,
            COUNT(CASE WHEN hs.submission_status = 'in_progress' THEN 1 END) as in_progress_count,
            COUNT(CASE WHEN hs.due_date < NOW() AND hs.submission_status != 'completed' THEN 1 END) as overdue_count,
            AVG(CASE WHEN hs.submission_status = 'completed' THEN hs.total_score END) as avg_score,
            MAX(CASE WHEN hs.submission_status = 'completed' THEN hs.total_score END) as max_score,
            MIN(CASE WHEN hs.submission_status = 'completed' THEN hs.total_score END) as min_score,
            AVG(CASE WHEN hs.submission_status = 'completed' THEN hs.completion_time END) as avg_completion_time
        FROM homework_assignments ha
        LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
        WHERE ha.homework_id = %s
    """, (homework_id,))
    
    stats = cursor.fetchone()
    
    return {
        'total_assignments': stats['total_assignments'] or 0,
        'completed_count': stats['completed_count'] or 0,
        'in_progress_count': stats['in_progress_count'] or 0,
        'overdue_count': stats['overdue_count'] or 0,
        'completion_rate': round((stats['completed_count'] or 0) / max(stats['total_assignments'] or 1, 1) * 100, 1),
        'average_score': round(stats['avg_score'] or 0, 1),
        'highest_score': stats['max_score'] or 0,
        'lowest_score': stats['min_score'] or 0,
        'average_completion_time': round(stats['avg_completion_time'] or 0, 1)
    }

def get_completion_analysis(cursor, homework_id):
    """获取完成率分析"""
    # 按时间段统计提交情况
    cursor.execute("""
        SELECT 
            DATE(hs.submitted_at) as submit_date,
            COUNT(*) as submissions_count
        FROM homework_submissions hs
        JOIN homework_assignments ha ON hs.assignment_id = ha.id
        WHERE ha.homework_id = %s AND hs.submission_status = 'completed'
        GROUP BY DATE(hs.submitted_at)
        ORDER BY submit_date
    """, (homework_id,))
    
    daily_submissions = cursor.fetchall()
    
    # 按班级统计完成率
    cursor.execute("""
        SELECT 
            c.name as class_name,
            COUNT(ha.id) as total_assignments,
            COUNT(CASE WHEN hs.submission_status = 'completed' THEN 1 END) as completed_assignments
        FROM classes c
        JOIN homework_assignments ha ON c.id = ha.class_id
        LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
        WHERE ha.homework_id = %s
        GROUP BY c.id, c.name
    """, (homework_id,))
    
    class_completion = cursor.fetchall()
    
    return {
        'daily_submissions': [
            {
                'date': item['submit_date'].isoformat() if item['submit_date'] else None,
                'count': item['submissions_count']
            }
            for item in daily_submissions
        ],
        'class_completion': [
            {
                'class_name': item['class_name'],
                'total': item['total_assignments'],
                'completed': item['completed_assignments'],
                'completion_rate': round((item['completed_assignments'] / max(item['total_assignments'], 1)) * 100, 1)
            }
            for item in class_completion
        ]
    }

def get_score_distribution(cursor, homework_id):
    """获取分数分布分析"""
    cursor.execute("""
        SELECT hs.total_score, h.total_score as max_score
        FROM homework_submissions hs
        JOIN homework_assignments ha ON hs.assignment_id = ha.id
        JOIN homeworks h ON ha.homework_id = h.id
        WHERE ha.homework_id = %s AND hs.submission_status = 'completed'
    """, (homework_id,))
    
    scores = cursor.fetchall()
    
    if not scores:
        return {
            'distribution': [],
            'statistics': {}
        }
    
    max_score = scores[0]['max_score'] if scores else 100
    score_values = [s['total_score'] for s in scores]
    
    # 分数段统计
    ranges = [
        (90, 100, '优秀'),
        (80, 89, '良好'),
        (70, 79, '中等'),
        (60, 69, '及格'),
        (0, 59, '不及格')
    ]
    
    distribution = []
    for min_score, max_range, label in ranges:
        count = len([s for s in score_values if min_score <= (s/max_score*100) <= max_range])
        distribution.append({
            'range': f'{min_score}-{max_range}',
            'label': label,
            'count': count,
            'percentage': round(count / len(score_values) * 100, 1) if score_values else 0
        })
    
    # 统计信息
    statistics = {
        'mean': round(sum(score_values) / len(score_values), 1) if score_values else 0,
        'median': sorted(score_values)[len(score_values)//2] if score_values else 0,
        'std_dev': round(calculate_std_dev(score_values), 1) if len(score_values) > 1 else 0,
        'total_students': len(score_values)
    }
    
    return {
        'distribution': distribution,
        'statistics': statistics
    }

def get_question_analysis(cursor, homework_id):
    """获取题目分析"""
    cursor.execute("""
        SELECT 
            q.id,
            q.question_order,
            q.question_type,
            q.question_content,
            q.score as max_score,
            COUNT(qa.id) as total_answers,
            COUNT(CASE WHEN qa.is_correct = 1 THEN 1 END) as correct_answers,
            AVG(qa.score_earned) as avg_score_earned
        FROM questions q
        LEFT JOIN question_answers qa ON q.id = qa.question_id
        LEFT JOIN homework_submissions hs ON qa.submission_id = hs.id
        WHERE q.homework_id = %s AND (hs.submission_status = 'completed' OR hs.submission_status IS NULL)
        GROUP BY q.id, q.question_order, q.question_type, q.question_content, q.score
        ORDER BY q.question_order
    """, (homework_id,))
    
    questions = cursor.fetchall()
    
    question_analysis = []
    for q in questions:
        correct_rate = (q['correct_answers'] / max(q['total_answers'], 1)) * 100 if q['total_answers'] else 0
        difficulty_level = 'easy' if correct_rate > 80 else 'medium' if correct_rate > 50 else 'hard'
        
        question_analysis.append({
            'question_id': q['id'],
            'question_order': q['question_order'],
            'question_type': q['question_type'],
            'question_content': q['question_content'][:100] + '...' if len(q['question_content']) > 100 else q['question_content'],
            'max_score': q['max_score'],
            'total_answers': q['total_answers'] or 0,
            'correct_answers': q['correct_answers'] or 0,
            'correct_rate': round(correct_rate, 1),
            'average_score': round(q['avg_score_earned'] or 0, 1),
            'difficulty_level': difficulty_level
        })
    
    return question_analysis

def get_knowledge_analysis(cursor, homework_id):
    """获取知识点掌握度分析"""
    cursor.execute("""
        SELECT 
            q.knowledge_points,
            COUNT(qa.id) as total_attempts,
            COUNT(CASE WHEN qa.is_correct = 1 THEN 1 END) as correct_attempts
        FROM questions q
        LEFT JOIN question_answers qa ON q.id = qa.question_id
        LEFT JOIN homework_submissions hs ON qa.submission_id = hs.id
        WHERE q.homework_id = %s AND (hs.submission_status = 'completed' OR hs.submission_status IS NULL)
        AND q.knowledge_points IS NOT NULL AND q.knowledge_points != '[]'
        GROUP BY q.knowledge_points
    """, (homework_id,))
    
    knowledge_data = cursor.fetchall()
    
    knowledge_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for item in knowledge_data:
        try:
            points = json.loads(item['knowledge_points'] or '[]')
            for point in points:
                knowledge_stats[point]['total'] += item['total_attempts'] or 0
                knowledge_stats[point]['correct'] += item['correct_attempts'] or 0
        except:
            continue
    
    knowledge_analysis = []
    for point, stats in knowledge_stats.items():
        mastery_rate = (stats['correct'] / max(stats['total'], 1)) * 100
        mastery_level = 'excellent' if mastery_rate > 85 else 'good' if mastery_rate > 70 else 'needs_improvement'
        
        knowledge_analysis.append({
            'knowledge_point': point,
            'total_attempts': stats['total'],
            'correct_attempts': stats['correct'],
            'mastery_rate': round(mastery_rate, 1),
            'mastery_level': mastery_level
        })
    
    return sorted(knowledge_analysis, key=lambda x: x['mastery_rate'])

def get_student_performance(cursor, homework_id):
    """获取学生表现分析"""
    cursor.execute("""
        SELECT 
            u.id,
            u.real_name,
            u.username,
            c.name as class_name,
            hs.total_score,
            hs.completion_time,
            hs.submitted_at,
            h.total_score as max_score
        FROM users u
        JOIN homework_assignments ha ON u.id = ha.student_id
        LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
        LEFT JOIN classes c ON ha.class_id = c.id
        JOIN homeworks h ON ha.homework_id = h.id
        WHERE ha.homework_id = %s
        ORDER BY hs.total_score DESC
    """, (homework_id,))
    
    students = cursor.fetchall()
    
    # 识别需要关注的学生
    struggling_students = []
    excellent_students = []
    
    for student in students:
        if student['total_score'] is not None:
            score_percentage = (student['total_score'] / student['max_score']) * 100
            
            if score_percentage < 60:
                struggling_students.append({
                    'student_id': student['id'],
                    'name': student['real_name'],
                    'class_name': student['class_name'],
                    'score': student['total_score'],
                    'score_percentage': round(score_percentage, 1)
                })
            elif score_percentage > 90:
                excellent_students.append({
                    'student_id': student['id'],
                    'name': student['real_name'],
                    'class_name': student['class_name'],
                    'score': student['total_score'],
                    'score_percentage': round(score_percentage, 1)
                })
    
    return {
        'total_students': len(students),
        'struggling_students': struggling_students,
        'excellent_students': excellent_students,
        'completion_summary': {
            'completed': len([s for s in students if s['total_score'] is not None]),
            'pending': len([s for s in students if s['total_score'] is None])
        }
    }

def generate_teaching_suggestions(basic_stats, score_distribution, question_analysis, knowledge_analysis):
    """生成教学建议"""
    suggestions = []
    
    # 基于完成率的建议
    completion_rate = basic_stats['completion_rate']
    if completion_rate < 70:
        suggestions.append({
            'type': 'completion',
            'priority': 'high',
            'title': '提高作业完成率',
            'content': f'当前完成率仅{completion_rate}%，建议：1) 适当延长截止时间；2) 提供更多指导；3) 简化题目难度',
            'action_items': ['延长截止时间', '增加课堂指导', '调整题目难度']
        })
    
    # 基于分数分布的建议
    stats = score_distribution['statistics']
    if stats['mean'] < 70:
        suggestions.append({
            'type': 'difficulty',
            'priority': 'high',
            'title': '降低题目难度',
            'content': f'平均分仅{stats["mean"]}分，题目可能过难，建议增加基础题目比例',
            'action_items': ['增加基础题目', '提供解题示例', '课前复习重点']
        })
    
    # 基于题目分析的建议
    difficult_questions = [q for q in question_analysis if q['correct_rate'] < 50]
    if difficult_questions:
        suggestions.append({
            'type': 'question_review',
            'priority': 'medium',
            'title': '重点讲解难题',
            'content': f'有{len(difficult_questions)}道题目正确率低于50%，需要重点讲解',
            'action_items': [f'讲解第{q["question_order"]}题' for q in difficult_questions[:3]]
        })
    
    # 基于知识点分析的建议
    weak_knowledge = [k for k in knowledge_analysis if k['mastery_rate'] < 60]
    if weak_knowledge:
        suggestions.append({
            'type': 'knowledge_reinforcement',
            'priority': 'high',
            'title': '加强薄弱知识点',
            'content': f'以下知识点掌握不足：{", ".join([k["knowledge_point"] for k in weak_knowledge[:3]])}',
            'action_items': [f'复习{k["knowledge_point"]}' for k in weak_knowledge[:3]]
        })
    
    return suggestions

def calculate_std_dev(values):
    """计算标准差"""
    if len(values) < 2:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

@analytics_bp.route('/homework/<int:homework_id>/export', methods=['POST'])
@require_auth
@require_teacher
def export_analytics(homework_id):
    """导出分析报告"""
    try:
        data = request.get_json()
        export_format = data.get('format', 'pdf')  # pdf, excel, csv
        
        # 这里应该实现实际的导出逻辑
        return jsonify({
            'success': True,
            'message': f'报告导出功能开发中 (格式: {export_format})',
            'download_url': f'/downloads/analytics_{homework_id}.{export_format}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'导出失败: {str(e)}'
        }), 500

@analytics_bp.route('/overview', methods=['GET'])
@require_auth
@require_teacher
def get_teacher_overview():
    """获取教师的整体分析概览"""
    try:
        teacher_id = request.current_user['id']
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 获取教师的作业统计
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_homeworks,
                        COUNT(CASE WHEN is_published = 1 THEN 1 END) as published_homeworks,
                        AVG(total_score) as avg_homework_score
                    FROM homeworks 
                    WHERE created_by = %s
                """, (teacher_id,))
                
                homework_stats = cursor.fetchone()
                
                # 获取最近的作业表现
                cursor.execute("""
                    SELECT 
                        h.title,
                        h.created_at,
                        COUNT(hs.id) as submissions,
                        AVG(hs.total_score) as avg_score
                    FROM homeworks h
                    LEFT JOIN homework_assignments ha ON h.id = ha.homework_id
                    LEFT JOIN homework_submissions hs ON ha.id = hs.assignment_id
                    WHERE h.created_by = %s
                    GROUP BY h.id, h.title, h.created_at
                    ORDER BY h.created_at DESC
                    LIMIT 5
                """, (teacher_id,))
                
                recent_homeworks = cursor.fetchall()
                
                overview_data = {
                    'summary': {
                        'total_homeworks': homework_stats['total_homeworks'] or 0,
                        'published_homeworks': homework_stats['published_homeworks'] or 0,
                        'average_score': round(homework_stats['avg_homework_score'] or 0, 1)
                    },
                    'recent_homeworks': [
                        {
                            'title': hw['title'],
                            'created_at': hw['created_at'].isoformat() if hw['created_at'] else None,
                            'submissions': hw['submissions'] or 0,
                            'average_score': round(hw['avg_score'] or 0, 1)
                        }
                        for hw in recent_homeworks
                    ],
                    'generated_at': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'data': overview_data
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取概览失败: {str(e)}'
        }), 500
