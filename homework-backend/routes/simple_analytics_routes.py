"""
简化版作业分析路由
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import json
from datetime import datetime

from models.database import DatabaseManager

simple_analytics_bp = Blueprint('simple_analytics', __name__, url_prefix='/api/simple-analytics')
db = DatabaseManager()

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'success': False, 'message': '缺少认证令牌'}), 401
            
            token = auth_header.split(' ')[1]
            request.current_user = {'id': 3, 'role': 'teacher'}  # 模拟教师用户
            
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'success': False, 'message': '认证失败'}), 401
    return decorated_function

def require_teacher(f):
    """教师权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.current_user.get('role') != 'teacher':
            return jsonify({'success': False, 'message': '需要教师权限'}), 403
        return f(*args, **kwargs)
    return decorated_function

@simple_analytics_bp.route('/homework/<int:homework_id>', methods=['GET'])
@require_auth
@require_teacher
def get_simple_homework_analytics(homework_id):
    """
    获取简化版作业分析
    """
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                
                # 获取作业基本信息
                cursor.execute("""
                    SELECT h.*, u.username as creator_name
                    FROM homeworks h
                    LEFT JOIN users u ON h.created_by = u.id
                    WHERE h.id = %s
                """, (homework_id,))
                
                homework = cursor.fetchone()
                if not homework:
                    return jsonify({
                        'success': False,
                        'message': '作业不存在'
                    }), 404
                
                # 获取提交统计
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_submissions,
                        COUNT(CASE WHEN status = 'graded' THEN 1 END) as graded_count,
                        AVG(CASE WHEN status = 'graded' THEN score END) as avg_score,
                        MAX(CASE WHEN status = 'graded' THEN score END) as max_score,
                        MIN(CASE WHEN status = 'graded' THEN score END) as min_score,
                        STDDEV(CASE WHEN status = 'graded' THEN score END) as std_dev
                    FROM homework_submissions
                    WHERE homework_id = %s
                """, (homework_id,))
                
                stats = cursor.fetchone()
                
                # 获取分数分布
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN score >= 90 THEN '90-100'
                            WHEN score >= 80 THEN '80-89'
                            WHEN score >= 70 THEN '70-79'
                            WHEN score >= 60 THEN '60-69'
                            ELSE '0-59'
                        END as score_range,
                        COUNT(*) as count
                    FROM homework_submissions
                    WHERE homework_id = %s AND status = 'graded'
                    GROUP BY score_range
                    ORDER BY score_range DESC
                """, (homework_id,))
                
                score_distribution = cursor.fetchall()
                
                # 获取题目分析
                cursor.execute("""
                    SELECT q.*
                    FROM questions q
                    WHERE q.homework_id = %s
                    ORDER BY q.order_index
                """, (homework_id,))
                
                questions = cursor.fetchall()
                
                # 获取学生表现
                cursor.execute("""
                    SELECT 
                        u.username,
                        u.real_name,
                        hs.score,
                        hs.submitted_at,
                        hs.time_spent,
                        hs.status
                    FROM homework_submissions hs
                    JOIN users u ON hs.student_id = u.id
                    WHERE hs.homework_id = %s
                    ORDER BY hs.score DESC
                """, (homework_id,))
                
                student_performances = cursor.fetchall()
                
                # 构建分析响应
                analytics_data = {
                    'homework_info': {
                        'id': homework_id,
                        'title': homework['title'],
                        'subject': homework['subject'],
                        'grade': homework['grade'],
                        'max_score': float(homework['max_score']) if homework['max_score'] else 100,
                        'created_by': homework.get('creator_name', '未知'),
                        'created_at': homework['created_at'].isoformat() if homework['created_at'] else None
                    },
                    'basic_statistics': {
                        'total_submissions': stats['total_submissions'] if stats else 0,
                        'graded_count': stats['graded_count'] if stats else 0,
                        'completion_rate': round((stats['graded_count'] / max(stats['total_submissions'], 1)) * 100, 1) if stats and stats['total_submissions'] else 0,
                        'average_score': round(float(stats['avg_score']), 1) if stats and stats['avg_score'] else 0,
                        'max_score': float(stats['max_score']) if stats and stats['max_score'] else 0,
                        'min_score': float(stats['min_score']) if stats and stats['min_score'] else 0,
                        'std_deviation': round(float(stats['std_dev']), 2) if stats and stats['std_dev'] else 0
                    },
                    'score_distribution': [
                        {
                            'range': dist['score_range'],
                            'count': dist['count'],
                            'percentage': round((dist['count'] / max(stats['graded_count'], 1)) * 100, 1) if stats and stats['graded_count'] else 0
                        }
                        for dist in score_distribution
                    ],
                    'question_analysis': [
                        {
                            'question_id': q['id'],
                            'question_order': q['order_index'],
                            'question_type': q['question_type'],
                            'question_content': q['content'][:100] + '...' if len(q['content']) > 100 else q['content'],
                            'max_score': float(q['score']) if q['score'] else 0,
                            'difficulty': q.get('difficulty', 1),
                            'knowledge_points': json.loads(q.get('knowledge_points', '[]')) if q.get('knowledge_points') else []
                        }
                        for q in questions
                    ],
                    'student_performance': [
                        {
                            'username': perf['username'],
                            'real_name': perf.get('real_name', ''),
                            'score': float(perf['score']) if perf['score'] else 0,
                            'submitted_at': perf['submitted_at'].isoformat() if perf['submitted_at'] else None,
                            'time_spent': perf.get('time_spent', 0),
                            'status': perf['status']
                        }
                        for perf in student_performances
                    ],
                    'teaching_suggestions': [
                        {
                            'type': 'general',
                            'title': '整体表现分析',
                            'content': f'本次作业共有{stats["total_submissions"] if stats else 0}人提交，平均分为{round(float(stats["avg_score"]), 1) if stats and stats["avg_score"] else 0}分。',
                            'priority': 'medium'
                        }
                    ],
                    'generated_at': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'data': analytics_data
                })
                
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取分析失败: {str(e)}'
        }), 500
