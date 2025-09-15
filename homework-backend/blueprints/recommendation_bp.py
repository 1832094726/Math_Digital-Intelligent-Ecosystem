#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统API蓝图
提供符号推荐、题目推荐、学习路径推荐等功能
"""

from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import jwt
from datetime import datetime
import logging

from services.symbol_recommendation_service import SymbolRecommendationService
from services.knowledge_recommendation_service import knowledge_recommendation_service
from models.database import get_db_connection

logger = logging.getLogger(__name__)

# 创建蓝图
recommendation_bp = Blueprint('recommendation', __name__, url_prefix='/api/recommend')

def token_required(f):
    """JWT令牌验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '缺少认证令牌'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': '令牌已过期'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': '无效令牌'}), 401
        
        return f(current_user_id, *args, **kwargs)
    return decorated

@recommendation_bp.route('/symbols', methods=['POST'])
@token_required
def recommend_symbols(current_user_id):
    """
    符号推荐API
    
    请求体:
    {
        "context": "求解方程 2x + 3 = 7",
        "question_id": 123,
        "limit": 10
    }
    
    响应:
    {
        "success": true,
        "recommendations": [
            {
                "id": 1,
                "symbol_text": "x",
                "symbol_name": "未知数x",
                "latex_code": "x",
                "category": "variable",
                "confidence": 0.95
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        # 验证请求参数
        if not data or 'context' not in data:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: context'
            }), 400
        
        context = data['context']
        question_id = data.get('question_id')
        limit = data.get('limit', 10)
        
        # 验证参数
        if not isinstance(context, str) or len(context.strip()) == 0:
            return jsonify({
                'success': False,
                'error': '上下文不能为空'
            }), 400
        
        if limit < 1 or limit > 50:
            return jsonify({
                'success': False,
                'error': '推荐数量限制在1-50之间'
            }), 400
        
        # 调用推荐服务
        symbol_service = SymbolRecommendationService()
        recommendations = symbol_service.recommend_symbols(
            context=context.strip(),
            user_id=current_user_id,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'total': len(recommendations),
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"符号推荐API错误: {e}")
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500

@recommendation_bp.route('/symbols/usage', methods=['POST'])
@token_required
def record_symbol_usage(current_user_id):
    """
    记录符号使用API
    
    请求体:
    {
        "symbol_id": 1,
        "symbol": "x",
        "context": "求解方程 2x + 3 = 7"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data or 'context' not in data:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: symbol, context'
            }), 400
        
        symbol = data['symbol']
        context = data['context']
        
        # 记录使用
        symbol_service = SymbolRecommendationService()
        success = symbol_service.record_symbol_usage(
            user_id=current_user_id,
            symbol=symbol,
            context=context
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': '符号使用记录成功'
            })
        else:
            return jsonify({
                'success': False,
                'error': '记录失败'
            }), 500
            
    except Exception as e:
        logger.error(f"记录符号使用API错误: {e}")
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500







@recommendation_bp.route('/stats', methods=['GET'])
@token_required
def get_recommendation_stats(current_user_id):
    """获取推荐系统统计信息"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 获取用户推荐统计
        cursor.execute("""
            SELECT 
                COUNT(*) as total_recommendations,
                COUNT(CASE WHEN selected_symbol IS NOT NULL THEN 1 END) as used_recommendations,
                AVG(success_rate) as avg_success_rate
            FROM symbol_recommendations 
            WHERE user_id = %s
        """, (current_user_id,))
        
        stats = cursor.fetchone()
        
        # 获取最常用符号
        cursor.execute("""
            SELECT selected_symbol, COUNT(*) as usage_count
            FROM symbol_recommendations 
            WHERE user_id = %s AND selected_symbol IS NOT NULL
            GROUP BY selected_symbol
            ORDER BY usage_count DESC
            LIMIT 5
        """, (current_user_id,))
        
        top_symbols = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_recommendations': stats['total_recommendations'] or 0,
                'used_recommendations': stats['used_recommendations'] or 0,
                'usage_rate': (stats['used_recommendations'] / max(stats['total_recommendations'], 1)) * 100,
                'avg_success_rate': float(stats['avg_success_rate'] or 0),
                'top_symbols': top_symbols
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"获取推荐统计API错误: {e}")
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500

@recommendation_bp.route('/knowledge', methods=['POST'])
@token_required
def recommend_knowledge_points(current_user_id):
    """
    知识点推荐API

    请求体:
    {
        "question_id": 123,  // 可选，题目ID
        "context": "解一元二次方程",  // 可选，上下文内容
        "limit": 5  // 可选，推荐数量限制，默认5
    }

    响应:
    {
        "success": true,
        "recommendations": [
            {
                "id": 1,
                "name": "一元二次方程",
                "description": "含有一个未知数的二次方程",
                "grade_level": 2,
                "difficulty_level": 3,
                "cognitive_type": "procedural",
                "relevance_score": 0.9,
                "recommendation_reason": "与当前题目直接相关",
                "related_points": [...]
            }
        ],
        "total": 5,
        "timestamp": "2023-06-15T10:30:00"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400

        question_id = data.get('question_id')
        context = data.get('context', '')
        limit = data.get('limit', 5)

        # 验证参数
        if limit > 20:
            limit = 20
        elif limit < 1:
            limit = 5

        # 调用知识点推荐服务
        result = knowledge_recommendation_service.recommend_knowledge_points(
            user_id=current_user_id,
            question_id=question_id,
            context=context,
            limit=limit
        )

        if result['success']:
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '推荐失败')
            }), 500

    except Exception as e:
        logger.error(f"知识点推荐API错误: {e}")
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500

@recommendation_bp.route('/exercises', methods=['POST'])
@token_required
def recommend_exercises(current_user_id):
    """
    练习题推荐API

    请求体:
    {
        "knowledge_point_id": 123,  // 可选，知识点ID
        "difficulty_level": 3,  // 可选，难度级别 1-5
        "limit": 10  // 可选，推荐数量限制，默认10
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400

        knowledge_point_id = data.get('knowledge_point_id')
        difficulty_level = data.get('difficulty_level')
        limit = data.get('limit', 10)

        # 验证参数
        if limit > 50:
            limit = 50
        elif limit < 1:
            limit = 10

        # TODO: 实现练习题推荐逻辑
        # 这里先返回模拟数据
        recommendations = [
            {
                'id': i,
                'title': f'练习题 {i}',
                'content': f'这是第{i}道练习题的内容',
                'difficulty_level': difficulty_level or 3,
                'knowledge_points': [knowledge_point_id] if knowledge_point_id else [],
                'estimated_time': 5,
                'recommendation_reason': '基于您的学习水平推荐'
            }
            for i in range(1, min(limit + 1, 6))
        ]

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'total': len(recommendations),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"练习题推荐API错误: {e}")
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500

@recommendation_bp.route('/learning-path', methods=['POST'])
@token_required
def recommend_learning_path(current_user_id):
    """
    学习路径推荐API

    请求体:
    {
        "target_knowledge_point": 123,  // 目标知识点ID
        "current_level": 2,  // 当前水平级别
        "learning_style": "visual"  // 学习风格
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400

        target_kp_id = data.get('target_knowledge_point')
        current_level = data.get('current_level', 1)
        learning_style = data.get('learning_style', 'balanced')

        # TODO: 实现学习路径推荐逻辑
        # 这里先返回模拟数据
        learning_path = {
            'path_id': f'path_{current_user_id}_{target_kp_id}',
            'target_knowledge_point': target_kp_id,
            'estimated_duration': 30,  # 预计学习时长（天）
            'steps': [
                {
                    'step': 1,
                    'knowledge_point_id': 1,
                    'knowledge_point_name': '基本运算',
                    'estimated_time': 5,  # 天
                    'resources': ['视频教程', '练习题集'],
                    'prerequisites': []
                },
                {
                    'step': 2,
                    'knowledge_point_id': 2,
                    'knowledge_point_name': '代数表达式',
                    'estimated_time': 8,
                    'resources': ['互动课件', '实践练习'],
                    'prerequisites': [1]
                }
            ],
            'learning_style_adaptations': {
                'visual': ['图表展示', '动画演示'],
                'auditory': ['语音讲解', '讨论互动'],
                'kinesthetic': ['实践操作', '实验验证']
            }
        }

        return jsonify({
            'success': True,
            'learning_path': learning_path,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"学习路径推荐API错误: {e}")
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500
