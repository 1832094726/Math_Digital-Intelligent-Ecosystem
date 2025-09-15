# -*- coding: utf-8 -*-
"""
增强的符号推荐API路由
"""

from flask import Blueprint, request, jsonify
from services.enhanced_symbol_service import EnhancedSymbolService
import logging

# 创建蓝图
enhanced_symbol_bp = Blueprint('enhanced_symbol', __name__, url_prefix='/api/symbols')

# 创建服务实例
symbol_service = EnhancedSymbolService()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@enhanced_symbol_bp.route('/recommend', methods=['POST'])
def get_symbol_recommendations():
    """获取符号推荐"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': '用户ID不能为空'}), 400
        
        # 构建上下文
        context = {
            'question_text': data.get('question_text', ''),
            'current_input': data.get('current_input', ''),
            'current_topic': data.get('current_topic', ''),
            'difficulty_level': data.get('difficulty_level', 'medium'),
            'cursor_position': data.get('cursor_position', 0),
            'usage_history': data.get('usage_history', {})
        }
        
        # 获取推荐
        recommendations = symbol_service.get_symbol_recommendations(user_id, context)
        
        logger.info(f"Generated {len(recommendations.get('symbols', []))} symbol recommendations for user {user_id}")
        
        return jsonify({
            'success': True,
            'data': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in get_symbol_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取符号推荐失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/recommend/explained', methods=['POST'])
def get_explained_symbol_recommendations():
    """获取带解释的符号推荐"""
    try:
        data = request.get_json()

        # 验证必需参数
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': '用户ID不能为空'}), 400

        # 构建上下文
        context = {
            'question_text': data.get('question_text', ''),
            'current_input': data.get('current_input', ''),
            'current_topic': data.get('current_topic', ''),
            'difficulty_level': data.get('difficulty_level', 'medium'),
            'cursor_position': data.get('cursor_position', 0),
            'usage_history': data.get('usage_history', {})
        }

        # 获取带解释的推荐
        recommendations = symbol_service.get_user_recommendations_with_explanation(user_id, context)

        logger.info(f"Generated explained recommendations for user {user_id}")

        return jsonify({
            'success': True,
            'data': recommendations
        })

    except Exception as e:
        logger.error(f"Error in get_explained_symbol_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取解释推荐失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/complete', methods=['POST'])
def get_symbol_completions():
    """获取符号补全建议"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        user_id = data.get('user_id')
        partial_input = data.get('partial_input', '')
        
        if not user_id:
            return jsonify({'error': '用户ID不能为空'}), 400
        
        if not partial_input:
            return jsonify({'completions': []})
        
        # 构建上下文
        context = {
            'question_text': data.get('question_text', ''),
            'current_input': data.get('context', ''),
            'max_suggestions': data.get('max_suggestions', 10)
        }
        
        # 获取补全建议
        completions = symbol_service.get_symbol_completions(user_id, partial_input, context)
        
        logger.info(f"Generated {len(completions.get('completions', []))} completions for '{partial_input}' for user {user_id}")
        
        return jsonify({
            'success': True,
            'data': completions
        })
        
    except Exception as e:
        logger.error(f"Error in get_symbol_completions: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取符号补全失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/context', methods=['POST'])
def get_context_aware_recommendations():
    """获取上下文感知的符号推荐"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': '用户ID不能为空'}), 400
        
        # 构建上下文
        context = {
            'current_input': data.get('current_input', ''),
            'question_text': data.get('question_text', ''),
            'math_context': data.get('math_context', {}),
            'cursor_position': data.get('cursor_position', 0),
            'usage_patterns': data.get('usage_patterns', {})
        }
        
        # 获取上下文感知推荐
        recommendations = symbol_service.get_symbol_recommendations(user_id, context)
        
        # 过滤出上下文相关的推荐
        context_recommendations = [
            rec for rec in recommendations.get('symbols', [])
            if rec.get('source') in ['context', 'personalized']
        ]
        
        logger.info(f"Generated {len(context_recommendations)} context-aware recommendations for user {user_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'symbols': context_recommendations,
                'context': recommendations.get('context', {}),
                'total_count': len(context_recommendations)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_context_aware_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取上下文推荐失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/usage', methods=['POST'])
def record_symbol_usage():
    """记录符号使用"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        user_id = data.get('user_id')
        symbol_id = data.get('symbol_id')
        
        if not user_id or not symbol_id:
            return jsonify({'error': '用户ID和符号ID不能为空'}), 400
        
        # 构建使用上下文
        context = {
            'question_text': data.get('question_text', ''),
            'current_topic': data.get('current_topic', ''),
            'current_input': data.get('current_input', ''),
            'timestamp': data.get('timestamp')
        }
        
        # 记录使用
        symbol_service.record_symbol_usage(user_id, symbol_id, context)
        
        logger.info(f"Recorded symbol usage: user {user_id}, symbol {symbol_id}")
        
        return jsonify({
            'success': True,
            'message': '符号使用记录成功'
        })
        
    except Exception as e:
        logger.error(f"Error in record_symbol_usage: {str(e)}")
        return jsonify({
            'success': False,
            'error': '记录符号使用失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/categories', methods=['GET'])
def get_symbol_categories():
    """获取符号分类"""
    try:
        categories = [
            {
                'id': 'basic',
                'name': '基本运算',
                'icon': 'icon-plus',
                'description': '加减乘除等基本运算符号'
            },
            {
                'id': 'relation',
                'name': '关系符号',
                'icon': 'icon-equals',
                'description': '等于、不等于、大于小于等关系符号'
            },
            {
                'id': 'greek',
                'name': '希腊字母',
                'icon': 'icon-alpha',
                'description': 'α、β、π等希腊字母'
            },
            {
                'id': 'calculus',
                'name': '微积分',
                'icon': 'icon-integral',
                'description': '积分、求和、极限等微积分符号'
            },
            {
                'id': 'geometry',
                'name': '几何符号',
                'icon': 'icon-triangle',
                'description': '角度、三角形、圆等几何符号'
            },
            {
                'id': 'algebra',
                'name': '代数符号',
                'icon': 'icon-function',
                'description': '函数、集合、逻辑等代数符号'
            },
            {
                'id': 'special',
                'name': '特殊符号',
                'icon': 'icon-star',
                'description': '无穷大、根号等特殊数学符号'
            }
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'categories': categories,
                'total_count': len(categories)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_symbol_categories: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取符号分类失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/category/<category_id>', methods=['GET'])
def get_symbols_by_category(category_id):
    """根据分类获取符号"""
    try:
        # 从符号数据中筛选指定分类的符号
        category_symbols = [
            symbol for symbol in symbol_service.symbols_data
            if symbol.get('category', '').lower() == category_id.lower() or
               category_id.lower() in symbol.get('category', '').lower()
        ]
        
        # 按使用频率和重要性排序
        category_symbols.sort(key=lambda x: x.get('difficulty', 0.5))
        
        return jsonify({
            'success': True,
            'data': {
                'symbols': category_symbols,
                'category_id': category_id,
                'total_count': len(category_symbols)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_symbols_by_category: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取分类符号失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/search', methods=['POST'])
def search_symbols():
    """搜索符号"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'symbols': []})
        
        # 搜索符号
        matching_symbols = []
        query_lower = query.lower()
        
        for symbol in symbol_service.symbols_data:
            # 搜索符号本身
            if query_lower in symbol.get('symbol', '').lower():
                matching_symbols.append({**symbol, 'match_type': 'symbol'})
                continue
            
            # 搜索LaTeX命令
            if query_lower in symbol.get('latex', '').lower():
                matching_symbols.append({**symbol, 'match_type': 'latex'})
                continue
            
            # 搜索描述
            if query_lower in symbol.get('description', '').lower():
                matching_symbols.append({**symbol, 'match_type': 'description'})
                continue
            
            # 搜索相关知识点
            related_knowledge = symbol.get('related_knowledge', [])
            if any(query_lower in knowledge.lower() for knowledge in related_knowledge):
                matching_symbols.append({**symbol, 'match_type': 'knowledge'})
        
        # 按匹配类型排序（符号 > LaTeX > 描述 > 知识点）
        match_priority = {'symbol': 4, 'latex': 3, 'description': 2, 'knowledge': 1}
        matching_symbols.sort(key=lambda x: match_priority.get(x.get('match_type', ''), 0), reverse=True)
        
        return jsonify({
            'success': True,
            'data': {
                'symbols': matching_symbols[:20],  # 限制结果数量
                'query': query,
                'total_count': len(matching_symbols)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in search_symbols: {str(e)}")
        return jsonify({
            'success': False,
            'error': '搜索符号失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/stats/<int:user_id>', methods=['GET'])
def get_user_symbol_stats(user_id):
    """获取用户符号使用统计"""
    try:
        user_patterns = symbol_service.usage_patterns.get(user_id, {})
        
        if not user_patterns:
            return jsonify({
                'success': True,
                'data': {
                    'total_usage': 0,
                    'unique_symbols': 0,
                    'most_used': [],
                    'recent_usage': []
                }
            })
        
        # 计算统计信息
        total_usage = sum(pattern.get('count', 0) for pattern in user_patterns.values())
        unique_symbols = len(user_patterns)
        
        # 最常用符号
        most_used = []
        for symbol_id, pattern in user_patterns.items():
            symbol_info = next((s for s in symbol_service.symbols_data if str(s.get('id')) == symbol_id), None)
            if symbol_info:
                most_used.append({
                    'symbol': symbol_info.get('symbol'),
                    'description': symbol_info.get('description'),
                    'count': pattern.get('count', 0),
                    'last_used': pattern.get('last_used')
                })
        
        most_used.sort(key=lambda x: x['count'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': {
                'user_id': user_id,
                'total_usage': total_usage,
                'unique_symbols': unique_symbols,
                'most_used': most_used[:10],
                'usage_patterns': user_patterns
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_user_symbol_stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取用户统计失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/analytics/<int:user_id>', methods=['GET'])
def get_user_learning_analytics(user_id):
    """获取用户学习分析"""
    try:
        analytics = symbol_service.get_user_learning_analytics(user_id)

        return jsonify({
            'success': True,
            'data': analytics
        })

    except Exception as e:
        logger.error(f"Error in get_user_learning_analytics: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取学习分析失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/recommend/adaptive', methods=['POST'])
def get_adaptive_recommendations():
    """获取自适应推荐"""
    try:
        data = request.get_json()

        # 验证必需参数
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': '用户ID不能为空'}), 400

        # 构建上下文
        context = {
            'question_text': data.get('question_text', ''),
            'current_input': data.get('current_input', ''),
            'current_topic': data.get('current_topic', ''),
            'difficulty_level': data.get('difficulty_level', 'medium'),
            'cursor_position': data.get('cursor_position', 0),
            'usage_history': data.get('usage_history', {})
        }

        # 获取自适应推荐
        recommendations = symbol_service.get_adaptive_recommendations(user_id, context)

        logger.info(f"Generated adaptive recommendations for user {user_id}")

        return jsonify({
            'success': True,
            'data': recommendations
        })

    except Exception as e:
        logger.error(f"Error in get_adaptive_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取自适应推荐失败',
            'message': str(e)
        }), 500


@enhanced_symbol_bp.route('/learning-insights/<int:user_id>', methods=['GET'])
def get_learning_insights(user_id):
    """获取学习洞察"""
    try:
        # 获取学习分析
        analytics = symbol_service.get_user_learning_analytics(user_id)

        if 'error' in analytics:
            return jsonify({
                'success': False,
                'error': analytics['error'],
                'message': analytics.get('message', '')
            }), 500

        # 提取洞察信息
        insights = analytics.get('insights', {})
        learning_pattern = analytics.get('learning_pattern', {})

        # 生成简化的洞察报告
        simplified_insights = {
            'user_id': user_id,
            'overall_assessment': insights.get('overall_assessment', ''),
            'strengths': insights.get('strengths', []),
            'areas_for_improvement': insights.get('areas_for_improvement', []),
            'learning_recommendations': insights.get('learning_recommendations', []),
            'progress_indicators': insights.get('progress_indicators', {}),
            'learning_style': learning_pattern.get('learning_style', 'unknown'),
            'activity_level': learning_pattern.get('activity_level', 'unknown'),
            'preferred_categories': learning_pattern.get('preferred_categories', [])
        }

        return jsonify({
            'success': True,
            'data': simplified_insights
        })

    except Exception as e:
        logger.error(f"Error in get_learning_insights: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取学习洞察失败',
            'message': str(e)
        }), 500


# 错误处理
@enhanced_symbol_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': '接口不存在',
        'message': 'API endpoint not found'
    }), 404


@enhanced_symbol_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': '请求方法不允许',
        'message': 'Method not allowed'
    }), 405


@enhanced_symbol_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '服务器内部错误',
        'message': 'Internal server error'
    }), 500
