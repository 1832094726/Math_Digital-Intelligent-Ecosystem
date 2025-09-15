#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符号推荐服务
实现基于上下文的数学符号智能推荐功能
"""

import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.database import get_db_connection
import logging

logger = logging.getLogger(__name__)

class SymbolRecommendationService:
    """符号推荐服务类"""
    
    def __init__(self):
        """初始化符号推荐服务"""
        self.symbol_database = self._load_symbol_database()
        self.context_patterns = self._load_context_patterns()
    
    def _load_symbol_database(self) -> Dict[str, Any]:
        """加载数学符号数据库"""
        return {
            # 基础运算符号
            'basic_operators': [
                {'symbol': '+', 'name': '加号', 'latex': '+', 'category': 'arithmetic', 'frequency': 100},
                {'symbol': '-', 'name': '减号', 'latex': '-', 'category': 'arithmetic', 'frequency': 100},
                {'symbol': '×', 'name': '乘号', 'latex': '\\times', 'category': 'arithmetic', 'frequency': 90},
                {'symbol': '÷', 'name': '除号', 'latex': '\\div', 'category': 'arithmetic', 'frequency': 90},
                {'symbol': '=', 'name': '等号', 'latex': '=', 'category': 'relation', 'frequency': 100},
            ],
            # 分数和根号
            'fractions_roots': [
                {'symbol': '½', 'name': '二分之一', 'latex': '\\frac{1}{2}', 'category': 'fraction', 'frequency': 80},
                {'symbol': '⅓', 'name': '三分之一', 'latex': '\\frac{1}{3}', 'category': 'fraction', 'frequency': 70},
                {'symbol': '¼', 'name': '四分之一', 'latex': '\\frac{1}{4}', 'category': 'fraction', 'frequency': 70},
                {'symbol': '√', 'name': '根号', 'latex': '\\sqrt{}', 'category': 'root', 'frequency': 85},
                {'symbol': '∛', 'name': '立方根', 'latex': '\\sqrt[3]{}', 'category': 'root', 'frequency': 60},
            ],
            # 几何符号
            'geometry': [
                {'symbol': '∠', 'name': '角', 'latex': '\\angle', 'category': 'geometry', 'frequency': 75},
                {'symbol': '△', 'name': '三角形', 'latex': '\\triangle', 'category': 'geometry', 'frequency': 80},
                {'symbol': '⊙', 'name': '圆', 'latex': '\\odot', 'category': 'geometry', 'frequency': 70},
                {'symbol': '∥', 'name': '平行', 'latex': '\\parallel', 'category': 'geometry', 'frequency': 65},
                {'symbol': '⊥', 'name': '垂直', 'latex': '\\perp', 'category': 'geometry', 'frequency': 65},
            ],
            # 代数符号
            'algebra': [
                {'symbol': 'x', 'name': '未知数x', 'latex': 'x', 'category': 'variable', 'frequency': 95},
                {'symbol': 'y', 'name': '未知数y', 'latex': 'y', 'category': 'variable', 'frequency': 90},
                {'symbol': 'a', 'name': '参数a', 'latex': 'a', 'category': 'parameter', 'frequency': 85},
                {'symbol': 'b', 'name': '参数b', 'latex': 'b', 'category': 'parameter', 'frequency': 85},
                {'symbol': 'n', 'name': '自然数n', 'latex': 'n', 'category': 'parameter', 'frequency': 80},
            ],
            # 比较符号
            'comparison': [
                {'symbol': '>', 'name': '大于', 'latex': '>', 'category': 'relation', 'frequency': 85},
                {'symbol': '<', 'name': '小于', 'latex': '<', 'category': 'relation', 'frequency': 85},
                {'symbol': '≥', 'name': '大于等于', 'latex': '\\geq', 'category': 'relation', 'frequency': 80},
                {'symbol': '≤', 'name': '小于等于', 'latex': '\\leq', 'category': 'relation', 'frequency': 80},
                {'symbol': '≠', 'name': '不等于', 'latex': '\\neq', 'category': 'relation', 'frequency': 75},
            ],
            # 函数符号
            'functions': [
                {'symbol': 'sin', 'name': '正弦函数', 'latex': '\\sin', 'category': 'function', 'frequency': 70},
                {'symbol': 'cos', 'name': '余弦函数', 'latex': '\\cos', 'category': 'function', 'frequency': 70},
                {'symbol': 'tan', 'name': '正切函数', 'latex': '\\tan', 'category': 'function', 'frequency': 65},
                {'symbol': 'log', 'name': '对数函数', 'latex': '\\log', 'category': 'function', 'frequency': 60},
            ]
        }
    
    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """加载上下文模式"""
        return {
            'equation': ['方程', '等式', '解', '求解', '未知数'],
            'geometry': ['三角形', '圆', '角', '面积', '周长', '体积'],
            'fraction': ['分数', '分子', '分母', '约分', '通分'],
            'inequality': ['不等式', '大于', '小于', '范围'],
            'function': ['函数', '图像', '定义域', '值域'],
            'arithmetic': ['计算', '运算', '加', '减', '乘', '除']
        }
    
    def recommend_symbols(self, context: str, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        基于上下文推荐符号
        
        Args:
            context: 输入上下文
            user_id: 用户ID
            limit: 推荐数量限制
            
        Returns:
            推荐符号列表
        """
        try:
            # 1. 分析上下文
            context_analysis = self._analyze_context(context)
            
            # 2. 获取用户历史偏好
            user_preferences = self._get_user_preferences(user_id)
            
            # 3. 生成候选符号
            candidates = self._generate_candidates(context_analysis, user_preferences)
            
            # 4. 计算推荐分数
            scored_symbols = self._calculate_scores(candidates, context_analysis, user_preferences)
            
            # 5. 排序和过滤
            recommendations = sorted(scored_symbols, key=lambda x: x['score'], reverse=True)[:limit]
            
            # 6. 记录推荐日志
            self._log_recommendation(user_id, context, recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"符号推荐失败: {e}")
            return self._get_default_recommendations(limit)
    
    def _analyze_context(self, context: str) -> Dict[str, Any]:
        """分析输入上下文"""
        analysis = {
            'categories': [],
            'keywords': [],
            'complexity': 1,
            'math_type': 'basic'
        }
        
        context_lower = context.lower()
        
        # 检测数学类型
        for category, keywords in self.context_patterns.items():
            for keyword in keywords:
                if keyword in context_lower:
                    analysis['categories'].append(category)
                    analysis['keywords'].append(keyword)
        
        # 检测复杂度
        if any(word in context_lower for word in ['函数', '导数', '积分', '微分']):
            analysis['complexity'] = 3
            analysis['math_type'] = 'advanced'
        elif any(word in context_lower for word in ['方程', '不等式', '几何']):
            analysis['complexity'] = 2
            analysis['math_type'] = 'intermediate'
        
        return analysis
    
    def _get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """获取用户历史偏好"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            # 查询用户最近的符号使用记录
            cursor.execute("""
                SELECT selected_symbol, COUNT(*) as usage_count
                FROM symbol_recommendations 
                WHERE user_id = %s AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY selected_symbol
                ORDER BY usage_count DESC
                LIMIT 20
            """, (user_id,))
            
            usage_history = cursor.fetchall()
            
            # 查询用户年级信息
            cursor.execute("SELECT grade FROM users WHERE id = %s", (user_id,))
            user_info = cursor.fetchone()
            
            preferences = {
                'frequently_used': [item['selected_symbol'] for item in usage_history],
                'usage_counts': {item['selected_symbol']: item['usage_count'] for item in usage_history},
                'grade_level': user_info['grade'] if user_info else 6
            }
            
            cursor.close()
            conn.close()
            
            return preferences
            
        except Exception as e:
            logger.error(f"获取用户偏好失败: {e}")
            return {'frequently_used': [], 'usage_counts': {}, 'grade_level': 6}
    
    def _generate_candidates(self, context_analysis: Dict[str, Any], user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成候选符号"""
        candidates = []
        
        # 根据上下文类别添加相关符号
        for category in context_analysis['categories']:
            if category in ['equation', 'algebra']:
                candidates.extend(self.symbol_database['algebra'])
                candidates.extend(self.symbol_database['basic_operators'])
            elif category == 'geometry':
                candidates.extend(self.symbol_database['geometry'])
            elif category == 'fraction':
                candidates.extend(self.symbol_database['fractions_roots'])
            elif category == 'inequality':
                candidates.extend(self.symbol_database['comparison'])
            elif category == 'function':
                candidates.extend(self.symbol_database['functions'])
        
        # 如果没有特定类别，添加基础符号
        if not context_analysis['categories']:
            candidates.extend(self.symbol_database['basic_operators'])
            candidates.extend(self.symbol_database['algebra'])
        
        # 去重
        seen = set()
        unique_candidates = []
        for symbol in candidates:
            if symbol['symbol'] not in seen:
                seen.add(symbol['symbol'])
                unique_candidates.append(symbol)
        
        return unique_candidates
    
    def _calculate_scores(self, candidates: List[Dict[str, Any]], context_analysis: Dict[str, Any], user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计算推荐分数"""
        scored_symbols = []
        
        for symbol in candidates:
            score = 0.0
            
            # 基础频率分数 (0-40分)
            score += (symbol['frequency'] / 100) * 40
            
            # 用户历史偏好分数 (0-30分)
            if symbol['symbol'] in user_preferences['frequently_used']:
                usage_count = user_preferences['usage_counts'].get(symbol['symbol'], 0)
                score += min(usage_count * 5, 30)
            
            # 上下文相关性分数 (0-30分)
            if symbol['category'] in context_analysis['categories']:
                score += 30
            elif any(keyword in symbol['name'] for keyword in context_analysis['keywords']):
                score += 15
            
            scored_symbols.append({
                'id': len(scored_symbols) + 1,
                'symbol_text': symbol['symbol'],
                'symbol_name': symbol['name'],
                'latex_code': symbol['latex'],
                'category': symbol['category'],
                'confidence': min(score / 100, 1.0),
                'score': score
            })
        
        return scored_symbols
    
    def _log_recommendation(self, user_id: int, context: str, recommendations: List[Dict[str, Any]]):
        """记录推荐日志"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            recommended_symbols = [
                {
                    'symbol': rec['symbol_text'],
                    'confidence': rec['confidence'],
                    'category': rec['category']
                }
                for rec in recommendations
            ]
            
            cursor.execute("""
                INSERT INTO symbol_recommendations 
                (user_id, context, recommended_symbols, created_at)
                VALUES (%s, %s, %s, %s)
            """, (user_id, context, json.dumps(recommended_symbols), datetime.now()))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"记录推荐日志失败: {e}")
    
    def _get_default_recommendations(self, limit: int) -> List[Dict[str, Any]]:
        """获取默认推荐"""
        default_symbols = self.symbol_database['basic_operators'][:limit]
        return [
            {
                'id': i + 1,
                'symbol_text': symbol['symbol'],
                'symbol_name': symbol['name'],
                'latex_code': symbol['latex'],
                'category': symbol['category'],
                'confidence': 0.8
            }
            for i, symbol in enumerate(default_symbols)
        ]
    
    def record_symbol_usage(self, user_id: int, symbol: str, context: str) -> bool:
        """记录符号使用"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 更新推荐记录中的选择符号
            cursor.execute("""
                UPDATE symbol_recommendations 
                SET selected_symbol = %s, usage_frequency = usage_frequency + 1
                WHERE user_id = %s AND context = %s 
                ORDER BY created_at DESC LIMIT 1
            """, (symbol, user_id, context))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"记录符号使用失败: {e}")
            return False
