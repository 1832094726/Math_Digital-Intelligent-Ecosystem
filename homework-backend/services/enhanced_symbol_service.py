# -*- coding: utf-8 -*-
"""
增强的符号推荐服务
集成学科符号动态键盘的核心功能
"""

import os
import re
import json
import jieba
import jieba.analyse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
from .collaborative_filtering import CollaborativeFiltering
from .knowledge_graph import KnowledgeGraph, KnowledgeGraphRecommender
from .learning_analytics import LearningAnalytics

class EnhancedSymbolService:
    """增强的符号推荐服务类"""
    
    def __init__(self):
        self.symbols_data = self._load_symbols_data()
        self.knowledge_data = self._load_knowledge_data()
        self.usage_patterns = defaultdict(dict)
        self.context_analyzer = ContextAnalyzer()
        self.completion_engine = CompletionEngine(self.symbols_data)

        # 初始化协同过滤推荐系统
        self.collaborative_filter = CollaborativeFiltering()

        # 初始化知识图谱推荐系统
        self.knowledge_graph = KnowledgeGraph()
        self.kg_recommender = KnowledgeGraphRecommender(self.knowledge_graph)

        # 初始化学习分析系统
        self.learning_analytics = LearningAnalytics()

        # 初始化jieba
        self._init_jieba()
    
    def _load_symbols_data(self) -> List[Dict]:
        """加载符号数据"""
        try:
            with open('data/symbols.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_symbols()
    
    def _load_knowledge_data(self) -> List[Dict]:
        """加载知识点数据"""
        try:
            with open('data/knowledge_points.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _get_default_symbols(self) -> List[Dict]:
        """获取默认符号数据"""
        return [
            {"id": 1, "symbol": "+", "latex": "+", "category": "基本运算", "description": "加号"},
            {"id": 2, "symbol": "−", "latex": "-", "category": "基本运算", "description": "减号"},
            {"id": 3, "symbol": "×", "latex": "\\times", "category": "基本运算", "description": "乘号"},
            {"id": 4, "symbol": "÷", "latex": "\\div", "category": "基本运算", "description": "除号"},
            {"id": 5, "symbol": "=", "latex": "=", "category": "关系符号", "description": "等号"},
            {"id": 6, "symbol": "≠", "latex": "\\neq", "category": "关系符号", "description": "不等于"},
            {"id": 7, "symbol": "≤", "latex": "\\leq", "category": "关系符号", "description": "小于等于"},
            {"id": 8, "symbol": "≥", "latex": "\\geq", "category": "关系符号", "description": "大于等于"},
            {"id": 9, "symbol": "α", "latex": "\\alpha", "category": "希腊字母", "description": "希腊字母alpha"},
            {"id": 10, "symbol": "β", "latex": "\\beta", "category": "希腊字母", "description": "希腊字母beta"},
            {"id": 11, "symbol": "π", "latex": "\\pi", "category": "希腊字母", "description": "圆周率"},
            {"id": 12, "symbol": "∫", "latex": "\\int", "category": "微积分", "description": "积分符号"},
            {"id": 13, "symbol": "∑", "latex": "\\sum", "category": "微积分", "description": "求和符号"},
            {"id": 14, "symbol": "√", "latex": "\\sqrt", "category": "根号", "description": "平方根"},
            {"id": 15, "symbol": "∞", "latex": "\\infty", "category": "特殊符号", "description": "无穷大"},
        ]
    
    def _init_jieba(self):
        """初始化jieba分词"""
        # 加载自定义词典
        knowledge_words = [kp.get('name', '') for kp in self.knowledge_data if kp.get('name')]
        for word in knowledge_words:
            jieba.add_word(word)
    
    def get_symbol_recommendations(self, user_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取符号推荐"""
        question_text = context.get('question_text', '')
        current_input = context.get('current_input', '')
        current_topic = context.get('current_topic', '')
        difficulty_level = context.get('difficulty_level', 'medium')
        
        # 分析数学上下文
        math_context = self.context_analyzer.analyze_context(question_text, current_input)
        
        # 获取基础推荐
        basic_recommendations = self._get_basic_recommendations(question_text, math_context)
        
        # 获取上下文感知推荐
        context_recommendations = self._get_context_aware_recommendations(
            current_input, question_text, math_context
        )
        
        # 获取个性化推荐
        personalized_recommendations = self._get_personalized_recommendations(
            user_id, basic_recommendations + context_recommendations, context
        )

        # 获取协同过滤推荐
        collaborative_recommendations = self._get_collaborative_recommendations(user_id, context)

        # 获取知识图谱推荐
        knowledge_graph_recommendations = self._get_knowledge_graph_recommendations(context)

        # 合并和排序
        all_recommendations = self._merge_and_rank_recommendations(
            basic_recommendations, context_recommendations,
            personalized_recommendations, collaborative_recommendations,
            knowledge_graph_recommendations
        )
        
        return {
            "symbols": all_recommendations[:12],
            "context": math_context,
            "total_count": len(all_recommendations)
        }
    
    def get_symbol_completions(self, user_id: int, partial_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取符号补全建议"""
        return self.completion_engine.get_completions(
            partial_input, context, self.usage_patterns.get(user_id, {})
        )
    
    def _get_basic_recommendations(self, question_text: str, math_context: Dict) -> List[Dict]:
        """获取基础推荐"""
        recommendations = []
        
        # 基于关键词匹配
        keywords = jieba.analyse.extract_tags(question_text, topK=10)
        
        for symbol in self.symbols_data:
            score = 0.5  # 基础分数
            
            # 关键词匹配
            symbol_keywords = symbol.get('related_knowledge', [])
            for keyword in keywords:
                if any(keyword in sk for sk in symbol_keywords):
                    score += 0.2
            
            # 类别匹配
            category_scores = {
                '几何符号': 0.3 if math_context.get('has_geometry') else 0,
                '微积分': 0.4 if math_context.get('has_calculus') else 0,
                '基本运算': 0.2 if math_context.get('has_algebra') else 0.1,
                '关系符号': 0.2 if math_context.get('has_equations') else 0,
            }
            
            category = symbol.get('category', '')
            score += category_scores.get(category, 0)
            
            recommendations.append({
                **symbol,
                'score': min(1.0, score),
                'source': 'basic'
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def _get_context_aware_recommendations(self, current_input: str, question_text: str, math_context: Dict) -> List[Dict]:
        """获取上下文感知推荐"""
        recommendations = []
        
        # 分析当前输入的符号模式
        input_patterns = self._analyze_input_patterns(current_input)
        
        for symbol in self.symbols_data:
            score = 0.3  # 基础分数
            
            # 基于输入模式的推荐
            if input_patterns.get('has_fractions') and symbol.get('category') == '分数':
                score += 0.4
            if input_patterns.get('has_roots') and symbol.get('category') == '根号':
                score += 0.4
            if input_patterns.get('has_greek') and symbol.get('category') == '希腊字母':
                score += 0.3
            
            # 基于上下文的推荐
            if math_context.get('complexity') == 'high' and symbol.get('category') in ['微积分', '高级符号']:
                score += 0.3
            elif math_context.get('complexity') == 'low' and symbol.get('category') == '基本运算':
                score += 0.2
            
            if score > 0.3:  # 只返回有意义的推荐
                recommendations.append({
                    **symbol,
                    'score': min(1.0, score),
                    'source': 'context',
                    'reason': self._generate_context_reason(symbol, input_patterns, math_context)
                })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def _get_personalized_recommendations(self, user_id: int, base_recommendations: List[Dict], context: Dict) -> List[Dict]:
        """获取个性化推荐"""
        user_patterns = self.usage_patterns.get(user_id, {})
        
        personalized = []
        for rec in base_recommendations:
            symbol_id = rec.get('id')
            usage_count = user_patterns.get(str(symbol_id), {}).get('count', 0)
            last_used = user_patterns.get(str(symbol_id), {}).get('last_used')
            
            # 基于使用频率调整分数
            frequency_bonus = min(0.3, usage_count * 0.02)
            
            # 基于最近使用时间调整分数
            recency_bonus = 0
            if last_used:
                try:
                    last_used_time = datetime.fromisoformat(last_used)
                    time_diff = datetime.now() - last_used_time
                    if time_diff < timedelta(hours=1):
                        recency_bonus = 0.2
                    elif time_diff < timedelta(days=1):
                        recency_bonus = 0.1
                except:
                    pass
            
            adjusted_score = rec['score'] + frequency_bonus + recency_bonus
            
            personalized.append({
                **rec,
                'score': min(1.0, adjusted_score),
                'source': 'personalized',
                'usage_count': usage_count,
                'personalization_reason': self._generate_personalization_reason(
                    usage_count, recency_bonus > 0
                )
            })
        
        return sorted(personalized, key=lambda x: x['score'], reverse=True)

    def _get_collaborative_recommendations(self, user_id: int, context: Dict[str, Any]) -> List[Dict]:
        """获取协同过滤推荐"""
        try:
            # 获取混合协同过滤推荐
            cf_recommendations = self.collaborative_filter.get_hybrid_recommendations(
                str(user_id), n=15
            )

            recommendations = []
            for symbol_id, cf_score in cf_recommendations:
                # 查找符号信息
                symbol_info = None
                for symbol in self.symbols_data:
                    if str(symbol.get('id')) == symbol_id:
                        symbol_info = symbol
                        break

                if symbol_info:
                    # 结合协同过滤分数和符号基础信息
                    base_score = 0.4  # 基础分数

                    # 协同过滤分数归一化到0-0.6范围
                    normalized_cf_score = min(0.6, cf_score / 10.0)

                    final_score = base_score + normalized_cf_score

                    recommendations.append({
                        **symbol_info,
                        'score': final_score,
                        'source': 'collaborative_filtering',
                        'cf_score': cf_score,
                        'recommendation_reason': self._generate_cf_reason(cf_score)
                    })

            return sorted(recommendations, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            print(f"协同过滤推荐失败: {e}")
            return []

    def _generate_cf_reason(self, cf_score: float) -> str:
        """生成协同过滤推荐原因"""
        if cf_score > 8.0:
            return "与您相似的用户强烈推荐"
        elif cf_score > 5.0:
            return "相似用户经常使用"
        elif cf_score > 2.0:
            return "其他用户也在使用"
        else:
            return "基于用户行为推荐"

    def _get_knowledge_graph_recommendations(self, context: Dict[str, Any]) -> List[Dict]:
        """获取知识图谱推荐"""
        try:
            # 使用知识图谱推荐器获取推荐
            kg_recommendations = self.kg_recommender.recommend_symbols(context)

            recommendations = []
            for kg_rec in kg_recommendations:
                symbol_id = kg_rec.get('id')

                # 查找符号信息
                symbol_info = None
                for symbol in self.symbols_data:
                    if str(symbol.get('id')) == symbol_id:
                        symbol_info = symbol
                        break

                if symbol_info:
                    # 结合知识图谱分数和符号基础信息
                    kg_score = kg_rec.get('score', 0.5)
                    base_score = 0.3

                    # 知识图谱分数归一化
                    normalized_kg_score = min(0.7, kg_score)
                    final_score = base_score + normalized_kg_score

                    recommendations.append({
                        **symbol_info,
                        'score': final_score,
                        'source': 'knowledge_graph',
                        'kg_score': kg_score,
                        'kg_reason': kg_rec.get('kg_reason', ''),
                        'original_kg_score': kg_rec.get('original_score', 0)
                    })

            return sorted(recommendations, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            print(f"知识图谱推荐失败: {e}")
            return []

    def _merge_and_rank_recommendations(self, *recommendation_lists) -> List[Dict]:
        """合并和排序推荐结果"""
        all_recommendations = []
        seen_symbols = set()

        # 为不同来源的推荐设置权重
        source_weights = {
            'basic': 1.0,
            'context': 1.2,
            'personalized': 1.3,
            'collaborative_filtering': 1.1,
            'knowledge_graph': 1.25
        }

        for rec_list in recommendation_lists:
            for rec in rec_list:
                symbol_key = rec.get('symbol', '') + rec.get('latex', '')
                if symbol_key not in seen_symbols:
                    seen_symbols.add(symbol_key)

                    # 应用来源权重
                    source = rec.get('source', 'basic')
                    weight = source_weights.get(source, 1.0)
                    weighted_score = rec.get('score', 0) * weight

                    rec_copy = rec.copy()
                    rec_copy['weighted_score'] = weighted_score
                    rec_copy['original_score'] = rec.get('score', 0)

                    all_recommendations.append(rec_copy)

        return sorted(all_recommendations, key=lambda x: x.get('weighted_score', 0), reverse=True)
    
    def _analyze_input_patterns(self, current_input: str) -> Dict[str, bool]:
        """分析输入模式"""
        return {
            'has_fractions': '\\frac' in current_input or '/' in current_input,
            'has_roots': '\\sqrt' in current_input,
            'has_greek': bool(re.search(r'\\[a-zA-Z]+', current_input)),
            'has_equations': '=' in current_input,
            'has_integrals': '\\int' in current_input,
            'has_sums': '\\sum' in current_input,
        }
    
    def _generate_context_reason(self, symbol: Dict, input_patterns: Dict, math_context: Dict) -> str:
        """生成上下文推荐原因"""
        reasons = []
        
        if input_patterns.get('has_fractions') and symbol.get('category') == '分数':
            reasons.append('检测到分数输入')
        if math_context.get('has_geometry') and symbol.get('category') == '几何符号':
            reasons.append('几何题目相关')
        if math_context.get('has_calculus') and symbol.get('category') == '微积分':
            reasons.append('微积分内容相关')
        
        return '，'.join(reasons) if reasons else '基于上下文推荐'
    
    def _generate_personalization_reason(self, usage_count: int, recently_used: bool) -> str:
        """生成个性化推荐原因"""
        reasons = []
        
        if usage_count > 10:
            reasons.append('您经常使用')
        elif usage_count > 0:
            reasons.append('您之前使用过')
        
        if recently_used:
            reasons.append('最近使用过')
        
        return '，'.join(reasons) if reasons else '个性化推荐'
    
    def record_symbol_usage(self, user_id: int, symbol_id: int, context: Dict[str, Any]):
        """记录符号使用"""
        if user_id not in self.usage_patterns:
            self.usage_patterns[user_id] = {}

        symbol_key = str(symbol_id)
        if symbol_key not in self.usage_patterns[user_id]:
            self.usage_patterns[user_id][symbol_key] = {'count': 0, 'contexts': []}

        self.usage_patterns[user_id][symbol_key]['count'] += 1
        last_used = datetime.now().isoformat()
        self.usage_patterns[user_id][symbol_key]['last_used'] = last_used

        # 记录使用上下文（保留最近10次）
        context_record = {
            'timestamp': last_used,
            'question_text': context.get('question_text', ''),
            'current_topic': context.get('current_topic', '')
        }

        contexts = self.usage_patterns[user_id][symbol_key]['contexts']
        contexts.append(context_record)
        if len(contexts) > 10:
            contexts.pop(0)

        # 更新协同过滤系统
        try:
            usage_count = self.usage_patterns[user_id][symbol_key]['count']
            self.collaborative_filter.update_user_rating(
                str(user_id), symbol_key, usage_count, last_used
            )
        except Exception as e:
            print(f"更新协同过滤数据失败: {e}")

    def get_user_recommendations_with_explanation(self, user_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取带解释的用户推荐"""
        recommendations = self.get_symbol_recommendations(user_id, context)

        # 添加推荐解释
        explained_recommendations = []
        for rec in recommendations.get('symbols', []):
            explanation = self._generate_recommendation_explanation(rec, user_id)
            rec_with_explanation = rec.copy()
            rec_with_explanation['explanation'] = explanation
            explained_recommendations.append(rec_with_explanation)

        # 获取用户统计信息
        user_stats = self.collaborative_filter.get_user_statistics(str(user_id))

        return {
            'symbols': explained_recommendations,
            'context': recommendations.get('context', {}),
            'user_statistics': user_stats,
            'total_count': len(explained_recommendations)
        }

    def _generate_recommendation_explanation(self, recommendation: Dict, user_id: int) -> str:
        """生成推荐解释"""
        explanations = []

        source = recommendation.get('source', '')
        score = recommendation.get('score', 0)

        if source == 'collaborative_filtering':
            cf_reason = recommendation.get('recommendation_reason', '')
            explanations.append(f"协同过滤: {cf_reason}")

        if source == 'context':
            context_reason = recommendation.get('reason', '')
            if context_reason:
                explanations.append(f"上下文分析: {context_reason}")

        if source == 'personalized':
            personalization_reason = recommendation.get('personalization_reason', '')
            if personalization_reason:
                explanations.append(f"个性化: {personalization_reason}")

        if source == 'knowledge_graph':
            kg_reason = recommendation.get('kg_reason', '')
            if kg_reason:
                explanations.append(f"知识图谱: {kg_reason}")

        if score > 0.8:
            explanations.append("高度推荐")
        elif score > 0.6:
            explanations.append("推荐使用")

        return " | ".join(explanations) if explanations else "系统推荐"

    def get_user_learning_analytics(self, user_id: int) -> Dict[str, Any]:
        """获取用户学习分析"""
        try:
            # 更新学习分析数据
            self.learning_analytics.usage_data[str(user_id)] = self.usage_patterns.get(user_id, {})

            # 生成学习分析报告
            analytics_summary = self.learning_analytics.get_user_statistics_summary(str(user_id))

            return analytics_summary

        except Exception as e:
            print(f"获取学习分析失败: {e}")
            return {
                'user_id': str(user_id),
                'error': '学习分析数据获取失败',
                'message': str(e)
            }

    def get_adaptive_recommendations(self, user_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取自适应推荐（基于学习分析）"""
        try:
            # 获取学习模式
            learning_pattern = self.learning_analytics.analyze_user_learning_pattern(str(user_id))

            # 根据学习模式调整推荐策略
            adapted_context = self._adapt_context_by_learning_pattern(context, learning_pattern)

            # 获取基础推荐
            base_recommendations = self.get_symbol_recommendations(user_id, adapted_context)

            # 根据学习模式过滤和调整推荐
            adaptive_recommendations = self._filter_recommendations_by_learning_pattern(
                base_recommendations.get('symbols', []), learning_pattern
            )

            # 生成学习建议
            learning_suggestions = self._generate_learning_suggestions(learning_pattern, adaptive_recommendations)

            return {
                'symbols': adaptive_recommendations,
                'learning_pattern': learning_pattern,
                'learning_suggestions': learning_suggestions,
                'adaptation_applied': True,
                'context': base_recommendations.get('context', {}),
                'total_count': len(adaptive_recommendations)
            }

        except Exception as e:
            print(f"自适应推荐失败: {e}")
            # 回退到基础推荐
            return self.get_symbol_recommendations(user_id, context)

    def _adapt_context_by_learning_pattern(self, context: Dict[str, Any], learning_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """根据学习模式调整上下文"""
        adapted_context = context.copy()

        # 根据学习风格调整
        learning_style = learning_pattern.get('learning_style', 'balanced')

        if learning_style == 'explorer':
            # 探索型学习者：增加新符号的权重
            adapted_context['exploration_bonus'] = 0.3
        elif learning_style == 'specialist':
            # 专精型学习者：增加熟悉符号的权重
            adapted_context['familiarity_bonus'] = 0.3
        elif learning_style == 'focused':
            # 专注型学习者：基于偏好类别推荐
            preferred_categories = learning_pattern.get('preferred_categories', [])
            if preferred_categories:
                adapted_context['preferred_categories'] = preferred_categories

        # 根据活动水平调整
        activity_level = learning_pattern.get('activity_level', 'medium')
        if activity_level in ['low', 'very_low']:
            # 低活跃度用户：推荐更简单的符号
            adapted_context['difficulty_adjustment'] = -0.2
        elif activity_level in ['high', 'very_high']:
            # 高活跃度用户：可以推荐更复杂的符号
            adapted_context['difficulty_adjustment'] = 0.2

        # 根据掌握水平调整
        mastery_levels = learning_pattern.get('mastery_levels', {})
        if mastery_levels:
            adapted_context['mastery_levels'] = mastery_levels

        return adapted_context

    def _filter_recommendations_by_learning_pattern(self, recommendations: List[Dict], learning_pattern: Dict[str, Any]) -> List[Dict]:
        """根据学习模式过滤推荐"""
        if not recommendations:
            return recommendations

        filtered_recommendations = []
        learning_style = learning_pattern.get('learning_style', 'balanced')
        mastery_levels = learning_pattern.get('mastery_levels', {})

        for rec in recommendations:
            symbol_id = str(rec.get('id', ''))
            current_mastery = mastery_levels.get(symbol_id, 0.0)

            # 根据学习风格调整分数
            adjusted_score = rec.get('score', 0.5)

            if learning_style == 'explorer':
                # 探索型：新符号加分，已掌握符号减分
                if current_mastery < 0.3:
                    adjusted_score *= 1.3
                elif current_mastery > 0.7:
                    adjusted_score *= 0.7
            elif learning_style == 'specialist':
                # 专精型：已掌握符号加分
                if current_mastery > 0.5:
                    adjusted_score *= 1.2
                elif current_mastery < 0.2:
                    adjusted_score *= 0.8
            elif learning_style == 'focused':
                # 专注型：中等掌握度符号加分
                if 0.3 <= current_mastery <= 0.7:
                    adjusted_score *= 1.2

            # 更新推荐项
            filtered_rec = rec.copy()
            filtered_rec['adapted_score'] = adjusted_score
            filtered_rec['original_score'] = rec.get('score', 0.5)
            filtered_rec['mastery_level'] = current_mastery
            filtered_rec['learning_adaptation'] = self._generate_adaptation_reason(learning_style, current_mastery)

            filtered_recommendations.append(filtered_rec)

        # 按调整后的分数重新排序
        filtered_recommendations.sort(key=lambda x: x.get('adapted_score', 0), reverse=True)

        return filtered_recommendations

    def _generate_adaptation_reason(self, learning_style: str, mastery_level: float) -> str:
        """生成适应性调整原因"""
        if learning_style == 'explorer' and mastery_level < 0.3:
            return "适合探索型学习者的新符号"
        elif learning_style == 'specialist' and mastery_level > 0.5:
            return "基于专精型学习偏好推荐"
        elif learning_style == 'focused' and 0.3 <= mastery_level <= 0.7:
            return "适合专注型学习者的进阶符号"
        else:
            return "基于学习模式调整"

    def _generate_learning_suggestions(self, learning_pattern: Dict[str, Any], recommendations: List[Dict]) -> List[str]:
        """生成学习建议"""
        suggestions = []

        activity_level = learning_pattern.get('activity_level', 'medium')
        consistency = learning_pattern.get('learning_consistency', 0.0)
        retention_rate = learning_pattern.get('retention_rate', 0.0)

        # 基于活动水平的建议
        if activity_level in ['low', 'very_low']:
            suggestions.append("建议增加练习频率，每天至少使用15分钟")
        elif activity_level in ['high', 'very_high']:
            suggestions.append("保持良好的学习节奏，可以尝试更具挑战性的符号")

        # 基于一致性的建议
        if consistency < 0.5:
            suggestions.append("建议建立规律的学习时间，提高学习一致性")
        elif consistency > 0.8:
            suggestions.append("您的学习习惯很好，继续保持")

        # 基于保持率的建议
        if retention_rate < 0.5:
            suggestions.append("建议定期复习已学符号，加强记忆巩固")
        elif retention_rate > 0.8:
            suggestions.append("您的符号掌握能力很强，可以学习更多新符号")

        # 基于推荐符号的建议
        if recommendations:
            high_score_symbols = [rec for rec in recommendations if rec.get('adapted_score', 0) > 0.8]
            if high_score_symbols:
                suggestions.append(f"重点关注推荐的{len(high_score_symbols)}个高匹配度符号")

        return suggestions[:5]  # 限制建议数量


class ContextAnalyzer:
    """上下文分析器"""
    
    def analyze_context(self, question_text: str, current_input: str) -> Dict[str, Any]:
        """分析数学上下文"""
        context = {
            'has_equations': self._has_equations(question_text, current_input),
            'has_geometry': self._has_geometry(question_text),
            'has_calculus': self._has_calculus(question_text, current_input),
            'has_algebra': self._has_algebra(question_text),
            'has_statistics': self._has_statistics(question_text),
            'complexity': self._calculate_complexity(question_text, current_input)
        }
        
        return context
    
    def _has_equations(self, question_text: str, current_input: str) -> bool:
        return '=' in current_input or bool(re.search(r'方程|等式', question_text))
    
    def _has_geometry(self, question_text: str) -> bool:
        return bool(re.search(r'三角形|圆|角度|面积|周长|几何', question_text))
    
    def _has_calculus(self, question_text: str, current_input: str) -> bool:
        return (bool(re.search(r'导数|积分|极限|微积分', question_text)) or 
                bool(re.search(r'\\int|\\sum|\\lim', current_input)))
    
    def _has_algebra(self, question_text: str) -> bool:
        return bool(re.search(r'代数|多项式|因式分解|方程', question_text))
    
    def _has_statistics(self, question_text: str) -> bool:
        return bool(re.search(r'概率|统计|平均|方差', question_text))
    
    def _calculate_complexity(self, question_text: str, current_input: str) -> str:
        score = 0
        
        # 基于符号复杂度
        if re.search(r'\\int|\\sum|\\prod', current_input):
            score += 3
        if re.search(r'\\frac|\\sqrt', current_input):
            score += 2
        if re.search(r'[α-ωΑ-Ω]', current_input):
            score += 1
        
        # 基于题目复杂度
        if re.search(r'微积分|导数|积分', question_text):
            score += 3
        if re.search(r'三角函数|对数', question_text):
            score += 2
        if re.search(r'方程|不等式', question_text):
            score += 1
        
        if score >= 5:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'


class CompletionEngine:
    """符号补全引擎"""
    
    def __init__(self, symbols_data: List[Dict]):
        self.symbols_data = symbols_data
        self.symbol_index = self._build_symbol_index()
    
    def _build_symbol_index(self) -> Dict[str, List[Dict]]:
        """构建符号索引"""
        index = defaultdict(list)
        
        for symbol in self.symbols_data:
            # 索引符号本身
            symbol_text = symbol.get('symbol', '').lower()
            if symbol_text:
                index[symbol_text].append(symbol)
            
            # 索引LaTeX命令
            latex = symbol.get('latex', '').lower()
            if latex and latex.startswith('\\'):
                latex_name = latex[1:]  # 去掉反斜杠
                index[latex_name].append(symbol)
            
            # 索引描述中的关键词
            description = symbol.get('description', '').lower()
            for word in description.split():
                if len(word) > 1:
                    index[word].append(symbol)
        
        return index
    
    def get_completions(self, partial_input: str, context: Dict, usage_patterns: Dict) -> Dict[str, Any]:
        """获取补全建议"""
        if not partial_input or len(partial_input) < 1:
            return {"completions": []}
        
        # 处理LaTeX命令
        if partial_input.startswith('\\'):
            search_term = partial_input[1:].lower()
            is_latex = True
        else:
            search_term = partial_input.lower()
            is_latex = False
        
        # 搜索匹配的符号
        matches = []
        for key, symbols in self.symbol_index.items():
            if key.startswith(search_term):
                for symbol in symbols:
                    if symbol not in matches:
                        matches.append(symbol)
        
        # 计算补全建议的分数
        completions = []
        for symbol in matches[:20]:  # 限制结果数量
            score = self._calculate_completion_score(
                symbol, partial_input, search_term, usage_patterns
            )
            
            insert_text = symbol.get('latex', symbol.get('symbol', ''))
            if is_latex and not insert_text.startswith('\\'):
                insert_text = '\\' + insert_text
            
            completions.append({
                'id': symbol.get('id'),
                'symbol': symbol.get('symbol'),
                'latex': symbol.get('latex'),
                'description': symbol.get('description'),
                'score': score,
                'insertText': insert_text,
                'replaceLength': len(partial_input)
            })
        
        # 按分数排序
        completions.sort(key=lambda x: x['score'], reverse=True)
        
        return {"completions": completions[:10]}
    
    def _calculate_completion_score(self, symbol: Dict, partial_input: str, search_term: str, usage_patterns: Dict) -> float:
        """计算补全分数"""
        score = 0.5  # 基础分数
        
        # 精确匹配加分
        symbol_text = symbol.get('symbol', '').lower()
        latex_text = symbol.get('latex', '').lower().lstrip('\\')
        
        if symbol_text == search_term or latex_text == search_term:
            score += 0.4
        elif symbol_text.startswith(search_term) or latex_text.startswith(search_term):
            score += 0.3
        
        # 使用频率加分
        symbol_id = str(symbol.get('id', ''))
        usage_count = usage_patterns.get(symbol_id, {}).get('count', 0)
        score += min(0.2, usage_count * 0.01)
        
        # 符号长度惩罚（更短的符号优先）
        symbol_length = len(symbol.get('symbol', ''))
        if symbol_length <= 2:
            score += 0.1
        elif symbol_length > 5:
            score -= 0.1
        
        return min(1.0, score)
