# -*- coding: utf-8 -*-
"""
学习分析服务
用于分析学生的符号使用模式和学习行为
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math


class LearningAnalytics:
    """学习分析类"""
    
    def __init__(self, data_path: str = 'data'):
        self.data_path = data_path
        self.usage_data = {}  # 用户使用数据
        self.learning_patterns = {}  # 学习模式
        self.performance_metrics = {}  # 性能指标
        
        # 加载数据
        self._load_usage_data()
        self._load_learning_patterns()
    
    def _load_usage_data(self):
        """加载使用数据"""
        usage_file = os.path.join(self.data_path, 'symbol_usage.json')
        
        if os.path.exists(usage_file):
            try:
                with open(usage_file, 'r', encoding='utf-8') as f:
                    self.usage_data = json.load(f)
            except Exception as e:
                print(f"加载使用数据失败: {e}")
                self.usage_data = {}
    
    def _load_learning_patterns(self):
        """加载学习模式数据"""
        patterns_file = os.path.join(self.data_path, 'learning_patterns.json')
        
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    self.learning_patterns = json.load(f)
            except Exception as e:
                print(f"加载学习模式失败: {e}")
                self.learning_patterns = {}
    
    def analyze_user_learning_pattern(self, user_id: str) -> Dict[str, Any]:
        """分析用户学习模式"""
        if user_id not in self.usage_data:
            return self._get_default_learning_pattern()
        
        user_data = self.usage_data[user_id]
        
        # 计算各种学习指标
        pattern = {
            'user_id': user_id,
            'activity_level': self._calculate_activity_level(user_data),
            'learning_consistency': self._calculate_learning_consistency(user_data),
            'symbol_diversity': self._calculate_symbol_diversity(user_data),
            'difficulty_progression': self._analyze_difficulty_progression(user_data),
            'usage_frequency_pattern': self._analyze_usage_frequency(user_data),
            'learning_velocity': self._calculate_learning_velocity(user_data),
            'retention_rate': self._calculate_retention_rate(user_data),
            'preferred_categories': self._identify_preferred_categories(user_data),
            'learning_style': self._identify_learning_style(user_data),
            'mastery_levels': self._calculate_mastery_levels(user_data),
            'recommendation_effectiveness': self._analyze_recommendation_effectiveness(user_data)
        }
        
        # 缓存学习模式
        self.learning_patterns[user_id] = pattern
        
        return pattern
    
    def _get_default_learning_pattern(self) -> Dict[str, Any]:
        """获取默认学习模式"""
        return {
            'user_id': '',
            'activity_level': 'low',
            'learning_consistency': 0.0,
            'symbol_diversity': 0.0,
            'difficulty_progression': 'stable',
            'usage_frequency_pattern': 'irregular',
            'learning_velocity': 0.0,
            'retention_rate': 0.0,
            'preferred_categories': [],
            'learning_style': 'unknown',
            'mastery_levels': {},
            'recommendation_effectiveness': 0.0
        }
    
    def _calculate_activity_level(self, user_data: Dict) -> str:
        """计算活动水平"""
        total_usage = sum(symbol_data.get('count', 0) for symbol_data in user_data.values())
        unique_symbols = len(user_data)
        
        # 计算活动分数
        activity_score = total_usage * 0.7 + unique_symbols * 0.3
        
        if activity_score >= 100:
            return 'very_high'
        elif activity_score >= 50:
            return 'high'
        elif activity_score >= 20:
            return 'medium'
        elif activity_score >= 5:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_learning_consistency(self, user_data: Dict) -> float:
        """计算学习一致性"""
        if not user_data:
            return 0.0
        
        # 分析使用时间的一致性
        usage_times = []
        for symbol_data in user_data.values():
            contexts = symbol_data.get('contexts', [])
            for context in contexts:
                timestamp = context.get('timestamp')
                if timestamp:
                    try:
                        usage_times.append(datetime.fromisoformat(timestamp))
                    except:
                        continue
        
        if len(usage_times) < 2:
            return 0.0
        
        # 计算使用间隔的标准差
        usage_times.sort()
        intervals = []
        for i in range(1, len(usage_times)):
            interval = (usage_times[i] - usage_times[i-1]).total_seconds() / 3600  # 小时
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        # 一致性 = 1 - (标准差 / 平均值)，归一化到0-1
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 0.0
        
        consistency = max(0, 1 - (std_interval / mean_interval))
        return min(1.0, consistency)
    
    def _calculate_symbol_diversity(self, user_data: Dict) -> float:
        """计算符号多样性"""
        if not user_data:
            return 0.0
        
        # 使用香农熵计算多样性
        total_usage = sum(symbol_data.get('count', 0) for symbol_data in user_data.values())
        
        if total_usage == 0:
            return 0.0
        
        entropy = 0.0
        for symbol_data in user_data.values():
            count = symbol_data.get('count', 0)
            if count > 0:
                probability = count / total_usage
                entropy -= probability * math.log2(probability)
        
        # 归一化到0-1范围
        max_entropy = math.log2(len(user_data)) if len(user_data) > 1 else 1
        diversity = entropy / max_entropy if max_entropy > 0 else 0
        
        return min(1.0, diversity)
    
    def _analyze_difficulty_progression(self, user_data: Dict) -> str:
        """分析难度进展"""
        # 这里需要符号难度信息，暂时使用简化逻辑
        symbol_counts = [(symbol_id, data.get('count', 0)) for symbol_id, data in user_data.items()]
        symbol_counts.sort(key=lambda x: x[1], reverse=True)
        
        # 简单的难度分析：假设符号ID越大难度越高
        if len(symbol_counts) < 3:
            return 'insufficient_data'
        
        # 分析最常用的符号
        top_symbols = [int(symbol_id) for symbol_id, _ in symbol_counts[:3] if symbol_id.isdigit()]
        
        if not top_symbols:
            return 'unknown'
        
        avg_difficulty = np.mean(top_symbols)
        
        if avg_difficulty > 10:
            return 'advanced'
        elif avg_difficulty > 5:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _analyze_usage_frequency(self, user_data: Dict) -> str:
        """分析使用频率模式"""
        if not user_data:
            return 'no_data'
        
        # 分析使用时间分布
        usage_times = []
        for symbol_data in user_data.values():
            contexts = symbol_data.get('contexts', [])
            for context in contexts:
                timestamp = context.get('timestamp')
                if timestamp:
                    try:
                        usage_times.append(datetime.fromisoformat(timestamp))
                    except:
                        continue
        
        if len(usage_times) < 5:
            return 'insufficient_data'
        
        # 按天分组
        daily_usage = defaultdict(int)
        for usage_time in usage_times:
            day_key = usage_time.date()
            daily_usage[day_key] += 1
        
        usage_counts = list(daily_usage.values())
        
        # 分析模式
        if len(usage_counts) < 3:
            return 'irregular'
        
        # 计算变异系数
        mean_usage = np.mean(usage_counts)
        std_usage = np.std(usage_counts)
        
        if mean_usage == 0:
            return 'irregular'
        
        cv = std_usage / mean_usage
        
        if cv < 0.3:
            return 'very_regular'
        elif cv < 0.6:
            return 'regular'
        elif cv < 1.0:
            return 'somewhat_irregular'
        else:
            return 'irregular'
    
    def _calculate_learning_velocity(self, user_data: Dict) -> float:
        """计算学习速度"""
        if not user_data:
            return 0.0
        
        # 计算新符号学习速度
        first_usage_times = []
        for symbol_data in user_data.values():
            contexts = symbol_data.get('contexts', [])
            if contexts:
                try:
                    first_time = datetime.fromisoformat(contexts[0].get('timestamp', ''))
                    first_usage_times.append(first_time)
                except:
                    continue
        
        if len(first_usage_times) < 2:
            return 0.0
        
        first_usage_times.sort()
        
        # 计算学习新符号的时间间隔
        total_days = (first_usage_times[-1] - first_usage_times[0]).days
        
        if total_days == 0:
            return len(first_usage_times)  # 一天内学习的符号数
        
        # 每天学习新符号的平均数量
        velocity = len(first_usage_times) / total_days
        
        return min(10.0, velocity)  # 限制最大值
    
    def _calculate_retention_rate(self, user_data: Dict) -> float:
        """计算保持率"""
        if not user_data:
            return 0.0
        
        # 分析符号的重复使用情况
        total_symbols = len(user_data)
        retained_symbols = 0
        
        for symbol_data in user_data.values():
            count = symbol_data.get('count', 0)
            contexts = symbol_data.get('contexts', [])
            
            # 如果符号使用超过1次，或者在不同时间使用过，认为是保持的
            if count > 1 or len(contexts) > 1:
                retained_symbols += 1
        
        return retained_symbols / total_symbols if total_symbols > 0 else 0.0
    
    def _identify_preferred_categories(self, user_data: Dict) -> List[str]:
        """识别偏好类别"""
        # 这里需要符号类别信息，暂时使用简化逻辑
        category_usage = defaultdict(int)
        
        for symbol_id, symbol_data in user_data.items():
            count = symbol_data.get('count', 0)
            
            # 简化的类别映射
            if symbol_id in ['1', '2', '3', '4']:
                category_usage['基本运算'] += count
            elif symbol_id in ['5', '6', '7', '8']:
                category_usage['关系符号'] += count
            elif symbol_id in ['9', '10', '11']:
                category_usage['希腊字母'] += count
            elif symbol_id in ['12', '13']:
                category_usage['微积分'] += count
            else:
                category_usage['其他'] += count
        
        # 按使用量排序
        sorted_categories = sorted(category_usage.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前3个类别
        return [category for category, _ in sorted_categories[:3] if category != '其他']
    
    def _identify_learning_style(self, user_data: Dict) -> str:
        """识别学习风格"""
        if not user_data:
            return 'unknown'
        
        total_usage = sum(symbol_data.get('count', 0) for symbol_data in user_data.values())
        unique_symbols = len(user_data)
        
        # 计算探索性指标
        exploration_ratio = unique_symbols / total_usage if total_usage > 0 else 0
        
        # 分析使用分布
        usage_counts = [symbol_data.get('count', 0) for symbol_data in user_data.values()]
        usage_variance = np.var(usage_counts) if usage_counts else 0
        
        # 基于指标判断学习风格
        if exploration_ratio > 0.7:
            return 'explorer'  # 探索型：喜欢尝试新符号
        elif exploration_ratio < 0.3:
            return 'specialist'  # 专精型：专注于少数符号
        elif usage_variance > np.mean(usage_counts) * 2:
            return 'focused'  # 专注型：有明显偏好
        else:
            return 'balanced'  # 平衡型：均衡使用
    
    def _calculate_mastery_levels(self, user_data: Dict) -> Dict[str, float]:
        """计算掌握水平"""
        mastery_levels = {}
        
        for symbol_id, symbol_data in user_data.items():
            count = symbol_data.get('count', 0)
            contexts = symbol_data.get('contexts', [])
            
            # 基于使用次数和使用场景多样性计算掌握度
            frequency_score = min(1.0, count / 20.0)  # 使用20次为满分
            diversity_score = min(1.0, len(contexts) / 5.0)  # 5个不同场景为满分
            
            # 时间衰减：最近使用的权重更高
            time_score = self._calculate_time_decay_score(symbol_data)
            
            # 综合掌握度
            mastery = (frequency_score * 0.4 + diversity_score * 0.3 + time_score * 0.3)
            mastery_levels[symbol_id] = mastery
        
        return mastery_levels
    
    def _calculate_time_decay_score(self, symbol_data: Dict) -> float:
        """计算时间衰减分数"""
        last_used = symbol_data.get('last_used')
        if not last_used:
            return 0.0
        
        try:
            last_time = datetime.fromisoformat(last_used)
            days_since = (datetime.now() - last_time).days
            
            # 指数衰减：30天半衰期
            decay_score = math.exp(-days_since / 30.0)
            return max(0.1, decay_score)
        except:
            return 0.5
    
    def _analyze_recommendation_effectiveness(self, user_data: Dict) -> float:
        """分析推荐效果"""
        # 这里需要推荐历史数据，暂时返回默认值
        # 在实际应用中，应该分析用户对推荐符号的采用率
        return 0.7  # 默认70%的推荐效果
    
    def generate_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """生成学习洞察"""
        pattern = self.analyze_user_learning_pattern(user_id)
        
        insights = {
            'user_id': user_id,
            'overall_assessment': self._generate_overall_assessment(pattern),
            'strengths': self._identify_strengths(pattern),
            'areas_for_improvement': self._identify_improvement_areas(pattern),
            'learning_recommendations': self._generate_learning_recommendations(pattern),
            'progress_indicators': self._calculate_progress_indicators(pattern),
            'comparative_analysis': self._generate_comparative_analysis(user_id, pattern)
        }
        
        return insights
    
    def _generate_overall_assessment(self, pattern: Dict) -> str:
        """生成总体评估"""
        activity = pattern.get('activity_level', 'low')
        consistency = pattern.get('learning_consistency', 0.0)
        diversity = pattern.get('symbol_diversity', 0.0)
        
        if activity in ['high', 'very_high'] and consistency > 0.7 and diversity > 0.6:
            return "优秀的学习者：活跃度高，学习一致性强，符号使用多样化"
        elif activity in ['medium', 'high'] and consistency > 0.5:
            return "良好的学习者：有规律的学习习惯，持续进步"
        elif activity in ['low', 'medium'] and diversity > 0.5:
            return "探索型学习者：喜欢尝试不同符号，但需要提高学习频率"
        else:
            return "需要改进：建议增加学习频率和一致性"
    
    def _identify_strengths(self, pattern: Dict) -> List[str]:
        """识别优势"""
        strengths = []
        
        if pattern.get('activity_level') in ['high', 'very_high']:
            strengths.append("学习活跃度高")
        
        if pattern.get('learning_consistency', 0) > 0.7:
            strengths.append("学习习惯一致")
        
        if pattern.get('symbol_diversity', 0) > 0.6:
            strengths.append("符号使用多样化")
        
        if pattern.get('retention_rate', 0) > 0.7:
            strengths.append("知识保持能力强")
        
        if pattern.get('learning_velocity', 0) > 2.0:
            strengths.append("学习速度快")
        
        return strengths if strengths else ["正在建立学习模式"]
    
    def _identify_improvement_areas(self, pattern: Dict) -> List[str]:
        """识别改进领域"""
        improvements = []
        
        if pattern.get('activity_level') in ['low', 'very_low']:
            improvements.append("增加学习频率")
        
        if pattern.get('learning_consistency', 0) < 0.5:
            improvements.append("建立规律的学习习惯")
        
        if pattern.get('symbol_diversity', 0) < 0.4:
            improvements.append("尝试更多类型的符号")
        
        if pattern.get('retention_rate', 0) < 0.5:
            improvements.append("加强符号复习和巩固")
        
        return improvements
    
    def _generate_learning_recommendations(self, pattern: Dict) -> List[str]:
        """生成学习建议"""
        recommendations = []
        
        learning_style = pattern.get('learning_style', 'unknown')
        activity_level = pattern.get('activity_level', 'low')
        
        if learning_style == 'explorer':
            recommendations.append("继续保持探索精神，同时加强对常用符号的练习")
        elif learning_style == 'specialist':
            recommendations.append("在掌握核心符号的基础上，尝试学习相关的新符号")
        elif learning_style == 'focused':
            recommendations.append("保持专注的学习方式，逐步扩展到相关领域")
        
        if activity_level in ['low', 'very_low']:
            recommendations.append("建议每天至少练习15分钟，建立稳定的学习节奏")
        
        preferred_categories = pattern.get('preferred_categories', [])
        if preferred_categories:
            recommendations.append(f"基于您对{', '.join(preferred_categories)}的偏好，推荐学习相关的高级符号")
        
        return recommendations
    
    def _calculate_progress_indicators(self, pattern: Dict) -> Dict[str, float]:
        """计算进度指标"""
        return {
            'overall_progress': self._calculate_overall_progress(pattern),
            'consistency_score': pattern.get('learning_consistency', 0.0),
            'diversity_score': pattern.get('symbol_diversity', 0.0),
            'retention_score': pattern.get('retention_rate', 0.0),
            'velocity_score': min(1.0, pattern.get('learning_velocity', 0.0) / 5.0)
        }
    
    def _calculate_overall_progress(self, pattern: Dict) -> float:
        """计算总体进度"""
        indicators = [
            pattern.get('learning_consistency', 0.0),
            pattern.get('symbol_diversity', 0.0),
            pattern.get('retention_rate', 0.0),
            min(1.0, pattern.get('learning_velocity', 0.0) / 5.0)
        ]
        
        return np.mean(indicators)
    
    def _generate_comparative_analysis(self, user_id: str, pattern: Dict) -> Dict[str, Any]:
        """生成对比分析"""
        # 与其他用户对比
        all_patterns = list(self.learning_patterns.values())
        
        if len(all_patterns) < 2:
            return {"message": "数据不足，无法进行对比分析"}
        
        user_activity = self._activity_level_to_score(pattern.get('activity_level', 'low'))
        user_consistency = pattern.get('learning_consistency', 0.0)
        user_diversity = pattern.get('symbol_diversity', 0.0)
        
        # 计算排名
        activity_scores = [self._activity_level_to_score(p.get('activity_level', 'low')) for p in all_patterns]
        consistency_scores = [p.get('learning_consistency', 0.0) for p in all_patterns]
        diversity_scores = [p.get('symbol_diversity', 0.0) for p in all_patterns]
        
        activity_rank = self._calculate_percentile(user_activity, activity_scores)
        consistency_rank = self._calculate_percentile(user_consistency, consistency_scores)
        diversity_rank = self._calculate_percentile(user_diversity, diversity_scores)
        
        return {
            'activity_percentile': activity_rank,
            'consistency_percentile': consistency_rank,
            'diversity_percentile': diversity_rank,
            'overall_ranking': np.mean([activity_rank, consistency_rank, diversity_rank])
        }
    
    def _activity_level_to_score(self, activity_level: str) -> float:
        """将活动水平转换为分数"""
        mapping = {
            'very_low': 0.1,
            'low': 0.3,
            'medium': 0.5,
            'high': 0.8,
            'very_high': 1.0
        }
        return mapping.get(activity_level, 0.3)
    
    def _calculate_percentile(self, value: float, all_values: List[float]) -> float:
        """计算百分位数"""
        if not all_values:
            return 50.0
        
        rank = sum(1 for v in all_values if v < value)
        percentile = (rank / len(all_values)) * 100
        
        return percentile
    
    def save_learning_patterns(self):
        """保存学习模式数据"""
        patterns_file = os.path.join(self.data_path, 'learning_patterns.json')
        
        try:
            os.makedirs(self.data_path, exist_ok=True)
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_patterns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存学习模式失败: {e}")
    
    def get_user_statistics_summary(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计摘要"""
        pattern = self.analyze_user_learning_pattern(user_id)
        insights = self.generate_learning_insights(user_id)
        
        return {
            'user_id': user_id,
            'learning_pattern': pattern,
            'insights': insights,
            'last_updated': datetime.now().isoformat()
        }
