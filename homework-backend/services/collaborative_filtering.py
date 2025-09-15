# -*- coding: utf-8 -*-
"""
协同过滤推荐算法实现
用于符号推荐系统的个性化推荐
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from datetime import datetime, timedelta
import math


class CollaborativeFiltering:
    """协同过滤推荐算法类"""
    
    def __init__(self, data_path: str = 'data'):
        self.data_path = data_path
        self.user_item_matrix = {}  # 用户-符号使用矩阵
        self.user_similarity_cache = {}  # 用户相似度缓存
        self.item_similarity_cache = {}  # 符号相似度缓存
        self.user_profiles = {}  # 用户画像
        self.symbol_profiles = {}  # 符号画像
        
        # 加载数据
        self._load_usage_data()
        self._build_user_profiles()
        self._build_symbol_profiles()
    
    def _load_usage_data(self):
        """加载用户符号使用数据"""
        usage_file = os.path.join(self.data_path, 'symbol_usage.json')
        
        if os.path.exists(usage_file):
            try:
                with open(usage_file, 'r', encoding='utf-8') as f:
                    usage_data = json.load(f)
                    
                # 构建用户-符号矩阵
                for user_id, user_data in usage_data.items():
                    if user_id not in self.user_item_matrix:
                        self.user_item_matrix[user_id] = {}
                    
                    for symbol_id, symbol_data in user_data.items():
                        # 使用频率作为评分
                        count = symbol_data.get('count', 0)
                        last_used = symbol_data.get('last_used')
                        
                        # 计算时间衰减因子
                        time_decay = self._calculate_time_decay(last_used)
                        
                        # 综合评分 = 使用次数 * 时间衰减
                        rating = count * time_decay
                        self.user_item_matrix[user_id][symbol_id] = rating
                        
            except Exception as e:
                print(f"加载使用数据失败: {e}")
                self.user_item_matrix = {}
        else:
            # 生成模拟数据用于测试
            self._generate_mock_data()
    
    def _calculate_time_decay(self, last_used_str: str) -> float:
        """计算时间衰减因子"""
        if not last_used_str:
            return 0.5  # 默认衰减
        
        try:
            last_used = datetime.fromisoformat(last_used_str.replace('Z', '+00:00'))
            now = datetime.now()
            days_diff = (now - last_used).days
            
            # 指数衰减：越近期使用，权重越高
            decay = math.exp(-days_diff / 30.0)  # 30天半衰期
            return max(0.1, decay)  # 最小保持0.1的权重
        except:
            return 0.5
    
    def _generate_mock_data(self):
        """生成模拟数据用于测试"""
        # 模拟10个用户对15个符号的使用数据
        users = [f"user_{i}" for i in range(1, 11)]
        symbols = [f"symbol_{i}" for i in range(1, 16)]
        
        np.random.seed(42)  # 确保可重现
        
        for user in users:
            self.user_item_matrix[user] = {}
            # 每个用户随机使用5-10个符号
            used_symbols = np.random.choice(symbols, size=np.random.randint(5, 11), replace=False)
            
            for symbol in used_symbols:
                # 使用次数在1-50之间
                count = np.random.randint(1, 51)
                # 时间衰减在0.3-1.0之间
                time_decay = np.random.uniform(0.3, 1.0)
                rating = count * time_decay
                self.user_item_matrix[user][symbol] = rating
    
    def _build_user_profiles(self):
        """构建用户画像"""
        for user_id, user_ratings in self.user_item_matrix.items():
            if not user_ratings:
                continue
                
            # 计算用户的平均评分
            avg_rating = np.mean(list(user_ratings.values()))
            
            # 计算用户的评分标准差（反映用户的选择性）
            rating_std = np.std(list(user_ratings.values()))
            
            # 用户活跃度（使用的符号数量）
            activity_level = len(user_ratings)
            
            # 用户偏好的符号类别（需要符号类别信息）
            preferred_categories = self._get_user_preferred_categories(user_id, user_ratings)
            
            self.user_profiles[user_id] = {
                'avg_rating': avg_rating,
                'rating_std': rating_std,
                'activity_level': activity_level,
                'preferred_categories': preferred_categories,
                'total_usage': sum(user_ratings.values())
            }
    
    def _build_symbol_profiles(self):
        """构建符号画像"""
        # 统计每个符号的使用情况
        symbol_stats = defaultdict(list)
        
        for user_ratings in self.user_item_matrix.values():
            for symbol_id, rating in user_ratings.items():
                symbol_stats[symbol_id].append(rating)
        
        for symbol_id, ratings in symbol_stats.items():
            if not ratings:
                continue
                
            self.symbol_profiles[symbol_id] = {
                'avg_rating': np.mean(ratings),
                'rating_std': np.std(ratings),
                'popularity': len(ratings),  # 使用该符号的用户数
                'total_usage': sum(ratings)
            }
    
    def _get_user_preferred_categories(self, user_id: str, user_ratings: Dict) -> List[str]:
        """获取用户偏好的符号类别"""
        # 这里需要符号类别信息，暂时返回空列表
        # 在实际应用中，应该根据符号ID查找对应的类别
        return []
    
    def calculate_user_similarity(self, user1: str, user2: str) -> float:
        """计算两个用户之间的相似度（皮尔逊相关系数）"""
        cache_key = f"{min(user1, user2)}_{max(user1, user2)}"
        
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]
        
        if user1 not in self.user_item_matrix or user2 not in self.user_item_matrix:
            return 0.0
        
        user1_ratings = self.user_item_matrix[user1]
        user2_ratings = self.user_item_matrix[user2]
        
        # 找到两个用户都使用过的符号
        common_symbols = set(user1_ratings.keys()) & set(user2_ratings.keys())
        
        if len(common_symbols) < 2:  # 至少需要2个共同符号
            similarity = 0.0
        else:
            # 计算皮尔逊相关系数
            user1_common = [user1_ratings[symbol] for symbol in common_symbols]
            user2_common = [user2_ratings[symbol] for symbol in common_symbols]
            
            similarity = self._pearson_correlation(user1_common, user2_common)
        
        # 缓存结果
        self.user_similarity_cache[cache_key] = similarity
        return similarity
    
    def calculate_item_similarity(self, item1: str, item2: str) -> float:
        """计算两个符号之间的相似度（余弦相似度）"""
        cache_key = f"{min(item1, item2)}_{max(item1, item2)}"
        
        if cache_key in self.item_similarity_cache:
            return self.item_similarity_cache[cache_key]
        
        # 获取使用这两个符号的用户及其评分
        item1_users = {}
        item2_users = {}
        
        for user_id, user_ratings in self.user_item_matrix.items():
            if item1 in user_ratings:
                item1_users[user_id] = user_ratings[item1]
            if item2 in user_ratings:
                item2_users[user_id] = user_ratings[item2]
        
        # 找到使用过这两个符号的共同用户
        common_users = set(item1_users.keys()) & set(item2_users.keys())
        
        if len(common_users) < 2:
            similarity = 0.0
        else:
            # 计算余弦相似度
            item1_vector = [item1_users[user] for user in common_users]
            item2_vector = [item2_users[user] for user in common_users]
            
            similarity = self._cosine_similarity(item1_vector, item2_vector)
        
        # 缓存结果
        self.item_similarity_cache[cache_key] = similarity
        return similarity
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """计算皮尔逊相关系数"""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_sq = sum(xi * xi for xi in x)
        sum_y_sq = sum(yi * yi for yi in y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _cosine_similarity(self, x: List[float], y: List[float]) -> float:
        """计算余弦相似度"""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        dot_product = sum(xi * yi for xi, yi in zip(x, y))
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        norm_y = math.sqrt(sum(yi * yi for yi in y))
        
        if norm_x == 0 or norm_y == 0:
            return 0.0
        
        return dot_product / (norm_x * norm_y)
    
    def get_user_based_recommendations(self, target_user: str, k: int = 5, n: int = 10) -> List[Tuple[str, float]]:
        """基于用户的协同过滤推荐"""
        if target_user not in self.user_item_matrix:
            return []
        
        target_ratings = self.user_item_matrix[target_user]
        
        # 计算与目标用户的相似度
        user_similarities = []
        for user_id in self.user_item_matrix:
            if user_id != target_user:
                similarity = self.calculate_user_similarity(target_user, user_id)
                if similarity > 0:
                    user_similarities.append((user_id, similarity))
        
        # 选择最相似的k个用户
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = user_similarities[:k]
        
        if not similar_users:
            return []
        
        # 计算推荐分数
        recommendations = defaultdict(float)
        similarity_sums = defaultdict(float)
        
        for similar_user, similarity in similar_users:
            similar_ratings = self.user_item_matrix[similar_user]
            
            for symbol_id, rating in similar_ratings.items():
                if symbol_id not in target_ratings:  # 只推荐用户未使用过的符号
                    recommendations[symbol_id] += similarity * rating
                    similarity_sums[symbol_id] += similarity
        
        # 归一化推荐分数
        final_recommendations = []
        for symbol_id, weighted_sum in recommendations.items():
            if similarity_sums[symbol_id] > 0:
                normalized_score = weighted_sum / similarity_sums[symbol_id]
                final_recommendations.append((symbol_id, normalized_score))
        
        # 按分数排序并返回前n个
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:n]
    
    def get_item_based_recommendations(self, target_user: str, n: int = 10) -> List[Tuple[str, float]]:
        """基于物品的协同过滤推荐"""
        if target_user not in self.user_item_matrix:
            return []
        
        target_ratings = self.user_item_matrix[target_user]
        
        if not target_ratings:
            return []
        
        # 计算推荐分数
        recommendations = defaultdict(float)
        similarity_sums = defaultdict(float)
        
        # 对于用户使用过的每个符号
        for used_symbol, user_rating in target_ratings.items():
            # 找到与该符号相似的其他符号
            for candidate_symbol in self.symbol_profiles:
                if candidate_symbol not in target_ratings:  # 只推荐未使用过的符号
                    similarity = self.calculate_item_similarity(used_symbol, candidate_symbol)
                    if similarity > 0:
                        recommendations[candidate_symbol] += similarity * user_rating
                        similarity_sums[candidate_symbol] += similarity
        
        # 归一化推荐分数
        final_recommendations = []
        for symbol_id, weighted_sum in recommendations.items():
            if similarity_sums[symbol_id] > 0:
                normalized_score = weighted_sum / similarity_sums[symbol_id]
                final_recommendations.append((symbol_id, normalized_score))
        
        # 按分数排序并返回前n个
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:n]
    
    def get_hybrid_recommendations(self, target_user: str, n: int = 10, 
                                 user_weight: float = 0.6, item_weight: float = 0.4) -> List[Tuple[str, float]]:
        """混合推荐（用户协同过滤 + 物品协同过滤）"""
        user_recs = self.get_user_based_recommendations(target_user, n=n*2)
        item_recs = self.get_item_based_recommendations(target_user, n=n*2)
        
        # 合并推荐结果
        combined_scores = defaultdict(float)
        
        # 用户协同过滤结果
        for symbol_id, score in user_recs:
            combined_scores[symbol_id] += user_weight * score
        
        # 物品协同过滤结果
        for symbol_id, score in item_recs:
            combined_scores[symbol_id] += item_weight * score
        
        # 排序并返回
        final_recommendations = [(symbol_id, score) for symbol_id, score in combined_scores.items()]
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return final_recommendations[:n]
    
    def update_user_rating(self, user_id: str, symbol_id: str, usage_count: int, last_used: str = None):
        """更新用户对符号的使用数据"""
        if user_id not in self.user_item_matrix:
            self.user_item_matrix[user_id] = {}
        
        # 计算时间衰减
        time_decay = self._calculate_time_decay(last_used) if last_used else 1.0
        
        # 更新评分
        rating = usage_count * time_decay
        self.user_item_matrix[user_id][symbol_id] = rating
        
        # 清除相关缓存
        self._clear_user_cache(user_id)
        self._clear_item_cache(symbol_id)
        
        # 更新用户画像
        self._update_user_profile(user_id)
        self._update_symbol_profile(symbol_id)
    
    def _clear_user_cache(self, user_id: str):
        """清除用户相关的相似度缓存"""
        keys_to_remove = []
        for key in self.user_similarity_cache:
            if user_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.user_similarity_cache[key]
    
    def _clear_item_cache(self, symbol_id: str):
        """清除符号相关的相似度缓存"""
        keys_to_remove = []
        for key in self.item_similarity_cache:
            if symbol_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.item_similarity_cache[key]
    
    def _update_user_profile(self, user_id: str):
        """更新单个用户的画像"""
        if user_id not in self.user_item_matrix:
            return
        
        user_ratings = self.user_item_matrix[user_id]
        if not user_ratings:
            return
        
        avg_rating = np.mean(list(user_ratings.values()))
        rating_std = np.std(list(user_ratings.values()))
        activity_level = len(user_ratings)
        preferred_categories = self._get_user_preferred_categories(user_id, user_ratings)
        
        self.user_profiles[user_id] = {
            'avg_rating': avg_rating,
            'rating_std': rating_std,
            'activity_level': activity_level,
            'preferred_categories': preferred_categories,
            'total_usage': sum(user_ratings.values())
        }
    
    def _update_symbol_profile(self, symbol_id: str):
        """更新单个符号的画像"""
        ratings = []
        for user_ratings in self.user_item_matrix.values():
            if symbol_id in user_ratings:
                ratings.append(user_ratings[symbol_id])
        
        if ratings:
            self.symbol_profiles[symbol_id] = {
                'avg_rating': np.mean(ratings),
                'rating_std': np.std(ratings),
                'popularity': len(ratings),
                'total_usage': sum(ratings)
            }
    
    def save_data(self):
        """保存数据到文件"""
        # 保存用户-符号矩阵
        usage_file = os.path.join(self.data_path, 'symbol_usage.json')
        
        # 转换数据格式以便保存
        save_data = {}
        for user_id, user_ratings in self.user_item_matrix.items():
            save_data[user_id] = {}
            for symbol_id, rating in user_ratings.items():
                save_data[user_id][symbol_id] = {
                    'rating': rating,
                    'last_updated': datetime.now().isoformat()
                }
        
        try:
            os.makedirs(self.data_path, exist_ok=True)
            with open(usage_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        
        # 计算用户在所有用户中的排名
        all_activity_levels = [p['activity_level'] for p in self.user_profiles.values()]
        activity_rank = sorted(all_activity_levels, reverse=True).index(profile['activity_level']) + 1
        
        return {
            'user_id': user_id,
            'profile': profile,
            'activity_rank': activity_rank,
            'total_users': len(self.user_profiles),
            'similar_users_count': len([u for u in self.user_item_matrix if u != user_id and 
                                      self.calculate_user_similarity(user_id, u) > 0.3])
        }
