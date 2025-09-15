# -*- coding: utf-8 -*-
"""
轻量级深度学习推荐器
不依赖BERT模型的简化版本，用于快速测试和部署
"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightMLPRecommender:
    """轻量级MLP推荐器"""
    
    def __init__(self, embedding_dim: int = 32, hidden_dims: List[int] = [64, 32, 16]):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        
        # 简化的权重矩阵（随机初始化）
        self.user_embeddings = {}
        self.symbol_embeddings = {}
        self.context_features = {}
        
        # 预定义的符号特征
        self._init_symbol_features()
        
        # 上下文分析器
        self.context_analyzer = ContextAnalyzer()
    
    def _init_symbol_features(self):
        """初始化符号特征"""
        # 基于符号类型的特征向量
        symbol_types = {
            'operator': [1.0, 0.0, 0.0, 0.0, 0.0],
            'number': [0.0, 1.0, 0.0, 0.0, 0.0],
            'variable': [0.0, 0.0, 1.0, 0.0, 0.0],
            'function': [0.0, 0.0, 0.0, 1.0, 0.0],
            'geometry': [0.0, 0.0, 0.0, 0.0, 1.0]
        }
        
        # 常见符号的分类
        self.symbol_categories = {
            '+': 'operator', '-': 'operator', '*': 'operator', '/': 'operator', '=': 'operator',
            '²': 'operator', '^': 'operator', '√': 'operator', '∫': 'operator', '∑': 'operator',
            'x': 'variable', 'y': 'variable', 'z': 'variable', 'a': 'variable', 'b': 'variable',
            'sin': 'function', 'cos': 'function', 'tan': 'function', 'log': 'function', 'ln': 'function',
            '∠': 'geometry', '°': 'geometry', '△': 'geometry', '⊥': 'geometry', '∥': 'geometry',
            '0': 'number', '1': 'number', '2': 'number', '3': 'number', '4': 'number',
            '5': 'number', '6': 'number', '7': 'number', '8': 'number', '9': 'number'
        }
        
        # 为每个符号生成特征向量
        for symbol, category in self.symbol_categories.items():
            base_features = symbol_types.get(category, [0.2, 0.2, 0.2, 0.2, 0.2])
            # 添加随机噪声使特征更加多样化
            noise = np.random.normal(0, 0.1, len(base_features))
            features = np.array(base_features) + noise
            features = np.clip(features, 0, 1)  # 确保在[0,1]范围内
            
            self.symbol_embeddings[symbol] = features.tolist()
    
    def get_user_embedding(self, user_id: str) -> List[float]:
        """获取用户嵌入向量"""
        if user_id not in self.user_embeddings:
            # 为新用户生成随机嵌入
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            self.user_embeddings[user_id] = embedding.tolist()
        
        return self.user_embeddings[user_id]
    
    def get_symbol_embedding(self, symbol: str) -> List[float]:
        """获取符号嵌入向量"""
        if symbol in self.symbol_embeddings:
            return self.symbol_embeddings[symbol]
        
        # 为未知符号生成默认嵌入
        default_embedding = [0.2] * 5  # 中性特征
        self.symbol_embeddings[symbol] = default_embedding
        return default_embedding
    
    def get_context_features(self, context_text: str) -> List[float]:
        """提取上下文特征"""
        features = self.context_analyzer.analyze(context_text)
        
        # 转换为数值特征向量
        feature_vector = [
            1.0 if features.get('has_equation', False) else 0.0,
            1.0 if features.get('has_fraction', False) else 0.0,
            1.0 if features.get('has_integral', False) else 0.0,
            1.0 if features.get('has_geometry', False) else 0.0,
            1.0 if features.get('has_trigonometry', False) else 0.0,
            features.get('complexity_score', 0.5),
            features.get('math_level', 0.5)
        ]
        
        return feature_vector
    
    def predict_score(self, user_id: str, symbol: str, context_text: str) -> float:
        """预测用户对符号的偏好分数"""
        try:
            # 获取特征向量
            user_emb = self.get_user_embedding(user_id)
            symbol_emb = self.get_symbol_embedding(symbol)
            context_features = self.get_context_features(context_text)
            
            # 简化的神经网络前向传播
            # 拼接所有特征
            all_features = user_emb[:5] + symbol_emb + context_features  # 限制用户嵌入长度
            
            # 第一层：线性变换 + ReLU
            hidden1 = self._linear_relu(all_features, self.hidden_dims[0])
            
            # 第二层
            hidden2 = self._linear_relu(hidden1, self.hidden_dims[1])
            
            # 输出层：sigmoid激活
            output = self._linear_sigmoid(hidden2, 1)
            
            return float(output[0])
            
        except Exception as e:
            logger.error(f"预测分数失败: {e}")
            return 0.5  # 默认分数
    
    def _linear_relu(self, inputs: List[float], output_dim: int) -> List[float]:
        """线性变换 + ReLU激活"""
        input_dim = len(inputs)
        
        # 简化的权重矩阵（随机生成）
        weights = np.random.normal(0, 0.1, (input_dim, output_dim))
        bias = np.random.normal(0, 0.01, output_dim)
        
        # 线性变换
        output = np.dot(inputs, weights) + bias
        
        # ReLU激活
        output = np.maximum(0, output)
        
        return output.tolist()
    
    def _linear_sigmoid(self, inputs: List[float], output_dim: int) -> List[float]:
        """线性变换 + Sigmoid激活"""
        input_dim = len(inputs)
        
        # 简化的权重矩阵
        weights = np.random.normal(0, 0.1, (input_dim, output_dim))
        bias = np.random.normal(0, 0.01, output_dim)
        
        # 线性变换
        output = np.dot(inputs, weights) + bias
        
        # Sigmoid激活
        output = 1 / (1 + np.exp(-np.clip(output, -500, 500)))  # 防止溢出
        
        return output.tolist()


class ContextAnalyzer:
    """上下文分析器"""
    
    def __init__(self):
        # 数学关键词模式
        self.patterns = {
            'equation': [r'方程', r'=', r'解', r'求解'],
            'fraction': [r'分数', r'\\frac', r'/', r'分子', r'分母'],
            'integral': [r'积分', r'\\int', r'∫', r'微积分'],
            'geometry': [r'几何', r'三角形', r'圆', r'角度', r'∠', r'°'],
            'trigonometry': [r'三角函数', r'sin', r'cos', r'tan', r'正弦', r'余弦', r'正切'],
            'algebra': [r'代数', r'多项式', r'因式分解'],
            'calculus': [r'导数', r'微分', r'极限', r'连续']
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """分析文本上下文"""
        features = {}
        
        # 检测数学主题
        for topic, patterns in self.patterns.items():
            has_topic = any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
            features[f'has_{topic}'] = has_topic
        
        # 计算复杂度分数
        complexity_indicators = [
            r'\\', r'∫', r'∑', r'∏', r'√', r'²', r'³',  # 复杂符号
            r'微积分', r'导数', r'极限', r'矩阵', r'向量'  # 高级概念
        ]
        
        complexity_count = sum(1 for indicator in complexity_indicators 
                             if re.search(indicator, text, re.IGNORECASE))
        features['complexity_score'] = min(1.0, complexity_count / 5.0)
        
        # 估算数学水平
        level_indicators = {
            'elementary': [r'加法', r'减法', r'乘法', r'除法', r'整数'],
            'middle': [r'分数', r'小数', r'百分比', r'比例', r'方程'],
            'high': [r'函数', r'三角函数', r'对数', r'指数'],
            'advanced': [r'微积分', r'导数', r'积分', r'极限', r'矩阵']
        }
        
        level_scores = {}
        for level, indicators in level_indicators.items():
            score = sum(1 for indicator in indicators 
                       if re.search(indicator, text, re.IGNORECASE))
            level_scores[level] = score
        
        # 确定主要数学水平
        max_level = max(level_scores, key=level_scores.get)
        level_mapping = {'elementary': 0.2, 'middle': 0.4, 'high': 0.6, 'advanced': 0.8}
        features['math_level'] = level_mapping.get(max_level, 0.5)
        
        return features


class LightweightDeepLearningRecommender:
    """轻量级深度学习推荐器"""
    
    def __init__(self, model_path: str = 'models/lightweight'):
        self.model_path = model_path
        self.mlp_recommender = LightweightMLPRecommender()
        
        # 确保模型目录存在
        os.makedirs(model_path, exist_ok=True)
        
        logger.info("轻量级深度学习推荐器初始化完成")
    
    def get_recommendations(self, user_id: str, context_text: str, 
                          candidate_symbols: List[Dict], top_k: int = 10) -> List[Dict]:
        """获取推荐结果"""
        if not candidate_symbols:
            return []
        
        try:
            recommendations = []
            
            for symbol_info in candidate_symbols:
                symbol = symbol_info.get('symbol', symbol_info.get('latex', ''))
                
                # 使用MLP模型预测分数
                score = self.mlp_recommender.predict_score(user_id, symbol, context_text)
                
                # 添加一些随机性以模拟真实的推荐多样性
                score += np.random.normal(0, 0.05)  # 小幅随机调整
                score = max(0.0, min(1.0, score))  # 确保在[0,1]范围内
                
                recommendations.append({
                    **symbol_info,
                    'score': score,
                    'source': 'deep_learning',
                    'model_type': 'Lightweight-MLP',
                    'dl_confidence': score
                })
            
            # 按分数排序并返回top_k
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"轻量级深度学习推荐失败: {e}")
            return []
    
    def save_model(self):
        """保存模型（轻量级版本主要保存用户嵌入）"""
        try:
            model_file = os.path.join(self.model_path, 'user_embeddings.json')
            
            data = {
                'user_embeddings': self.mlp_recommender.user_embeddings,
                'symbol_embeddings': self.mlp_recommender.symbol_embeddings,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(model_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info("轻量级模型保存成功")
            
        except Exception as e:
            logger.error(f"保存轻量级模型失败: {e}")
    
    def load_model(self):
        """加载模型"""
        try:
            model_file = os.path.join(self.model_path, 'user_embeddings.json')
            
            if os.path.exists(model_file):
                with open(model_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.mlp_recommender.user_embeddings = data.get('user_embeddings', {})
                saved_symbol_embeddings = data.get('symbol_embeddings', {})
                
                # 合并保存的符号嵌入
                self.mlp_recommender.symbol_embeddings.update(saved_symbol_embeddings)
                
                logger.info("轻量级模型加载成功")
            
        except Exception as e:
            logger.error(f"加载轻量级模型失败: {e}")
