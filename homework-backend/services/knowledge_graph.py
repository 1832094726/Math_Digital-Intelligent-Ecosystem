# -*- coding: utf-8 -*-
"""
知识图谱推理系统
用于基于知识关系的符号推荐
"""

import json
import os
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, deque
import networkx as nx
import numpy as np


class KnowledgeGraph:
    """知识图谱类"""
    
    def __init__(self, data_path: str = 'data'):
        self.data_path = data_path
        self.graph = nx.DiGraph()  # 有向图
        self.concept_symbols = defaultdict(set)  # 概念到符号的映射
        self.symbol_concepts = defaultdict(set)  # 符号到概念的映射
        self.concept_hierarchy = {}  # 概念层次结构
        
        # 加载知识图谱数据
        self._load_knowledge_graph()
        self._build_concept_symbol_mapping()
        self._calculate_concept_importance()
    
    def _load_knowledge_graph(self):
        """加载知识图谱数据"""
        kg_file = os.path.join(self.data_path, 'knowledge_graph.json')
        
        if os.path.exists(kg_file):
            try:
                with open(kg_file, 'r', encoding='utf-8') as f:
                    kg_data = json.load(f)
                    self._build_graph_from_data(kg_data)
            except Exception as e:
                print(f"加载知识图谱失败: {e}")
                self._create_default_knowledge_graph()
        else:
            self._create_default_knowledge_graph()
    
    def _build_graph_from_data(self, kg_data: Dict):
        """从数据构建知识图谱"""
        # 添加节点
        for concept_data in kg_data.get('concepts', []):
            concept_id = concept_data['id']
            self.graph.add_node(concept_id, **concept_data)
        
        # 添加边（关系）
        for relation in kg_data.get('relations', []):
            source = relation['source']
            target = relation['target']
            relation_type = relation.get('type', 'related_to')
            weight = relation.get('weight', 1.0)
            
            self.graph.add_edge(source, target, 
                              relation_type=relation_type, 
                              weight=weight)
    
    def _create_default_knowledge_graph(self):
        """创建默认知识图谱"""
        # 数学概念节点
        concepts = [
            {'id': 'arithmetic', 'name': '算术', 'level': 1, 'difficulty': 0.2},
            {'id': 'algebra', 'name': '代数', 'level': 2, 'difficulty': 0.5},
            {'id': 'geometry', 'name': '几何', 'level': 2, 'difficulty': 0.4},
            {'id': 'calculus', 'name': '微积分', 'level': 3, 'difficulty': 0.8},
            {'id': 'statistics', 'name': '统计学', 'level': 2, 'difficulty': 0.6},
            {'id': 'trigonometry', 'name': '三角学', 'level': 2, 'difficulty': 0.6},
            {'id': 'linear_algebra', 'name': '线性代数', 'level': 3, 'difficulty': 0.7},
            {'id': 'number_theory', 'name': '数论', 'level': 3, 'difficulty': 0.9},
            
            # 具体概念
            {'id': 'addition', 'name': '加法', 'level': 1, 'difficulty': 0.1, 'parent': 'arithmetic'},
            {'id': 'subtraction', 'name': '减法', 'level': 1, 'difficulty': 0.1, 'parent': 'arithmetic'},
            {'id': 'multiplication', 'name': '乘法', 'level': 1, 'difficulty': 0.2, 'parent': 'arithmetic'},
            {'id': 'division', 'name': '除法', 'level': 1, 'difficulty': 0.2, 'parent': 'arithmetic'},
            
            {'id': 'equations', 'name': '方程', 'level': 2, 'difficulty': 0.4, 'parent': 'algebra'},
            {'id': 'inequalities', 'name': '不等式', 'level': 2, 'difficulty': 0.5, 'parent': 'algebra'},
            {'id': 'functions', 'name': '函数', 'level': 2, 'difficulty': 0.6, 'parent': 'algebra'},
            
            {'id': 'triangles', 'name': '三角形', 'level': 2, 'difficulty': 0.3, 'parent': 'geometry'},
            {'id': 'circles', 'name': '圆', 'level': 2, 'difficulty': 0.4, 'parent': 'geometry'},
            {'id': 'angles', 'name': '角度', 'level': 2, 'difficulty': 0.3, 'parent': 'geometry'},
            
            {'id': 'derivatives', 'name': '导数', 'level': 3, 'difficulty': 0.8, 'parent': 'calculus'},
            {'id': 'integrals', 'name': '积分', 'level': 3, 'difficulty': 0.9, 'parent': 'calculus'},
            {'id': 'limits', 'name': '极限', 'level': 3, 'difficulty': 0.7, 'parent': 'calculus'},
        ]
        
        # 添加节点
        for concept in concepts:
            self.graph.add_node(concept['id'], **concept)
        
        # 添加层次关系
        relations = [
            # 父子关系
            ('arithmetic', 'addition', 'contains', 1.0),
            ('arithmetic', 'subtraction', 'contains', 1.0),
            ('arithmetic', 'multiplication', 'contains', 1.0),
            ('arithmetic', 'division', 'contains', 1.0),
            
            ('algebra', 'equations', 'contains', 1.0),
            ('algebra', 'inequalities', 'contains', 1.0),
            ('algebra', 'functions', 'contains', 1.0),
            
            ('geometry', 'triangles', 'contains', 1.0),
            ('geometry', 'circles', 'contains', 1.0),
            ('geometry', 'angles', 'contains', 1.0),
            
            ('calculus', 'derivatives', 'contains', 1.0),
            ('calculus', 'integrals', 'contains', 1.0),
            ('calculus', 'limits', 'contains', 1.0),
            
            # 前置关系
            ('arithmetic', 'algebra', 'prerequisite', 0.8),
            ('algebra', 'calculus', 'prerequisite', 0.9),
            ('geometry', 'trigonometry', 'prerequisite', 0.7),
            ('trigonometry', 'calculus', 'prerequisite', 0.6),
            
            # 相关关系
            ('geometry', 'trigonometry', 'related_to', 0.8),
            ('algebra', 'linear_algebra', 'related_to', 0.7),
            ('calculus', 'statistics', 'related_to', 0.5),
            ('functions', 'calculus', 'related_to', 0.8),
            ('equations', 'functions', 'related_to', 0.6),
        ]
        
        # 添加边
        for source, target, relation_type, weight in relations:
            self.graph.add_edge(source, target, 
                              relation_type=relation_type, 
                              weight=weight)
    
    def _build_concept_symbol_mapping(self):
        """构建概念与符号的映射关系"""
        # 加载符号数据
        symbols_file = os.path.join(self.data_path, 'symbols.json')
        
        if os.path.exists(symbols_file):
            try:
                with open(symbols_file, 'r', encoding='utf-8') as f:
                    symbols_data = json.load(f)
                    
                for symbol in symbols_data:
                    symbol_id = str(symbol.get('id'))
                    related_knowledge = symbol.get('related_knowledge', [])
                    category = symbol.get('category', '')
                    
                    # 基于相关知识点建立映射
                    for knowledge in related_knowledge:
                        concept_id = self._find_concept_by_name(knowledge)
                        if concept_id:
                            self.concept_symbols[concept_id].add(symbol_id)
                            self.symbol_concepts[symbol_id].add(concept_id)
                    
                    # 基于类别建立映射
                    concept_id = self._map_category_to_concept(category)
                    if concept_id:
                        self.concept_symbols[concept_id].add(symbol_id)
                        self.symbol_concepts[symbol_id].add(concept_id)
                        
            except Exception as e:
                print(f"构建概念符号映射失败: {e}")
        
        # 如果没有映射数据，创建默认映射
        if not self.concept_symbols:
            self._create_default_concept_symbol_mapping()
    
    def _find_concept_by_name(self, name: str) -> Optional[str]:
        """根据名称查找概念ID"""
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('name', '') == name:
                return node_id
        return None
    
    def _map_category_to_concept(self, category: str) -> Optional[str]:
        """将符号类别映射到概念"""
        category_mapping = {
            '基本运算': 'arithmetic',
            '关系符号': 'algebra',
            '希腊字母': 'algebra',
            '几何符号': 'geometry',
            '微积分': 'calculus',
            '统计符号': 'statistics',
            '三角函数': 'trigonometry'
        }
        return category_mapping.get(category)
    
    def _create_default_concept_symbol_mapping(self):
        """创建默认的概念符号映射"""
        # 基于符号ID的简单映射
        mappings = {
            'arithmetic': ['1', '2', '3', '4'],  # +, -, ×, ÷
            'algebra': ['5', '6', '7', '8', '9', '10'],  # =, ≠, ≤, ≥, α, β
            'geometry': ['14'],  # 几何相关符号
            'calculus': ['12', '13'],  # ∫, ∑
            'statistics': ['15'],  # 统计相关符号
        }
        
        for concept_id, symbol_ids in mappings.items():
            for symbol_id in symbol_ids:
                self.concept_symbols[concept_id].add(symbol_id)
                self.symbol_concepts[symbol_id].add(concept_id)
    
    def _calculate_concept_importance(self):
        """计算概念重要性"""
        # 使用PageRank算法计算概念重要性
        try:
            pagerank_scores = nx.pagerank(self.graph, weight='weight')
            
            # 将重要性分数添加到节点属性中
            for node_id, score in pagerank_scores.items():
                self.graph.nodes[node_id]['importance'] = score
                
        except Exception as e:
            print(f"计算概念重要性失败: {e}")
            # 使用默认重要性分数
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]['importance'] = 0.1
    
    def get_related_concepts(self, concept_id: str, max_depth: int = 2, 
                           relation_types: List[str] = None) -> List[Tuple[str, float]]:
        """获取相关概念"""
        if concept_id not in self.graph:
            return []
        
        if relation_types is None:
            relation_types = ['contains', 'prerequisite', 'related_to']
        
        related_concepts = []
        visited = set()
        queue = deque([(concept_id, 0, 1.0)])  # (concept_id, depth, weight)
        
        while queue:
            current_concept, depth, current_weight = queue.popleft()
            
            if current_concept in visited or depth > max_depth:
                continue
            
            visited.add(current_concept)
            
            if current_concept != concept_id:
                related_concepts.append((current_concept, current_weight))
            
            # 探索邻居节点
            for neighbor in self.graph.neighbors(current_concept):
                if neighbor not in visited:
                    edge_data = self.graph[current_concept][neighbor]
                    relation_type = edge_data.get('relation_type', '')
                    edge_weight = edge_data.get('weight', 1.0)
                    
                    if relation_type in relation_types:
                        new_weight = current_weight * edge_weight * 0.8  # 衰减因子
                        queue.append((neighbor, depth + 1, new_weight))
        
        # 按权重排序
        related_concepts.sort(key=lambda x: x[1], reverse=True)
        return related_concepts
    
    def get_concept_symbols(self, concept_id: str, include_related: bool = True) -> List[Tuple[str, float]]:
        """获取概念相关的符号"""
        symbols_with_scores = []
        
        # 直接相关的符号
        direct_symbols = self.concept_symbols.get(concept_id, set())
        for symbol_id in direct_symbols:
            symbols_with_scores.append((symbol_id, 1.0))
        
        if include_related:
            # 相关概念的符号
            related_concepts = self.get_related_concepts(concept_id, max_depth=2)
            
            for related_concept, concept_weight in related_concepts:
                related_symbols = self.concept_symbols.get(related_concept, set())
                for symbol_id in related_symbols:
                    if symbol_id not in direct_symbols:  # 避免重复
                        symbols_with_scores.append((symbol_id, concept_weight * 0.7))
        
        # 按分数排序并去重
        unique_symbols = {}
        for symbol_id, score in symbols_with_scores:
            if symbol_id not in unique_symbols or unique_symbols[symbol_id] < score:
                unique_symbols[symbol_id] = score
        
        result = [(symbol_id, score) for symbol_id, score in unique_symbols.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    def infer_concepts_from_text(self, text: str) -> List[Tuple[str, float]]:
        """从文本推断相关概念"""
        concepts_scores = defaultdict(float)
        
        # 关键词匹配
        text_lower = text.lower()
        
        for node_id, node_data in self.graph.nodes(data=True):
            concept_name = node_data.get('name', '').lower()
            
            # 直接匹配概念名称
            if concept_name in text_lower:
                importance = node_data.get('importance', 0.1)
                concepts_scores[node_id] += importance * 2.0
            
            # 模糊匹配（包含关系）
            elif any(word in text_lower for word in concept_name.split()):
                importance = node_data.get('importance', 0.1)
                concepts_scores[node_id] += importance * 1.0
        
        # 基于关键词的推断
        keyword_concept_mapping = {
            '加': 'addition', '减': 'subtraction', '乘': 'multiplication', '除': 'division',
            '方程': 'equations', '不等式': 'inequalities', '函数': 'functions',
            '三角形': 'triangles', '圆': 'circles', '角': 'angles',
            '导数': 'derivatives', '积分': 'integrals', '极限': 'limits',
            '求和': 'integrals', '微分': 'derivatives'
        }
        
        for keyword, concept_id in keyword_concept_mapping.items():
            if keyword in text:
                if concept_id in self.graph:
                    importance = self.graph.nodes[concept_id].get('importance', 0.1)
                    concepts_scores[concept_id] += importance * 1.5
        
        # 转换为列表并排序
        result = [(concept_id, score) for concept_id, score in concepts_scores.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    def get_symbol_recommendations_by_knowledge(self, question_text: str, 
                                              current_input: str = '', 
                                              max_recommendations: int = 10) -> List[Tuple[str, float, str]]:
        """基于知识图谱推理获取符号推荐"""
        # 从问题文本推断概念
        inferred_concepts = self.infer_concepts_from_text(question_text + ' ' + current_input)
        
        if not inferred_concepts:
            return []
        
        # 获取推荐符号
        symbol_recommendations = []
        
        for concept_id, concept_score in inferred_concepts[:5]:  # 取前5个最相关的概念
            concept_symbols = self.get_concept_symbols(concept_id, include_related=True)
            
            for symbol_id, symbol_score in concept_symbols:
                final_score = concept_score * symbol_score
                concept_name = self.graph.nodes[concept_id].get('name', concept_id)
                reason = f"与{concept_name}相关"
                
                symbol_recommendations.append((symbol_id, final_score, reason))
        
        # 去重并排序
        unique_recommendations = {}
        for symbol_id, score, reason in symbol_recommendations:
            if symbol_id not in unique_recommendations or unique_recommendations[symbol_id][0] < score:
                unique_recommendations[symbol_id] = (score, reason)
        
        final_recommendations = [
            (symbol_id, score, reason) 
            for symbol_id, (score, reason) in unique_recommendations.items()
        ]
        
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return final_recommendations[:max_recommendations]
    
    def get_learning_path(self, start_concept: str, target_concept: str) -> List[str]:
        """获取学习路径"""
        try:
            # 使用最短路径算法
            path = nx.shortest_path(self.graph, start_concept, target_concept, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_concept_difficulty(self, concept_id: str) -> float:
        """获取概念难度"""
        if concept_id in self.graph:
            return self.graph.nodes[concept_id].get('difficulty', 0.5)
        return 0.5
    
    def save_knowledge_graph(self):
        """保存知识图谱到文件"""
        kg_file = os.path.join(self.data_path, 'knowledge_graph.json')
        
        # 准备数据
        concepts = []
        relations = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            concept_data = {'id': node_id}
            concept_data.update(node_data)
            concepts.append(concept_data)
        
        for source, target, edge_data in self.graph.edges(data=True):
            relation_data = {
                'source': source,
                'target': target
            }
            relation_data.update(edge_data)
            relations.append(relation_data)
        
        kg_data = {
            'concepts': concepts,
            'relations': relations,
            'concept_symbols': {k: list(v) for k, v in self.concept_symbols.items()},
            'symbol_concepts': {k: list(v) for k, v in self.symbol_concepts.items()}
        }
        
        try:
            os.makedirs(self.data_path, exist_ok=True)
            with open(kg_file, 'w', encoding='utf-8') as f:
                json.dump(kg_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存知识图谱失败: {e}")


class KnowledgeGraphRecommender:
    """基于知识图谱的推荐器"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
    
    def recommend_symbols(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于知识图谱推荐符号"""
        question_text = context.get('question_text', '')
        current_input = context.get('current_input', '')
        difficulty_level = context.get('difficulty_level', 'medium')
        
        # 获取知识图谱推荐
        kg_recommendations = self.kg.get_symbol_recommendations_by_knowledge(
            question_text, current_input, max_recommendations=15
        )
        
        recommendations = []
        for symbol_id, score, reason in kg_recommendations:
            # 根据难度级别调整分数
            adjusted_score = self._adjust_score_by_difficulty(score, symbol_id, difficulty_level)
            
            recommendations.append({
                'id': symbol_id,
                'score': adjusted_score,
                'source': 'knowledge_graph',
                'kg_reason': reason,
                'original_score': score
            })
        
        return recommendations
    
    def _adjust_score_by_difficulty(self, score: float, symbol_id: str, difficulty_level: str) -> float:
        """根据难度级别调整分数"""
        # 获取符号相关概念的平均难度
        symbol_concepts = self.kg.symbol_concepts.get(symbol_id, set())
        
        if not symbol_concepts:
            return score
        
        avg_difficulty = np.mean([
            self.kg.get_concept_difficulty(concept_id) 
            for concept_id in symbol_concepts
        ])
        
        # 根据用户难度级别调整
        difficulty_multipliers = {
            'beginner': 0.3,
            'easy': 0.5,
            'medium': 0.7,
            'hard': 0.9,
            'expert': 1.0
        }
        
        target_difficulty = difficulty_multipliers.get(difficulty_level, 0.7)
        
        # 如果符号难度与用户水平匹配，给予加成
        difficulty_diff = abs(avg_difficulty - target_difficulty)
        
        if difficulty_diff < 0.2:
            adjustment = 1.2  # 20%加成
        elif difficulty_diff < 0.4:
            adjustment = 1.0  # 无调整
        else:
            adjustment = 0.8  # 20%惩罚
        
        return score * adjustment
