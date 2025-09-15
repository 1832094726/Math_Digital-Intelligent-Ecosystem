#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点推荐服务
基于知识图谱和用户学习状态提供个性化知识点推荐
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from models.database import get_db_connection

logger = logging.getLogger(__name__)

class KnowledgeRecommendationService:
    """知识点推荐服务类"""
    
    def __init__(self):
        self.logger = logger
        
    def recommend_knowledge_points(self, user_id: int, question_id: Optional[int] = None, 
                                 context: str = "", limit: int = 5) -> Dict[str, Any]:
        """
        推荐知识点
        
        Args:
            user_id: 用户ID
            question_id: 题目ID（可选）
            context: 上下文内容
            limit: 推荐数量限制
            
        Returns:
            推荐结果字典
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 获取推荐知识点
            recommendations = []
            
            if question_id:
                # 基于题目推荐相关知识点
                recommendations = self._recommend_by_question(cursor, question_id, user_id, limit)
            elif context:
                # 基于上下文推荐知识点
                recommendations = self._recommend_by_context(cursor, context, user_id, limit)
            else:
                # 基于用户学习状态推荐
                recommendations = self._recommend_by_user_state(cursor, user_id, limit)
            
            # 记录推荐日志
            self._log_recommendation(cursor, user_id, question_id, context, recommendations)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return {
                'success': True,
                'recommendations': recommendations,
                'total': len(recommendations),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"知识点推荐失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': [],
                'total': 0
            }
    
    def _recommend_by_question(self, cursor, question_id: int, user_id: int, limit: int) -> List[Dict]:
        """基于题目推荐知识点"""
        try:
            # 查询题目相关的知识点
            cursor.execute("""
                SELECT DISTINCT kp.id, kp.name, kp.description, kp.grade_level, 
                       kp.difficulty_level, kp.cognitive_type
                FROM knowledge_points kp
                JOIN question_knowledge_points qkp ON kp.id = qkp.knowledge_point_id
                WHERE qkp.question_id = %s
                ORDER BY qkp.relevance_score DESC
                LIMIT %s
            """, (question_id, limit))
            
            direct_kps = cursor.fetchall()
            
            if not direct_kps:
                # 如果没有直接关联，基于题目内容推荐
                return self._recommend_by_content_analysis(cursor, question_id, user_id, limit)
            
            recommendations = []
            for kp in direct_kps:
                # 获取相关知识点
                related_kps = self._get_related_knowledge_points(cursor, kp['id'], 3)
                
                recommendations.append({
                    'id': kp['id'],
                    'name': kp['name'],
                    'description': kp['description'],
                    'grade_level': kp['grade_level'],
                    'difficulty_level': kp['difficulty_level'],
                    'cognitive_type': kp['cognitive_type'],
                    'relevance_score': 0.9,  # 直接关联的相关度很高
                    'recommendation_reason': '与当前题目直接相关',
                    'related_points': related_kps
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"基于题目推荐失败: {e}")
            return []
    
    def _recommend_by_context(self, cursor, context: str, user_id: int, limit: int) -> List[Dict]:
        """基于上下文推荐知识点"""
        try:
            # 简单的关键词匹配推荐
            context_lower = context.lower()
            recommendations = []
            
            # 定义关键词到知识点的映射
            keyword_mappings = {
                '方程': ['一元一次方程', '代数表达式'],
                '运算': ['基本运算', '分数运算'],
                '几何': ['几何图形'],
                '代数': ['代数表达式', '一元一次方程'],
                '分数': ['分数运算', '基本运算']
            }
            
            matched_kp_names = set()
            for keyword, kp_names in keyword_mappings.items():
                if keyword in context_lower:
                    matched_kp_names.update(kp_names)
            
            if matched_kp_names:
                # 查询匹配的知识点
                placeholders = ','.join(['%s'] * len(matched_kp_names))
                cursor.execute(f"""
                    SELECT id, name, description, grade_level, difficulty_level, cognitive_type
                    FROM knowledge_points
                    WHERE name IN ({placeholders})
                    ORDER BY difficulty_level ASC
                    LIMIT %s
                """, list(matched_kp_names) + [limit])
                
                kps = cursor.fetchall()
                
                for kp in kps:
                    related_kps = self._get_related_knowledge_points(cursor, kp['id'], 2)
                    
                    recommendations.append({
                        'id': kp['id'],
                        'name': kp['name'],
                        'description': kp['description'],
                        'grade_level': kp['grade_level'],
                        'difficulty_level': kp['difficulty_level'],
                        'cognitive_type': kp['cognitive_type'],
                        'relevance_score': 0.8,
                        'recommendation_reason': '与输入内容相关',
                        'related_points': related_kps
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"基于上下文推荐失败: {e}")
            return []
    
    def _recommend_by_user_state(self, cursor, user_id: int, limit: int) -> List[Dict]:
        """基于用户学习状态推荐知识点"""
        try:
            # 获取用户信息
            cursor.execute("SELECT grade FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()
            
            if not user or not user.get('grade'):
                # 如果没有年级信息，推荐基础知识点
                cursor.execute("""
                    SELECT id, name, description, grade_level, difficulty_level, cognitive_type
                    FROM knowledge_points
                    WHERE grade_level = 1
                    ORDER BY difficulty_level ASC
                    LIMIT %s
                """, (limit,))
            else:
                # 基于用户年级推荐适合的知识点
                user_grade = user['grade']
                cursor.execute("""
                    SELECT id, name, description, grade_level, difficulty_level, cognitive_type
                    FROM knowledge_points
                    WHERE grade_level <= %s
                    ORDER BY grade_level DESC, difficulty_level ASC
                    LIMIT %s
                """, (user_grade, limit))
            
            kps = cursor.fetchall()
            recommendations = []
            
            for kp in kps:
                related_kps = self._get_related_knowledge_points(cursor, kp['id'], 2)
                
                recommendations.append({
                    'id': kp['id'],
                    'name': kp['name'],
                    'description': kp['description'],
                    'grade_level': kp['grade_level'],
                    'difficulty_level': kp['difficulty_level'],
                    'cognitive_type': kp['cognitive_type'],
                    'relevance_score': 0.7,
                    'recommendation_reason': '适合您当前的学习水平',
                    'related_points': related_kps
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"基于用户状态推荐失败: {e}")
            return []
    
    def _get_related_knowledge_points(self, cursor, kp_id: int, limit: int = 3) -> List[Dict]:
        """获取相关知识点"""
        try:
            cursor.execute("""
                SELECT kp.id, kp.name, kr.relationship_type, kr.strength
                FROM knowledge_points kp
                JOIN knowledge_relationships kr ON (
                    (kr.source_point_id = %s AND kr.target_point_id = kp.id) OR
                    (kr.target_point_id = %s AND kr.source_point_id = kp.id)
                )
                WHERE kp.id != %s
                ORDER BY kr.strength DESC
                LIMIT %s
            """, (kp_id, kp_id, kp_id, limit))
            
            related = cursor.fetchall()
            return [
                {
                    'id': rel['id'],
                    'name': rel['name'],
                    'relationship_type': rel['relationship_type'],
                    'strength': float(rel['strength'])
                }
                for rel in related
            ]
            
        except Exception as e:
            self.logger.error(f"获取相关知识点失败: {e}")
            return []
    
    def _recommend_by_content_analysis(self, cursor, question_id: int, user_id: int, limit: int) -> List[Dict]:
        """基于题目内容分析推荐知识点"""
        try:
            # 获取题目内容
            cursor.execute("SELECT content FROM questions WHERE id = %s", (question_id,))
            question = cursor.fetchone()
            
            if question:
                # 使用上下文推荐方法
                return self._recommend_by_context(cursor, question['content'], user_id, limit)
            
            return []
            
        except Exception as e:
            self.logger.error(f"基于内容分析推荐失败: {e}")
            return []
    
    def _log_recommendation(self, cursor, user_id: int, question_id: Optional[int], 
                          context: str, recommendations: List[Dict]):
        """记录推荐日志"""
        try:
            # 这里可以记录推荐日志用于后续分析
            pass
        except Exception as e:
            self.logger.error(f"记录推荐日志失败: {e}")

# 全局服务实例
knowledge_recommendation_service = KnowledgeRecommendationService()
