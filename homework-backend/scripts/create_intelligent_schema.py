#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能化系统数据库表创建脚本
创建知识图谱、AI推荐、学习分析、错误分析、自适应学习等智能化功能的数据表
"""

import pymysql
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

def create_intelligent_tables():
    """创建智能化系统的数据表"""
    
    # 数据库连接配置
    db_config = config['development'].DATABASE_CONFIG
    
    try:
        # 连接数据库
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()
        
        print("🚀 开始创建智能化系统数据表...")
        print(f"📅 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 1. 创建知识图谱系统表
        print("\n📚 创建知识图谱系统表...")
        
        # 知识点表
        print("  - 创建 knowledge_points 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `knowledge_points` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '知识点ID',
          `name` varchar(200) NOT NULL COMMENT '知识点名称',
          `description` text DEFAULT NULL COMMENT '详细描述',
          `subject_id` bigint(20) DEFAULT NULL COMMENT '学科ID',
          `grade_level` int(11) NOT NULL COMMENT '年级层次',
          `difficulty_level` int(11) NOT NULL DEFAULT '3' COMMENT '难度等级(1-5)',
          `cognitive_type` enum('conceptual','procedural','metacognitive') DEFAULT 'conceptual' COMMENT '认知类型',
          `bloom_level` int(11) DEFAULT '1' COMMENT '布鲁姆分类等级(1-6)',
          `prerequisites` json DEFAULT NULL COMMENT '前置知识点',
          `learning_objectives` json DEFAULT NULL COMMENT '学习目标',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          KEY `idx_subject_id` (`subject_id`),
          KEY `idx_grade_level` (`grade_level`),
          KEY `idx_difficulty_level` (`difficulty_level`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点定义表'
        """)
        
        # 知识点关系表
        print("  - 创建 knowledge_relationships 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `knowledge_relationships` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '关系ID',
          `source_point_id` bigint(20) NOT NULL COMMENT '源知识点ID',
          `target_point_id` bigint(20) NOT NULL COMMENT '目标知识点ID',
          `relationship_type` enum('prerequisite','related','extends','applies_to','contradicts') NOT NULL COMMENT '关系类型',
          `strength` decimal(3,2) DEFAULT '0.50' COMMENT '关系强度(0-1)',
          `confidence` decimal(3,2) DEFAULT '0.50' COMMENT '置信度(0-1)',
          `evidence_count` int(11) DEFAULT '0' COMMENT '证据数量',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_source_point` (`source_point_id`),
          KEY `idx_target_point` (`target_point_id`),
          KEY `idx_relationship_type` (`relationship_type`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识点关系表'
        """)
        
        # 概念图表
        print("  - 创建 concept_maps 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `concept_maps` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '概念图ID',
          `title` varchar(200) NOT NULL COMMENT '概念图标题',
          `description` text DEFAULT NULL COMMENT '描述',
          `subject_id` bigint(20) DEFAULT NULL COMMENT '学科ID',
          `grade_level` int(11) NOT NULL COMMENT '年级层次',
          `map_data` json DEFAULT NULL COMMENT '图结构数据(节点和边)',
          `layout_config` json DEFAULT NULL COMMENT '布局配置',
          `created_by` bigint(20) NOT NULL COMMENT '创建者ID',
          `is_public` tinyint(1) DEFAULT '0' COMMENT '是否公开',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          KEY `idx_subject_id` (`subject_id`),
          KEY `idx_created_by` (`created_by`),
          KEY `idx_grade_level` (`grade_level`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='概念图表'
        """)
        
        # 2. 创建智能推荐系统表
        print("\n🤖 创建智能推荐系统表...")
        
        # 符号推荐表
        print("  - 创建 symbol_recommendations 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `symbol_recommendations` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '推荐ID',
          `user_id` bigint(20) NOT NULL COMMENT '用户ID',
          `context` text DEFAULT NULL COMMENT '输入上下文',
          `recommended_symbols` json DEFAULT NULL COMMENT '推荐的符号列表',
          `selected_symbol` varchar(50) DEFAULT NULL COMMENT '用户选择的符号',
          `usage_frequency` int(11) DEFAULT '0' COMMENT '使用频率',
          `success_rate` decimal(5,2) DEFAULT '0.00' COMMENT '推荐成功率',
          `response_time` int(11) DEFAULT '0' COMMENT '响应时间(毫秒)',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_user_id` (`user_id`),
          KEY `idx_created_at` (`created_at`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='符号推荐表'
        """)
        
        # 题目推荐表
        print("  - 创建 problem_recommendations 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `problem_recommendations` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '推荐ID',
          `user_id` bigint(20) NOT NULL COMMENT '用户ID',
          `question_id` bigint(20) NOT NULL COMMENT '题目ID',
          `recommendation_reason` text DEFAULT NULL COMMENT '推荐理由',
          `difficulty_match` decimal(3,2) DEFAULT '0.50' COMMENT '难度匹配度(0-1)',
          `knowledge_gap_target` json DEFAULT NULL COMMENT '目标知识缺口',
          `predicted_success_rate` decimal(3,2) DEFAULT '0.50' COMMENT '预测成功率',
          `actual_result` enum('correct','incorrect','skipped','timeout') DEFAULT NULL COMMENT '实际结果',
          `user_feedback` int(11) DEFAULT NULL COMMENT '用户反馈评分(1-5)',
          `recommended_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '推荐时间',
          `completed_at` datetime DEFAULT NULL COMMENT '完成时间',
          PRIMARY KEY (`id`),
          KEY `idx_user_id` (`user_id`),
          KEY `idx_question_id` (`question_id`),
          KEY `idx_recommended_at` (`recommended_at`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='题目推荐表'
        """)
        
        # 学习路径推荐表
        print("  - 创建 learning_path_recommendations 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `learning_path_recommendations` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '推荐ID',
          `user_id` bigint(20) NOT NULL COMMENT '用户ID',
          `current_knowledge_state` json DEFAULT NULL COMMENT '当前知识状态',
          `recommended_path` json DEFAULT NULL COMMENT '推荐路径',
          `path_type` enum('remedial','advancement','review','exploration') DEFAULT 'advancement' COMMENT '路径类型',
          `estimated_duration` int(11) DEFAULT '0' COMMENT '预计时长(分钟)',
          `success_prediction` decimal(3,2) DEFAULT '0.50' COMMENT '成功预测概率',
          `adaptation_triggers` json DEFAULT NULL COMMENT '适应触发条件',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
          PRIMARY KEY (`id`),
          KEY `idx_user_id` (`user_id`),
          KEY `idx_path_type` (`path_type`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学习路径推荐表'
        """)
        
        # 3. 创建学习分析系统表
        print("\n📊 创建学习分析系统表...")
        
        # 学习行为表
        print("  - 创建 learning_behaviors 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `learning_behaviors` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '行为ID',
          `user_id` bigint(20) NOT NULL COMMENT '用户ID',
          `session_id` varchar(100) DEFAULT NULL COMMENT '会话ID',
          `behavior_type` enum('click','hover','input','submit','pause','review','help_seek') NOT NULL COMMENT '行为类型',
          `behavior_data` json DEFAULT NULL COMMENT '行为详细数据',
          `context_info` json DEFAULT NULL COMMENT '上下文信息',
          `timestamp` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '时间戳',
          `duration` int(11) DEFAULT '0' COMMENT '持续时间(毫秒)',
          `device_info` json DEFAULT NULL COMMENT '设备信息',
          PRIMARY KEY (`id`),
          KEY `idx_user_id` (`user_id`),
          KEY `idx_session_id` (`session_id`),
          KEY `idx_behavior_type` (`behavior_type`),
          KEY `idx_timestamp` (`timestamp`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学习行为表'
        """)
        
        # 交互日志表
        print("  - 创建 interaction_logs 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `interaction_logs` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '日志ID',
          `user_id` bigint(20) NOT NULL COMMENT '用户ID',
          `homework_id` bigint(20) DEFAULT NULL COMMENT '作业ID',
          `question_id` bigint(20) DEFAULT NULL COMMENT '题目ID',
          `interaction_type` enum('view','attempt','submit','hint','skip','review') NOT NULL COMMENT '交互类型',
          `interaction_data` json DEFAULT NULL COMMENT '交互数据',
          `response_time` int(11) DEFAULT '0' COMMENT '响应时间(毫秒)',
          `accuracy` decimal(3,2) DEFAULT NULL COMMENT '准确率',
          `hint_used` tinyint(1) DEFAULT '0' COMMENT '是否使用提示',
          `attempts_count` int(11) DEFAULT '1' COMMENT '尝试次数',
          `final_answer` text DEFAULT NULL COMMENT '最终答案',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          KEY `idx_user_id` (`user_id`),
          KEY `idx_homework_id` (`homework_id`),
          KEY `idx_question_id` (`question_id`),
          KEY `idx_interaction_type` (`interaction_type`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='交互日志表'
        """)
        
        # 参与度指标表
        print("  - 创建 engagement_metrics 表...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `engagement_metrics` (
          `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '指标ID',
          `user_id` bigint(20) NOT NULL COMMENT '用户ID',
          `date` date NOT NULL COMMENT '日期',
          `session_duration` int(11) DEFAULT '0' COMMENT '会话时长(分钟)',
          `questions_attempted` int(11) DEFAULT '0' COMMENT '尝试题目数',
          `completion_rate` decimal(5,2) DEFAULT '0.00' COMMENT '完成率',
          `focus_score` decimal(3,2) DEFAULT '0.00' COMMENT '专注度评分(0-1)',
          `persistence_score` decimal(3,2) DEFAULT '0.00' COMMENT '坚持度评分(0-1)',
          `help_seeking_frequency` decimal(5,2) DEFAULT '0.00' COMMENT '求助频率',
          `self_regulation_score` decimal(3,2) DEFAULT '0.00' COMMENT '自我调节评分(0-1)',
          `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
          PRIMARY KEY (`id`),
          UNIQUE KEY `uk_user_date` (`user_id`,`date`),
          KEY `idx_user_id` (`user_id`),
          KEY `idx_date` (`date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='参与度指标表'
        """)
        
        # 提交更改
        connection.commit()
        
        print("\n✅ 智能化系统数据表创建完成!")
        print("=" * 60)
        print("📋 创建的表:")
        print("  📚 知识图谱系统:")
        print("    - knowledge_points (知识点定义)")
        print("    - knowledge_relationships (知识点关系)")
        print("    - concept_maps (概念图)")
        print("  🤖 智能推荐系统:")
        print("    - symbol_recommendations (符号推荐)")
        print("    - problem_recommendations (题目推荐)")
        print("    - learning_path_recommendations (学习路径推荐)")
        print("  📊 学习分析系统:")
        print("    - learning_behaviors (学习行为)")
        print("    - interaction_logs (交互日志)")
        print("    - engagement_metrics (参与度指标)")
        
        # 插入一些示例数据
        print("\n🌱 插入示例数据...")
        insert_sample_data(cursor, connection)
        
        print("\n🎉 智能化系统初始化完成!")
        
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        if connection:
            connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    
    return True

def insert_sample_data(cursor, connection):
    """插入示例数据"""
    try:
        # 插入示例知识点
        print("  - 插入示例知识点...")
        sample_knowledge_points = [
            ('基本运算', '加减乘除四则运算', 1, 1, 'procedural', 1, '[]', '["掌握四则运算法则", "能够进行基本计算"]'),
            ('代数表达式', '用字母和数字表示的数学表达式', 2, 2, 'conceptual', 2, '["基本运算"]', '["理解代数表达式的概念", "能够化简代数表达式"]'),
            ('一元一次方程', '含有一个未知数的一次方程', 2, 3, 'procedural', 3, '["代数表达式"]', '["掌握一元一次方程的解法", "能够验证方程的解"]'),
            ('几何图形', '平面几何基本图形', 1, 2, 'conceptual', 2, '[]', '["认识基本几何图形", "理解图形的性质"]'),
            ('分数运算', '分数的加减乘除运算', 1, 3, 'procedural', 2, '["基本运算"]', '["掌握分数运算法则", "能够进行分数计算"]')
        ]
        
        for kp in sample_knowledge_points:
            cursor.execute("""
                INSERT IGNORE INTO knowledge_points 
                (name, description, grade_level, difficulty_level, cognitive_type, bloom_level, prerequisites, learning_objectives)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, kp)
        
        # 插入知识点关系
        print("  - 插入知识点关系...")
        cursor.execute("""
            INSERT IGNORE INTO knowledge_relationships 
            (source_point_id, target_point_id, relationship_type, strength, confidence)
            SELECT 
                kp1.id, kp2.id, 'prerequisite', 0.9, 0.95
            FROM knowledge_points kp1, knowledge_points kp2
            WHERE kp1.name = '基本运算' AND kp2.name = '代数表达式'
        """)
        
        cursor.execute("""
            INSERT IGNORE INTO knowledge_relationships 
            (source_point_id, target_point_id, relationship_type, strength, confidence)
            SELECT 
                kp1.id, kp2.id, 'prerequisite', 0.85, 0.9
            FROM knowledge_points kp1, knowledge_points kp2
            WHERE kp1.name = '代数表达式' AND kp2.name = '一元一次方程'
        """)
        
        connection.commit()
        print("  ✅ 示例数据插入完成")
        
    except Exception as e:
        print(f"  ❌ 插入示例数据失败: {e}")
        connection.rollback()

if __name__ == "__main__":
    success = create_intelligent_tables()
    if success:
        print("\n🚀 智能化系统数据库初始化成功!")
        print("💡 现在可以使用符号推荐、知识图谱等智能化功能了")
    else:
        print("\n❌ 智能化系统数据库初始化失败!")
        sys.exit(1)
