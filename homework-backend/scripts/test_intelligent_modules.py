#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能模块关系说明测试脚本
测试学习路径、符号推荐、自适应路径、掌握度追踪等智能模块的关系说明完成情况
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data_js():
    """加载data.js文件内容"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_js_path = os.path.join(project_root, 'database-visualization', 'data.js')
    
    with open(data_js_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取tables对象
    start_marker = 'tables: {'
    end_marker = '    },\n\n    // 关系定义'
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("无法找到tables定义")
    
    tables_content = content[start_idx + len(start_marker):end_idx] + '    }'
    
    # 简单的JavaScript对象解析（针对我们的特定格式）
    return parse_tables_content(tables_content)

def parse_tables_content(content):
    """解析tables内容"""
    tables = {}
    lines = content.split('\n')
    current_table = None
    current_relationships = []
    in_relationships = False
    brace_count = 0
    
    for line in lines:
        line = line.strip()
        
        # 检测表定义开始
        if line.endswith(': {') and not in_relationships:
            current_table = line[:-3].strip()
            current_relationships = []
            in_relationships = False
            continue
        
        # 检测relationships开始
        if line == 'relationships: [':
            in_relationships = True
            brace_count = 0
            continue
        
        # 在relationships中
        if in_relationships:
            if '{' in line:
                brace_count += line.count('{')
            if '}' in line:
                brace_count -= line.count('}')
            
            # 检测单个关系对象
            if 'type:' in line and 'belongsTo' in line:
                relationship = {'has_design_reason': False}
                current_relationships.append(relationship)
            
            # 检测designReason
            if 'designReason:' in line and current_relationships:
                current_relationships[-1]['has_design_reason'] = True
            
            # relationships结束
            if line == ']' and brace_count == 0:
                in_relationships = False
                if current_table:
                    tables[current_table] = {
                        'relationships': current_relationships
                    }
    
    return tables

def test_intelligent_modules():
    """测试智能模块的关系说明"""
    print("🧠 智能模块关系说明测试")
    print("=" * 50)
    
    # 智能模块表列表
    intelligent_modules = {
        'learning_paths': '个性化学习路径',
        'symbol_recommendations': '符号推荐',
        'problem_recommendations': '题目推荐', 
        'learning_path_recommendations': '学习路径推荐',
        'adaptive_paths': '自适应路径',
        'mastery_tracking': '掌握度跟踪',
        'learning_behaviors': '学习行为',
        'interaction_logs': '交互日志',
        'user_behavior_logs': '用户行为日志',
        'recommendation_results': '推荐结果'
    }
    
    try:
        tables = load_data_js()
        
        total_relationships = 0
        detailed_relationships = 0
        
        print("📊 智能模块关系说明统计:")
        print()
        
        for table_name, display_name in intelligent_modules.items():
            if table_name in tables:
                table_data = tables[table_name]
                relationships = table_data.get('relationships', [])
                
                table_total = len(relationships)
                table_detailed = sum(1 for rel in relationships if rel.get('has_design_reason', False))
                
                total_relationships += table_total
                detailed_relationships += table_detailed
                
                status = "✅" if table_detailed == table_total and table_total > 0 else "⚠️" if table_detailed > 0 else "❌"
                completion_rate = (table_detailed / table_total * 100) if table_total > 0 else 0
                
                print(f"{status} {display_name} ({table_name})")
                print(f"   关系总数: {table_total}")
                print(f"   详细说明: {table_detailed}")
                print(f"   完成度: {completion_rate:.1f}%")
                print()
            else:
                print(f"❌ {display_name} ({table_name}) - 表不存在")
                print()
        
        print("=" * 50)
        print("📈 智能模块总体统计:")
        overall_completion = (detailed_relationships / total_relationships * 100) if total_relationships > 0 else 0
        print(f"总关系数: {total_relationships}")
        print(f"详细说明: {detailed_relationships}")
        print(f"完成度: {overall_completion:.1f}%")
        
        if overall_completion >= 80:
            print("🎉 智能模块关系说明质量优秀！")
        elif overall_completion >= 60:
            print("👍 智能模块关系说明质量良好！")
        elif overall_completion >= 40:
            print("⚠️ 智能模块关系说明需要继续改进")
        else:
            print("❌ 智能模块关系说明亟需完善")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_intelligent_modules()
