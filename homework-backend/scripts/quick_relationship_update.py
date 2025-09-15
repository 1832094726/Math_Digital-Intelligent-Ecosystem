#!/usr/bin/env python3
"""
快速批量更新关系说明
"""

import re
import os

def quick_update_relationships():
    """快速批量更新关系说明"""
    
    data_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'database-visualization', 'data.js'
    )
    
    print("⚡ 开始快速批量更新关系说明...")
    
    # 简单的关系说明模板
    simple_updates = [
        # 学习分析系统
        {
            "pattern": "{ type: 'belongsTo', table: 'users', foreignKey: 'user_id', description: '属于用户' }",
            "replacement": """{ 
                    type: 'belongsTo', 
                    table: 'users', 
                    foreignKey: 'user_id', 
                    description: '记录属于特定用户',
                    designReason: {
                        fieldPurpose: 'user_id外键标识记录所属的用户',
                        businessLogic: '需要将数据与具体用户关联，支持个性化分析',
                        dataIntegrity: '确保数据与用户身份绑定，保护用户隐私',
                        performanceBenefit: '便于按用户查询相关数据，支持个性化服务'
                    }
                }"""
        },
        
        # 作业系统
        {
            "pattern": "{ type: 'belongsTo', table: 'homeworks', foreignKey: 'homework_id', description: '属于作业' }",
            "replacement": """{ 
                    type: 'belongsTo', 
                    table: 'homeworks', 
                    foreignKey: 'homework_id', 
                    description: '记录属于特定作业',
                    designReason: {
                        fieldPurpose: 'homework_id外键标识记录关联的作业',
                        businessLogic: '需要将记录与具体作业关联，支持作业管理',
                        dataIntegrity: '确保记录与作业内容匹配，支持数据一致性',
                        performanceBenefit: '便于按作业查询相关记录，支持作业分析'
                    }
                }"""
        },
        
        # 学科系统
        {
            "pattern": "{ type: 'belongsTo', table: 'subjects', foreignKey: 'subject_id', description: '属于学科' }",
            "replacement": """{ 
                    type: 'belongsTo', 
                    table: 'subjects', 
                    foreignKey: 'subject_id', 
                    description: '记录属于特定学科',
                    designReason: {
                        fieldPurpose: 'subject_id外键标识记录所属的学科领域',
                        businessLogic: '需要按学科分类管理，支持学科级别的操作',
                        dataIntegrity: '确保记录与学科的正确分类，支持学科管理',
                        performanceBenefit: '便于按学科查询记录，支持学科级别的统计'
                    }
                }"""
        },
        
        # 班级系统
        {
            "pattern": "{ type: 'belongsTo', table: 'classes', foreignKey: 'class_id', description: '属于班级' }",
            "replacement": """{ 
                    type: 'belongsTo', 
                    table: 'classes', 
                    foreignKey: 'class_id', 
                    description: '记录属于特定班级',
                    designReason: {
                        fieldPurpose: 'class_id外键标识记录所属的班级',
                        businessLogic: '需要按班级管理，支持班级级别的操作',
                        dataIntegrity: '确保记录与班级的正确关联，支持班级管理',
                        performanceBenefit: '便于按班级查询记录，支持班级统计分析'
                    }
                }"""
        }
    ]
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计更新前的状态
        before_count = content.count('designReason')
        print(f"📊 更新前详细关系说明数量: {before_count}")
        
        # 执行批量更新
        updated_count = 0
        for update in simple_updates:
            pattern = update['pattern']
            replacement = update['replacement']
            
            # 计算匹配次数
            matches = content.count(pattern)
            if matches > 0:
                content = content.replace(pattern, replacement)
                updated_count += matches
                print(f"  ✅ 更新了 {matches} 个匹配项: {pattern[:50]}...")
            else:
                print(f"  ⚠️  未找到匹配项: {pattern[:50]}...")
        
        # 统计更新后的状态
        after_count = content.count('designReason')
        print(f"📊 更新后详细关系说明数量: {after_count}")
        print(f"🎯 本次新增详细说明: {after_count - before_count}")
        
        # 写回文件
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 快速批量更新完成，共更新 {updated_count} 个关系")
        return True
        
    except Exception as e:
        print(f"❌ 快速批量更新失败: {e}")
        return False

def main():
    print("🚀 开始快速批量更新数据库关系说明")
    print("=" * 50)
    
    success = quick_update_relationships()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 快速批量更新成功完成")
    else:
        print("⚠️  快速批量更新遇到问题")

if __name__ == '__main__':
    main()
