#!/usr/bin/env python3
"""
批量更新数据库关系说明
"""

import re
import os

def update_relationships():
    """批量更新关系说明"""
    
    data_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'database-visualization', 'data.js'
    )
    
    print("🔧 开始批量更新关系说明...")
    
    # 需要更新的关系映射
    relationship_updates = {
        # classes表的关系
        "classes.*belongsTo.*schools": {
            "old": "{ type: 'belongsTo', table: 'schools', foreignKey: 'school_id', description: '班级属于学校' }",
            "new": """{ 
                    type: 'belongsTo', 
                    table: 'schools', 
                    foreignKey: 'school_id', 
                    description: '班级属于特定学校',
                    designReason: {
                        fieldPurpose: 'school_id外键标识班级所属的学校',
                        businessLogic: '班级是学校的基本教学单位，必须归属于具体学校',
                        dataIntegrity: '确保班级与学校的正确关联，支持多校区管理',
                        performanceBenefit: '便于按学校查询班级列表，支持学校级别的管理'
                    }
                }"""
        },
        
        # courses表的关系
        "courses.*belongsTo.*subjects": {
            "old": "{ type: 'belongsTo', table: 'subjects', foreignKey: 'subject_id', description: '课程属于学科' }",
            "new": """{ 
                    type: 'belongsTo', 
                    table: 'subjects', 
                    foreignKey: 'subject_id', 
                    description: '课程属于特定学科',
                    designReason: {
                        fieldPurpose: 'subject_id外键标识课程所属的学科领域',
                        businessLogic: '课程必须归属于具体学科，如数学、语文等',
                        dataIntegrity: '确保课程与学科的正确分类，支持学科管理',
                        performanceBenefit: '便于按学科查询课程列表，支持学科级别的统计'
                    }
                }"""
        },
        
        # knowledge_points表的关系
        "knowledge_points.*hasMany.*knowledge_relationships": {
            "old": "{ type: 'hasMany', table: 'knowledge_relationships', localKey: 'id', description: '知识点有多个关系' }",
            "new": """{ 
                    type: 'hasMany', 
                    table: 'knowledge_relationships', 
                    localKey: 'id', 
                    foreignKey: 'source_point_id',
                    description: '知识点作为源点有多个关系',
                    designReason: {
                        fieldPurpose: 'source_point_id外键标识关系的起始知识点',
                        businessLogic: '知识点之间存在前置、包含、相关等多种关系',
                        dataIntegrity: '构建完整的知识图谱，支持学习路径规划',
                        performanceBenefit: '便于查询知识点的关联关系，支持智能推荐'
                    }
                }"""
        }
    }
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计更新前的状态
        before_count = content.count('designReason')
        print(f"📊 更新前详细关系说明数量: {before_count}")
        
        # 执行批量更新
        updated_count = 0
        for pattern, update_info in relationship_updates.items():
            old_pattern = update_info['old']
            new_content = update_info['new']
            
            if old_pattern in content:
                content = content.replace(old_pattern, new_content)
                updated_count += 1
                print(f"  ✅ 更新了 {pattern}")
            else:
                print(f"  ⚠️  未找到 {pattern}")
        
        # 统计更新后的状态
        after_count = content.count('designReason')
        print(f"📊 更新后详细关系说明数量: {after_count}")
        print(f"🎯 本次新增详细说明: {after_count - before_count}")
        
        # 写回文件
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 批量更新完成，共更新 {updated_count} 个关系")
        return True
        
    except Exception as e:
        print(f"❌ 批量更新失败: {e}")
        return False

def main():
    print("🚀 开始批量更新数据库关系说明")
    print("=" * 50)
    
    success = update_relationships()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 批量更新成功完成")
    else:
        print("⚠️  批量更新遇到问题")
    
    print("\n💡 下一步建议:")
    print("  1. 运行测试脚本验证更新效果")
    print("  2. 在浏览器中查看可视化效果")
    print("  3. 继续为更多表添加详细关系说明")

if __name__ == '__main__':
    main()
