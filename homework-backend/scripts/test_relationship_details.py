#!/usr/bin/env python3
"""
测试数据库关系详细说明
"""

import json
import os

def test_relationship_details():
    """测试关系说明的详细程度"""
    
    # 读取数据文件
    data_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'database-visualization', 'data.js'
    )
    
    print("🔍 检查数据库关系说明的详细程度...")
    print(f"📁 数据文件: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计关系说明
        total_relationships = content.count('type:')
        detailed_relationships = content.count('designReason')
        foreign_key_mentions = content.count('foreignKey')
        
        print(f"\n📊 关系统计:")
        print(f"  总关系数量: {total_relationships}")
        print(f"  详细说明数量: {detailed_relationships}")
        print(f"  外键字段说明: {foreign_key_mentions}")
        
        # 检查关键字段
        key_fields = [
            'fieldPurpose',
            'businessLogic', 
            'dataIntegrity',
            'performanceBenefit'
        ]
        
        print(f"\n🔍 详细说明字段检查:")
        for field in key_fields:
            count = content.count(field)
            print(f"  {field}: {count}个")
        
        # 检查具体表的关系说明
        key_tables = [
            'user_sessions',
            'homework_submissions',
            'questions',
            'notifications'
        ]
        
        print(f"\n📋 关键表关系检查:")
        for table in key_tables:
            if f'{table}:' in content and 'designReason' in content:
                print(f"  ✅ {table}: 包含详细关系说明")
            else:
                print(f"  ❌ {table}: 缺少详细关系说明")
        
        # 评估完成度
        completion_rate = (detailed_relationships / max(total_relationships, 1)) * 100
        print(f"\n🎯 关系说明完成度: {completion_rate:.1f}%")
        
        if completion_rate >= 80:
            print("✅ 关系说明详细程度良好")
        elif completion_rate >= 50:
            print("⚠️  关系说明需要进一步完善")
        else:
            print("❌ 关系说明过于简单，需要大幅改进")
            
        # 示例展示
        print(f"\n📝 关系说明示例:")
        if 'user_sessions' in content and 'designReason' in content:
            start = content.find('user_sessions')
            end = content.find('}', start + 500)
            if end > start:
                sample = content[start:end+1]
                if 'designReason' in sample:
                    print("  找到详细的user_sessions关系说明 ✅")
                    # 提取设计原因部分
                    reason_start = sample.find('designReason')
                    if reason_start > 0:
                        reason_part = sample[reason_start:reason_start+200]
                        print(f"  示例内容: {reason_part[:100]}...")
        
        return completion_rate >= 80
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("🚀 开始测试数据库关系详细说明")
    print("=" * 50)
    
    success = test_relationship_details()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试通过：关系说明详细程度符合要求")
    else:
        print("⚠️  测试提醒：建议进一步完善关系说明")
    
    print("\n💡 改进建议:")
    print("  1. 每个外键字段都应该有具体的设计原因")
    print("  2. 说明应该包含业务逻辑、数据完整性、性能考虑")
    print("  3. 避免使用泛化的描述，提供具体的使用场景")

if __name__ == '__main__':
    main()
