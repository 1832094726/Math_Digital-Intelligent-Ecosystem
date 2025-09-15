#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集质量验证工具
验证JSONL格式数据集是否符合硅基流动微调要求
"""

import json
import os
from typing import List, Dict, Any, Tuple

class DatasetValidator:
    def __init__(self):
        self.required_roles = ['system', 'user', 'assistant']
        self.valid_roles = ['system', 'user', 'assistant']
        self.max_lines = 5000
        
    def validate_json_format(self, line: str, line_num: int) -> Tuple[bool, str, Dict]:
        """验证JSON格式"""
        try:
            data = json.loads(line.strip())
            return True, "", data
        except json.JSONDecodeError as e:
            return False, f"第{line_num}行JSON格式错误: {e}", {}
    
    def validate_messages_structure(self, data: Dict, line_num: int) -> Tuple[bool, List[str]]:
        """验证messages结构"""
        errors = []
        
        # 检查是否包含messages字段
        if 'messages' not in data:
            errors.append(f"第{line_num}行缺少'messages'字段")
            return False, errors
        
        messages = data['messages']
        
        # 检查messages是否为数组且不为空
        if not isinstance(messages, list) or len(messages) == 0:
            errors.append(f"第{line_num}行'messages'必须是非空数组")
            return False, errors
        
        # 检查每个消息的结构
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"第{line_num}行messages[{i}]必须是对象")
                continue
            
            # 检查必需字段
            if 'role' not in msg:
                errors.append(f"第{line_num}行messages[{i}]缺少'role'字段")
            if 'content' not in msg:
                errors.append(f"第{line_num}行messages[{i}]缺少'content'字段")
            
            # 检查role值
            if 'role' in msg and msg['role'] not in self.valid_roles:
                errors.append(f"第{line_num}行messages[{i}]的role值'{msg['role']}'无效")
        
        return len(errors) == 0, errors
    
    def validate_role_sequence(self, messages: List[Dict], line_num: int) -> Tuple[bool, List[str]]:
        """验证角色序列"""
        errors = []
        roles = [msg.get('role') for msg in messages]
        
        # 检查system角色位置
        system_indices = [i for i, role in enumerate(roles) if role == 'system']
        if system_indices and system_indices[0] != 0:
            errors.append(f"第{line_num}行system角色消息必须在数组首位")
        
        # 检查第一条非system消息
        non_system_roles = [role for role in roles if role != 'system']
        if non_system_roles and non_system_roles[0] != 'user':
            errors.append(f"第{line_num}行第一条非system消息必须是user角色")
        
        # 检查user和assistant交替出现
        user_assistant_sequence = [role for role in roles if role in ['user', 'assistant']]
        if len(user_assistant_sequence) < 2:
            errors.append(f"第{line_num}行user和assistant消息不少于1对")
        else:
            for i in range(len(user_assistant_sequence) - 1):
                current_role = user_assistant_sequence[i]
                next_role = user_assistant_sequence[i + 1]
                
                if current_role == next_role:
                    errors.append(f"第{line_num}行user和assistant角色必须交替出现")
                    break
        
        return len(errors) == 0, errors
    
    def validate_content_quality(self, messages: List[Dict], line_num: int) -> Tuple[bool, List[str]]:
        """验证内容质量"""
        errors = []
        warnings = []
        
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            role = msg.get('role', '')
            
            # 检查内容长度
            if len(content.strip()) == 0:
                errors.append(f"第{line_num}行messages[{i}]内容为空")
            elif len(content) < 10:
                warnings.append(f"第{line_num}行messages[{i}]内容过短")
            elif len(content) > 2000:
                warnings.append(f"第{line_num}行messages[{i}]内容过长")
            
            # 检查数学相关内容
            if role == 'user':
                if not any(keyword in content for keyword in ['方程', '计算', '求', '多少', '面积', '周长', '体积', '解', '=', '+', '-', '×', '÷']):
                    warnings.append(f"第{line_num}行user消息可能不是数学题目")
            
            elif role == 'assistant':
                if not any(keyword in content for keyword in ['方程', '公式', '=', '设', '根据', '因此']):
                    warnings.append(f"第{line_num}行assistant消息可能不包含方程解析")
        
        return len(errors) == 0, errors + warnings
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """验证整个文件"""
        if not os.path.exists(file_path):
            return {
                'valid': False,
                'errors': [f"文件不存在: {file_path}"],
                'warnings': [],
                'stats': {}
            }
        
        errors = []
        warnings = []
        line_count = 0
        valid_samples = 0
        
        role_stats = {'system': 0, 'user': 0, 'assistant': 0}
        content_lengths = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                
                if line_count > self.max_lines:
                    errors.append(f"数据集超过{self.max_lines}行限制")
                    break
                
                # 验证JSON格式
                json_valid, json_error, data = self.validate_json_format(line, line_num)
                if not json_valid:
                    errors.append(json_error)
                    continue
                
                # 验证messages结构
                struct_valid, struct_errors = self.validate_messages_structure(data, line_num)
                if not struct_valid:
                    errors.extend(struct_errors)
                    continue
                
                messages = data['messages']
                
                # 验证角色序列
                role_valid, role_errors = self.validate_role_sequence(messages, line_num)
                if not role_valid:
                    errors.extend(role_errors)
                
                # 验证内容质量
                content_valid, content_issues = self.validate_content_quality(messages, line_num)
                if not content_valid:
                    errors.extend([issue for issue in content_issues if '错误' in issue])
                warnings.extend([issue for issue in content_issues if '警告' in issue or '过短' in issue or '过长' in issue or '可能' in issue])
                
                # 统计信息
                if struct_valid and role_valid and content_valid:
                    valid_samples += 1
                
                for msg in messages:
                    role = msg.get('role')
                    if role in role_stats:
                        role_stats[role] += 1
                    content_lengths.append(len(msg.get('content', '')))
        
        # 计算统计信息
        stats = {
            'total_lines': line_count,
            'valid_samples': valid_samples,
            'invalid_samples': line_count - valid_samples,
            'role_distribution': role_stats,
            'avg_content_length': sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            'max_content_length': max(content_lengths) if content_lengths else 0,
            'min_content_length': min(content_lengths) if content_lengths else 0
        }
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def print_validation_report(self, result: Dict[str, Any], file_path: str):
        """打印验证报告"""
        print(f"\n{'='*60}")
        print(f"数据集验证报告: {file_path}")
        print(f"{'='*60}")
        
        # 总体状态
        status = "✅ 通过" if result['valid'] else "❌ 失败"
        print(f"验证状态: {status}")
        
        # 统计信息
        stats = result['stats']
        print(f"\n📊 统计信息:")
        print(f"  总行数: {stats.get('total_lines', 0)}")
        print(f"  有效样本: {stats.get('valid_samples', 0)}")
        print(f"  无效样本: {stats.get('invalid_samples', 0)}")
        print(f"  平均内容长度: {stats.get('avg_content_length', 0):.1f} 字符")
        
        # 角色分布
        role_dist = stats.get('role_distribution', {})
        print(f"\n👥 角色分布:")
        for role, count in role_dist.items():
            print(f"  {role}: {count}")
        
        # 错误信息
        if result['errors']:
            print(f"\n❌ 错误 ({len(result['errors'])}):")
            for error in result['errors'][:10]:  # 只显示前10个错误
                print(f"  • {error}")
            if len(result['errors']) > 10:
                print(f"  ... 还有 {len(result['errors']) - 10} 个错误")
        
        # 警告信息
        if result['warnings']:
            print(f"\n⚠️  警告 ({len(result['warnings'])}):")
            for warning in result['warnings'][:5]:  # 只显示前5个警告
                print(f"  • {warning}")
            if len(result['warnings']) > 5:
                print(f"  ... 还有 {len(result['warnings']) - 5} 个警告")
        
        # 建议
        print(f"\n💡 建议:")
        if result['valid']:
            print("  • 数据集格式正确，可以用于微调")
        else:
            print("  • 请修复上述错误后重新验证")
        
        if stats.get('total_lines', 0) < 100:
            print("  • 建议增加更多训练样本以提高模型效果")
        
        if stats.get('avg_content_length', 0) < 50:
            print("  • 建议增加内容的详细程度")

def main():
    """主函数"""
    validator = DatasetValidator()
    
    # 验证数据集文件
    dataset_files = [
        "data/math_equation_training_dataset.jsonl",
        "data/complete_math_equation_dataset.jsonl",
        "data/math_equation_fine_tuning_dataset.jsonl"
    ]
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            print(f"验证文件: {file_path}")
            result = validator.validate_file(file_path)
            validator.print_validation_report(result, file_path)
        else:
            print(f"文件不存在: {file_path}")

if __name__ == "__main__":
    main()
