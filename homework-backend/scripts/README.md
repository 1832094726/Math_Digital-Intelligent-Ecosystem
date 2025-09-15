# 系统维护脚本说明

本目录包含了K-12数学教育系统的各种维护和管理脚本。

## 📁 文件分类

### 🗄️ 数据库初始化脚本
这些脚本用于创建和初始化数据库表结构：

- `create_homework_schema.py` - 创建作业相关表结构
- `create_homework_simple.py` - 创建简化的作业表结构
- `create_assignment_tables.py` - 创建作业分配表
- `create_auth_schema.py` - 创建认证相关表
- `create_basic_tables.py` - 创建基础表结构
- `create_database.py` - 数据库创建脚本
- `create_missing_tables.py` - 创建缺失的表
- `create_remaining_tables.py` - 创建剩余表结构
- `create_simple_schema.py` - 创建简化模式
- `create_student_homework_tables.py` - 创建学生作业表
- `init_database.py` - 数据库初始化主脚本

### 🧪 测试脚本
这些脚本用于测试API接口和功能：

- `test_api.py` - 通用API测试
- `test_assignment_api.py` - 作业分配API测试
- `test_auth.py` - 认证功能测试
- `test_homework_api.py` - 作业管理API测试
- `test_simple_student_homework.py` - 简化学生作业测试
- `test_student_homework_api.py` - 学生作业API测试
- `test_submission_api.py` - 提交功能API测试

### 🔍 调试和验证脚本
这些脚本用于调试和验证系统状态：

- `check_existing_tables.py` - 检查现有表结构
- `check_schema.py` - 检查数据库模式
- `debug_schema.py` - 调试数据库模式
- `validate_dataset.py` - 验证数据集

### 🎯 数据生成和处理脚本
这些脚本用于生成测试数据和处理数据：

- `create_test_assignment.py` - 创建测试作业
- `generate_equation_dataset.py` - 生成方程数据集
- `setup_test_student.py` - 设置测试学生

## 🚀 使用说明

### 初始化数据库
```bash
# 1. 首先运行主初始化脚本
python init_database.py

# 2. 如果需要创建特定表，运行对应的create_*.py脚本
python create_homework_schema.py
```

### 运行测试
```bash
# 测试所有API
python test_api.py

# 测试特定功能
python test_homework_api.py
python test_auth.py
```

### 调试问题
```bash
# 检查数据库状态
python check_existing_tables.py
python debug_schema.py
```

## ⚠️ 注意事项

1. **运行顺序**：数据库初始化脚本需要按依赖关系运行
2. **环境要求**：确保数据库连接配置正确
3. **测试数据**：测试脚本会创建测试数据，注意清理
4. **备份**：运行数据库脚本前建议备份现有数据

## 🔄 维护建议

- 定期运行测试脚本确保功能正常
- 新增功能时添加对应的测试脚本
- 保持脚本的文档更新
