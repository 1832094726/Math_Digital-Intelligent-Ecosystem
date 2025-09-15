# 数字化智能生态系统 for K-12 数学教育

## 大文件下载说明

由于 GitHub 的文件大小限制（100MB），以下文件未包含在代码仓库中：

1. 模型文件：
   - `Subject_symbol_dynamic_keyboard/test/best_models/BERT-ncf-mlp(01-17-09-40).pt` (415.23 MB)
   - `Subject_symbol_dynamic_keyboard/test/chinese-bert-wwm-ext/pytorch_model.bin` (392.51 MB)

2. 数据集：
   - `Subject_symbol_dynamic_keyboard/test/new_data/train/user_similarity.csv` (860.17 MB)

这些文件可以通过以下方式获取：

1. 联系项目维护者获取完整数据
2. 运行相关训练脚本重新生成模型文件
3. 按照数据处理脚本重新生成数据集

## 项目结构

- `architect/`: 系统架构设计文档
- `database-visualization/`: 数据库可视化模块
- `homework_system/`: 作业系统前端
- `homework-backend/`: 作业系统后端
- `Subject_symbol_dynamic_keyboard/`: 数学符号输入系统

## 本地开发设置

1. 克隆仓库
2. 按照各个子系统的 README 进行环境配置
3. 获取所需的模型文件和数据集
4. 启动相应的服务

## 注意事项

- 大文件已通过 .gitignore 配置排除在版本控制之外
- 请确保在开发过程中不要提交超过 100MB 的文件到 Git 仓库
