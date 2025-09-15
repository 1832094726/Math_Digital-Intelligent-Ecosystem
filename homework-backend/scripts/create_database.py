# -*- coding: utf-8 -*-
"""
数据库创建脚本
执行整合版SQL脚本创建完整的数据库结构
"""
import os
import pymysql
from config import config

def create_database_schema():
    """执行SQL脚本创建数据库结构"""
    
    # 获取SQL脚本路径
    sql_script_path = os.path.join(os.path.dirname(__file__), '..', 'architect', '04_数据模型设计_整合版.sql')
    
    if not os.path.exists(sql_script_path):
        print(f"错误: SQL脚本文件未找到: {sql_script_path}")
        return False

    print(f"正在读取SQL脚本: {sql_script_path}")
    
    # 读取SQL脚本
    try:
        with open(sql_script_path, 'r', encoding='utf8') as f:
            sql_script = f.read()
    except Exception as e:
        print(f"错误: 无法读取SQL脚本文件: {e}")
        return False

    # 解析数据库配置
    env = os.environ.get('FLASK_ENV', 'development')
    current_config = config[env]
    
    # 使用DATABASE_CONFIG配置
    db_config = current_config.DATABASE_CONFIG
    host = db_config['host']
    port = db_config['port']
    user = db_config['user']
    password = db_config['password']
    db_name = db_config['database']

    print(f"连接数据库: {host}:{port}/{db_name}")
    
    # 连接数据库
    connection = None
    cursor = None
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            passwd=password,
            db=db_name,
            charset='utf8mb4',
            autocommit=True  # 使用自动提交来避免外键检查问题
        )
        cursor = connection.cursor()
        
        print("数据库连接成功，开始执行SQL脚本...")
        
        # 分割SQL语句
        # 使用更智能的分割，处理存储过程中的分号
        sql_commands = []
        current_command = ""
        in_delimiter = False
        delimiter = ";"
        
        lines = sql_script.split('\n')
        for line in lines:
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('--') or line.startswith('/*'):
                continue
                
            # 处理DELIMITER语句
            if line.upper().startswith('DELIMITER'):
                if 'DELIMITER //' in line.upper():
                    delimiter = "//"
                    in_delimiter = True
                elif 'DELIMITER ;' in line.upper():
                    delimiter = ";"
                    in_delimiter = False
                continue
            
            current_command += line + "\n"
            
            # 检查是否到达命令结尾
            if line.endswith(delimiter):
                if delimiter == "//" or not in_delimiter:
                    # 移除结尾的分隔符
                    current_command = current_command.rstrip().rstrip(delimiter).strip()
                    if current_command:
                        sql_commands.append(current_command)
                    current_command = ""
        
        # 添加最后一个命令（如果有的话）
        if current_command.strip():
            sql_commands.append(current_command.strip())
        
        print(f"解析到 {len(sql_commands)} 个SQL命令")
        
        # 执行SQL命令
        success_count = 0
        for i, command in enumerate(sql_commands, 1):
            if not command.strip():
                continue
                
            try:
                # 显示执行进度
                print(f"[{i}/{len(sql_commands)}] 执行: {command[:60]}...")
                
                cursor.execute(command)
                success_count += 1
                
            except pymysql.err.OperationalError as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print(f"  警告: {e} (跳过)")
                    continue
                else:
                    print(f"  错误: {e}")
                    print(f"  命令: {command[:200]}...")
                    return False
                    
            except Exception as e:
                print(f"  错误: {e}")
                print(f"  命令: {command[:200]}...")
                return False
        
        print(f"\n✅ 数据库创建成功！")
        print(f"   - 成功执行 {success_count} 个SQL命令")
        print(f"   - 数据库: {db_name}")
        print(f"   - 主机: {host}:{port}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库创建失败: {e}")
        return False
        
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def check_database_status():
    """检查数据库创建状态"""
    try:
        from models.database import execute_query
        
        # 检查表数量
        tables = execute_query("SHOW TABLES")
        print(f"\n数据库状态检查:")
        print(f"  - 总表数: {len(tables)}")
        
        # 检查关键表
        key_tables = ['users', 'schools', 'courses', 'homeworks', 'knowledge_points']
        for table_name in key_tables:
            try:
                count_result = execute_query(f"SELECT COUNT(*) as count FROM {table_name}", fetch_one=True)
                count = count_result['count'] if count_result else 0
                print(f"  - {table_name}: {count} 条记录")
            except:
                print(f"  - {table_name}: 表不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库状态检查失败: {e}")
        return False

def main():
    """主函数"""
    print("=== K-12数学教育智能生态系统 - 数据库创建 ===\n")
    
    # 创建数据库结构
    if create_database_schema():
        print("\n" + "="*50)
        
        # 检查创建结果
        check_database_status()
        
        print("\n🎉 数据库初始化完成！")
        print("现在可以运行以下命令测试系统：")
        print("  python init_database.py    # 检查数据库连接和表结构")
        print("  python test_auth.py        # 测试认证API")
        print("  python app.py              # 启动应用服务器")
        
    else:
        print("\n❌ 数据库创建失败，请检查错误信息并重试")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
