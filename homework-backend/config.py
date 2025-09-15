# -*- coding: utf-8 -*-
"""
系统配置文件
"""
import os
from datetime import timedelta

class Config:
    """基础配置"""
    # 数据库配置 - 支持环境变量覆盖
    DATABASE_CONFIG = {
        'host': os.environ.get('DB_HOST', 'obmt6zg485miazb4-mi.aliyun-cn-beijing-internet.oceanbase.cloud'),
        'port': int(os.environ.get('DB_PORT', 3306)),
        'user': os.environ.get('DB_USER', 'hcj'),
        'password': os.environ.get('DB_PASSWORD', 'Xv0Mu8_:'),
        'database': os.environ.get('DB_NAME', 'testccnu'),
        'charset': 'utf8mb4'
    }
    
    # JWT配置
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'your-secret-key-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    JSON_AS_ASCII = False
    
    # 文件上传配置
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # 密码配置
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_REQUIRE_SPECIAL = True
    
    # 登录配置
    MAX_LOGIN_ATTEMPTS = 5
    LOGIN_LOCKOUT_DURATION = timedelta(minutes=30)

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    
class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


