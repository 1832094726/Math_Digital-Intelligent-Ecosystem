#!/usr/bin/env python3
"""
Flask应用启动脚本
确保在Docker容器中正确监听所有网络接口
"""

import os
from app import app

if __name__ == '__main__':
    # 从环境变量获取配置
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print(f"🚀 启动Flask应用...")
    print(f"📍 监听地址: {host}:{port}")
    print(f"🔧 调试模式: {debug}")
    print(f"🌐 环境: {os.environ.get('FLASK_ENV', 'production')}")
    
    # 启动应用
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )
