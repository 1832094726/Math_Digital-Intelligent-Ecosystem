@echo off
echo 启动数据库可视化API服务器...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

REM 检查依赖包
echo 检查依赖包...
python -c "import flask, flask_cors, pymysql" >nul 2>&1
if errorlevel 1 (
    echo 安装依赖包...
    pip install flask flask-cors pymysql
    if errorlevel 1 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
)

echo 依赖包检查完成
echo.

REM 启动API服务器
echo 启动API服务器...
python api-server.py

pause
