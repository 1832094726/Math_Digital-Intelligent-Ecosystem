@echo off
chcp 65001 >nul
echo 🚀 K12数学教育生态系统 - Windows全栈构建
echo ==========================================

:: 检查Docker
echo [信息] 检查Docker环境...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker未安装或未启动
    echo 请先安装Docker Desktop for Windows
    pause
    exit /b 1
)

:: 检查前端项目
echo [信息] 检查前端项目...
if not exist "homework_system\package.json" (
    echo [错误] 未找到Vue前端项目
    pause
    exit /b 1
)

:: 检查后端项目
echo [信息] 检查后端项目...
if not exist "homework-backend\requirements.txt" (
    echo [错误] 未找到Flask后端项目
    pause
    exit /b 1
)

:: 设置镜像名称
set IMAGE_NAME=matheco/k12-math-ecosystem
set VERSION=latest
set FULL_IMAGE_NAME=%IMAGE_NAME%:%VERSION%

echo [信息] 构建全栈镜像: %FULL_IMAGE_NAME%

:: 切换到项目根目录
cd /d "%~dp0\.."

:: 构建镜像（多阶段构建）
echo [信息] 开始多阶段构建...
echo [信息] 阶段1: 构建Vue前端...
echo [信息] 阶段2: 构建Flask后端并整合前端...
docker build -t %FULL_IMAGE_NAME% -f docker/Dockerfile .

if %errorlevel% neq 0 (
    echo [错误] 镜像构建失败
    pause
    exit /b 1
)

echo [成功] 镜像构建完成

:: 询问是否推送到Docker Hub
set /p PUSH_CHOICE="是否推送到Docker Hub? (y/N): "
if /i "%PUSH_CHOICE%"=="y" (
    echo [信息] 登录Docker Hub...
    docker login
    
    if %errorlevel% neq 0 (
        echo [错误] Docker Hub登录失败
        pause
        exit /b 1
    )
    
    echo [信息] 推送镜像到Docker Hub...
    docker push %FULL_IMAGE_NAME%
    
    if %errorlevel% neq 0 (
        echo [错误] 镜像推送失败
        pause
        exit /b 1
    )
    
    echo [成功] 全栈镜像已推送到Docker Hub
    echo.
    echo 🎉 全栈镜像已发布: %FULL_IMAGE_NAME%
    echo.
    echo 📋 包含组件:
    echo    ✅ Vue.js前端 (homework_system)
    echo    ✅ Flask后端 (homework-backend)
    echo    ✅ 数学符号键盘
    echo    ✅ 静态资源
    echo.
    echo 🚀 现在可以在Linux服务器上一键部署:
    echo docker pull %FULL_IMAGE_NAME%
    echo docker run -d -p 8080:5000 %FULL_IMAGE_NAME%
) else (
    echo [信息] 跳过推送到Docker Hub
    echo.
    echo 💡 本地镜像已构建完成
    echo 可以使用以下命令导出镜像:
    echo docker save %FULL_IMAGE_NAME% -o k12-math-ecosystem.tar
    echo 然后将tar文件传输到Linux服务器并导入:
    echo docker load -i k12-math-ecosystem.tar
)

echo.
echo 📋 镜像信息:
docker images | findstr %IMAGE_NAME%

echo.
echo [完成] 构建流程结束
pause
