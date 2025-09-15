@echo off
chcp 65001 >nul
echo 🏗️ Windows Docker镜像构建脚本
echo ================================

:: 检查Docker
echo [信息] 检查Docker环境...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker未安装或未启动
    echo 请先安装Docker Desktop for Windows
    pause
    exit /b 1
)

:: 设置镜像名称
set IMAGE_NAME=matheco/k12-math-ecosystem
set VERSION=latest
set FULL_IMAGE_NAME=%IMAGE_NAME%:%VERSION%

echo [信息] 构建镜像: %FULL_IMAGE_NAME%

:: 切换到项目根目录
cd /d "%~dp0\.."

:: 构建镜像
echo [信息] 开始构建Docker镜像...
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
    
    echo [成功] 镜像已推送到Docker Hub
    echo.
    echo 🎉 现在可以在Linux服务器上使用以下命令部署:
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
