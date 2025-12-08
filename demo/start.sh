#!/bin/bash
# 智慧课堂系统启动脚本

echo "========================================="
echo "      智慧课堂系统启动脚本"
echo "========================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖是否安装
echo "检查依赖..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "警告: Flask未安装，正在安装依赖..."
    pip install -r requirements.txt
fi

# 创建必要的文件夹
mkdir -p uploads
mkdir -p faces_database
mkdir -p templates
mkdir -p static/css
mkdir -p static/js

echo ""
echo "启动服务器..."
echo "服务器地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动Flask应用
python3 app.py

