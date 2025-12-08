# 智慧课堂系统

一个基于YOLO和Flask的智慧课堂管理系统，包含人脸识别打卡、实时人脸检测识别、学生行为检测等功能。

## 功能特性

1. **人脸识别打卡**
   - 注册学生人脸信息
   - 人脸识别打卡功能
   - 查看已注册人员列表

2. **课堂实时人脸检测与识别**
   - 支持上传视频文件进行检测
   - 支持RTSP流实时检测
   - 实时显示检测结果并识别人脸

3. **课堂学生行为检测**
   - 支持上传视频文件进行检测
   - 支持RTSP流实时检测
   - 检测学生行为（低头、转头等）

## 系统要求

- Python 3.8+
- CUDA（可选，用于GPU加速）
- 足够的磁盘空间用于存储上传的视频文件

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: `dlib` 和 `face-recognition` 的安装可能需要额外步骤：

- **Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
pip install face-recognition
```

- **macOS**:
```bash
brew install cmake
pip install dlib
pip install face-recognition
```

### 2. 准备模型文件

确保以下模型文件存在：
- `yolo11n.pt` - 人脸检测模型（如果不存在会自动下载）
- `weights/best.pt` - 行为检测模型（可选，如果不存在会使用默认模型）

### 3. 创建必要的文件夹

系统会自动创建以下文件夹：
- `uploads/` - 存储上传的视频文件
- `faces_database/` - 存储注册的人脸数据

## 使用方法

### 启动服务器

```bash
python app.py
```

服务器将在 `http://0.0.0.0:5000` 启动。

### 访问Web界面

在浏览器中打开：`http://localhost:5000`

## 使用说明

### 人脸识别打卡

1. **注册人脸**:
   - 输入学生姓名
   - 上传包含清晰人脸的图片
   - 点击"注册"按钮

2. **人脸打卡**:
   - 上传或拍摄包含人脸的图片
   - 点击"打卡"按钮
   - 系统会自动识别并显示打卡结果

### 课堂实时人脸检测

1. **选择视频源**:
   - **上传视频**: 选择本地视频文件并上传
   - **RTSP流**: 输入RTSP流地址（格式：`rtsp://username:password@ip:port/stream`）

2. **开始检测**:
   - 点击"开始检测"按钮
   - 系统会实时显示检测结果，识别人脸并标注姓名

### 课堂学生行为检测

1. **选择视频源**:
   - **上传视频**: 选择本地视频文件并上传
   - **RTSP流**: 输入RTSP流地址

2. **开始检测**:
   - 点击"开始检测"按钮
   - 系统会实时检测学生行为（如低头、转头等）

## API接口

### 人脸识别相关

- `POST /api/face/register` - 注册人脸
- `POST /api/face/checkin` - 人脸打卡
- `GET /api/face/list` - 获取已注册人脸列表

### 视频流相关

- `POST /api/stream/face/start` - 启动人脸检测视频流
- `GET /api/stream/face/video` - 获取人脸检测视频流
- `POST /api/stream/behavior/start` - 启动行为检测视频流
- `GET /api/stream/behavior/video` - 获取行为检测视频流

### 文件上传

- `POST /api/upload/video` - 上传视频文件

## 技术栈

- **后端**: Flask
- **AI模型**: YOLO (Ultralytics)
- **人脸识别**: face-recognition (基于dlib)
- **视频处理**: OpenCV
- **前端**: HTML, CSS, JavaScript

## 注意事项

1. **性能**: 如果使用CPU进行推理，处理速度可能较慢。建议使用GPU加速。

2. **RTSP流**: 确保RTSP流地址正确且网络可达。某些RTSP流可能需要特定的编解码器支持。

3. **视频格式**: 支持常见的视频格式（mp4, avi, mov, mkv等）。

4. **人脸识别精度**: 人脸识别精度取决于：
   - 注册时图片的质量和清晰度
   - 打卡时图片的光照条件
   - 人脸的角度和遮挡情况

5. **存储空间**: 上传的视频文件会保存在 `uploads/` 文件夹中，请定期清理。

## 故障排除

### 问题：无法安装dlib

**解决方案**: 
- 确保已安装CMake和必要的编译工具
- 尝试使用conda安装：`conda install -c conda-forge dlib`

### 问题：RTSP流无法连接

**解决方案**:
- 检查RTSP地址格式是否正确
- 确认网络连接正常
- 某些RTSP流可能需要特定的认证信息

### 问题：检测速度慢

**解决方案**:
- 使用GPU版本的PyTorch
- 降低视频分辨率
- 调整检测置信度阈值

## 开发说明

### 项目结构

```
ai-class/
├── app.py                 # Flask主应用
├── templates/             # HTML模板
│   └── index.html
├── static/               # 静态文件
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── uploads/              # 上传文件存储
├── faces_database/       # 人脸数据库
├── weights/              # 模型权重文件
└── requirements.txt      # 依赖列表
```

## 许可证

本项目仅供学习和研究使用。

## 更新日志

### v1.0.0 (2024)
- 初始版本
- 实现人脸识别打卡功能
- 实现实时人脸检测与识别
- 实现学生行为检测
- 支持视频上传和RTSP流
