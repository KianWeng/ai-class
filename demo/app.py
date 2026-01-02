#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智慧课堂系统 - Flask后端应用
包含人脸识别打卡、实时人脸检测、学生行为检测功能
"""

# # 设置库路径（解决 dlib CUDNN 问题）
# # 必须在导入任何可能使用 dlib 的模块之前设置
import os
import cv2
import numpy as np
import base64
import json
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

try:
    from facenet_pytorch import InceptionResnetV1
    from facenet_pytorch import fixed_image_standardization
    import torch
    from PIL import Image, ImageDraw, ImageFont
    FACE_RECOGNITION_AVAILABLE = True
    # 初始化 FaceNet 模型用于特征提取
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet = resnet.to(device)
    print(f"FaceNet 模型已加载，使用设备: {device}")
    if device == 'cuda':
        print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except ImportError as e:
    print(f"警告: facenet-pytorch库未安装，人脸识别功能将受限: {e}")
    FACE_RECOGNITION_AVAILABLE = False
    resnet = None
    fixed_image_standardization = None
except Exception as e:
    print(f"警告: facenet-pytorch库导入失败: {e}")
    import traceback
    traceback.print_exc()
    FACE_RECOGNITION_AVAILABLE = False
    resnet = None
    fixed_image_standardization = None
from pathlib import Path
import pickle

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}
FACES_FOLDER = 'faces_database'
MODEL_FACE_PATH = 'weights/face_detect/yolo11n-face.pt'  # 人脸检测模型
MODEL_BEHAVIOR_PATH = 'weights/scb/bth_best.pt'  # 行为检测模型（如果存在）- 低头、转头
MODEL_BEHAVIOR_HRW_PATH = 'weights/scb/hrw_best.pt'  # 行为检测模型（如果存在）- 举手、阅读、写字

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 创建必要的文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# 全局变量
face_model = None
behavior_model = None  # 低头、转头模型
behavior_model_hrw = None  # 举手、阅读、写字模型
face_encodings_db = {}  # 存储人脸编码 {name: encoding}
rtsp_streams = {}  # 存储RTSP流 {stream_id: cap}
video_streams = {}  # 存储视频流 {stream_id: cap}
behavior_statistics = {}  # 存储行为统计 {stream_id: {class_name: count}}
face_statistics = {}  # 存储人脸统计 {stream_id: {name: count}}
face_detection_list = {}  # 存储检测到的人脸列表 {stream_id: [face_info, ...]}

# 初始化模型
def init_models():
    global face_model, behavior_model, behavior_model_hrw
    
    # 加载人脸检测模型
    if os.path.exists(MODEL_FACE_PATH):
        print(f"加载人脸检测模型: {MODEL_FACE_PATH}")
        face_model = YOLO(MODEL_FACE_PATH)
    else:
        print(f"警告: 人脸检测模型不存在: {MODEL_FACE_PATH}")
        face_model = YOLO('yolo11n.pt')  # 使用默认模型
    
    # 加载行为检测模型1：低头、转头
    if os.path.exists(MODEL_BEHAVIOR_PATH):
        print(f"加载行为检测模型1: {MODEL_BEHAVIOR_PATH} (低头、转头)")
        behavior_model = YOLO(MODEL_BEHAVIOR_PATH)
    else:
        print(f"警告: 行为检测模型1不存在: {MODEL_BEHAVIOR_PATH}")
        behavior_model = None
    
    # 加载行为检测模型2：举手、阅读、写字
    if os.path.exists(MODEL_BEHAVIOR_HRW_PATH):
        print(f"加载行为检测模型2: {MODEL_BEHAVIOR_HRW_PATH} (举手、阅读、写字)")
        behavior_model_hrw = YOLO(MODEL_BEHAVIOR_HRW_PATH)
    else:
        print(f"警告: 行为检测模型2不存在: {MODEL_BEHAVIOR_HRW_PATH}")
        behavior_model_hrw = None
    
    # 加载已注册的人脸
    load_face_database()

def load_face_database():
    """加载已注册的人脸数据库"""
    global face_encodings_db
    
    db_file = os.path.join(FACES_FOLDER, 'face_database.pkl')
    if os.path.exists(db_file):
        with open(db_file, 'rb') as f:
            face_encodings_db = pickle.load(f)
        print(f"已加载 {len(face_encodings_db)} 个已注册人脸")
    else:
        face_encodings_db = {}
        print("人脸数据库为空")

def save_face_database():
    """保存人脸数据库"""
    db_file = os.path.join(FACES_FOLDER, 'face_database.pkl')
    with open(db_file, 'wb') as f:
        pickle.dump(face_encodings_db, f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 全局变量：缓存字体，避免重复加载
_chinese_font_cache = None

def get_chinese_font(font_size=30):
    """获取中文字体，如果找不到则返回默认字体"""
    global _chinese_font_cache
    
    # 如果已经缓存了字体，直接返回
    if _chinese_font_cache is not None:
        try:
            return ImageFont.truetype(_chinese_font_cache, font_size)
        except:
            pass
    
    # 尝试加载中文字体
    font_paths = [
        # Linux 中文字体（按优先级排序）
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Windows 中文字体
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        # macOS 中文字体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                _chinese_font_cache = font_path  # 缓存字体路径
                return font
        except:
            continue
    
    # 如果所有字体都加载失败，使用默认字体
    if not hasattr(get_chinese_font, '_warned'):
        print("=" * 60)
        print("警告: 未找到中文字体，中文可能显示为方块或乱码")
        print("建议安装中文字体，运行以下命令：")
        print("  Ubuntu/Debian: sudo apt-get install fonts-wqy-microhei")
        print("  CentOS/RHEL: sudo yum install wqy-microhei-fonts")
        print("=" * 60)
        get_chinese_font._warned = True
    
    return ImageFont.load_default()

def draw_chinese_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """
    在图像上绘制中文文字
    :param img: OpenCV 图像 (BGR格式)
    :param text: 要绘制的文字
    :param position: 文字位置 (x, y)
    :param font_size: 字体大小
    :param color: 文字颜色 (BGR格式)
    :return: 绘制后的图像
    """
    # 转换为 PIL Image (RGB格式)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # 获取字体
    font = get_chinese_font(font_size)
    
    # 转换颜色格式 (BGR -> RGB)
    rgb_color = (color[2], color[1], color[0])
    
    # 绘制文字
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转换回 OpenCV 格式 (RGB -> BGR)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr
    
    # 转换颜色格式 (BGR -> RGB)
    rgb_color = (color[2], color[1], color[0])
    
    # 绘制文字
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转换回 OpenCV 格式 (RGB -> BGR)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def detect_faces_yolo(image):
    """使用YOLO检测人脸"""
    if face_model is None:
        return []
    
    results = face_model.predict(image, conf=0.25, verbose=False)
    faces = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            faces.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence
            })
    
    return faces

def align_face(img, box, target_size=(160, 160)):
    """
    裁剪并标准化人脸图像（FaceNet 要求 160×160，且像素标准化）
    :param img: 原始 BGR 图像
    :param box: 人脸边界框 [x1,y1,x2,y2]
    :param target_size: FaceNet 输入尺寸
    :return: 对齐后的张量（可直接输入 FaceNet）
    """
    # 裁剪人脸区域（防止越界）
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    face_img = img[y1:y2, x1:x2]
    
    # 转换为 RGB（OpenCV 是 BGR，FaceNet 要求 RGB）
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # 缩放至目标尺寸
    face_img_resized = cv2.resize(face_img_rgb, target_size)
    
    # 转换为张量并标准化（FaceNet 要求的标准化方式）
    face_tensor = torch.tensor(face_img_resized).permute(2, 0, 1).float()  # [H,W,C] → [C,H,W]
    face_tensor = fixed_image_standardization(face_tensor)  # 标准化（关键，否则特征不准）
    
    return face_tensor

def extract_features_batch(face_tensors):
    """批量提取人脸特征向量"""
    if not FACE_RECOGNITION_AVAILABLE or resnet is None:
        return []
    
    if len(face_tensors) == 0:
        return []
    
    try:
        # 将所有张量堆叠成批次
        valid_tensors = []
        valid_indices = []
        
        for i, face_tensor in enumerate(face_tensors):
            if face_tensor is not None:
                # 确保是 3D (channels, height, width)
                if face_tensor.dim() == 3:
                    valid_tensors.append(face_tensor)
                    valid_indices.append(i)
        
        if len(valid_tensors) == 0:
            return []
        
        # 堆叠成批次 (batch, channels, height, width)
        batch_tensor = torch.stack(valid_tensors)
        
        # 批量提取特征
        with torch.no_grad():
            device = next(resnet.parameters()).device
            face_encodings = resnet(batch_tensor.to(device))
            # 归一化特征向量
            face_encodings = torch.nn.functional.normalize(face_encodings, p=2, dim=1)
            face_encodings = face_encodings.cpu().numpy()
        
        # 创建完整的结果列表（包含 None）
        results = [None] * len(face_tensors)
        for idx, encoding in zip(valid_indices, face_encodings):
            results[idx] = encoding
        
        return results
    except Exception as e:
        print(f"批量特征提取错误: {e}")
        import traceback
        traceback.print_exc()
        return []

def recognize_face_from_encoding(face_encoding):
    """从特征向量识别人脸"""
    if face_encoding is None:
        return None, 0.0
    
    try:
        # 与数据库中的所有人脸比较（使用余弦相似度）
        best_match = None
        best_similarity = -1.0
        all_similarities = []  # 记录所有相似度，用于调试
        
        for name, known_encoding in face_encodings_db.items():
            # 计算余弦相似度
            similarity = np.dot(face_encoding, known_encoding) / (
                np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding)
            )
            all_similarities.append((name, similarity))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # 打印前3个最相似的结果（用于调试）
        if len(all_similarities) > 0:
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            top3 = all_similarities[:3]
            top3_str = ", ".join([f"{name}:{sim:.3f}" for name, sim in top3])
            print(f"  [相似度排名] 前3名: {top3_str}")
        
        # 如果相似度大于0.6，认为是同一个人
        if best_similarity > 0.6:
            return best_match, best_similarity
        
        return None, best_similarity  # 返回最高相似度，即使未达到阈值
    except Exception as e:
        print(f"人脸识别错误: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def detect_behaviors(image, stream_id=None):
    """检测学生行为（支持两个模型）"""
    behaviors = []
    
    # 模型1：低头、转头
    if behavior_model is not None:
        results = behavior_model.predict(image, conf=0.25, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # 根据类别名称
                if hasattr(behavior_model, 'names') and behavior_model.names:
                    class_names = behavior_model.names
                else:
                    class_names = ['BowHead', 'TurnHead']
                
                class_name = class_names[cls] if cls < len(class_names) else f'行为{cls}'
                
                behaviors.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name
                })
    
    # 模型2：举手、阅读、写字
    if behavior_model_hrw is not None:
        results = behavior_model_hrw.predict(image, conf=0.25, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # 根据类别名称
                if hasattr(behavior_model_hrw, 'names') and behavior_model_hrw.names:
                    class_names = behavior_model_hrw.names
                else:
                    class_names = ['RaiseHand', 'Reading', 'Writing']  # 举手、阅读、写字
                
                class_name = class_names[cls] if cls < len(class_names) else f'行为{cls}'
                
                behaviors.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name
                })
    
    # 更新统计
    if stream_id and len(behaviors) > 0:
        if stream_id not in behavior_statistics:
            behavior_statistics[stream_id] = {}
        
        for behavior in behaviors:
            class_name = behavior['class']
            behavior_statistics[stream_id][class_name] = behavior_statistics[stream_id].get(class_name, 0) + 1
    
    return behaviors


def process_frame_face_detection(frame, stream_id=None):
    """处理帧进行人脸检测和识别（YOLO检测 -> 裁剪对齐 -> 批量FaceNet识别）"""
    # 步骤1: 使用YOLO检测人脸
    detection_start_time = time.time()
    faces = detect_faces_yolo(frame)
    detection_time = time.time() - detection_start_time
    print(f"[人脸检测] YOLO检测时间: {detection_time*1000:.2f} ms, 检测到 {len(faces)} 个人脸")
    
    # 绘制结果
    annotated_frame = frame.copy()
    results_data = []
    
    if len(faces) == 0:
        return annotated_frame, results_data
    
    # 步骤2: 对齐所有人脸（裁剪和缩放）
    alignment_start_time = time.time()
    face_tensors = []
    
    for i, face in enumerate(faces):
        try:
            # 使用 align_face 函数对齐人脸
            face_tensor = align_face(frame, face['bbox'])
            face_tensors.append(face_tensor)
        except Exception as e:
            print(f"[警告] 人脸 {i+1} 对齐失败: {e}")
            face_tensors.append(None)
    
    alignment_time = time.time() - alignment_start_time
    print(f"[人脸对齐] 对齐时间: {alignment_time*1000:.2f} ms")
    
    # 步骤3: 批量提取特征
    feature_extraction_start_time = time.time()
    face_encodings = extract_features_batch(face_tensors)
    feature_extraction_time = time.time() - feature_extraction_start_time
    print(f"[特征提取] 批量提取时间: {feature_extraction_time*1000:.2f} ms")
    
    # 步骤4: 与数据库比较（批量识别）
    recognition_start_time = time.time()
    recognition_results = []
    
    for i, (face, encoding) in enumerate(zip(faces, face_encodings)):
        if encoding is not None:
            name, similarity = recognize_face_from_encoding(encoding)
        else:
            name, similarity = None, 0.0
        
        # 打印识别结果
        if name:
            print(f"[识别结果] 人脸 {i+1}: 识别为 {name}, 相似度: {similarity:.4f}, 检测置信度: {face['confidence']:.4f}")
        else:
            print(f"[识别结果] 人脸 {i+1}: 未识别, 最高相似度: {similarity:.4f}, 检测置信度: {face['confidence']:.4f}")
        
        recognition_results.append({
            'index': i,
            'bbox': face['bbox'],
            'confidence': face['confidence'],
            'name': name,
            'similarity': similarity
        })
    
    recognition_time = time.time() - recognition_start_time
    print(f"[人脸识别] 识别时间: {recognition_time*1000:.2f} ms, 识别成功: {sum(1 for r in recognition_results if r['name'])}/{len(recognition_results)}")
    
    # 绘制结果
    for result in recognition_results:
        x1, y1, x2, y2 = result['bbox']
        confidence = result['confidence']
        name = result['name']
        similarity = result['similarity']
        
        # 绘制边界框
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签（使用支持中文的函数）
        if name:
            label = f"{name}: {similarity:.2f}"
        else:
            label = f"未知: {confidence:.2f}"
        
        # 使用支持中文的绘制函数
        annotated_frame = draw_chinese_text(annotated_frame, label, (x1, y1 - 25), font_size=20, color=color)
        
        results_data.append({
            'bbox': result['bbox'],
            'confidence': confidence,
            'name': name,
            'similarity': similarity
        })
    
    # 打印总时间统计和识别摘要
    total_time = detection_time + alignment_time + feature_extraction_time + recognition_time
    recognized_count = sum(1 for r in recognition_results if r['name'])
    print(f"[总计] 单帧总处理时间: {total_time*1000:.2f} ms (检测: {detection_time*1000:.2f} ms + 对齐: {alignment_time*1000:.2f} ms + 特征提取: {feature_extraction_time*1000:.2f} ms + 识别: {recognition_time*1000:.2f} ms)")
    print(f"[识别摘要] 检测到 {len(faces)} 个人脸, 成功识别 {recognized_count} 个, 未识别 {len(faces) - recognized_count} 个")
    if recognized_count > 0:
        recognized_names = [r['name'] for r in recognition_results if r['name']]
        print(f"[识别摘要] 识别到的人员: {', '.join(recognized_names)}")
    print("-" * 60)  # 分隔线
    
    # 更新统计和列表
    if stream_id:
        # 初始化统计字典
        if stream_id not in face_statistics:
            face_statistics[stream_id] = {}
        if stream_id not in face_detection_list:
            face_detection_list[stream_id] = []
        
        # 为每个检测到的人脸添加时间戳并添加到列表
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for result in recognition_results:
            face_info = {
                'timestamp': current_time,
                'name': result['name'] if result['name'] else '未知',
                'similarity': result['similarity'],
                'confidence': result['confidence'],
                'bbox': result['bbox']
            }
            face_detection_list[stream_id].append(face_info)
            
            # 更新统计计数
            name_key = result['name'] if result['name'] else '未知'
            face_statistics[stream_id][name_key] = face_statistics[stream_id].get(name_key, 0) + 1
    
    return annotated_frame, results_data

def process_frame_behavior_detection(frame, stream_id=None):
    """处理帧进行行为检测"""
    behaviors = detect_behaviors(frame, stream_id=stream_id)
    
    # 绘制结果
    annotated_frame = frame.copy()
    
    for behavior in behaviors:
        x1, y1, x2, y2 = behavior['bbox']
        confidence = behavior['confidence']
        class_name = behavior['class']
        
        # 绘制边界框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 绘制标签（使用支持中文的函数）
        label = f"{class_name}: {confidence:.2f}"
        annotated_frame = draw_chinese_text(annotated_frame, label, (x1, y1 - 25), font_size=20, color=(255, 0, 0))
    
    return annotated_frame, behaviors

def generate_frames_face(source_type, source_path, stream_id):
    """生成人脸检测视频流"""
    cap = None
    
    # 初始化统计
    if stream_id:
        face_statistics[stream_id] = {}
        face_detection_list[stream_id] = []
    
    try:
        if source_type == 'rtsp':
            cap = cv2.VideoCapture(source_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif source_type == 'video':
            cap = cv2.VideoCapture(source_path)
        else:
            return
        
        if not cap.isOpened():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9' + b'\r\n')
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧（传递stream_id用于统计）
            processed_frame, _ = process_frame_face_detection(frame, stream_id=stream_id)
            
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    except Exception as e:
        print(f"生成帧错误: {e}")
    finally:
        if cap:
            cap.release()

def generate_frames_behavior(source_type, source_path, stream_id):
    """生成行为检测视频流"""
    cap = None
    
    # 初始化统计
    if stream_id:
        behavior_statistics[stream_id] = {}
    
    try:
        if source_type == 'rtsp':
            cap = cv2.VideoCapture(source_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif source_type == 'video':
            cap = cv2.VideoCapture(source_path)
        else:
            return
        
        if not cap.isOpened():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9' + b'\r\n')
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧（传递stream_id用于统计）
            processed_frame, _ = process_frame_behavior_detection(frame, stream_id=stream_id)
            
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    except Exception as e:
        print(f"生成帧错误: {e}")
    finally:
        if cap:
            cap.release()

# 路由
@app.route('/')
def index():
    return render_template('index.html')

# 人脸识别打卡相关API
@app.route('/api/face/register', methods=['POST'])
def register_face():
    """注册人脸"""
    try:
        if 'image' not in request.files or 'name' not in request.form:
            return jsonify({'success': False, 'message': '缺少必要参数'}), 400
        
        file = request.files['image']
        name = request.form['name']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 读取图片
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'success': False, 'message': '无法读取图片'}), 400
            
            if not FACE_RECOGNITION_AVAILABLE or resnet is None:
                return jsonify({'success': False, 'message': 'facenet-pytorch库未安装，请先安装该库'}), 400
            
            # 使用 YOLO 检测人脸
            faces = detect_faces_yolo(image)
            if len(faces) == 0:
                return jsonify({'success': False, 'message': '未检测到人脸'}), 400
            
            # 使用第一个检测到的人脸
            face_box = faces[0]['bbox']
            
            # 对齐人脸
            face_tensor = align_face(image, face_box)
            
            # 确保 face_tensor 是 4D (batch, channels, height, width)
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            # 提取人脸特征向量
            with torch.no_grad():
                # 确保 face_tensor 在正确的设备上
                device = next(resnet.parameters()).device
                face_encoding = resnet(face_tensor.to(device))
                # 归一化特征向量
                face_encoding = torch.nn.functional.normalize(face_encoding, p=2, dim=1)
                face_encoding = face_encoding.cpu().numpy()[0]
            
            # 保存人脸编码
            face_encodings_db[name] = face_encoding
            save_face_database()
            
            # 保存图片
            filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            filepath = os.path.join(FACES_FOLDER, filename)
            cv2.imwrite(filepath, image)
            
            return jsonify({
                'success': True,
                'message': f'成功注册 {name}',
                'name': name
            })
        
        return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/face/checkin', methods=['POST'])
def checkin_face():
    """人脸打卡"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': '缺少图片'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 读取图片
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'success': False, 'message': '无法读取图片'}), 400
            
            if not FACE_RECOGNITION_AVAILABLE or resnet is None:
                return jsonify({
                    'success': False,
                    'message': 'facenet-pytorch库未安装，请先安装该库',
                    'name': None
                })
            
            # 使用 YOLO 检测人脸
            faces = detect_faces_yolo(image)
            if len(faces) == 0:
                return jsonify({
                    'success': False,
                    'message': '未检测到人脸',
                    'name': None
                })
            
            # 使用第一个检测到的人脸
            face_box = faces[0]['bbox']
            
            # 对齐人脸
            face_tensor = align_face(image, face_box)
            
            # 确保 face_tensor 是 4D (batch, channels, height, width)
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            # 提取人脸特征向量
            with torch.no_grad():
                # 确保 face_tensor 在正确的设备上
                device = next(resnet.parameters()).device
                face_encoding = resnet(face_tensor.to(device))
                # 归一化特征向量
                face_encoding = torch.nn.functional.normalize(face_encoding, p=2, dim=1)
                face_encoding = face_encoding.cpu().numpy()[0]
            
            # 与数据库比较（使用余弦相似度）
            best_match = None
            best_similarity = -1.0
            
            for name, known_encoding in face_encodings_db.items():
                # 计算余弦相似度
                similarity = np.dot(face_encoding, known_encoding) / (
                    np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # 如果相似度大于0.6，认为是同一个人
            if best_similarity > 0.6:
                return jsonify({
                    'success': True,
                    'message': f'打卡成功: {best_match}',
                    'name': best_match,
                    'similarity': best_similarity,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '未识别到已注册的人脸',
                    'name': None,
                    'similarity': best_similarity
                })
        
        return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/face/list', methods=['GET'])
def list_faces():
    """列出所有已注册的人脸"""
    return jsonify({
        'success': True,
        'faces': list(face_encodings_db.keys()),
        'count': len(face_encodings_db)
    })

# 视频流相关API
@app.route('/api/stream/face/start', methods=['POST'])
def start_face_stream():
    """启动人脸检测视频流"""
    try:
        data = request.json
        source_type = data.get('type', 'video')  # 'video' or 'rtsp'
        source_path = data.get('path', '')
        
        if not source_path:
            return jsonify({'success': False, 'message': '缺少路径参数'}), 400
        
        stream_id = f"face_{int(time.time())}"
        
        return jsonify({
            'success': True,
            'stream_id': stream_id,
            'url': f'/api/stream/face/video?type={source_type}&path={source_path}&id={stream_id}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stream/face/video')
def face_video_stream():
    """人脸检测视频流"""
    source_type = request.args.get('type', 'video')
    source_path = request.args.get('path', '')
    stream_id = request.args.get('id', '')
    
    if source_type == 'rtsp':
        return Response(generate_frames_face('rtsp', source_path, stream_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif source_type == 'video':
        # 如果是视频文件，需要先上传
        if source_path.startswith('uploads/'):
            filepath = source_path
        else:
            filepath = os.path.join(UPLOAD_FOLDER, source_path)
        
        return Response(generate_frames_face('video', filepath, stream_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'error': '不支持的类型'}), 400

@app.route('/api/stream/behavior/start', methods=['POST'])
def start_behavior_stream():
    """启动行为检测视频流"""
    try:
        data = request.json
        source_type = data.get('type', 'video')
        source_path = data.get('path', '')
        
        if not source_path:
            return jsonify({'success': False, 'message': '缺少路径参数'}), 400
        
        stream_id = f"behavior_{int(time.time())}"
        
        return jsonify({
            'success': True,
            'stream_id': stream_id,
            'url': f'/api/stream/behavior/video?type={source_type}&path={source_path}&id={stream_id}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stream/behavior/video')
def behavior_video_stream():
    """行为检测视频流"""
    source_type = request.args.get('type', 'video')
    source_path = request.args.get('path', '')
    stream_id = request.args.get('id', '')
    
    if source_type == 'rtsp':
        return Response(generate_frames_behavior('rtsp', source_path, stream_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif source_type == 'video':
        if source_path.startswith('uploads/'):
            filepath = source_path
        else:
            filepath = os.path.join(UPLOAD_FOLDER, source_path)
        
        return Response(generate_frames_behavior('video', filepath, stream_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'error': '不支持的类型'}), 400

@app.route('/api/upload/video', methods=['POST'])
def upload_video():
    """上传视频文件"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': '缺少视频文件'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{int(time.time())}_{file.filename}")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'message': '上传成功',
                'filename': filename,
                'path': f'uploads/{filename}'
            })
        
        return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/upload/video/list', methods=['GET'])
def list_uploaded_videos():
    """列出所有已上传的视频文件"""
    try:
        videos = []
        upload_path = Path(UPLOAD_FOLDER)
        
        if upload_path.exists():
            # 获取所有视频文件
            video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
            for file_path in upload_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower().lstrip('.') in video_extensions:
                    file_stat = file_path.stat()
                    videos.append({
                        'filename': file_path.name,
                        'path': f'uploads/{file_path.name}',
                        'size': file_stat.st_size,
                        'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                        'modified_time': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # 按修改时间倒序排列（最新的在前）
            videos.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'videos': videos,
            'count': len(videos)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# 人脸检测统计API
@app.route('/api/face/statistics', methods=['GET'])
def get_face_statistics():
    """获取人脸检测统计"""
    stream_id = request.args.get('stream_id', '')
    
    if not stream_id:
        return jsonify({'success': False, 'message': '缺少stream_id参数'}), 400
    
    if stream_id in face_statistics:
        stats = face_statistics[stream_id]
        total = sum(stats.values())
        return jsonify({
            'success': True,
            'statistics': stats,
            'total': total,
            'stream_id': stream_id
        })
    else:
        return jsonify({
            'success': True,
            'statistics': {},
            'total': 0,
            'stream_id': stream_id
        })

@app.route('/api/face/list', methods=['GET'])
def get_face_detection_list():
    """获取检测到的人脸列表"""
    stream_id = request.args.get('stream_id', '')
    limit = request.args.get('limit', type=int)  # 可选：限制返回数量
    
    if not stream_id:
        return jsonify({'success': False, 'message': '缺少stream_id参数'}), 400
    
    if stream_id in face_detection_list:
        face_list = face_detection_list[stream_id]
        
        # 如果指定了limit，只返回最新的N条记录
        if limit and limit > 0:
            face_list = face_list[-limit:]
        
        return jsonify({
            'success': True,
            'faces': face_list,
            'count': len(face_list),
            'total': len(face_detection_list[stream_id]),
            'stream_id': stream_id
        })
    else:
        return jsonify({
            'success': True,
            'faces': [],
            'count': 0,
            'total': 0,
            'stream_id': stream_id
        })

@app.route('/api/face/statistics/reset', methods=['POST'])
def reset_face_statistics():
    """重置人脸检测统计"""
    data = request.json if request.is_json else {}
    stream_id = data.get('stream_id', '') or request.form.get('stream_id', '')
    
    if not stream_id:
        return jsonify({'success': False, 'message': '缺少stream_id参数'}), 400
    
    if stream_id in face_statistics:
        face_statistics[stream_id] = {}
        face_detection_list[stream_id] = []
        return jsonify({
            'success': True,
            'message': '统计已重置',
            'stream_id': stream_id
        })
    else:
        return jsonify({
            'success': True,
            'message': '统计不存在或已为空',
            'stream_id': stream_id
        })

# 行为检测统计API
@app.route('/api/behavior/statistics', methods=['GET'])
def get_behavior_statistics():
    """获取行为统计"""
    stream_id = request.args.get('stream_id', '')
    
    if not stream_id:
        return jsonify({'success': False, 'message': '缺少stream_id参数'}), 400
    
    if stream_id in behavior_statistics:
        stats = behavior_statistics[stream_id]
        total = sum(stats.values())
        return jsonify({
            'success': True,
            'statistics': stats,
            'total': total,
            'stream_id': stream_id
        })
    else:
        return jsonify({
            'success': True,
            'statistics': {},
            'total': 0,
            'stream_id': stream_id
        })

@app.route('/api/behavior/statistics/reset', methods=['POST'])
def reset_behavior_statistics():
    """重置行为统计"""
    data = request.json if request.is_json else {}
    stream_id = data.get('stream_id', '') or request.form.get('stream_id', '')
    
    if not stream_id:
        return jsonify({'success': False, 'message': '缺少stream_id参数'}), 400
    
    if stream_id in behavior_statistics:
        behavior_statistics[stream_id] = {}
        return jsonify({
            'success': True,
            'message': '统计已重置',
            'stream_id': stream_id
        })
    else:
        return jsonify({
            'success': True,
            'message': '统计不存在或已为空',
            'stream_id': stream_id
        })

if __name__ == '__main__':
    print("正在初始化模型...")
    init_models()
    print("模型初始化完成!")
    print("启动Flask服务器...")
    app.run(host='0.0.0.0', port=8188, debug=True, threaded=True)

