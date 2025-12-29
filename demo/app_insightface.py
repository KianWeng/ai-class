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
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import insightface
    from insightface.app import FaceAnalysis
    FACE_RECOGNITION_AVAILABLE = True
    # 初始化两个 ArcFace 模型
    # 1. 检测模型：用于实时人脸检测（det_size=1280x1280，检测精度更高）
    # 2. 注册模型：用于注册和特征提取（det_size=640x640，速度更快）
    try:
        # 检测模型
        face_analysis_detect = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_analysis_detect.prepare(ctx_id=1, det_size=(1280, 1280))
        if not hasattr(face_analysis_detect, 'get') or face_analysis_detect.get is None:
            raise AttributeError("face_analysis_detect.get 方法不可用，检测模型初始化失败")
        print(f"✓ 检测模型已加载 (det_size=1280x1280)")
        
        # 注册模型
        face_analysis_register = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_analysis_register.prepare(ctx_id=1, det_size=(640, 640))
        if not hasattr(face_analysis_register, 'get') or face_analysis_register.get is None:
            raise AttributeError("face_analysis_register.get 方法不可用，注册模型初始化失败")
        print(f"✓ 注册模型已加载 (det_size=640x640)")
        
        # 为了兼容性，保留face_analysis指向检测模型
        face_analysis = face_analysis_detect
        
        if torch.cuda.is_available():
            print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    except Exception as init_error:
        print(f"警告: ArcFace 模型初始化失败: {init_error}")
        import traceback
        traceback.print_exc()
        FACE_RECOGNITION_AVAILABLE = False
        face_analysis = None
        face_analysis_detect = None
        face_analysis_register = None
except ImportError as e:
    print(f"警告: insightface库未安装，人脸识别功能将受限: {e}")
    FACE_RECOGNITION_AVAILABLE = False
    face_analysis = None
    face_analysis_detect = None
    face_analysis_register = None
except Exception as e:
    print(f"警告: insightface库导入失败: {e}")
    import traceback
    traceback.print_exc()
    FACE_RECOGNITION_AVAILABLE = False
    face_analysis = None
    face_analysis_detect = None
    face_analysis_register = None
from pathlib import Path
import pickle

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}
FACES_FOLDER = 'faces_database'
MODEL_BEHAVIOR_PATH = 'weights/scb/bth_best.pt'  # 行为检测模型（如果存在）- 低头、转头
MODEL_BEHAVIOR_HRW_PATH = 'weights/scb/hrw_best.pt'  # 行为检测模型（如果存在）- 举手、阅读、写字

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 创建必要的文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# 全局变量
behavior_model = None  # 低头、转头模型
behavior_model_hrw = None  # 举手、阅读、写字模型
face_encodings_db = {}  # 存储人脸编码 {name: encoding}
rtsp_streams = {}  # 存储RTSP流 {stream_id: cap}
video_streams = {}  # 存储视频流 {stream_id: cap}
behavior_statistics = {}  # 存储行为统计 {stream_id: {class_name: count}}

# 初始化模型
def init_models():
    global behavior_model, behavior_model_hrw
    
    # InsightFace模型已在导入时初始化，这里只需要检查是否可用
    if FACE_RECOGNITION_AVAILABLE and face_analysis_detect is not None and face_analysis_register is not None:
        print("✓ InsightFace检测模型已就绪 (det_size=1280x1280)")
        print("✓ InsightFace注册模型已就绪 (det_size=640x640)")
    else:
        print("⚠ InsightFace模型不可用，人脸识别功能将受限")
    
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
            loaded_db = pickle.load(f)
        
        print(f"从文件加载了 {len(loaded_db)} 个已注册人脸")
        
        # 清理和验证数据库
        face_encodings_db = {}
        invalid_count = 0
        
        for name, encoding in loaded_db.items():
            try:
                # 如果是numpy数组
                if isinstance(encoding, np.ndarray):
                    # 检查是否是0维数组或object类型
                    if encoding.shape == () or encoding.dtype == object:
                        # 尝试从0维数组中提取实际值
                        if encoding.shape == ():
                            actual_value = encoding.item()
                            if actual_value is None:
                                invalid_count += 1
                                continue
                            encoding = np.array(actual_value)
                        else:
                            encoding = np.array(encoding, dtype=np.float32)
                    
                    # 检查维度
                    if encoding.ndim == 0:
                        invalid_count += 1
                        continue
                    
                    # 确保是一维数组
                    if encoding.ndim > 1:
                        encoding = encoding.flatten()
                    
                    # 检查是否为空
                    if encoding.size == 0:
                        invalid_count += 1
                        continue
                    
                    # 转换为float32并归一化
                    encoding = encoding.astype(np.float32)
                    norm = np.linalg.norm(encoding)
                    if norm > 0:
                        encoding = encoding / norm
                        face_encodings_db[name] = encoding
                    else:
                        invalid_count += 1
                else:
                    # 不是numpy数组，尝试转换
                    try:
                        encoding_array = np.array(encoding, dtype=np.float32)
                        if encoding_array.size == 0 or encoding_array.ndim == 0:
                            invalid_count += 1
                            continue
                        
                        # 归一化
                        norm = np.linalg.norm(encoding_array)
                        if norm > 0:
                            encoding_array = encoding_array / norm
                            face_encodings_db[name] = encoding_array
                        else:
                            invalid_count += 1
                    except Exception:
                        invalid_count += 1
            except Exception:
                invalid_count += 1
        
        # 如果有无效记录，保存清理后的数据库
        if invalid_count > 0:
            save_face_database()
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

def detect_faces_insightface(image):
    """使用InsightFace检测人脸"""
    if not FACE_RECOGNITION_AVAILABLE or face_analysis_detect is None:
        return []
    
    try:
        # 使用检测模型检测人脸
        faces = face_analysis_detect.get(image)
        results = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            results.append({
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'confidence': float(face.det_score),
                'kps': face.kps.astype(int).tolist() if hasattr(face, 'kps') and face.kps is not None else None
            })
        
        return results
    except Exception as e:
        print(f"[警告] InsightFace检测失败: {e}")
        return []

def align_face_arcface(img, face_obj):
    """
    使用InsightFace的face对象直接获取对齐后的人脸
    :param img: 原始 BGR 图像
    :param face_obj: InsightFace检测到的face对象
    :return: 对齐后的人脸图像 (112, 112, 3) BGR格式
    """
    if face_obj is None:
        return None
    
    try:
        # 使用InsightFace的对齐功能，直接获取112x112的对齐人脸
        aligned_face = face_obj.norm_crop(img)
        return aligned_face
    except Exception as e:
        print(f"[警告] 人脸对齐失败: {e}")
        # 如果对齐失败，使用bbox裁剪
        bbox = face_obj.bbox.astype(int)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), min(w, int(bbox[2])), min(h, int(bbox[3]))
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        return cv2.resize(face_img, (112, 112))

def extract_features_from_faces(face_objects):
    """从InsightFace检测到的face对象直接提取特征向量"""
    if not FACE_RECOGNITION_AVAILABLE:
        return []
    
    if len(face_objects) == 0:
        return []
    
    try:
        face_encodings = []
        
        for i, face_obj in enumerate(face_objects):
            if face_obj is None:
                face_encodings.append(None)
                continue
            
            try:
                embedding = None
                
                # 尝试多种可能的属性名（不同版本的insightface可能使用不同的属性名）
                # 优先级: normed_embedding > embedding > norm_embedding
                if hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is not None:
                    embedding = face_obj.normed_embedding
                elif hasattr(face_obj, 'embedding') and face_obj.embedding is not None:
                    embedding = face_obj.embedding
                elif hasattr(face_obj, 'norm_embedding') and face_obj.norm_embedding is not None:
                    embedding = face_obj.norm_embedding
                else:
                    face_encodings.append(None)
                    continue
                
                if embedding is not None:
                    # 确保是numpy数组
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding)
                    
                    # 计算范数
                    norm = np.linalg.norm(embedding)
                    
                    # 如果范数不为1，需要归一化（除非使用的是normed_embedding）
                    if hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is embedding:
                        # 已经归一化，但检查一下
                        if abs(norm - 1.0) > 0.01:
                            if norm > 0:
                                embedding = embedding / norm
                    else:
                        # 需要归一化
                        if norm > 0:
                            embedding = embedding / norm
                        else:
                            face_encodings.append(None)
                            continue
                    
                    face_encodings.append(embedding)
                else:
                    face_encodings.append(None)
            except Exception:
                face_encodings.append(None)
        
        return face_encodings
    except Exception:
        return []

def recognize_face_from_encoding(face_encoding):
    """从特征向量识别人脸"""
    if face_encoding is None:
        return None, 0.0
    
    try:
        # 检查数据库是否为空
        if len(face_encodings_db) == 0:
            return None, 0.0
        
        # 与数据库中的所有人脸比较（使用余弦相似度）
        best_match = None
        best_similarity = -1.0
        
        for name, known_encoding in face_encodings_db.items():
            # 验证数据库中的特征向量
            if known_encoding is None:
                continue
            
            # 确保是numpy数组
            if not isinstance(known_encoding, np.ndarray):
                try:
                    known_encoding = np.array(known_encoding, dtype=np.float32)
                except Exception:
                    continue
            
            # 检查是否是有效的数组
            if known_encoding.shape == () or known_encoding.dtype == object:
                continue
            
            if known_encoding.size == 0:
                continue
            
            # 确保输入特征向量也是numpy数组
            if not isinstance(face_encoding, np.ndarray):
                face_encoding = np.array(face_encoding, dtype=np.float32)
            
            # 检查维度是否匹配
            if face_encoding.shape != known_encoding.shape:
                similarity = 0.0
            else:
                # 计算点积
                dot_product = np.dot(face_encoding, known_encoding)
                norm_input = np.linalg.norm(face_encoding)
                norm_known = np.linalg.norm(known_encoding)
                
                # 计算余弦相似度
                if norm_input > 0 and norm_known > 0:
                    similarity = dot_product / (norm_input * norm_known)
                else:
                    similarity = 0.0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # ArcFace通常使用0.6-0.7的阈值，这里使用0.6
        # 如果相似度大于0.6，认为是同一个人
        if best_similarity > 0.4:
            return best_match, best_similarity
        
        return None, best_similarity  # 返回最高相似度，即使未达到阈值
    except Exception:
        return None, 0.0

def detect_behaviors(image, stream_id=None):
    """检测学生行为（支持两个模型）"""
    behaviors = []
    
    # 模型1：低头、转头
    if behavior_model is not None:
        detection_start_time = time.time()
        results = behavior_model.predict(image, conf=0.25, verbose=False)
        detection_time = time.time() - detection_start_time
        
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
        detection_start_time = time.time()
        results = behavior_model_hrw.predict(image, conf=0.25, verbose=False)
        detection_time = time.time() - detection_start_time
        
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
    
    if len(behaviors) > 0:
        print(f"[行为检测] 检测到 {len(behaviors)} 个行为")
        for i, behavior in enumerate(behaviors):
            print(f"  - 行为{i+1}: {behavior['class']} | 置信度: {behavior['confidence']:.4f}")
    
    return behaviors


def process_frame_face_detection(frame):
    """处理帧进行人脸检测和识别（InsightFace检测 -> 直接提取特征 -> 识别）"""
    total_start_time = time.time()
    
    # 步骤1: 使用检测模型检测人脸
    detection_start_time = time.time()
    face_objects = face_analysis_detect.get(frame) if FACE_RECOGNITION_AVAILABLE and face_analysis_detect is not None else []
    detection_time = time.time() - detection_start_time
    
    # 绘制结果
    annotated_frame = frame.copy()
    results_data = []
    
    if len(face_objects) == 0:
        total_time = time.time() - total_start_time
        print(f"[人脸检测] 未检测到人脸 | 总耗时: {total_time*1000:.2f}ms")
        return annotated_frame, results_data
    
    # 步骤2: 直接从face对象提取特征（InsightFace已经包含特征向量）
    feature_extraction_start_time = time.time()
    face_encodings = extract_features_from_faces(face_objects)
    feature_extraction_time = time.time() - feature_extraction_start_time
    
    # 步骤3: 与数据库比较（批量识别）
    recognition_start_time = time.time()
    recognition_results = []
    
    for i, (face_obj, encoding) in enumerate(zip(face_objects, face_encodings)):
        bbox = face_obj.bbox.astype(int)
        confidence = float(face_obj.det_score)
        
        if encoding is not None:
            name, similarity = recognize_face_from_encoding(encoding)
        else:
            name, similarity = None, 0.0
        
        recognition_results.append({
            'index': i,
            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            'confidence': confidence,
            'name': name,
            'similarity': similarity
        })
    
    recognition_time = time.time() - recognition_start_time
    total_time = time.time() - total_start_time
    
    # 打印检测结果和耗时信息
    recognized_count = sum(1 for r in recognition_results if r['name'] is not None)
    print(f"[人脸检测] 检测到 {len(face_objects)} 个人脸 | "
          f"识别成功: {recognized_count} | "
          f"检测耗时: {detection_time*1000:.2f}ms | "
          f"特征提取耗时: {feature_extraction_time*1000:.2f}ms | "
          f"识别耗时: {recognition_time*1000:.2f}ms | "
          f"总耗时: {total_time*1000:.2f}ms")
    
    # 打印每个检测到的人脸详情
    for i, result in enumerate(recognition_results):
        if result['name']:
            print(f"  - 人脸{i+1}: {result['name']} | 相似度: {result['similarity']:.4f} | 检测置信度: {result['confidence']:.4f}")
        else:
            print(f"  - 人脸{i+1}: 未知 | 最高相似度: {result['similarity']:.4f} | 检测置信度: {result['confidence']:.4f}")
    
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
    
    
    return annotated_frame, results_data

def process_frame_behavior_detection(frame, stream_id=None):
    """处理帧进行行为检测"""
    total_start_time = time.time()
    behaviors = detect_behaviors(frame, stream_id=stream_id)
    total_time = time.time() - total_start_time
    
    # 如果没有检测到行为，也打印信息
    if len(behaviors) == 0:
        print(f"[行为检测] 未检测到行为 | 总耗时: {total_time*1000:.2f}ms")
    else:
        # 检测到行为时，打印总耗时信息（detect_behaviors已经打印了详细信息）
        print(f"[行为检测] 总耗时: {total_time*1000:.2f}ms")
    
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
            
            # 处理帧
            processed_frame, _ = process_frame_face_detection(frame)
            
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
        # 清理统计（可选，如果需要保留统计可以注释掉）
        # if stream_id and stream_id in behavior_statistics:
        #     del behavior_statistics[stream_id]

# 路由
@app.route('/')
def index():
    return render_template('index.html')

# 人脸识别打卡相关API
@app.route('/api/face/register', methods=['POST'])
def register_face():
    """注册人脸"""
    total_start_time = time.time()
    try:
        if 'image' not in request.files or 'name' not in request.form:
            return jsonify({'success': False, 'message': '缺少必要参数'}), 400
        
        file = request.files['image']
        name = request.form['name']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 读取图片
            read_start_time = time.time()
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            read_time = time.time() - read_start_time
            
            if image is None:
                return jsonify({'success': False, 'message': '无法读取图片'}), 400
            
            if not FACE_RECOGNITION_AVAILABLE or face_analysis_register is None:
                return jsonify({'success': False, 'message': 'insightface库未安装，请先安装该库'}), 400
            
            # 检查图像尺寸
            h, w = image.shape[:2]
            min_size = 128  # 最小尺寸要求（像素）
            original_size = (w, h)
            resized = False
            
            # 如果图像太小，尝试放大
            if w < min_size or h < min_size:
                # 计算放大比例，使最小边至少达到min_size
                scale = max(min_size / w, min_size / h) * 1.5  # 放大1.5倍以确保足够大
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized = True
                print(f"[人脸注册] 图像尺寸过小 ({w}x{h})，已放大至 {new_w}x{new_h}")
            
            # 使用注册模型检测人脸并提取特征
            detection_start_time = time.time()
            face_objects = face_analysis_register.get(image)
            detection_time = time.time() - detection_start_time
            
            if len(face_objects) == 0:
                total_time = time.time() - total_start_time
                size_info = f"原始尺寸: {original_size[0]}x{original_size[1]}"
                if resized:
                    size_info += f", 放大后: {image.shape[1]}x{image.shape[0]}"
                print(f"[人脸注册] 姓名: {name} | 未检测到人脸 | {size_info} | 总耗时: {total_time*1000:.2f}ms")
                return jsonify({
                    'success': False, 
                    'message': f'未检测到人脸。图像尺寸: {original_size[0]}x{original_size[1]}。建议使用至少128x128像素的清晰人脸照片。'
                }), 400
            
            # 使用第一个检测到的人脸
            face_obj = face_objects[0]
            bbox = face_obj.bbox.astype(int)
            confidence = float(face_obj.det_score)
            
            # 直接从face对象提取特征向量（尝试多种可能的属性名）
            extraction_start_time = time.time()
            face_encoding = None
            if hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is not None:
                face_encoding = face_obj.normed_embedding
            elif hasattr(face_obj, 'embedding') and face_obj.embedding is not None:
                face_encoding = face_obj.embedding
            elif hasattr(face_obj, 'norm_embedding') and face_obj.norm_embedding is not None:
                face_encoding = face_obj.norm_embedding
            else:
                total_time = time.time() - total_start_time
                print(f"[人脸注册] 姓名: {name} | 无法提取人脸特征 | 总耗时: {total_time*1000:.2f}ms")
                return jsonify({'success': False, 'message': '无法提取人脸特征'}), 400
            
            # 确保是numpy数组
            if not isinstance(face_encoding, np.ndarray):
                face_encoding = np.array(face_encoding)
            
            # 归一化（如果不是normed_embedding）
            if not (hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is face_encoding):
                norm = np.linalg.norm(face_encoding)
                if norm > 0:
                    face_encoding = face_encoding / norm
                else:
                    total_time = time.time() - total_start_time
                    print(f"[人脸注册] 姓名: {name} | 特征向量范数为0 | 总耗时: {total_time*1000:.2f}ms")
                    return jsonify({'success': False, 'message': '特征向量范数为0'}), 400
            
            extraction_time = time.time() - extraction_start_time
            
            # 保存人脸编码
            save_start_time = time.time()
            face_encodings_db[name] = face_encoding
            save_face_database()
            save_time = time.time() - save_start_time
            
            # 保存图片
            filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            filepath = os.path.join(FACES_FOLDER, filename)
            cv2.imwrite(filepath, image)
            
            total_time = time.time() - total_start_time
            
            print(f"[人脸注册] 姓名: {name} | 注册成功 | "
                  f"检测到人脸: 1个 | 检测置信度: {confidence:.4f} | "
                  f"图片读取耗时: {read_time*1000:.2f}ms | "
                  f"人脸检测耗时: {detection_time*1000:.2f}ms | "
                  f"特征提取耗时: {extraction_time*1000:.2f}ms | "
                  f"保存耗时: {save_time*1000:.2f}ms | "
                  f"总耗时: {total_time*1000:.2f}ms | "
                  f"人脸位置: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            
            return jsonify({
                'success': True,
                'message': f'成功注册 {name}',
                'name': name
            })
        
        return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
    
    except Exception as e:
        total_time = time.time() - total_start_time
        print(f"[人脸注册] 注册失败 | 错误: {str(e)} | 总耗时: {total_time*1000:.2f}ms")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/face/checkin', methods=['POST'])
def checkin_face():
    """人脸打卡"""
    total_start_time = time.time()
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': '缺少图片'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 读取图片
            read_start_time = time.time()
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            read_time = time.time() - read_start_time
            
            if image is None:
                total_time = time.time() - total_start_time
                print(f"[人脸打卡] 无法读取图片 | 总耗时: {total_time*1000:.2f}ms")
                return jsonify({'success': False, 'message': '无法读取图片'}), 400
            
            if not FACE_RECOGNITION_AVAILABLE or face_analysis_register is None:
                total_time = time.time() - total_start_time
                print(f"[人脸打卡] insightface库未安装 | 总耗时: {total_time*1000:.2f}ms")
                return jsonify({
                    'success': False,
                    'message': 'insightface库未安装，请先安装该库',
                    'name': None
                })
            
            # 检查图像尺寸
            h, w = image.shape[:2]
            min_size = 128  # 最小尺寸要求（像素）
            original_size = (w, h)
            resized = False
            
            # 如果图像太小，尝试放大
            if w < min_size or h < min_size:
                # 计算放大比例，使最小边至少达到min_size
                scale = max(min_size / w, min_size / h) * 1.5  # 放大1.5倍以确保足够大
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized = True
                print(f"[人脸打卡] 图像尺寸过小 ({w}x{h})，已放大至 {new_w}x{new_h}")
            
            # 使用注册模型检测人脸并提取特征
            detection_start_time = time.time()
            face_objects = face_analysis_register.get(image)
            detection_time = time.time() - detection_start_time
            
            if len(face_objects) == 0:
                total_time = time.time() - total_start_time
                size_info = f"原始尺寸: {original_size[0]}x{original_size[1]}"
                if resized:
                    size_info += f", 放大后: {image.shape[1]}x{image.shape[0]}"
                print(f"[人脸打卡] 未检测到人脸 | {size_info} | "
                      f"图片读取耗时: {read_time*1000:.2f}ms | "
                      f"人脸检测耗时: {detection_time*1000:.2f}ms | "
                      f"总耗时: {total_time*1000:.2f}ms")
                return jsonify({
                    'success': False,
                    'message': f'未检测到人脸。图像尺寸: {original_size[0]}x{original_size[1]}。建议使用至少128x128像素的清晰人脸照片。',
                    'name': None
                })
            
            # 使用第一个检测到的人脸
            face_obj = face_objects[0]
            bbox = face_obj.bbox.astype(int)
            confidence = float(face_obj.det_score)
            
            # 直接从face对象提取特征向量（尝试多种可能的属性名）
            extraction_start_time = time.time()
            face_encoding = None
            if hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is not None:
                face_encoding = face_obj.normed_embedding
            elif hasattr(face_obj, 'embedding') and face_obj.embedding is not None:
                face_encoding = face_obj.embedding
            elif hasattr(face_obj, 'norm_embedding') and face_obj.norm_embedding is not None:
                face_encoding = face_obj.norm_embedding
            else:
                total_time = time.time() - total_start_time
                print(f"[人脸打卡] 无法提取人脸特征 | "
                      f"检测到人脸: {len(face_objects)}个 | 检测置信度: {confidence:.4f} | "
                      f"图片读取耗时: {read_time*1000:.2f}ms | "
                      f"人脸检测耗时: {detection_time*1000:.2f}ms | "
                      f"总耗时: {total_time*1000:.2f}ms")
                return jsonify({
                    'success': False,
                    'message': '无法提取人脸特征',
                    'name': None
                })
            
            # 确保是numpy数组
            if not isinstance(face_encoding, np.ndarray):
                face_encoding = np.array(face_encoding)
            
            # 归一化（如果不是normed_embedding）
            if not (hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is face_encoding):
                norm = np.linalg.norm(face_encoding)
                if norm > 0:
                    face_encoding = face_encoding / norm
                else:
                    total_time = time.time() - total_start_time
                    print(f"[人脸打卡] 特征向量范数为0 | "
                          f"检测到人脸: {len(face_objects)}个 | 检测置信度: {confidence:.4f} | "
                          f"图片读取耗时: {read_time*1000:.2f}ms | "
                          f"人脸检测耗时: {detection_time*1000:.2f}ms | "
                          f"特征提取耗时: {(time.time() - extraction_start_time)*1000:.2f}ms | "
                          f"总耗时: {total_time*1000:.2f}ms")
                    return jsonify({
                        'success': False,
                        'message': '特征向量范数为0',
                        'name': None
                    })
            
            extraction_time = time.time() - extraction_start_time
            
            # 与数据库比较（使用余弦相似度）
            recognition_start_time = time.time()
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
            
            recognition_time = time.time() - recognition_start_time
            total_time = time.time() - total_start_time
            
            # 如果相似度大于0.6，认为是同一个人
            if best_similarity > 0.6:
                print(f"[人脸打卡] 打卡成功: {best_match} | "
                      f"检测到人脸: {len(face_objects)}个 | 检测置信度: {confidence:.4f} | "
                      f"相似度: {best_similarity:.4f} | "
                      f"图片读取耗时: {read_time*1000:.2f}ms | "
                      f"人脸检测耗时: {detection_time*1000:.2f}ms | "
                      f"特征提取耗时: {extraction_time*1000:.2f}ms | "
                      f"识别耗时: {recognition_time*1000:.2f}ms | "
                      f"总耗时: {total_time*1000:.2f}ms | "
                      f"人脸位置: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                return jsonify({
                    'success': True,
                    'message': f'打卡成功: {best_match}',
                    'name': best_match,
                    'similarity': best_similarity,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                print(f"[人脸打卡] 未识别到已注册的人脸 | "
                      f"检测到人脸: {len(face_objects)}个 | 检测置信度: {confidence:.4f} | "
                      f"最高相似度: {best_similarity:.4f} | "
                      f"图片读取耗时: {read_time*1000:.2f}ms | "
                      f"人脸检测耗时: {detection_time*1000:.2f}ms | "
                      f"特征提取耗时: {extraction_time*1000:.2f}ms | "
                      f"识别耗时: {recognition_time*1000:.2f}ms | "
                      f"总耗时: {total_time*1000:.2f}ms | "
                      f"人脸位置: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                return jsonify({
                    'success': False,
                    'message': '未识别到已注册的人脸',
                    'name': None,
                    'similarity': best_similarity
                })
        
        return jsonify({'success': False, 'message': '不支持的文件格式'}), 400
    
    except Exception as e:
        total_time = time.time() - total_start_time
        print(f"[人脸打卡] 打卡失败 | 错误: {str(e)} | 总耗时: {total_time*1000:.2f}ms")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/face/list', methods=['GET'])
def list_faces():
    """列出所有已注册的人脸"""
    return jsonify({
        'success': True,
        'faces': list(face_encodings_db.keys()),
        'count': len(face_encodings_db)
    })

@app.route('/api/face/delete', methods=['POST'])
def delete_face():
    """删除已注册的人脸"""
    total_start_time = time.time()
    try:
        data = request.json
        if not data or 'name' not in data:
            return jsonify({'success': False, 'message': '缺少必要参数：name'}), 400
        
        name = data['name']
        
        # 检查人脸是否存在
        if name not in face_encodings_db:
            total_time = time.time() - total_start_time
            print(f"[人脸删除] 姓名: {name} | 未找到该人脸 | 总耗时: {total_time*1000:.2f}ms")
            return jsonify({'success': False, 'message': f'未找到名为 {name} 的已注册人脸'}), 404
        
        # 从数据库中删除
        delete_start_time = time.time()
        del face_encodings_db[name]
        save_face_database()
        delete_time = time.time() - delete_start_time
        
        # 尝试删除对应的图片文件
        deleted_images = []
        try:
            if os.path.exists(FACES_FOLDER):
                # 查找所有以该姓名开头的图片文件
                for filename in os.listdir(FACES_FOLDER):
                    if filename.startswith(f"{name}_") and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(FACES_FOLDER, filename)
                        try:
                            os.remove(filepath)
                            deleted_images.append(filename)
                            print(f"[人脸删除] 已删除图片文件: {filename}")
                        except Exception as e:
                            print(f"[人脸删除] 删除图片文件失败 {filename}: {e}")
        except Exception as e:
            print(f"[人脸删除] 查找图片文件时出错: {e}")
        
        total_time = time.time() - total_start_time
        
        print(f"[人脸删除] 姓名: {name} | 删除成功 | "
              f"删除耗时: {delete_time*1000:.2f}ms | "
              f"总耗时: {total_time*1000:.2f}ms | "
              f"删除的图片文件数: {len(deleted_images)}")
        
        return jsonify({
            'success': True,
            'message': f'成功删除 {name}',
            'name': name,
            'deleted_images': deleted_images,
            'deleted_images_count': len(deleted_images)
        })
    
    except Exception as e:
        total_time = time.time() - total_start_time
        print(f"[人脸删除] 删除失败 | 错误: {str(e)} | 总耗时: {total_time*1000:.2f}ms")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

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
    """
    启动行为检测视频流
    支持两种视频源：
    1. RTSP流：source_type='rtsp', source_path为RTSP流地址
    2. 已上传的视频文件：source_type='video', source_path为文件路径（可通过/api/upload/video/list获取已上传文件列表）
    """
    try:
        data = request.json
        source_type = data.get('type', 'video')
        source_path = data.get('path', '')
        
        if not source_path:
            return jsonify({
                'success': False, 
                'message': '缺少路径参数。对于视频文件，请使用 /api/upload/video/list 获取已上传的文件列表'
            }), 400
        
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
    """行为检测视频流（支持RTSP流和已上传的视频文件）"""
    source_type = request.args.get('type', 'video')
    source_path = request.args.get('path', '')
    stream_id = request.args.get('id', '')
    
    if source_type == 'rtsp':
        return Response(generate_frames_behavior('rtsp', source_path, stream_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif source_type == 'video':
        # 处理视频文件路径（支持已上传的文件列表中的文件）
        if source_path.startswith('uploads/'):
            filepath = source_path
        else:
            filepath = os.path.join(UPLOAD_FOLDER, source_path)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            return jsonify({
                'error': f'视频文件不存在: {filepath}',
                'message': '请确保文件已上传，或从已上传文件列表中选择'
            }), 404
        
        # 检查是否为有效的视频文件
        if not os.path.isfile(filepath):
            return jsonify({
                'error': f'无效的视频文件路径: {filepath}',
                'message': '请选择有效的视频文件'
            }), 400
        
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
    """
    列出所有已上传的视频文件
    该接口返回的视频文件列表可用于：
    1. 人脸检测视频流（/api/stream/face/start）
    2. 行为检测视频流（/api/stream/behavior/start）
    """
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

