#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智慧课堂系统 - Flask后端应用
包含人脸识别打卡、实时人脸检测、学生行为检测功能
"""

# # 设置库路径（解决 dlib CUDNN 问题）
# # 必须在导入任何可能使用 dlib 的模块之前设置
import os
conda_env_lib = '/data/miniconda3/envs/yolo_v11/lib'
if os.path.exists(conda_env_lib):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if conda_env_lib not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f'{conda_env_lib}:{current_ld_path}'
        # 注意：在 Python 中修改 LD_LIBRARY_PATH 对已加载的库无效
        # 需要使用 ctypes 重新加载动态链接器
        try:
            import ctypes
            if hasattr(ctypes, 'CDLL'):
                # 尝试预加载 CUDNN 库
                cudnn_lib = os.path.join(conda_env_lib, 'libcudnn.so.9')
                if os.path.exists(cudnn_lib):
                    try:
                        ctypes.CDLL(cudnn_lib, mode=ctypes.RTLD_GLOBAL)
                    except:
                        pass
        except:
            pass
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

# # 在导入 face_recognition (dlib) 之前预加载 CUDNN 库
# # 解决 dlib 编译时链接 CUDNN 但运行时找不到符号的问题
# try:
#     import ctypes
    
#     conda_env_lib = '/data/miniconda3/envs/yolo_v11/lib'
#     if os.path.exists(conda_env_lib):
#         # 预加载主要的 CUDNN 库
#         cudnn_lib = os.path.join(conda_env_lib, 'libcudnn.so.9')
#         if os.path.exists(cudnn_lib):
#             try:
#                 ctypes.CDLL(cudnn_lib, mode=ctypes.RTLD_GLOBAL)
#             except:
#                 pass  # 忽略加载错误
# except:
#     pass  # 如果预加载失败，继续执行

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("警告: face_recognition库未安装，人脸识别功能将受限")
    FACE_RECOGNITION_AVAILABLE = False
except Exception as e:
    print(f"警告: face_recognition库导入失败: {e}")
    FACE_RECOGNITION_AVAILABLE = False
from pathlib import Path
import pickle

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}
FACES_FOLDER = 'faces_database'
MODEL_FACE_PATH = 'weights/face_detect/best.pt'  # 人脸检测模型
MODEL_BEHAVIOR_PATH = 'weights/scb/best.pt'  # 行为检测模型（如果存在）

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 创建必要的文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# 全局变量
face_model = None
behavior_model = None
face_encodings_db = {}  # 存储人脸编码 {name: encoding}
rtsp_streams = {}  # 存储RTSP流 {stream_id: cap}
video_streams = {}  # 存储视频流 {stream_id: cap}

# 初始化模型
def init_models():
    global face_model, behavior_model
    
    # 加载人脸检测模型
    if os.path.exists(MODEL_FACE_PATH):
        print(f"加载人脸检测模型: {MODEL_FACE_PATH}")
        face_model = YOLO(MODEL_FACE_PATH)
    else:
        print(f"警告: 人脸检测模型不存在: {MODEL_FACE_PATH}")
        face_model = YOLO('yolo11n.pt')  # 使用默认模型
    
    # 加载行为检测模型
    if os.path.exists(MODEL_BEHAVIOR_PATH):
        print(f"加载行为检测模型: {MODEL_BEHAVIOR_PATH}")
        behavior_model = YOLO(MODEL_BEHAVIOR_PATH)
    else:
        print(f"警告: 行为检测模型不存在: {MODEL_BEHAVIOR_PATH}，将使用默认模型")
        behavior_model = None
    
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

def recognize_face(face_image):
    """识别人脸"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, 0.0
    
    try:
        # 转换颜色空间
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # 获取人脸编码
        encodings = face_recognition.face_encodings(rgb_image)
        
        if len(encodings) == 0:
            return None, 0.0
        
        face_encoding = encodings[0]
        
        # 与数据库中的所有人脸比较
        best_match = None
        best_distance = 1.0
        
        for name, known_encoding in face_encodings_db.items():
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # 如果距离小于0.6，认为是同一个人
        if best_distance < 0.6:
            return best_match, 1.0 - best_distance
        
        return None, 0.0
    except Exception as e:
        print(f"人脸识别错误: {e}")
        return None, 0.0

def detect_behaviors(image):
    """检测学生行为"""
    if behavior_model is None:
        return []
    
    results = behavior_model.predict(image, conf=0.25, verbose=False)
    behaviors = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # 根据类别名称
            class_names = ['BowHead', 'TurnHead'] if behavior_model.names else ['行为1', '行为2']
            class_name = class_names[cls] if cls < len(class_names) else f'行为{cls}'
            
            behaviors.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class': class_name
            })
    
    return behaviors

def _recognize_single_face(face_roi, face_index, face_bbox, confidence):
    """并行识别人脸的辅助函数"""
    recognition_start_time = time.time()
    name, similarity = recognize_face(face_roi)
    recognition_time = time.time() - recognition_start_time
    
    return {
        'index': face_index,
        'bbox': face_bbox,
        'confidence': confidence,
        'name': name,
        'similarity': similarity,
        'recognition_time': recognition_time
    }

def process_frame_face_detection(frame):
    """处理帧进行人脸检测和识别（并行识别）"""
    # 检测人脸 - 记录检测时间
    detection_start_time = time.time()
    faces = detect_faces_yolo(frame)
    detection_time = time.time() - detection_start_time
    print(f"[人脸检测] 检测时间: {detection_time*1000:.2f} ms, 检测到 {len(faces)} 个人脸")
    
    # 绘制结果
    annotated_frame = frame.copy()
    results_data = []
    
    if len(faces) == 0:
        return annotated_frame, results_data
    
    # 并行识别人脸 - 记录总识别时间
    recognition_start_time = time.time()
    
    # 准备人脸区域数据
    face_tasks = []
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face['bbox']
        face_roi = frame[y1:y2, x1:x2]
        face_tasks.append((face_roi, i, face['bbox'], face['confidence']))
    
    # 使用线程池并行识别（最多使用4个线程，避免过多线程导致性能下降）
    max_workers = min(4, len(faces))
    recognition_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有识别任务
        future_to_face = {
            executor.submit(_recognize_single_face, face_roi, idx, bbox, conf): idx 
            for face_roi, idx, bbox, conf in face_tasks
        }
        
        # 收集结果
        for future in as_completed(future_to_face):
            try:
                result = future.result()
                recognition_results.append(result)
                print(f"[人脸识别] 人脸 {result['index']+1} 识别时间: {result['recognition_time']*1000:.2f} ms, 结果: {result['name'] if result['name'] else '未知'}")
            except Exception as e:
                print(f"[人脸识别] 识别错误: {e}")
    
    # 按原始索引排序，保持顺序
    recognition_results.sort(key=lambda x: x['index'])
    
    total_recognition_time = time.time() - recognition_start_time
    
    # 绘制结果
    for result in recognition_results:
        x1, y1, x2, y2 = result['bbox']
        confidence = result['confidence']
        name = result['name']
        similarity = result['similarity']
        
        # 绘制边界框
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{name if name else '未知'}: {similarity:.2f}" if name else f"未知: {confidence:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        results_data.append({
            'bbox': result['bbox'],
            'confidence': confidence,
            'name': name,
            'similarity': similarity
        })
    
    # 打印总时间统计
    if len(recognition_results) > 0:
        avg_single_recognition_time = sum(r['recognition_time'] for r in recognition_results) / len(recognition_results)
        print(f"[统计] 并行识别总时间: {total_recognition_time*1000:.2f} ms, 平均单次识别时间: {avg_single_recognition_time*1000:.2f} ms/人脸")
        print(f"[性能] 并行加速比: {sum(r['recognition_time'] for r in recognition_results) / total_recognition_time:.2f}x")
    
    total_time = detection_time + total_recognition_time
    print(f"[总计] 单帧总处理时间: {total_time*1000:.2f} ms (检测: {detection_time*1000:.2f} ms + 识别: {total_recognition_time*1000:.2f} ms)")
    
    return annotated_frame, results_data

def process_frame_behavior_detection(frame):
    """处理帧进行行为检测"""
    behaviors = detect_behaviors(frame)
    
    # 绘制结果
    annotated_frame = frame.copy()
    
    for behavior in behaviors:
        x1, y1, x2, y2 = behavior['bbox']
        confidence = behavior['confidence']
        class_name = behavior['class']
        
        # 绘制边界框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 绘制标签
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
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
            processed_frame, _ = process_frame_behavior_detection(frame)
            
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
            
            if not FACE_RECOGNITION_AVAILABLE:
                return jsonify({'success': False, 'message': 'face_recognition库未安装，请先安装该库'}), 400
            
            # 转换为RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return jsonify({'success': False, 'message': '未检测到人脸'}), 400
            
            # 获取人脸编码
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_encodings) == 0:
                return jsonify({'success': False, 'message': '无法提取人脸特征'}), 400
            
            # 保存人脸编码
            face_encodings_db[name] = face_encodings[0]
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
            
            if not FACE_RECOGNITION_AVAILABLE:
                return jsonify({
                    'success': False,
                    'message': 'face_recognition库未安装，请先安装该库',
                    'name': None
                })
            
            # 识别人脸
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return jsonify({
                    'success': False,
                    'message': '未检测到人脸',
                    'name': None
                })
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_encodings) == 0:
                return jsonify({
                    'success': False,
                    'message': '无法提取人脸特征',
                    'name': None
                })
            
            face_encoding = face_encodings[0]
            
            # 与数据库比较
            best_match = None
            best_distance = 1.0
            
            for name, known_encoding in face_encodings_db.items():
                distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
            
            if best_distance < 0.6:
                return jsonify({
                    'success': True,
                    'message': f'打卡成功: {best_match}',
                    'name': best_match,
                    'similarity': 1.0 - best_distance,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '未识别到已注册的人脸',
                    'name': None,
                    'similarity': 1.0 - best_distance
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

if __name__ == '__main__':
    print("正在初始化模型...")
    init_models()
    print("模型初始化完成!")
    print("启动Flask服务器...")
    app.run(host='0.0.0.0', port=9100, debug=True, threaded=True)

