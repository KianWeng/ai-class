// 全局变量
let currentFaceStream = null;
let currentBehaviorStream = null;
let uploadedFaceVideoPath = null;
let uploadedBehaviorVideoPath = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 标签页切换
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const tab = this.getAttribute('data-tab');
            switchTab(tab);
        });
    });

    // 视频源切换
    document.querySelectorAll('input[name="face-source"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === 'video') {
                document.getElementById('face-video-upload').style.display = 'block';
                document.getElementById('face-rtsp-input').style.display = 'none';
            } else {
                document.getElementById('face-video-upload').style.display = 'none';
                document.getElementById('face-rtsp-input').style.display = 'block';
            }
        });
    });
    
    // 页面加载时自动加载已上传的视频列表
    loadFaceVideoList();
    loadBehaviorVideoList();

    document.querySelectorAll('input[name="behavior-source"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === 'video') {
                document.getElementById('behavior-video-upload').style.display = 'block';
                document.getElementById('behavior-rtsp-input').style.display = 'none';
            } else {
                document.getElementById('behavior-video-upload').style.display = 'none';
                document.getElementById('behavior-rtsp-input').style.display = 'block';
            }
        });
    });

    // 图片预览
    document.getElementById('register-image').addEventListener('change', function(e) {
        previewImage(e.target.files[0], 'register-preview');
    });

    document.getElementById('checkin-image').addEventListener('change', function(e) {
        previewImage(e.target.files[0], 'checkin-preview');
    });

    // 加载已注册人脸列表
    loadFaceList();
});

// 切换标签页
function switchTab(tabName) {
    // 隐藏所有标签内容
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // 移除所有按钮的active类
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // 显示选中的标签内容
    document.getElementById(tabName).classList.add('active');

    // 激活对应的按钮
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // 如果切换到人脸检测标签页，加载视频列表
    if (tabName === 'face-detection') {
        loadFaceVideoList();
    }
    // 如果切换到行为检测标签页，加载视频列表
    if (tabName === 'behavior-detection') {
        loadBehaviorVideoList();
    }
}

// 图片预览
function previewImage(file, previewId) {
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById(previewId);
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

// 注册人脸
async function registerFace() {
    const name = document.getElementById('register-name').value.trim();
    const imageInput = document.getElementById('register-image');

    if (!name) {
        alert('请输入姓名');
        return;
    }

    if (!imageInput.files || imageInput.files.length === 0) {
        alert('请选择图片');
        return;
    }

    const formData = new FormData();
    formData.append('name', name);
    formData.append('image', imageInput.files[0]);

    try {
        const response = await fetch('/api/face/register', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            alert(`成功注册: ${result.name}`);
            document.getElementById('register-name').value = '';
            document.getElementById('register-image').value = '';
            document.getElementById('register-preview').style.display = 'none';
            loadFaceList();
        } else {
            alert(`注册失败: ${result.message}`);
        }
    } catch (error) {
        alert(`注册失败: ${error.message}`);
    }
}

// 人脸打卡
async function checkinFace() {
    const imageInput = document.getElementById('checkin-image');
    const resultBox = document.getElementById('checkin-result');

    if (!imageInput.files || imageInput.files.length === 0) {
        alert('请选择图片');
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
        const response = await fetch('/api/face/checkin', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            resultBox.className = 'result-box success';
            resultBox.innerHTML = `
                <strong>打卡成功!</strong><br>
                姓名: ${result.name}<br>
                相似度: ${(result.similarity * 100).toFixed(2)}%<br>
                时间: ${result.time}
            `;
        } else {
            resultBox.className = 'result-box error';
            resultBox.innerHTML = `<strong>打卡失败:</strong> ${result.message}`;
        }
    } catch (error) {
        resultBox.className = 'result-box error';
        resultBox.innerHTML = `<strong>打卡失败:</strong> ${error.message}`;
    }
}

// 加载人脸列表
async function loadFaceList() {
    const faceList = document.getElementById('face-list');

    try {
        const response = await fetch('/api/face/list');
        const result = await response.json();

        if (result.success) {
            if (result.faces.length === 0) {
                faceList.innerHTML = '<p>暂无已注册人员</p>';
            } else {
                faceList.innerHTML = result.faces.map(name => 
                    `<div class="face-item">
                        <span style="flex: 1;">${name}</span>
                        <button class="btn btn-danger" style="padding: 5px 15px; font-size: 12px; margin: 0;" onclick="deleteFace('${name}')">删除</button>
                    </div>`
                ).join('');
            }
        } else {
            faceList.innerHTML = '<p>加载失败</p>';
        }
    } catch (error) {
        faceList.innerHTML = '<p>加载失败</p>';
    }
}

// 删除人脸
async function deleteFace(name) {
    if (!confirm(`确定要删除 "${name}" 的注册信息吗？此操作不可恢复。`)) {
        return;
    }

    try {
        const response = await fetch('/api/face/delete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: name })
        });

        const result = await response.json();

        if (result.success) {
            alert(`成功删除: ${result.name}`);
            if (result.deleted_images_count > 0) {
                console.log(`已删除 ${result.deleted_images_count} 个图片文件`);
            }
            // 刷新列表
            loadFaceList();
        } else {
            alert(`删除失败: ${result.message}`);
        }
    } catch (error) {
        alert(`删除失败: ${error.message}`);
    }
}

// 上传人脸检测视频
async function uploadFaceVideo() {
    const fileInput = document.getElementById('face-video-file');
    const statusBox = document.getElementById('face-video-status');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert('请选择视频文件');
        return;
    }

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);

    statusBox.className = 'status-box';
    statusBox.innerHTML = '正在上传...';
    statusBox.style.display = 'block';

    try {
        const response = await fetch('/api/upload/video', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            uploadedFaceVideoPath = result.path;
            statusBox.className = 'status-box success';
            statusBox.innerHTML = `上传成功: ${result.filename}`;
            // 上传成功后刷新视频列表
            loadFaceVideoList();
        } else {
            statusBox.className = 'status-box error';
            statusBox.innerHTML = `上传失败: ${result.message}`;
        }
    } catch (error) {
        statusBox.className = 'status-box error';
        statusBox.innerHTML = `上传失败: ${error.message}`;
    }
}

// 加载已上传的视频列表
async function loadFaceVideoList() {
    const videoListContainer = document.getElementById('face-video-list');
    
    videoListContainer.innerHTML = '<p style="color: #999; text-align: center; margin: 20px 0;">正在加载...</p>';
    
    try {
        const response = await fetch('/api/upload/video/list');
        const result = await response.json();
        
        if (result.success) {
            if (result.videos.length === 0) {
                videoListContainer.innerHTML = '<p style="color: #999; text-align: center; margin: 20px 0;">暂无已上传的视频</p>';
                return;
            }
            
            let html = '<div class="video-list-items">';
            result.videos.forEach(video => {
                const isSelected = uploadedFaceVideoPath === video.path ? 'selected' : '';
                html += `
                    <div class="video-item ${isSelected}" data-path="${video.path}" onclick="selectFaceVideo('${video.path}', '${video.filename}')" style="
                        padding: 10px;
                        margin: 5px 0;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        cursor: pointer;
                        transition: all 0.3s;
                        background: ${isSelected ? '#e3f2fd' : '#fff'};
                    " onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='${isSelected ? '#e3f2fd' : '#fff'}'">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #333;">${video.filename}</strong>
                                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                    大小: ${video.size_mb} MB | 修改时间: ${video.modified_time}
                                </div>
                            </div>
                            ${isSelected ? '<span style="color: #2196F3; font-weight: bold;">✓ 已选择</span>' : ''}
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            videoListContainer.innerHTML = html;
        } else {
            videoListContainer.innerHTML = `<p style="color: #f44336; text-align: center; margin: 20px 0;">加载失败: ${result.message}</p>`;
        }
    } catch (error) {
        videoListContainer.innerHTML = `<p style="color: #f44336; text-align: center; margin: 20px 0;">加载失败: ${error.message}</p>`;
    }
}

// 选择已上传的视频
function selectFaceVideo(path, filename) {
    uploadedFaceVideoPath = path;
    
    // 更新UI显示
    document.querySelectorAll('.video-item').forEach(item => {
        item.classList.remove('selected');
        item.style.background = '#fff';
        const selectedSpan = item.querySelector('span');
        if (selectedSpan) {
            selectedSpan.remove();
        }
    });
    
    const selectedItem = document.querySelector(`.video-item[data-path="${path}"]`);
    if (selectedItem) {
        selectedItem.classList.add('selected');
        selectedItem.style.background = '#e3f2fd';
        const nameDiv = selectedItem.querySelector('div > div');
        if (nameDiv && !nameDiv.querySelector('span')) {
            const span = document.createElement('span');
            span.style.cssText = 'color: #2196F3; font-weight: bold; margin-left: 10px;';
            span.textContent = '✓ 已选择';
            nameDiv.appendChild(span);
        }
    }
    
    // 更新状态提示
    const statusBox = document.getElementById('face-video-status');
    statusBox.className = 'status-box success';
    statusBox.innerHTML = `已选择视频: ${filename}`;
    statusBox.style.display = 'block';
}

// 上传行为检测视频
async function uploadBehaviorVideo() {
    const fileInput = document.getElementById('behavior-video-file');
    const statusBox = document.getElementById('behavior-video-status');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert('请选择视频文件');
        return;
    }

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);

    statusBox.className = 'status-box';
    statusBox.innerHTML = '正在上传...';
    statusBox.style.display = 'block';

    try {
        const response = await fetch('/api/upload/video', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            uploadedBehaviorVideoPath = result.path;
            statusBox.className = 'status-box success';
            statusBox.innerHTML = `上传成功: ${result.filename}`;
            // 上传成功后刷新视频列表
            loadBehaviorVideoList();
        } else {
            statusBox.className = 'status-box error';
            statusBox.innerHTML = `上传失败: ${result.message}`;
        }
    } catch (error) {
        statusBox.className = 'status-box error';
        statusBox.innerHTML = `上传失败: ${error.message}`;
    }
}

// 加载已上传的行为检测视频列表
async function loadBehaviorVideoList() {
    const videoListContainer = document.getElementById('behavior-video-list');
    
    if (!videoListContainer) {
        return; // 如果容器不存在，直接返回
    }
    
    videoListContainer.innerHTML = '<p style="color: #999; text-align: center; margin: 20px 0;">正在加载...</p>';
    
    try {
        const response = await fetch('/api/upload/video/list');
        const result = await response.json();
        
        if (result.success) {
            if (result.videos.length === 0) {
                videoListContainer.innerHTML = '<p style="color: #999; text-align: center; margin: 20px 0;">暂无已上传的视频</p>';
                return;
            }
            
            let html = '<div class="video-list-items">';
            result.videos.forEach(video => {
                const isSelected = uploadedBehaviorVideoPath === video.path ? 'selected' : '';
                html += `
                    <div class="video-item ${isSelected}" data-path="${video.path}" onclick="selectBehaviorVideo('${video.path}', '${video.filename}')" style="
                        padding: 10px;
                        margin: 5px 0;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        cursor: pointer;
                        transition: all 0.3s;
                        background: ${isSelected ? '#e3f2fd' : '#fff'};
                    " onmouseover="this.style.background='#f5f5f5'" onmouseout="this.style.background='${isSelected ? '#e3f2fd' : '#fff'}'">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #333;">${video.filename}</strong>
                                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                                    大小: ${video.size_mb} MB | 修改时间: ${video.modified_time}
                                </div>
                            </div>
                            ${isSelected ? '<span style="color: #2196F3; font-weight: bold;">✓ 已选择</span>' : ''}
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            videoListContainer.innerHTML = html;
        } else {
            videoListContainer.innerHTML = `<p style="color: #f44336; text-align: center; margin: 20px 0;">加载失败: ${result.message}</p>`;
        }
    } catch (error) {
        videoListContainer.innerHTML = `<p style="color: #f44336; text-align: center; margin: 20px 0;">加载失败: ${error.message}</p>`;
    }
}

// 选择已上传的行为检测视频
function selectBehaviorVideo(path, filename) {
    uploadedBehaviorVideoPath = path;
    
    // 更新UI显示（只更新行为检测部分的视频项）
    const behaviorVideoList = document.getElementById('behavior-video-list');
    if (behaviorVideoList) {
        behaviorVideoList.querySelectorAll('.video-item').forEach(item => {
            item.classList.remove('selected');
            item.style.background = '#fff';
            const selectedSpan = item.querySelector('span');
            if (selectedSpan) {
                selectedSpan.remove();
            }
        });
        
        const selectedItem = behaviorVideoList.querySelector(`.video-item[data-path="${path}"]`);
        if (selectedItem) {
            selectedItem.classList.add('selected');
            selectedItem.style.background = '#e3f2fd';
            const nameDiv = selectedItem.querySelector('div > div');
            if (nameDiv && !nameDiv.querySelector('span')) {
                const span = document.createElement('span');
                span.style.cssText = 'color: #2196F3; font-weight: bold; margin-left: 10px;';
                span.textContent = '✓ 已选择';
                nameDiv.appendChild(span);
            }
        }
    }
    
    // 更新状态提示
    const statusBox = document.getElementById('behavior-video-status');
    if (statusBox) {
        statusBox.className = 'status-box success';
        statusBox.innerHTML = `已选择视频: ${filename}`;
        statusBox.style.display = 'block';
    }
}

// 开始人脸检测
async function startFaceDetection() {
    const sourceType = document.querySelector('input[name="face-source"]:checked').value;
    const videoStream = document.getElementById('face-video-stream');
    const placeholder = document.getElementById('face-video-placeholder');

    let sourcePath = '';

    if (sourceType === 'video') {
        if (!uploadedFaceVideoPath) {
            alert('请先上传视频文件或从列表中选择已上传的视频');
            return;
        }
        sourcePath = uploadedFaceVideoPath;
    } else {
        const rtspUrl = document.getElementById('face-rtsp-url').value.trim();
        if (!rtspUrl) {
            alert('请输入RTSP流地址');
            return;
        }
        sourcePath = rtspUrl;
    }

    try {
        const response = await fetch('/api/stream/face/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: sourceType,
                path: sourcePath
            })
        });

        const result = await response.json();

        if (result.success) {
            // 显示视频流
            placeholder.style.display = 'none';
            videoStream.style.display = 'block';
            videoStream.src = result.url + '&t=' + new Date().getTime();
            currentFaceStream = result.stream_id;
            
            // 开始自动刷新人脸列表
            startFaceDetectionListAutoRefresh();
            // 立即刷新一次
            setTimeout(() => refreshFaceDetectionList(), 2000);
        } else {
            alert(`启动失败: ${result.message}`);
        }
    } catch (error) {
        alert(`启动失败: ${error.message}`);
    }
}

// 停止人脸检测
function stopFaceDetection() {
    const videoStream = document.getElementById('face-video-stream');
    const placeholder = document.getElementById('face-video-placeholder');

    videoStream.src = '';
    videoStream.style.display = 'none';
    placeholder.style.display = 'flex';
    
    // 停止自动刷新列表
    stopFaceDetectionListAutoRefresh();
    
    currentFaceStream = null;
}

// 开始行为检测
async function startBehaviorDetection() {
    const sourceType = document.querySelector('input[name="behavior-source"]:checked').value;
    const videoStream = document.getElementById('behavior-video-stream');
    const placeholder = document.getElementById('behavior-video-placeholder');

    let sourcePath = '';

    if (sourceType === 'video') {
        if (!uploadedBehaviorVideoPath) {
            alert('请先上传视频文件或从列表中选择已上传的视频');
            return;
        }
        sourcePath = uploadedBehaviorVideoPath;
    } else {
        const rtspUrl = document.getElementById('behavior-rtsp-url').value.trim();
        if (!rtspUrl) {
            alert('请输入RTSP流地址');
            return;
        }
        sourcePath = rtspUrl;
    }

    try {
        const response = await fetch('/api/stream/behavior/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: sourceType,
                path: sourcePath
            })
        });

        const result = await response.json();

        if (result.success) {
            // 显示视频流
            placeholder.style.display = 'none';
            videoStream.style.display = 'block';
            videoStream.src = result.url + '&t=' + new Date().getTime();
            currentBehaviorStream = result.stream_id;
            
            // 开始自动刷新统计
            startBehaviorStatisticsAutoRefresh();
            // 立即刷新一次
            setTimeout(() => refreshBehaviorStatistics(), 2000);
        } else {
            alert(`启动失败: ${result.message}`);
        }
    } catch (error) {
        alert(`启动失败: ${error.message}`);
    }
}

// 停止行为检测
function stopBehaviorDetection() {
    const videoStream = document.getElementById('behavior-video-stream');
    const placeholder = document.getElementById('behavior-video-placeholder');

    videoStream.src = '';
    videoStream.style.display = 'none';
    placeholder.style.display = 'flex';
    
    // 停止自动刷新统计
    stopBehaviorStatisticsAutoRefresh();
    
    currentBehaviorStream = null;
}

// 刷新行为统计
async function refreshBehaviorStatistics() {
    if (!currentBehaviorStream) {
        alert('请先开始检测');
        return;
    }

    try {
        const response = await fetch(`/api/behavior/statistics?stream_id=${currentBehaviorStream}`);
        const result = await response.json();

        if (result.success) {
            displayBehaviorStatistics(result.statistics, result.total);
        } else {
            alert(`获取统计失败: ${result.message}`);
        }
    } catch (error) {
        alert(`获取统计失败: ${error.message}`);
    }
}

// 重置行为统计
async function resetBehaviorStatistics() {
    if (!currentBehaviorStream) {
        alert('请先开始检测');
        return;
    }

    if (!confirm('确定要重置统计吗？')) {
        return;
    }

    try {
        const response = await fetch('/api/behavior/statistics/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                stream_id: currentBehaviorStream
            })
        });

        const result = await response.json();

        if (result.success) {
            alert('统计已重置');
            refreshBehaviorStatistics();
        } else {
            alert(`重置失败: ${result.message}`);
        }
    } catch (error) {
        alert(`重置失败: ${error.message}`);
    }
}

// 显示行为统计
function displayBehaviorStatistics(statistics, total) {
    const container = document.getElementById('behavior-statistics');
    
    if (total === 0) {
        container.innerHTML = '<p style="color: #999; text-align: center; margin: 20px 0;">暂无检测数据</p>';
        return;
    }

    // 行为名称映射（中文显示）
    const behaviorNames = {
        'BowHead': '低头',
        'TurnHead': '转头',
        'RaiseHand': '举手',
        'Reading': '阅读',
        'Writing': '写字'
    };

    let html = '<div class="statistics-table">';
    html += `<div class="statistics-header"><strong>总计: ${total} 次</strong></div>`;
    html += '<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">';
    html += '<thead><tr><th style="padding: 8px; border: 1px solid #ddd; background: #f5f5f5;">行为类型</th><th style="padding: 8px; border: 1px solid #ddd; background: #f5f5f5;">检测次数</th><th style="padding: 8px; border: 1px solid #ddd; background: #f5f5f5;">占比</th></tr></thead>';
    html += '<tbody>';

    // 按次数排序
    const sortedStats = Object.entries(statistics).sort((a, b) => b[1] - a[1]);

    for (const [behavior, count] of sortedStats) {
        const percentage = ((count / total) * 100).toFixed(1);
        const displayName = behaviorNames[behavior] || behavior;
        html += `<tr>`;
        html += `<td style="padding: 8px; border: 1px solid #ddd;">${displayName}</td>`;
        html += `<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${count}</td>`;
        html += `<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${percentage}%</td>`;
        html += `</tr>`;
    }

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// 自动刷新统计（每5秒）
let behaviorStatisticsInterval = null;

function startBehaviorStatisticsAutoRefresh() {
    if (behaviorStatisticsInterval) {
        clearInterval(behaviorStatisticsInterval);
    }
    behaviorStatisticsInterval = setInterval(() => {
        if (currentBehaviorStream) {
            refreshBehaviorStatistics();
        }
    }, 5000);
}

function stopBehaviorStatisticsAutoRefresh() {
    if (behaviorStatisticsInterval) {
        clearInterval(behaviorStatisticsInterval);
        behaviorStatisticsInterval = null;
    }
}

// 刷新人脸检测列表
async function refreshFaceDetectionList() {
    if (!currentFaceStream) {
        alert('请先开始检测');
        return;
    }

    try {
        // 获取统计信息
        const statsResponse = await fetch(`/api/face/statistics?stream_id=${currentFaceStream}`);
        const statsResult = await statsResponse.json();

        if (statsResult.success) {
            // 显示人员统计
            displayFaceStatistics(statsResult.statistics, statsResult.total);
        } else {
            alert(`获取统计失败: ${statsResult.message}`);
        }
    } catch (error) {
        alert(`获取统计失败: ${error.message}`);
    }
}

// 显示人员统计列表
function displayFaceStatistics(statistics, total) {
    const container = document.getElementById('face-statistics-list');
    
    if (total === 0 || Object.keys(statistics).length === 0) {
        container.innerHTML = '<p style="color: #999; text-align: center; margin: 20px 0;">暂无检测数据</p>';
        return;
    }

    // 按检测次数排序
    const sortedStats = Object.entries(statistics).sort((a, b) => b[1] - a[1]);

    let html = '<div class="face-statistics-content">';
    html += `<div style="margin-bottom: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px; text-align: center;">
        <strong style="font-size: 16px;">总计检测: ${total} 次</strong> | 识别到: ${sortedStats.length} 人
    </div>`;
    
    html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">';
    
    sortedStats.forEach(([name, count]) => {
        const percentage = ((count / total) * 100).toFixed(1);
        const isRecognized = name !== '未知';
        
        html += `
            <div style="
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                background: ${isRecognized ? '#e8f5e9' : '#fff3e0'};
                border-left: 4px solid ${isRecognized ? '#4caf50' : '#ff9800'};
                transition: transform 0.2s;
            " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 18px; margin-right: 8px;">${isRecognized ? '✓' : '○'}</span>
                    <strong style="color: ${isRecognized ? '#2e7d32' : '#e65100'}; font-size: 16px;">${name}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 20px; font-weight: bold; color: #2196F3;">${count}</span>
                    <span style="font-size: 12px; color: #666;">次</span>
                </div>
                <div style="margin-top: 6px;">
                    <div style="background: #e0e0e0; border-radius: 4px; height: 6px; overflow: hidden;">
                        <div style="background: ${isRecognized ? '#4caf50' : '#ff9800'}; height: 100%; width: ${percentage}%; transition: width 0.3s;"></div>
                    </div>
                    <div style="font-size: 11px; color: #666; margin-top: 4px; text-align: right;">${percentage}%</div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    html += '</div>';
    
    container.innerHTML = html;
}


// 重置人脸检测列表
async function resetFaceDetectionList() {
    if (!currentFaceStream) {
        alert('请先开始检测');
        return;
    }

    if (!confirm('确定要清空检测列表吗？')) {
        return;
    }

    try {
        const response = await fetch('/api/face/statistics/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                stream_id: currentFaceStream
            })
        });

        const result = await response.json();

        if (result.success) {
            alert('列表已清空');
            refreshFaceDetectionList();
        } else {
            alert(`清空失败: ${result.message}`);
        }
    } catch (error) {
        alert(`清空失败: ${error.message}`);
    }
}

// 自动刷新人脸检测列表（每3秒）
let faceDetectionListInterval = null;

function startFaceDetectionListAutoRefresh() {
    if (faceDetectionListInterval) {
        clearInterval(faceDetectionListInterval);
    }
    faceDetectionListInterval = setInterval(() => {
        if (currentFaceStream) {
            refreshFaceDetectionList();
        }
    }, 3000);
}

function stopFaceDetectionListAutoRefresh() {
    if (faceDetectionListInterval) {
        clearInterval(faceDetectionListInterval);
        faceDetectionListInterval = null;
    }
}

