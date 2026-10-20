**深度学习框架目标检测YOLO+DeepSeek垃圾分类检测系统**
![【亮点】：](https://i-blog.csdnimg.cn/direct/d98f35ec015c49ac8cea64feab97db5e.png)

【结合Deepseek，Qwen】
【可以上传文件夹批量检测，可以导出图片检测结果为PDF文件】
【附垃圾厨余等四类检测YOLO数据集模型14964张 4类，yolov8 300epoch训练好的模型】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a47dfb30f6fa4138a39d9b32e2b593a8.png)

可以实现检测图片、批量检测图片、视频检测、摄像头检测四种方式。
能够检测出[“可回收垃圾”, “厨余垃圾”, “有害垃圾”, “其他垃圾”]4种检测情况。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ec03b944bd1f4841a3daf9c02a9b5d6b.png)

结合DeepSeek、Qwen等大模型对检测结果给出相关建议，并可将检测报告导出为PDF文件。另外添加可视化界面对检测结果进行可视化显示。 技术栈使用YOLO、DeepSeek、Qwen、Pytorch、SpringBoot、MyBatis-Plus、Vue3、Echarts、Ts、element-plus、Flask、Axios、MySQL等技术。 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3afdead30dbe40e592c8e7c988dd2b40.png)
系统还支持分用户管理。分为管理员和普通用户，管理员可查看所有记录，并管理用户。普通用户只可查看自己的记录。 界面简洁、美观。支持主题色修改、布局样式修改、组件大小修改等。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f4fe014d2704480ba35acff20fdc74c2.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36dc9ebc39e0451caa52bb50911fe52a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae4225445d53483b804ebebc7aa40d73.png)
 **【YOLO + DeepSeek 垃圾分类检测系统】** 的 **简化版核心代码实现**，涵盖 **后端（Flask）+ 前端（Vue3 简化逻辑）+ 模型调用 + PDF 报告生成 + 大模型建议** 功能。完整系统可基于此扩展。
 以下文字及代码仅供参考。

> ✅ 特点：  
> - 支持 4 类垃圾检测（可回收、厨余、有害、其他）  
> - 支持单图/批量/视频/摄像头检测  
> - 调用 **DeepSeek / Qwen** 给出环保建议  
> - 自动生成 **PDF 检测报告**  
> - 用户权限管理（管理员 vs 普通用户）  
> - 提供训练好的 `best.pt`（14964 张数据集，YOLOv8，300 epoch）

---

## ✅ 一、后端核心代码（Flask）

```python
# app.py
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import os
import cv2
import base64
from datetime import datetime
import requests
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import json

app = Flask(__name__)

# 加载 YOLO 模型（替换为你自己的 best.pt）
model = YOLO('models/best.pt')  # 4类：['recyclable', 'kitchen', 'hazardous', 'other']

# 中文类别映射
class_map = {
    'recyclable': '可回收垃圾',
    'kitchen': '厨余垃圾',
    'hazardous': '有害垃圾',
    'other': '其他垃圾'
}

# 模拟用户数据库（实际用 MySQL + MyBatis-Plus）
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user1": {"password": "user123", "role": "user"}
}
records = []  # 检测记录

# DeepSeek/Qwen API 调用（示例）
def get_ai_advice(detections):
    prompt = f"这是一次垃圾分类检测结果：{detections}。请给出环保建议和处理方式，用中文回答，不超过100字。"
    try:
        # 使用 DeepSeek API（替换 YOUR_API_KEY）
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": "Bearer YOUR_DEEPSEEK_API_KEY"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return response.json()['choices'][0]['message']['content']
    except:
        # 回退到本地规则
        if any(d['label'] == '有害垃圾' for d in detections):
            return "⚠️ 有害垃圾请勿随意丢弃！应送至指定回收点。"
        else:
            return "✅ 分类正确，请投入对应垃圾桶。"

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user = users.get(username)
    if user and user['password'] == password:
        return jsonify({"success": True, "role": user['role'], "username": username})
    return jsonify({"success": False})

@app.route('/detect', methods=['POST'])
def detect_image():
    file = request.files['image']
    username = request.form.get('username')
    save_path = f"uploads/{file.filename}"
    file.save(save_path)

    # YOLO 检测
    results = model(save_path)
    annotated_img = results[0].plot()
    output_path = f"outputs/{file.filename}"
    cv2.imwrite(output_path, annotated_img)

    # 解析结果
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label_en = model.names[cls_id]
        label_cn = class_map[label_en]
        detections.append({
            "label": label_cn,
            "confidence": round(conf, 2),
            "bbox": box.xyxy[0].tolist()
        })

    # 获取 AI 建议
    advice = get_ai_advice(detections)

    # 保存记录
    record = {
        "id": len(records) + 1,
        "username": username,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections": detections,
        "advice": advice,
        "image": output_path
    }
    records.append(record)

    # 返回结果（含 base64 图像）
    with open(output_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    return jsonify({
        "success": True,
        "detections": detections,
        "advice": advice,
        "image": img_b64
    })

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    files = request.files.getlist('images')
    username = request.form.get('username')
    results_list = []

    for file in files:
        file.save(f"uploads/{file.filename}")
        results = model(f"uploads/{file.filename}")
        # ...（同上处理逻辑）
        # 此处简化，实际可循环处理
        results_list.append(file.filename)

    return jsonify({"success": True, "processed": len(results_list)})

@app.route('/export_pdf/<int:record_id>')
def export_pdf(record_id):
    record = next((r for r in records if r['id'] == record_id), None)
    if not record:
        return "Record not found", 404

    pdf_path = f"reports/report_{record_id}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # 标题
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "垃圾分类检测报告")
    c.line(50, height - 55, width - 50, height - 55)

    # 基本信息
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"用户: {record['username']}")
    c.drawString(50, height - 100, f"时间: {record['timestamp']}")

    # 检测结果表格
    data = [["类别", "置信度"]]
    for det in record['detections']:
        data.append([det['label'], str(det['confidence'])])
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    table.wrapOn(c, width - 100, height - 200)
    table.drawOn(c, 50, height - 250)

    # AI 建议
    c.drawString(50, height - 300, "AI 环保建议:")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 320, record['advice'])

    c.save()
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    app.run(debug=True, port=5000)
```

---

## ✅ 二、前端简化逻辑（Vue3 + Element Plus）

```vue
<!-- App.vue -->
<template>
  <el-container>
    <el-header>垃圾分类智能检测系统</el-header>
    <el-main>
      <el-upload
        action="/detect"
        :on-success="handleSuccess"
        :data="{ username: currentUser }"
        :show-file-list="false"
      >
        <el-button type="primary">上传图片检测</el-button>
      </el-upload>

      <div v-if="resultImage" class="result">
        <img :src="'data:image/jpeg;base64,' + resultImage" />
        <p><strong>AI建议：</strong>{{ aiAdvice }}</p>
        <el-button @click="exportPDF">导出PDF报告</el-button>
      </div>
    </el-main>
  </el-container>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const currentUser = 'user1' // 实际从登录获取
const resultImage = ref('')
const aiAdvice = ref('')
let currentRecordId = null

const handleSuccess = (response) => {
  if (response.success) {
    resultImage.value = response.image
    aiAdvice.value = response.advice
    // 实际应从后端返回 record.id
    currentRecordId = 1
  }
}

const exportPDF = () => {
  window.open(`/export_pdf/${currentRecordId}`, '_blank')
}
</script>

<style scoped>
.result { margin-top: 20px; text-align: center; }
.result img { max-width: 100%; border: 1px solid #ddd; }
</style>
```

---

## ✅ 三、依赖安装（requirements.txt）

```txt
flask==2.3.2
ultralytics==8.3.0
opencv-python==4.8.0
reportlab==4.0.4
requests==2.31.0
torch==2.0.1
```

安装命令：
```bash
pip install -r requirements.txt
```

---

## ✅ 四、目录结构

```
GarbageDetection/
├── app.py                 # Flask 后端
├── models/
│   └── best.pt            # YOLOv8 训练好的模型（14964张，4类）
├── uploads/               # 上传文件
├── outputs/               # 检测结果图
├── reports/               # PDF 报告
├── frontend/              # Vue3 项目（简化）
└── requirements.txt
```

---

## ✅ 五、功能说明

| 功能 | 实现方式 |
|------|----------|
| 📸 单图检测 | `POST /detect` |
| 📁 批量检测 | `POST /batch_detect` |
| 📄 PDF 导出 | `GET /export_pdf/<id>` |
| 💬 AI 建议 | 调用 DeepSeek/Qwen API |
| 👥 用户权限 | 模拟用户表（实际用 MySQL） |
| 🎨 主题切换 | Element Plus 支持动态主题 |

---




