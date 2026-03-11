import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from PIL import Image

# 1. 导入你之前保存的 model.py 中的模型结构
from model import ParaNet 

# ==================== 模块一：初始化设置 ====================
# 设置设备：如果有显卡就用显卡，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# 加载模型架构
# 注意：num_emotions=2 对应你的二分类（专注/不专注）
model = ParaNet(num_emotions=2).to(device)

# 加载训练好的权重 (如果你还没有训练，先把这行注释掉，用随机权重跑测试)
try:
    model.load_state_dict(torch.load("best_model_state.bin", map_location=device))
    print("成功加载模型权重！")
except FileNotFoundError:
    print("警告：未找到 'best_model_state.bin'。将使用随机初始化的模型进行演示。")

model.eval() # 切换到评估模式（固定参数，不进行训练）

# 初始化 MediaPipe 人脸检测器 (比 OpenCV 哈尔级联更稳)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# ==================== 模块二：定义预处理流水线 ====================
# 这个过程必须和训练时保持一模一样：转灰度 -> 缩放 -> 转Tensor
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 变成黑白 (因为你的模型只吃单通道)
    transforms.Resize((224, 224)),               # 缩放到 224x224 (模型要求的尺寸)
    transforms.ToTensor(),                       # 变成 0-1 之间的张量
])

# 类别标签映射
classes = {0: 'Engaged (Focus)', 1: 'Not Engaged (Distracted)'}

# ==================== 模块三：主循环 (系统启动) ====================
def main():
    # 打开摄像头 (0 通常是默认摄像头)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("系统启动中... 按 'q' 键退出")

    while True:
        # 1. 读取一帧画面
        ret, frame = cap.read()
        if not ret: break

        # 2. 转换颜色空间 BGR -> RGB (MediaPipe 需要 RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. 检测人脸
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                # 获取人脸的边界框 (Bounding Box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                x_min, y_min = max(0, x-20), max(0, y-50)
                x_max, y_max = min(iw, x+w+20), min(ih, y+h+20)
                
                # 截取人脸区域 
                face_img = frame[y_min:y_max, x_min:x_max]
                
                if face_img.size == 0: continue

                # ================= 核心修改 2：显示概率与调整阈值 =================
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                input_tensor = data_transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    # 将输出转化为百分比概率
                    probabilities = torch.softmax(outputs, dim=1)
                    engaged_prob = probabilities[0][0].item()    # 专注的概率
                    distracted_prob = probabilities[0][1].item() # 不专注的概率

                # 物理外挂：适当降低专注的阈值 (原来是 0.5 胜出，现在只要 > 0.4 就亮绿灯)
                # 因为模型可能对不专注的数据更敏感
                if engaged_prob > 0.40: 
                    label_text = f"Engaged ({engaged_prob*100:.0f}%)"
                    color = (0, 255, 0) # 绿色
                else:
                    label_text = f"Distracted ({distracted_prob*100:.0f}%)"
                    color = (0, 0, 255) # 红色
                
                # 在画面上绘制结果 (注意这里用 x_min, y_min 画框，也就是实际送给模型的框)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label_text, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 显示画面
        cv2.imshow('Student Concentration Analysis System', frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()