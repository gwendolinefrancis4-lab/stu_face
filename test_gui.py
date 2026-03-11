import sys
import os
import time
import sqlite3
import datetime
import pandas as pd
import cv2
import torch
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QMessageBox, QFileDialog, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from model import ParaNet

# ====================== 1. 数据库初始化（存储学生记录） ======================
def init_db():
    """初始化SQLite数据库，创建学生专注度记录表"""
    conn = sqlite3.connect("classroom_attention.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attention_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            detect_time TEXT NOT NULL,
            attention_status TEXT NOT NULL, 
            attention_duration REAL NOT NULL,  
            class_date TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_record(student_id, status, duration):
    """保存单条学生专注记录到数据库"""
    try:
        conn = sqlite3.connect("classroom_attention.db")
        cursor = conn.cursor()
        now = datetime.datetime.now()
        detect_time = now.strftime("%Y-%m-%d %H:%M:%S")
        class_date = now.strftime("%Y-%m-%d")
        cursor.execute('''
            INSERT INTO attention_records 
            (student_id, detect_time, attention_status, attention_duration, class_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (student_id, detect_time, status, duration, class_date))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"保存记录失败：{e}")
        return False

# ====================== 2. 视频检测线程（集成专注时长计算） ======================
class DetectionThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)  # 传递视频帧
    status_signal = pyqtSignal(str, float)  # 传递状态（Engaged/Distracted）和实时概率
    duration_signal = pyqtSignal(float, float)  # 传递累计专注/不专注时长（秒）

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.cap = None
        # 专注时长统计变量
        self.start_time = None  # 每段状态的开始时间
        self.last_status = None  # 上一帧的状态
        self.total_engaged = 0.0  # 累计专注时长（秒）
        self.total_distracted = 0.0  # 累计不专注时长（秒）
        self.student_id = "student_01"  # 默认学生ID（多人场景可按人脸ID区分）

        # 初始化模型和MediaPipe
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "best_model_state.bin")
        self.DEVICE = torch.device("cpu")
        self.model = ParaNet(num_emotions=2).to(self.DEVICE)
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.DEVICE))
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(None, "错误", f"模型加载失败：{e}")
        
        # MediaPipe人脸检测
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def run(self):
        """线程主循环：检测+专注时长计算"""
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.start_time = time.time()
        self.last_status = "Engaged"  # 初始状态

        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 人脸检测+模型推理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            current_status = "Distracted"  # 默认不专注
            prob = 0.0

            if results.detections:
                for detection in results.detections:
                    # 人脸框裁剪
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                                 int(bbox.width * iw), int(bbox.height * ih)
                    x_min, y_min = max(0, x-20), max(0, y-50)
                    x_max, y_max = min(iw, x+w+20), min(ih, y+h+20)
                    face_img = frame[y_min:y_max, x_min:x_max]

                    # 预处理+推理
                    try:
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape)==3 else face_img
                        gray_face = cv2.resize(gray_face, (224, 224))
                        img_tensor = torch.from_numpy(gray_face).unsqueeze(0).unsqueeze(0).float().to(self.DEVICE) / 255.0
                        
                        with torch.no_grad():
                            outputs = self.model(img_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            engaged_prob = probabilities[0][0].item()
                            distracted_prob = probabilities[0][1].item()
                            prob = engaged_prob

                            # 状态判定（阈值0.4）
                            if engaged_prob > 0.4:
                                current_status = "Engaged"
                            else:
                                current_status = "Distracted"

                            # 绘制人脸框和状态
                            color = (0, 255, 0) if current_status == "Engaged" else (0, 0, 255)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                            cv2.putText(frame, f"{current_status} ({prob*100:.0f}%)", 
                                        (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"推理失败：{e}")
            
            # ====================== 核心：专注时长计算 ======================
            current_time = time.time()
            time_diff = current_time - self.start_time  # 本次状态持续时长

            # 状态变化时，保存上一段状态的时长
            if current_status != self.last_status:
                # 保存上一段记录
                if self.last_status == "Engaged":
                    self.total_engaged += time_diff
                    save_record(self.student_id, "Engaged", time_diff)
                else:
                    self.total_distracted += time_diff
                    save_record(self.student_id, "Distracted", time_diff)
                
                # 重置计时
                self.start_time = current_time
                self.last_status = current_status

            # 实时传递状态和时长
            self.frame_signal.emit(frame)
            self.status_signal.emit(current_status, prob)
            self.duration_signal.emit(self.total_engaged, self.total_distracted)

        # 线程结束时，保存最后一段状态
        if self.last_status:
            final_diff = time.time() - self.start_time
            if self.last_status == "Engaged":
                self.total_engaged += final_diff
                save_record(self.student_id, "Engaged", final_diff)
            else:
                self.total_distracted += final_diff
                save_record(self.student_id, "Distracted", final_diff)

        # 释放资源
        if self.cap:
            self.cap.release()
        self.is_running = False

    def stop(self):
        """停止检测线程"""
        self.is_running = False
        self.wait()

# ====================== 3. 主界面（教师端+学生端） ======================
class AttentionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的课堂专注度识别系统")
        self.setGeometry(100, 100, 1000, 700)
        self.thread = None

        # 初始化数据库
        init_db()

        # 主布局
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 1. 学生端标签页
        self.student_tab = QWidget()
        self.init_student_tab()

        # 2. 教师端标签页
        self.teacher_tab = QWidget()
        self.init_teacher_tab()

        # 添加标签页
        self.tab_widget.addTab(self.student_tab, "学生端")
        self.tab_widget.addTab(self.teacher_tab, "教师端")

    def init_student_tab(self):
        """初始化学生端界面：实时检测+专注时长显示"""
        layout = QVBoxLayout()

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # 状态和时长显示
        status_layout = QHBoxLayout()
        self.status_label = QLabel("当前状态：未检测")
        self.status_label.setFont(QFont("Arial", 12))
        self.duration_label = QLabel("累计专注时长：0.0秒 | 累计不专注时长：0.0秒")
        self.duration_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.duration_label)
        layout.addLayout(status_layout)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.student_tab.setLayout(layout)

    def init_teacher_tab(self):
        """初始化教师端界面：查看学生记录+导出Excel"""
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("学生专注度历史记录")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # 记录表格
        self.record_table = QTableWidget()
        self.record_table.setColumnCount(6)
        self.record_table.setHorizontalHeaderLabels([
            "记录ID", "学生ID", "检测时间", "专注状态", "专注时长(秒)", "课堂日期"
        ])
        # 自动调整列宽
        self.record_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.record_table)

        # 按钮布局
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新记录")
        self.export_btn = QPushButton("导出Excel")
        self.clear_btn = QPushButton("清空记录")

        self.refresh_btn.clicked.connect(self.load_records)
        self.export_btn.clicked.connect(self.export_records)
        self.clear_btn.clicked.connect(self.clear_records)

        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        self.teacher_tab.setLayout(layout)

    def start_detection(self):
        """启动检测线程"""
        self.thread = DetectionThread()
        self.thread.frame_signal.connect(self.update_video)
        self.thread.status_signal.connect(self.update_status)
        self.thread.duration_signal.connect(self.update_duration)
        self.thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_detection(self):
        """停止检测线程"""
        if self.thread:
            self.thread.stop()
            self.thread = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("当前状态：已停止")

    def update_video(self, frame):
        """更新视频显示"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        ))

    def update_status(self, status, prob):
        """更新当前状态显示"""
        self.status_label.setText(f"当前状态：{status} (概率：{prob*100:.1f}%)")

    def update_duration(self, engaged, distracted):
        """更新累计时长显示"""
        self.duration_label.setText(
            f"累计专注时长：{engaged:.1f}秒 | 累计不专注时长：{distracted:.1f}秒"
        )

    def load_records(self):
        """加载数据库中的学生记录到表格"""
        try:
            conn = sqlite3.connect("classroom_attention.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attention_records ORDER BY detect_time DESC")
            records = cursor.fetchall()
            conn.close()

            # 清空表格
            self.record_table.setRowCount(0)

            # 填充数据
            for row_idx, record in enumerate(records):
                self.record_table.insertRow(row_idx)
                for col_idx, value in enumerate(record):
                    self.record_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

            QMessageBox.information(self, "成功", f"共加载 {len(records)} 条记录")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载记录失败：{e}")

    def export_records(self):
        """导出记录为Excel文件（需安装pandas和openpyxl）"""
        try:
            conn = sqlite3.connect("classroom_attention.db")
            df = pd.read_sql("SELECT * FROM attention_records", conn)
            conn.close()

            # 选择保存路径
            save_path, _ = QFileDialog.getSaveFileName(self, "导出Excel", "专注度记录.xlsx", "Excel Files (*.xlsx)")
            if save_path:
                df.to_excel(save_path, index=False)
                QMessageBox.information(self, "成功", f"记录已导出到：{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{e}")

    def clear_records(self):
        """清空所有记录（需确认）"""
        reply = QMessageBox.question(self, "确认", "是否清空所有专注度记录？此操作不可恢复！",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                conn = sqlite3.connect("classroom_attention.db")
                cursor = conn.cursor()
                cursor.execute("DELETE FROM attention_records")
                conn.commit()
                conn.close()
                self.load_records()  # 刷新表格
                QMessageBox.information(self, "成功", "所有记录已清空")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"清空失败：{e}")

    def closeEvent(self, event):
        """窗口关闭时停止线程"""
        if self.thread:
            self.thread.stop()
        event.accept()

# ====================== 程序入口 ======================
if __name__ == "__main__":
    # 屏蔽冗余日志
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

    app = QApplication(sys.argv)
    window = AttentionSystem()
    window.show()
    sys.exit(app.exec_())