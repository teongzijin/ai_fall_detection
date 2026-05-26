import cv2
import numpy as np
from ultralytics import YOLO

class RobustFallDetector:
    """
    An Edge-AI optimized Fall Detection engine utilizing YOLOv8x-Pose,
    featuring dynamic keypoint fallback for occlusion robustness.
    """
    def __init__(self, model_path='yolov8x-pose.pt', angle_threshold=45, consecutive_frames=5, min_conf=0.3):
        # 加载最强 Pose 模型
        self.model = YOLO(model_path)
        
        # 核心超参数（根据你在 Muar 现场实测的数据锁定的黄金值）
        self.angle_threshold = angle_threshold
        self.consecutive_frames = consecutive_frames
        self.min_conf = min_conf
        
        # 计数器状态管理
        self.fall_counter = 0

    @staticmethod
    def calculate_spine_angle(p_nose, p_lower):
        """计算脊椎轴线与水平线的夹角"""
        return np.abs(np.degrees(np.arctan2(p_nose[1] - p_lower[1], p_nose[0] - p_lower[0])))

    def process_frame(self, frame):
        """
        核心推理管道：输入原始帧，处理旋转、遮挡、时间平滑，返回是否触发报警和渲染图
        """
        # 1. 画面旋转纠正（解决手机直立拍摄、电脑画面地板在右边的商业痛点）
        frame_corrected = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # 2. AI 推理（生产环境关闭 verbose 打印，保持控制台干净）
        results = self.model(frame_corrected, verbose=False)
        
        annotated_frame = frame_corrected.copy()
        is_fall_detected_this_frame = False
        current_spine_angle = 90 # 默认站立角度

        for result in results:
            annotated_frame = result.plot() # 获取带有 YOLO 官方骨骼线的画面
            
            if result.keypoints is None: continue
            keypoints_data = result.keypoints.data.cpu().numpy()
            
            for person in keypoints_data:
                # 提取核心解剖学关键点
                p_nose = person[0]
                p_l_shoulder, p_r_shoulder = person[5], person[6]
                p_l_hip, p_r_hip = person[11], person[12]
                
                # --- [核心亮点：多点弹性骨骼补位逻辑] ---
                # 优先方案：鼻子到双髋中点（标准的脊椎角度）
                if p_l_hip[2] > 0.4 and p_r_hip[2] > 0.4:
                    p_lower_x = (p_l_hip[0] + r_hip[0]) / 2 if 'r_hip' in locals() else (p_l_hip[0] + p_r_hip[0]) / 2
                    p_lower_y = (p_l_hip[1] + p_r_hip[1]) / 2
                # 备用方案：如果下半身出了画面或被床脚遮挡，自动切换为“鼻子到双肩中点”
                elif p_l_shoulder[2] > 0.4 and p_r_shoulder[2] > 0.4:
                    p_lower_x = (p_l_shoulder[0] + p_r_shoulder[0]) / 2
                    p_lower_y = (p_l_shoulder[1] + p_r_shoulder[1]) / 2
                else:
                    continue # 遮挡太严重，跳过此人

                # 验证上半身可见度，确保置信度达标
                upper_body_conf = (p_nose[2] + p_l_shoulder[2] + p_r_shoulder[2]) / 3
                
                if upper_body_conf > self.min_conf:
                    current_spine_angle = self.calculate_spine_angle(p_nose, (p_lower_x, p_lower_y))
                    
                    # 角度判定
                    if current_spine_angle < self.angle_threshold:
                        is_fall_detected_this_frame = True
                        break # 只要发现一个人摔倒，此帧即标记为跌倒

        # --- [时间序列平滑滤波器] ---
        # 防止因快速弯腰或光线跳变导致的瞬时误报。只有连续多帧满足条件才算真摔
        if is_fall_detected_this_frame:
            self.fall_counter += 1
        else:
            self.fall_counter = 0 # 瞬间重置，保障极低的误报率

        # 触发最终报警
        alert_triggered = self.fall_counter >= self.consecutive_frames
        if alert_triggered:
            self.fall_counter = 0 # 触发后计数器归零进入冷却
            
        # 将 Debug 数据实时烧录在画面上，方便 Portfolio 展示
        cv2.putText(annotated_frame, f"Angle: {int(current_spine_angle)}deg | Counter: {self.fall_counter}/{self.consecutive_frames}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        return alert_triggered, annotated_frame