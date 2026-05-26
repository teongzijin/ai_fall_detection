import cv2
import time
import os
from core.detector import RobustFallDetector

# 这里的配置可以直接改手机的 IP webcam 地址
CAMERA_SOURCE = "http://192.168.1.3:8080/video" 
ALERT_IMAGE_DIR = r'D:\AI_Project\fall_alerts'
os.makedirs(ALERT_IMAGE_DIR, exist_ok=True)

def main():
    # 初始化检测引擎
    detector = RobustFallDetector(angle_threshold=45, consecutive_frames=5, min_conf=0.3)
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    print("BOSS AI Vision - Fall Detection Engine Active...")
    last_alert_time = 0
    COOLDOWN_SECONDS = 10 # 报警冷却时间，防止硬盘被图片塞爆

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 运行推理
        alert_triggered, processed_frame = detector.process_frame(frame)
        
        # 报警触发：本地日志 + 自动存图
        if alert_triggered:
            current_time = time.time()
            if current_time - last_alert_time > COOLDOWN_SECONDS:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_name = f"FALL_ALERT_{timestamp}.jpg"
                full_img_path = os.path.join(ALERT_IMAGE_DIR, img_name)
                
                # 自动保存带有 AI 骨骼线条的证据图
                cv2.imwrite(full_img_path, processed_frame)
                
                # 追加本地运行日志
                with open(os.path.join(ALERT_IMAGE_DIR, "alerts_history.log"), "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detected Fall! Saved Evidence: {img_name}\n")
                
                print(f"🚨 [ALERT] 已记录跌倒事件并存图: {img_name}")
                last_alert_time = current_time

        cv2.imshow("BOSS AI - Fall Monitoring SYSTEM", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()