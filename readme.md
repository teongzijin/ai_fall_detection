# 🏥 BOSS-AI: Occlusion-Robust Edge Fall Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Hardware: NVIDIA RTX 3080 Optimized](https://img.shields.io/badge/Hardware-RTX_3080_Optimized-green.svg)](https://www.nvidia.com/)

An enterprise-grade, privacy-first computer vision pipeline designed for real-time elderly fall detection in healthcare facilities.

Powered by state-of-the-art **YOLOv8x-Pose** and customized heuristics to handle real-world deployment challenges such as severe body occlusion, unstable pose tracking, and high false-positive environments.

---

# 🎬 Live Production Proof
<img width="1080" height="1920" alt="FALL_ALERT_20260512_170241" src="https://github.com/user-attachments/assets/9377f34f-44e5-40fe-a228-323e157e87c7" />

## Muar, Johor, Malaysia Test Scenario

> 💡 **Developer Note for Reviewers**  
> Place a 3-second animated GIF here showing a successful live fall detection scenario with skeleton overlays and alert visualization.

Example:

```markdown
![Demo GIF](./assets/demo.gif)
```

This dramatically improves recruiter engagement and project credibility.

---

# 🚀 Key Architectural Highlights

Most computer vision solutions fail in residential care environments due to two major issues:

- Severe body occlusions (furniture, beds, partial camera blocking)
- High false-positive rates from sudden movements or unstable tracking

BOSS-AI addresses these problems at the algorithmic level.

---

## 1️⃣ Topological Fallback Mechanism (TFM)

Traditional angle estimation pipelines fail when the lower body leaves the frame or becomes hidden behind obstacles.

BOSS-AI continuously monitors pose keypoint confidence scores and dynamically switches orientation calculation strategies:

### Primary Orientation Axis
- Spine Axis (`Nose → Hip`)

### Fallback Orientation Axis
- Upper Body Axis (`Nose → Shoulder`)

This allows uninterrupted posture estimation even under severe occlusion conditions.

---

## 2️⃣ Temporal Window Anti-Jitter Filtering

To eliminate false alerts caused by rapid normal movements (e.g. sitting quickly or bending down), the system applies a sliding temporal validation window.

A fall event is only confirmed when:

- A fall topology persists over **N consecutive frames**
- Confidence thresholds remain stable across the detection window

This significantly improves deployment reliability in real-world environments.

---

## 3️⃣ Privacy-by-Design (PDPA Compliance Ready)

The entire inference pipeline operates fully on localized edge hardware.

### Key Benefits

- No cloud video transmission
- No third-party inference APIs
- Reduced latency
- PDPA-friendly architecture
- Offline-capable deployment

Optimized for:

- NVIDIA RTX 3080
- CUDA acceleration
- Real-time local inference

---

# 🏗️ Project Structure

```text
boss-ai-fall-detection/
├── core/
│   ├── __init__.py
│   └── detector.py
│       # Main OOP Detection Pipeline
│       # Includes Occlusion Fallback Engine
│
├── fall_alerts/
│   └── alerts_history.log
│       # Local Detection Audit Logs
│
├── app.py
│   # Main application entry point
│   # Handles webcam/IP camera streams
│
└── requirements.txt
    # Verified Python dependencies
```

---

# ⚙️ Hyperparameters (Production Baseline)

The detector is fine-tuned based on empirical testing data.

| Parameter | Value | Description |
|---|---|---|
| `angle_threshold` | `45` | Angles below 45° relative to horizontal ground are classified as prone posture |
| `consecutive_frames` | `5` | Number of continuous frames required to confirm a fall |
| `min_conf` | `0.3` | Minimum pose confidence threshold to reduce structural noise |

---

# 🚀 Quick Start

## 1️⃣ Installation

Ensure your CUDA environment is properly configured before installation.

```bash
git clone https://github.com/yourusername/boss-ai-fall-detection.git

cd boss-ai-fall-detection

pip install -r requirements.txt
```

---

## 2️⃣ Execution

Connect your IP camera or local webcam stream and run:

```bash
python app.py
```

---

# 🧠 Core Technologies

- Python
- YOLOv8x-Pose
- OpenCV
- CUDA
- NVIDIA GPU Acceleration
- Edge AI Inference
- Real-Time Pose Estimation

---

# 📈 Future Roadmap

- [ ] FastAPI backend integration
- [ ] Real-time alert dashboard
- [ ] WebSocket live event streaming
- [ ] Multi-camera support
- [ ] Docker deployment
- [ ] ONNX/TensorRT optimization
- [ ] Edge device benchmarking

---

# 📄 License

Distributed under the MIT License.

See the `LICENSE` file for more information.

---

# 📞 Architecture & Contact

## Lead Architect

**Teong Zi Jin** — Malaysia

### Areas of Specialization

- Full-Stack Infrastructure
- Distributed Systems
- Edge-AI Pipeline Engineering
- Real-Time Computer Vision Systems
- AI Deployment Architecture

---

# ⭐ Why This Project Matters

Elderly fall detection is a high-impact healthcare problem where low latency, privacy preservation, and deployment reliability are critical.

BOSS-AI focuses on solving practical deployment issues instead of only optimizing benchmark accuracy, making it suitable for real-world edge environments.

