<<<<<<< HEAD
# Smart Traffic Monitoring System Using Computer Vision

**Academic Minor Project &mdash; B.Tech Computer Science and Engineering**

---

## 1. Project Overview

The **Smart Traffic Monitoring System** is an automated computer vision and deep learning application designed to monitor vehicular traffic from surveillance video footage and traffic photographs. It replaces error-prone, manual traffic observation with an automated detection, tracking, counting, and density classification pipeline.

### Core Workflow
```
Media Input (Video/Image) ──> YOLO Vehicle Detection ──> Temporal Tracking & Counting ──> Density Classification ──> Web Dashboard
```

1. **Media Input:** Upload or select recorded traffic surveillance videos (`.mp4`, `.avi`, `.mov`) or static traffic photographs (`.jpg`, `.png`).
2. **Vehicle Detection:** Detect vehicles using pretrained lightweight **YOLO26n** (with **YOLO11n** fallback), filtering exclusively for target classes: **Cars, Motorcycles, Buses, and Trucks**.
3. **Vehicle Counting:** Track individual vehicles across consecutive video frames using ByteTrack temporal tracking and virtual reference lines to prevent duplicate counting.
4. **Traffic Density Classification:** Classify the traffic state into **Low**, **Medium**, or **High** density using configurable, rule-based thresholds evaluated on active road zone occupancy.
5. **Monitoring Dashboard:** Present media preview, processed output with bounding boxes, category-wise vehicle counts, traffic density badges, and measured performance metrics.

---

## 2. Technology Stack

- **Programming Language:** Python 3.10+ (tested and verified on Python 3.13)
- **Object Detection Model:** YOLO26n (`yolo26n.pt`) pretrained on MS COCO dataset (with `yolo11n.pt` automatic fallback)
- **Deep Learning Framework:** PyTorch & Ultralytics
- **Computer Vision:** OpenCV (`opencv-python`)
- **Numerical Processing:** NumPy
- **Linear Assignment / Tracking:** `lapx` / ByteTrack
- **Backend Framework:** Flask (Lightweight Python web framework with background worker threads)
- **Frontend:** Semantic HTML5, Clean Academic CSS3, Vanilla JavaScript (Fetch API)
- **Development Environment:** VS Code
- **Version Control:** Git & GitHub

---

## 3. Project Structure

```
smart-traffic-monitoring/
│
├── app.py                      # Flask backend entry point, preloaded detector, and REST API routes
├── config.py                   # Centralized project configuration, thresholds, & file extensions
├── yolo26n.pt                  # Pretrained YOLO26n COCO model weights (Primary)
├── yolo11n.pt                  # Pretrained YOLO11n COCO model weights (Fallback)
├── requirements.txt            # Python dependencies (flask, ultralytics, opencv-python, lapx)
├── README.md                   # Project documentation & viva preparation guide
│
├── modules/                    # Modular Python application logic
│   ├── __init__.py
│   ├── detector.py             # YOLO vehicle detector with dynamic class mapping & fallback loader
│   ├── tracker_counter.py      # Unique vehicle tracking, bidirectional line counter, & active ROI
│   ├── density_classifier.py   # Rule-based traffic density classifier (Low/Med/High)
│   └── video_processor.py      # Image and video processing, frame annotation HUD, and video writing
│
├── templates/
│   └── index.html              # Academic dashboard UI (Dashboard, Input, Results, About)
│
├── static/
│   ├── css/
│   │   └── style.css           # Clean, professional academic UI stylesheet
│   ├── js/
│   │   └── main.js             # Upload handler, polling logic, and dynamic dashboard updates
│   ├── sample_videos/          # Bundled sample traffic videos (sample_traffic_1.mp4, highway)
│   ├── sample_images/          # Bundled sample images covering all 4 COCO classes
│   ├── uploads/                # Directory for uploaded raw videos & images
│   └── outputs/                # Directory for processed annotated videos & images
│
└── tests/
    ├── test_full_suite.py      # Comprehensive 8-part verification suite
    ├── test_audit.py           # Technical audit script (upload validation, density, tracking)
    ├── test_server.py          # Flask REST API endpoints & async job integration
    └── test_system.py          # Pipeline verification script on sample video
```

---

## 4. Key Engineering Highlights (Viva & Evaluation Points)

### A. Prevention of Duplicate Counting
- **The Problem:** In simple object detection, if a car appears across 100 consecutive frames of a 30 FPS video, naive per-frame counting would register 100 cars instead of 1.
- **The Solution:** The system integrates **ByteTrack temporal tracking** (`persist=True`). Each detected vehicle is assigned a persistent `track_id` maintained across frames.
- **The Virtual Line & Region Registration:** A horizontal reference line is calibrated across the roadway ($0.50 \times \text{Height}$). When a vehicle's centroid crosses the virtual line or enters the road zone, its unique track ID is recorded in a registered set (`Set[int]`). Once recorded, the ID cannot increment the counter again, even if the vehicle pauses or stays in frame.

### B. Separation of Cumulative Volume and Active Density
- **Cumulative Vehicle Count:** Measures total throughput (how many unique vehicles have traversed the scene over the entire observation period).
- **Traffic Density (Congestion Level):** Evaluated strictly on the **number of vehicles currently visible inside the defined monitored road region** ($0.10 \times H$ to $0.95 \times H$):
  - **Low Traffic:** Active vehicles in road zone $\le$ Low Limit (default $\le$ 2). Free-flowing traffic.
  - **Medium Traffic:** Active vehicles between Low and Medium Limits (default 3 &ndash; 5). Normal steady traffic.
  - **High Traffic:** Active vehicles $>$ Medium Limit (default $>$ 5). High vehicle density / traffic congestion.
- The dashboard displays both metrics separately so that a video with 50 total vehicles passing one by one is correctly classified as **Low Traffic**, rather than incorrectly flagged as High Traffic.

### C. Target Vehicle Classes (MS COCO)
Only the 4 standard vehicle categories from the MS COCO dataset are processed:
| Class ID | Vehicle Class | Color Code (BGR) | Description |
|:---:|:---:|:---:|:---|
| 2 | Car | `(255, 140, 0)` Sky Blue | Light passenger vehicles, sedans, hatchbacks |
| 3 | Motorcycle | `(0, 165, 255)` Orange | Two-wheelers, motorbikes, scooters |
| 5 | Bus | `(70, 180, 50)` Green | Heavy public transit buses |
| 7 | Truck | `(180, 50, 200)` Purple | Commercial freight trucks and lorries |

### D. Scope Adherence (Strict Minor Project Boundaries)
In strict accordance with B.Tech Minor Project guidelines, advanced enterprise extensions are deliberately excluded:
- **Excluded:** Emergency vehicle priority (ambulance detection / signal preemption).
- **Excluded:** Adaptive traffic signal phase optimization or hardware controllers.
- **Excluded:** Traffic violation detection (red light jumping, speeding, wrong-way driving).
- **Excluded:** Automatic Number Plate Recognition (ANPR / ALPR).
- **Excluded:** Multi-camera re-identification or city-scale traffic simulation.

---

## 5. Measured Hardware Performance Benchmarks

All benchmarks were measured on a standard consumer CPU (Intel Core / AMD x86_64, Windows OS) without GPU acceleration:

| Input Type | Resolution | Inference Engine | Measured Processing Speed | Memory Footprint |
|---|---|---|---|---|
| Static Image (`sample_car.jpg`) | 768 x 432 | YOLO26n CPU | **~150 ms (Single Inference)** | ~280 MB RAM |
| Full Video (`sample_traffic_1.mp4`) | 768 x 432 | YOLO26n + ByteTrack | **~9.9 &ndash; 18.6 FPS (CPU)** | ~350 MB RAM |
| Fast Video Mode (`frame_stride=2`) | 768 x 432 | YOLO26n + ByteTrack | **~25 &ndash; 30+ FPS Equivalent** | ~350 MB RAM |

*Note: Model loading takes ~1.8 seconds at server boot; subsequent requests reuse the preloaded shared detector (`SHARED_DETECTOR`), resulting in zero model reload latency.*

---

## 6. Installation and Setup Instructions

### Step 1: Navigate to the Project Directory
Open PowerShell or Command Prompt:
```powershell
cd C:\Users\Saiful\.gemini\antigravity\scratch\smart-traffic-monitoring
```

### Step 2: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Verification Test Suite
Execute the full 8-part automated verification suite to confirm model loading, tracking, image inference, video inference, and REST endpoints:
```powershell
python tests/test_full_suite.py
```

### Step 4: Launch the Web Application
Start the Flask web server:
```powershell
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. How to Use the Application

1. **Open Dashboard:** Browse to `http://127.0.0.1:5000`. You will see the system overview with zeroed statistic cards and model status.
2. **Select Media:**
   - **Upload:** Drag and drop an MP4/AVI video or JPG/PNG image into the drop zone.
   - **Bundled Samples:** Click any sample button (*Sample Traffic 1*, *Car Sample*, *Motorcycle Sample*, *Bus Sample*, *Truck Sample*) for instant testing.
3. **Inspect Preview:** The preview player displays the image or video along with decoded resolution, FPS, total frames, and duration.
4. **Adjust Thresholds (Optional):** Calibrate the Low and Medium traffic limits if desired.
5. **Start Analysis:** Click **"Start Traffic Analysis"**.
   - For images: Results appear instantly (<200 ms).
   - For videos: A live progress card displays the current frame, elapsed time, and real-time processing FPS.
6. **Review Results:**
   - Classified traffic density banner (Green for Low, Amber for Medium, Red for High).
   - Annotated media with bounding boxes, category labels, confidence scores, and dual HUD.
   - Download the annotated output or inspect the measured performance table.

---

## 8. REST API Reference

| Endpoint | Method | Payload / Params | Description |
|---|---|---|---|
| `/` | `GET` | None | Renders the HTML academic dashboard. |
| `/api/samples` | `GET` | None | Lists available bundled sample videos and images. |
| `/api/upload` | `POST` | `multipart/form-data` (`video` or `media`) | Uploads image/video, returns decoded metadata. |
| `/api/use-sample` | `POST` | JSON `{"filename": "sample_name"}` | Copies sample to uploads for analysis. |
| `/api/analyze` | `POST` | JSON `{"filename": "...", "low_threshold": 2, "medium_threshold": 5}` | Runs synchronous image analysis or spawns async video job. |
| `/api/status/<job_id>` | `GET` | `job_id` | Returns current progress, live counts, or final results. |

---

## 9. Viva Examination Questions & Answers

**Q1: Why did you choose YOLO26n / YOLO11n?**  
*Answer:* YOLO26n and YOLO11n are lightweight "Nano" variants of the YOLO architecture. They provide fast inference on standard consumer CPUs without requiring expensive dedicated GPUs, making them ideal for an academic Minor Project.

**Q2: How does the system avoid counting the same vehicle in every frame?**  
*Answer:* By combining YOLO object detection with ByteTrack temporal tracking, each detected vehicle retains a consistent ID across frames. A spatial virtual line or unique ID registration ensures each vehicle is counted only once.

**Q3: Why use rule-based thresholds instead of a deep-learning classifier for traffic density?**  
*Answer:* Traffic density is a direct mathematical consequence of vehicle volume relative to road capacity. Rule-based thresholding is explainable, transparent, requires zero training data, and allows immediate calibration based on the specific camera angle or road segment.

**Q4: What is the difference between Cumulative Vehicle Count and Traffic Density?**  
*Answer:* Cumulative vehicle count measures total throughput (how many unique vehicles passed the monitoring line over the observation period). Traffic density measures the instantaneous and average congestion level (how many vehicles are simultaneously occupying the monitored road region at any given point in time). The dashboard presents both separately.

---

## 10. License & Academic Attribution
Developed as an academic Minor Project for B.Tech Computer Science and Engineering. Built with open-source tools: Ultralytics, PyTorch, OpenCV, and Flask.
>>>>>>> 6aa808d (commit project)
