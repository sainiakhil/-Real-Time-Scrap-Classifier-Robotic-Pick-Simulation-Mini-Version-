# ♻️ Waste Sorting Conveyor Belt System with YOLOv8

An intelligent waste sorting system that uses a fine-tuned YOLOv8 model to detect and classify recyclable waste items on a conveyor belt in real-time. The system includes real-time detection simulation, pick point generation for robotic sorting, and a live monitoring dashboard.

---

## 🎯 Overview

This project implements an automated waste sorting system designed for conveyor belt applications. It uses computer vision and deep learning to:

1. **Detect** waste items in real-time video streams
2. **Classify** items into recyclable categories
3. **Generate pick points** for robotic arm coordination
4. **Monitor** system performance through a live dashboard

### Target Classes

The system is trained to detect three specific waste categories:
- 🍾 **Plastic Water Bottle**
- ☕ **Paper Cup**
- 🍴 **Disposable Plastic Cutlery**

---

## 📊 Dataset

**Dataset:** [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

### Dataset Details:
- **Source:** Kaggle
- **Total Classes Available:** Multiple waste categories
- **Classes Used:** 3 (plastic water bottle, paper cup, disposable plastic cutlery)
- **Format:** Images with YOLO annotation format
- **Purpose:** Fine-tuning YOLOv8 for waste detection

### Data Preparation:
```bash
# Directory structure
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## ✨ Features

### 1. **Real-time Detection**
- Processes video streams at configurable FPS (1-10 fps)
- Multiple input modes: synthetic, webcam, video file, image folder
- Confidence threshold adjustment

### 2. **Pick Point Generation**
- Calculates center coordinates of detected objects
- Displays crosshair overlay on detection
- Console logging of pick coordinates for robotic integration

### 3. **Live Dashboard**
- Real-time detection statistics
- Material-wise object counting
- Detection timeline visualization
- Timestamped detection logs

### 4. **Flexible Input Sources**
- **Synthetic Mode:** Simulated conveyor belt with animated waste items
- **Webcam:** Live camera feed
- **Video File:** Pre-recorded conveyor footage
- **Image Folder:** Batch processing of sequential images

---

## 📁 Project Structure

```
waste-sorting-system/
│
├── train.py                    # YOLOv8 model training script
├── realtime_simulation.py      # Real-time detection simulation
├── app.py                      # Streamlit dashboard application
├── best.pt                     # Trained YOLOv8 model weights
├── requirements.txt            # Python dependencies

```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Webcam (optional, for live testing)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/waste-sorting-system.git
cd waste-sorting-system
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
pillow>=10.0.0
```

---

## 🚀 Usage

### 1. Model Training

Train the YOLOv8 model on the waste classification dataset:

```bash
python train.py
```

**Training Configuration:**
- Base Model: YOLOv8n (nano) or YOLOv8s (small)
- Epochs: 50-100 (configurable)
- Image Size: 640x640
- Batch Size: 16 (adjust based on GPU memory)

**Output:**
- Trained weights saved to `runs/detect/train/weights/best.pt`
- Training metrics and plots in `runs/detect/train/`

### 2. Real-time Simulation

Run the conveyor belt simulation with real-time detection:

```bash
python realtime_simulation.py
```

**Controls:**
- **q:** Quit application
- **s:** Save current frame
- **p:** Pause/Resume

**Configuration Options:**

```python
# In realtime_simulation.py, modify:
MODEL_PATH = "best.pt"              # Your trained model
MODE = 'synthetic'                   # Options: synthetic, webcam, video, folder
SOURCE = None                        # Video file or folder path (if applicable)
FPS = 3                             # Target FPS
DISPLAY_WIDTH = 1280                # Window width
DISPLAY_HEIGHT = 720                # Window height
```

**Example Usage:**

```python
# Synthetic Mode (Demo)
simulator = WasteConveyorSimulator(
    model_path="best.pt",
    mode='synthetic',
    fps=3
)
simulator.run(display_width=1280, display_height=720)

# Webcam Mode
simulator = WasteConveyorSimulator(
    model_path="best.pt",
    mode='webcam',
    fps=3
)
simulator.run()

# Video File Mode
simulator = WasteConveyorSimulator(
    model_path="best.pt",
    mode='video',
    source='conveyor_video.mp4',
    fps=3
)
simulator.run()
```

### 3. Dashboard

Launch the Streamlit monitoring dashboard:

```bash
streamlit run app.py
```

**Dashboard Features:**
- Live video feed with bounding boxes
- Real-time detection counts per material
- Detection timeline graph
- Timestamped detection logs with pick points

**Access:** Opens automatically in browser at `http://localhost:8501`

---

## 🎓 Model Training

### Training Process

The `train.py` script fine-tunes YOLOv8 on the selected waste classes:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy

# Train
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640
)
```

### data.yaml Configuration

```yaml
path: dataset
train: train/images
val: val/images

nc: 3  # number of classes
names: ['plastic water bottle', 'paper cup', 'disposable plastic cutlery']
```

### Training Tips

1. **Data Augmentation:** YOLOv8 applies augmentation automatically
2. **Learning Rate:** Auto-adjusted by YOLOv8
3. **Early Stopping:** Use `patience` parameter to avoid overfitting
4. **GPU Memory:** Reduce batch size if out of memory errors occur

### Evaluation Metrics

After training, check:
- **mAP@0.5:** Mean Average Precision at IoU threshold 0.5
- **Precision:** Accuracy of positive predictions
- **Recall:** Ability to find all positive instances
- **Confusion Matrix:** Class-wise performance

---

## 🎥 Real-time Simulation

### Features

The `realtime_simulation.py` provides a complete conveyor belt simulation:

#### 1. Synthetic Mode
- Animated conveyor belt with moving waste items
- Realistic scrolling effect
- Random object placement and movement

#### 2. Detection Overlay
- **Bounding Boxes:** Color-coded by class
- **Confidence Scores:** Displayed on each detection
- **Pick Points:** Red crosshair at object center
- **Coordinates:** (x, y) displayed for robotic integration

#### 3. Console Output
```
[Frame 42] plastic water bottle | Pick Point: (345, 234) | Confidence: 0.892 | BBox: (310, 201, 70, 66)
[Frame 42] paper cup | Pick Point: (567, 189) | Confidence: 0.764 | BBox: (540, 165, 54, 48)
```

#### 4. Performance Monitoring
- Real-time FPS counter
- Frame count tracker
- Detection count display

### Integration with Robotic Systems

The pick points can be directly used for robotic arm control:

```python
# Example robot integration
for detection in detections:
    pick_x, pick_y = detection['pick_point']
    class_name = detection['class']
    
    # Send to robot controller
    robot.move_to(pick_x, pick_y)
    robot.classify_and_sort(class_name)
```

---

## 📊 Dashboard

### Streamlit Application (`app.py`)

#### Sidebar Controls
- **Model Path:** Select trained model weights
- **Input Mode:** Choose video source
- **Confidence Threshold:** Adjust detection sensitivity (0.0-1.0)
- **Target FPS:** Set processing speed (1-10)
- **Control Buttons:** Start, Stop, Reset

#### Main Display

##### Statistics Panel
- 🎯 **Total Detections:** Cumulative count
- 🍾 **Plastic Bottles:** Class-specific count
- ☕ **Paper Cups:** Class-specific count
- 🍴 **Plastic Cutlery:** Class-specific count

##### Timeline Graph
- X-axis: Frame number
- Y-axis: Objects detected per frame
- Shows detection trends over last 50 frames

##### Detection Log
- Last 10 detections with timestamps
- Format: `● HH:MM:SS.mmm | class | Conf: 0.XX | Pick: (x, y)`
- Color-coded by material type

### Dashboard Layout

```
┌────────────────────────────────────────────────────┐
│        ♻️ Waste Sorting Dashboard                  │
├──────────────────────┬─────────────────────────────┤
│                      │   📊 Statistics             │
│   📹 Live Feed       │   ┌─────┬─────┬─────┬─────┐│
│                      │   │Total│Bottle│Cup│Cutlery││
│   [Video Stream]     │   │ 142 │ 45  │ 38 │  59  ││
│                      │   └─────┴─────┴─────┴─────┘│
│   [Bounding Boxes]   │                             │
│   [Pick Points]      │   [Detection Timeline]      │
│                      │                             │
├──────────────────────┴─────────────────────────────┤
│   📝 Recent Detections                             │
│   ● 14:23:45.123 | plastic water bottle | ...     │
│   ● 14:23:45.456 | paper cup | ...                │
│   ● 14:23:45.789 | disposable plastic cutlery ... │
└────────────────────────────────────────────────────┘
```



