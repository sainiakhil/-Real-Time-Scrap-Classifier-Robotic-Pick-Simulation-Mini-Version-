"""
Waste Sorting Conveyor Belt Dashboard with Streamlit
Real-time monitoring of waste detection with live statistics
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Waste Sorting Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #10B981;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .detection-log {
        font-family: monospace;
        font-size: 0.9rem;
        background-color: #111827;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'material_counts' not in st.session_state:
    st.session_state.material_counts = defaultdict(int)
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'running' not in st.session_state:
    st.session_state.running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Class configuration for your model
CLASS_COLORS = {
    'plastic water bottle': '#3B82F6',
    'paper cup': '#10B981',
    'disposable plastic cutlery': '#EF4444'
}

def load_model(model_path):
    """Load YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_frame(frame, model, conf_threshold=0.25):
    """Process frame and return detections"""
    results = model(frame, conf=conf_threshold, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            pick_x = int((x1 + x2) / 2)
            pick_y = int((y1 + y2) / 2)
            
            detection = {
                'bbox': (x1, y1, x2, y2),
                'class': class_name,
                'confidence': confidence,
                'pick_point': (pick_x, pick_y),
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            detections.append(detection)
    
    return detections, results

def draw_detections(frame, detections):
    """Draw bounding boxes and pick points on frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        pick_x, pick_y = det['pick_point']
        
        # Color mapping
        color_hex = CLASS_COLORS.get(class_name, '#10B981')
        color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), color_bgr, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw pick point
        cv2.drawMarker(frame, (pick_x, pick_y), (0, 0, 255), 
                      cv2.MARKER_CROSS, 20, 2)
        cv2.circle(frame, (pick_x, pick_y), 8, (0, 0, 255), 2)
        
        # Draw coordinates
        coord_text = f"({pick_x},{pick_y})"
        cv2.putText(frame, coord_text, (pick_x + 12, pick_y - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return frame

def update_statistics(detections):
    """Update detection statistics"""
    for det in detections:
        class_name = det['class']
        st.session_state.material_counts[class_name] += 1
        st.session_state.total_detections += 1
        
        # Add to history (keep last 50)
        st.session_state.detection_history.append(det)
        if len(st.session_state.detection_history) > 50:
            st.session_state.detection_history.pop(0)

def create_timeline_chart():
    """Create detection timeline"""
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        
        # Count detections per timestamp
        timeline = df.groupby('timestamp').size().reset_index(name='count')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(timeline))),
            y=timeline['count'],
            mode='lines+markers',
            name='Detections',
            line=dict(color='#10B981', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Detection Timeline (Last 50)',
            xaxis_title='Frame',
            yaxis_title='Objects Detected',
            height=250,
            showlegend=False
        )
        return fig
    return None

def main():
    # Header
    st.markdown('<p class="main-header">♻️ Waste Sorting Conveyor Belt Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_path = st.text_input(
            "Model Path", 
            value="best.pt",
            help="Path to your trained YOLOv8 model"
        )
        
        input_mode = st.selectbox(
            "Input Mode",
            ["Webcam", "Video File", "Image Folder", "Synthetic Demo"]
        )
        
        if input_mode == "Video File":
            video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        elif input_mode == "Image Folder":
            image_folder = st.text_input("Image Folder Path")
        
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        fps_target = st.slider("Target FPS", 1, 10, 3, 1)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.running = True
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.running = False
        
        if st.button("🔄 Reset Stats", use_container_width=True):
            st.session_state.detection_history = []
            st.session_state.material_counts = defaultdict(int)
            st.session_state.total_detections = 0
            st.session_state.frame_count = 0
            st.rerun()
    
    # Load model
    model = load_model("finetune_model.pt")
    if model is None:
        st.error("⚠️ Please provide a valid model path")
        return
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        total_metric = metric_col1.empty()
        bottle_metric = metric_col2.empty()
        cup_metric = metric_col3.empty()
        cutlery_metric = st.empty()
        
        st.divider()
        
        # Timeline Chart
        chart_placeholder = st.empty()
    
    # Detection log section
    st.subheader("📝 Recent Detections")
    log_placeholder = st.empty()
    
    # Initialize video capture based on mode
    if input_mode == "Webcam":
        cap = cv2.VideoCapture(0)
    elif input_mode == "Video File" and video_file is not None:
        # Save uploaded file temporarily
        temp_file = f"temp_video.{video_file.name.split('.')[-1]}"
        with open(temp_file, 'wb') as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_file)
    else:
        cap = None
    
    # Main processing loop
    if st.session_state.running and cap is not None:
        frame_interval = 1.0 / fps_target
        
        while st.session_state.running:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                st.warning("End of stream")
                break
            
            # Resize frame
            frame = cv2.resize(frame, (800, 480))
            
            # Process frame
            detections, _ = process_frame(frame, model, conf_threshold)
            
            # Draw detections
            frame = draw_detections(frame, detections)
            
            # Update statistics
            if detections:
                update_statistics(detections)
            
            st.session_state.frame_count += 1
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update metrics
            total_metric.metric("🎯 Total Detections", st.session_state.total_detections)
            bottle_metric.metric("🍾 Plastic Bottles", 
                               st.session_state.material_counts['plastic water bottle'])
            cup_metric.metric("☕ Paper Cups", 
                            st.session_state.material_counts['paper cup'])
            cutlery_metric.metric("🍴 Plastic Cutlery", 
                                st.session_state.material_counts['disposable plastic cutlery'])
            
            # Update charts
            timeline_chart = create_timeline_chart()
            if timeline_chart:
                chart_placeholder.plotly_chart(timeline_chart, use_container_width=True, key=f"timeline_{st.session_state.frame_count}")
            
            # Update detection log
            if st.session_state.detection_history:
                log_html = ""
                for det in reversed(st.session_state.detection_history[-10:]):
                    color = CLASS_COLORS.get(det['class'], '#10B981')
                    log_html += f"""
                    <div class="detection-log">
                        <span style="color: {color};">●</span> 
                        <b>{det['timestamp']}</b> | 
                        {det['class']} | 
                        Conf: {det['confidence']:.2f} | 
                        Pick: ({det['pick_point'][0]}, {det['pick_point'][1]})
                    </div>
                    """
                log_placeholder.markdown(log_html, unsafe_allow_html=True)
            
            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)
            
            # Small delay to allow UI updates
            time.sleep(0.01)
    
    elif st.session_state.running and input_mode == "Synthetic Demo":
        st.info("🎬 Synthetic demo mode - Use OpenCV script for full synthetic simulation")
    
    # Cleanup
    if cap is not None:
        cap.release()

if __name__ == "__main__":
    main()