
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import os

class WasteConveyorSimulator:
    def __init__(self, model_path, mode='synthetic', source=None, fps=3):
        """
        Initialize the conveyor simulator
        
        Args:
            model_path: Path to your fine-tuned YOLOv8 model (.pt file)
            mode: 'synthetic', 'webcam', 'video', or 'folder'
            source: Path to video file or image folder (if applicable)
            fps: Target FPS for processing (default: 3)
        """
        self.model = YOLO(model_path)
        self.mode = mode
        self.source = source
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        
        # Waste classes with colors (BGR format for OpenCV)
        # Updated for your 3-class model
        self.class_colors = {
            'plastic water bottle': (255, 130, 59),  # Blue
            'paper cup': (129, 185, 16),              # Green
            'disposable plastic cutlery': (68, 68, 239)  # Red
        }
        
        # Synthetic conveyor properties
        self.conveyor_offset = 0
        self.synthetic_items = []
        self.frame_count = 0
        
        # Initialize based on mode
        if mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
        elif mode == 'video':
            self.cap = cv2.VideoCapture(source)
        elif mode == 'folder':
            self.images = sorted(list(Path(source).glob('*.jpg')) + 
                               list(Path(source).glob('*.png')))
            self.image_idx = 0
        elif mode == 'synthetic':
            self.cap = None
            self.init_synthetic_items()
    
    def init_synthetic_items(self):
        """Initialize synthetic waste items on conveyor"""
        # Updated to use your 3 classes
        classes = ['plastic water bottle', 'paper cup', 'disposable plastic cutlery']
        for i in range(6):
            item = {
                'id': i,
                'class': np.random.choice(classes),
                'x': np.random.randint(100, 500),
                'y': np.random.randint(100, 300),
                'width': np.random.randint(50, 100),
                'height': np.random.randint(50, 100),
                'speed': np.random.uniform(1, 3)
            }
            self.synthetic_items.append(item)
    
    def create_synthetic_frame(self, width=800, height=480):
        """Create a synthetic conveyor frame with waste items"""
        # Create base frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        
        # Draw conveyor belt
        belt_top = int(height * 0.15)
        belt_bottom = int(height * 0.85)
        cv2.rectangle(frame, (0, belt_top), (width, belt_bottom), (60, 60, 60), -1)
        
        # Draw conveyor lines (moving effect)
        self.conveyor_offset = (self.conveyor_offset + 3) % 50
        for x in range(-50 + self.conveyor_offset, width, 50):
            cv2.line(frame, (x, belt_top), (x, belt_bottom), (80, 80, 80), 2)
        
        # Update and draw synthetic items
        for item in self.synthetic_items:
            # Move item
            item['x'] += item['speed']
            
            # Reset if off screen
            if item['x'] > width + 50:
                item['x'] = -item['width']
                item['y'] = np.random.randint(100, 300)
                # Use your 3 classes
                classes = ['plastic water bottle', 'paper cup', 'disposable plastic cutlery']
                item['class'] = np.random.choice(classes)
            
            # Draw item (rectangle with recycle symbol)
            color = self.class_colors.get(item['class'], (150, 150, 150))
            x, y, w, h = int(item['x']), int(item['y']), item['width'], item['height']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Add recycle symbol
            cv2.putText(frame, '♻', (x + w//2 - 15, y + h//2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return frame
    
    def get_frame(self):
        """Get next frame based on mode"""
        if self.mode == 'synthetic':
            return self.create_synthetic_frame()
        
        elif self.mode in ['webcam', 'video']:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        
        elif self.mode == 'folder':
            if self.image_idx >= len(self.images):
                return None
            frame = cv2.imread(str(self.images[self.image_idx]))
            self.image_idx += 1
            return frame
        
        return None
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes, labels, and pick points on frame
        
        Args:
            frame: Input frame
            results: YOLO detection results
        """
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Calculate pick point (center of bounding box)
                pick_x = int((x1 + x2) / 2)
                pick_y = int((y1 + y2) / 2)
                
                # Get color for this class
                color = self.class_colors.get(class_name, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with confidence
                label = f"{class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1), color, -1)
                
                # Label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw pick point crosshair
                cv2.line(frame, (pick_x - 20, pick_y), (pick_x + 20, pick_y), 
                        (0, 0, 255), 2)
                cv2.line(frame, (pick_x, pick_y - 20), (pick_x, pick_y + 20), 
                        (0, 0, 255), 2)
                cv2.circle(frame, (pick_x, pick_y), 10, (0, 0, 255), 2)
                
                # Draw pick point coordinates
                coord_text = f"({pick_x}, {pick_y})"
                cv2.putText(frame, coord_text, (pick_x + 15, pick_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Print pick point to console
                print(f"[Frame {self.frame_count}] {class_name} | "
                      f"Pick Point: ({pick_x}, {pick_y}) | "
                      f"Confidence: {confidence:.3f} | "
                      f"BBox: ({x1}, {y1}, {x2-x1}, {y2-y1})")
        
        return frame
    
    def add_info_overlay(self, frame, fps, detection_count):
        """Add information overlay to frame"""
        # Semi-transparent background for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add text information
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add mode indicator
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (frame.shape[1] - 250, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        return frame
    
    def run(self, display_width=1280, display_height=720):
        """
        Main simulation loop
        
        Args:
            display_width: Maximum width for display window (default: 1280)
            display_height: Maximum height for display window (default: 720)
        """
        print("=" * 60)
        print("Waste Sorting Conveyor Belt Simulator")
        print("=" * 60)
        print(f"Mode: {self.mode}")
        print(f"Target FPS: {self.target_fps}")
        print(f"Model: {self.model.model_name}")
        print(f"Display Size: {display_width}x{display_height}")
        print("Press 'q' to quit, 's' to save frame, 'p' to pause")
        print("=" * 60)
        
        last_time = time.time()
        paused = False
        
        # Create named window with specific size
        cv2.namedWindow('Waste Sorting Conveyor Belt', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Waste Sorting Conveyor Belt', display_width, display_height)
        
        while True:
            if not paused:
                # Get frame
                frame = self.get_frame()
                if frame is None:
                    print("End of stream reached")
                    break
                
                # Resize frame to fit display while maintaining aspect ratio
                h, w = frame.shape[:2]
                
                # Calculate scaling factor
                scale_w = display_width / w
                scale_h = display_height / h
                scale = min(scale_w, scale_h)
                
                # Only resize if frame is larger than display size
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Run YOLO detection
                results = self.model(frame, conf=0.6, verbose=False)
                
                # Draw detections and pick points
                frame = self.draw_detections(frame, results)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
                last_time = current_time
                
                # Get detection count
                detection_count = len(results[0].boxes) if results else 0
                
                # Add info overlay
                frame = self.add_info_overlay(frame, fps, detection_count)
                
                self.frame_count += 1
                
                # Display frame
                cv2.imshow('Waste Sorting Conveyor Belt', frame)
                
                # Control frame rate
                time.sleep(max(0, self.frame_interval - (time.time() - current_time)))
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_frame_{self.frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        # Cleanup
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nSimulation completed!")
        print(f"Total frames processed: {self.frame_count}")


def main():
    """
    Example usage - modify parameters as needed
    """
    
    # Configuration
    MODEL_PATH = "finetune_model.pt"  # Change this!
    
    # Choose mode and configure
    MODE = 'video'  # Options: 'synthetic', 'webcam', 'video', 'folder'
    SOURCE = '4058067-uhd_2160_4096_25fps.mp4'       # Path to video file or image folder 
    FPS = 35            # Target FPS
    
    # Display settings (adjust based on your screen size)
    DISPLAY_WIDTH = 1280   # Window width
    DISPLAY_HEIGHT = 720   # Window height
    
        # Initialize and run simulator
    simulator = WasteConveyorSimulator(
        model_path=MODEL_PATH,
        mode=MODE,
        source=SOURCE,
        fps=FPS
    )
    
    simulator.run(display_width=DISPLAY_WIDTH, display_height=DISPLAY_HEIGHT)


if __name__ == "__main__":
    main()