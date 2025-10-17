from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# 2. Train the model
results = model.train(data='/Scrap-detection-5/data.yaml', epochs=100, imgsz=640)

# After training, the best model is automatically saved in a 'runs' folder.
print("Training complete. Best model saved at:", results.save_dir)