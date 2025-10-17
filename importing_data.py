from roboflow import Roboflow

rf = Roboflow(api_key="API_KEY_HERE")
project = rf.workspace("tracking-and-detection").project("scrap-detection-vqlvk")
version = project.version(5)
dataset = version.download("yolov8-obb")