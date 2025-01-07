from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')
model.train(data='Dataset10.0/data.yaml', epochs=100, imgsz=(848,480), batch=0.95)