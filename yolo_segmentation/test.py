from ultralytics import YOLO

model = YOLO('trained_model.pt')

results = model('path/to/image.jpg')    
annotated_frame = results[0].plot()