from ultralytics import RTDETR

model = RTDETR("runs/detect/train-42/weights/best.pt")

print(model.model)