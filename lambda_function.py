from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
#model.train(data='coco128.yaml', epochs=3)  # train the model
results = model('https://ultralytics.com/images/zidane.jpg')  # predict on an image

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
    print(boxes)
    print(masks)
    print(probs)






""" for result in f:
    detection_count = result.boxes.shape[0]
    print(result.boxes)

    for i in range(detection_count):
        cls = int(result.boxes.cls[i].item())
        name = result.names[cls]
        confidence = float(result.boxes.conf[i].item())
        bounding_box = result.boxes.xyxy[i].cpu().numpy()

        x = int(bounding_box[0])
        y = int(bounding_box[1])
        width = int(bounding_box[2] - x)
        height = int(bounding_box[3] - y) """