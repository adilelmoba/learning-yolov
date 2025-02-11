import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # 'yolov8n.pt' is the smallest model, use 'yolov8s.pt' for better accuracy

# Open the webcam
cap = cv2.VideoCapture(0)  # Change to 1 or another index if you have multiple cameras

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Render results on the frame
    for result in results:
        annotated_frame = result.plot()

    # Show the output
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
