import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/input_video.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("outputs_detection/annotated_video.mp4", cv2.VideoWriter.fourcc(*"MP4v"), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error occurred")
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls.item())
        if results.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == 27: # 27 = escape
        break

    out.write(frame)

cap.release()
cv2.destroyAllWindows()
out.release()

