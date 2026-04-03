import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("data/input_video.mp4")
if not cap.isOpened():
    print("Error opening video")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("outputs/annotated_video.mp4", cv2.VideoWriter.fourcc(*"MP4v"), fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])[0] # Class 0 = person
    detections = []

    for box in results.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if results.names[cls] == "person" and conf > 0.5:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        if track.hits < 2:
            continue

        if track.time_since_update > 1:
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        color = (int(track_id * 37) % 255, int(track_id * 17) % 255, int(track_id * 29) % 255)

        cv2.rectangle(frame, (l,t), (r,b), color, 2)
        cv2.putText(frame, f"ID {track_id}", (l,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

    out.write(frame)

cap.release()
cv2.destroyAllWindows()
out.release()