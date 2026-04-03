import cv2
from detection import PlayerDetector
from tracking import PlayerTracker
from utils import draw_tracks

video_path = "data/input_video.mp4"
output_path = "outputs/annotated_video.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter.fourcc(*"mp4v"),
                      fps,
                      (width, height))

detector = PlayerDetector()
tracker = PlayerTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)

    frame = draw_tracks(frame, tracks)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()