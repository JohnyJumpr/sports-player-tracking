import cv2

def draw_tracks(frame, tracks):
    for track_id, x1, y1, x2, y2 in tracks:
        color = (int(track_id * 37) % 255, int(track_id * 17) % 255, int(track_id * 29) % 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame