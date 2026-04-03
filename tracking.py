from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)

        tracked_players = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            if track.hits < 2:
                continue

            if track.time_since_update > 1:
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            tracked_players.append((track_id, l, t, r, b))

        return tracked_players