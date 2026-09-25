import os
from rfdetr import RFDETRMedium
import supervision as sv
import cv2
from utils import save_stub, load_stub

DEFAULT_WEIGHTS = "models/object_detection.pth"

class Tracker:
    def __init__(self, weights_path=None):
        weights_path = weights_path or os.environ.get("BASKETBALL_CV_WEIGHTS", DEFAULT_WEIGHTS)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. They are not in the repository: "
                "see 'Running it' in README.md, then pass the path to Tracker() "
                "or set BASKETBALL_CV_WEIGHTS."
            )
        self.model = RFDETRMedium(pretrain_weights=weights_path)
        self.tracker = sv.ByteTrack()
        self.class_names = self.model.class_names

    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            batch_frames=frames[i:i+batch_size]
            batch_frames_rgb=[cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) for frame in batch_frames]

            batch_detections=self.model.predict(batch_frames_rgb, threshold=0.5)
            if isinstance(batch_detections, sv.Detections):
                detections.append(batch_detections)
            else:
                detections.extend(batch_detections)
        return detections
    
    def get_object_tracks(self,frames, read_from_stub=False, stub_path=None):

        tracks = load_stub(read_from_stub, stub_path)

        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for detection in detections:

            tracked = self.tracker.update_with_detections(detection)

            frame_tracks = {}

            for i in range(len(tracked.xyxy)):
                bbox = tracked.xyxy[i].tolist()
                cls_id = int(tracked.class_id[i])
                track_id = int(tracked.tracker_id[i])

                frame_tracks[track_id] = {
                    "bbox": bbox,
                    "class": self.class_names.get(cls_id, str(cls_id)),  # map id → name
                }

            tracks.append(frame_tracks)

        unique_classes = set()

        for frame_dict in tracks:
            for obj in frame_dict.values():
                unique_classes.add(obj["class"])

        print("\nUnique classes detected in this video:")
        for cls in sorted(unique_classes):
            print(" -", cls)
        print()
        save_stub(stub_path, tracks)
        return tracks
    