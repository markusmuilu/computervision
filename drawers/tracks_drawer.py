from .utils import draw_object
import cv2 

class TracksDrawer:
    
    def __init__(self):
        pass

    def draw(self, video_frames, tracks):

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            obj_dict = tracks[frame_num]

            # Draw bounding boxes and labels for each tracked object
            for track_id, obj in obj_dict.items():
                class_name = obj["class"]
                bbox = obj["bbox"]


                frame = draw_object(frame, bbox, class_name, track_id)

            output_video_frames.append(frame)
        return output_video_frames