import argparse

from trackers import Tracker
from utils import read_video, save_video
from drawers import TracksDrawer

def main():
    parser = argparse.ArgumentParser(description="Detect, track and draw players, ball and events on basketball video.")
    parser.add_argument("--input", default="input_video/nba.mp4", help="input video")
    parser.add_argument("--output", default="output/result.avi", help="output video (XVID .avi)")
    parser.add_argument("--weights", default=None, help="model weights (default: BASKETBALL_CV_WEIGHTS or models/object_detection.pth)")
    parser.add_argument("--stub", default="stubs/object_track_stubs.pkl", help="where detections are cached")
    parser.add_argument("--fresh", action="store_true", help="ignore the cached stub and re-run detection")
    args = parser.parse_args()

    # Read video frames
    video_frames, fps = read_video(args.input)

    # Initialize tracker and get object tracks
    print("Tracking objects in video...")
    object_tracker = Tracker(weights_path=args.weights)
    tracks = object_tracker.get_object_tracks(video_frames,
                                              read_from_stub=not args.fresh,
                                              stub_path=args.stub
                                              )

    print("Drawing tracks on video frames...")
    # Draw tracks on video frames
    tracks_drawer = TracksDrawer()
    video_frames = tracks_drawer.draw(video_frames, tracks)

    # Save processed video
    save_video(video_frames, args.output, fps)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
