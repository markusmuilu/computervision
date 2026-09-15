from trackers import Tracker
from utils import read_video, save_video
from drawers import TracksDrawer

def main():
    video_path = "input_video/nba.mp4"

    # Read video frames
    video_frames = read_video(video_path)

    # Initialize tracker and get object tracks
    print("Tracking objects in video...")
    object_tracker = Tracker()
    tracks = object_tracker.get_object_tracks(video_frames, 
                                              read_from_stub=True, 
                                              stub_path="stubs/object_track_stubs.pkl"
                                              )

    print("Drawing tracks on video frames...")
    # Draw tracks on video frames
    tracks_drawer = TracksDrawer()
    video_frames = tracks_drawer.draw(video_frames, tracks)


    # Save processed video (if needed)
    output_video_path = "output/result.AVI"
    save_video(video_frames, output_video_path)

if __name__ == "__main__":
    main()
