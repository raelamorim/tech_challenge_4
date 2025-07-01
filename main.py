from app.video_processor import process_video
import sys

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    process_video(source)
