from moviepy import VideoFileClip


def extract_audio(video_path):
    content = VideoFileClip(video_path)
    content.audio.write_audiofile("converted_audio.mp3")


if __name__ == "__main__":
    extract_audio("test_video.mp4")
