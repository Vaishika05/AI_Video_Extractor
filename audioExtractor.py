import os
from moviepy import VideoFileClip


def extract_audio(video_path):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} does not exist.")

    # Extract base name (without extension)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = "audios"
    os.makedirs(output_dir, exist_ok=True)

    output_audio = os.path.join(output_dir, f"{base_name}.mp3")

    if os.path.exists(output_audio):
        print("Audio already exists. Skipping extraction.")
        return output_audio

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio)

    print(f"Audio saved at {output_audio}")
    return output_audio
