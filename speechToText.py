import torch
import librosa
from transformers import pipeline
import os


device = 0 if torch.cuda.is_available() else -1


def speech_to_text(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"{audio_path} does not exist.")

    pipe = pipeline(
        "automatic-speech-recognition", model="openai/whisper-small", device=device
    )

    audio, sr = librosa.load(audio_path, sr=16000)

    result = pipe(audio, chunk_length_s=30, stride_length_s=5)

    transcript = result["text"]

    # Extract base name from audio
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    output_dir = "videoTranscripts"
    os.makedirs(output_dir, exist_ok=True)

    transcript_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(transcript)

    print(f"Transcript saved at {transcript_path}")

    return transcript
