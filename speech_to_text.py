import torch
import librosa
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1


def speech_to_text(audio_path):

    pipe = pipeline(
        "automatic-speech-recognition", model="openai/whisper-small", device=device
    )

    # Load audio manually (no ffmpeg required)
    audio, sr = librosa.load(audio_path, sr=16000)

    # Transcribe with chunking
    result = pipe(audio, chunk_length_s=30, stride_length_s=5)

    print(result["text"])


# if __name__ == "__main__":
#     speech_to_text("converted_audio.mp3")
