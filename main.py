import os
import audioExtractor
import speechToText
import textSummarizer


if __name__ == "__main__":

    input_video = "test_video.mp4"

    # 1️ Extract audio
    audio_file = audioExtractor.extract_audio(input_video)

    # 2️ Speech to text
    transcript = speechToText.speech_to_text(audio_file)

    notes = textSummarizer.transcript_to_notes(transcript, input_video)

    print("✅ Notes generation completed successfully.")
