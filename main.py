import os
import audioExtractor
import speechToText
import textSummarizer


if __name__ == "__main__":

    input_video = "test_video.mp4"

    # 1️⃣ Extract audio
    audio_file = audioExtractor.extract_audio(input_video)

    # 2️⃣ Speech to text
    transcript = speechToText.speech_to_text(audio_file)

    # 3️⃣ Chunk transcript (use returned transcript directly)
    chunks = textSummarizer.chunk_transcript(transcript)

    # 4️⃣ Load summarizer once
    summarizer = textSummarizer.load_llm()

    # 5️⃣ Summarize chunks
    chunk_summaries = textSummarizer.chunk_summarizer(chunks, summarizer)

    # 6️⃣ Generate final notes
    textSummarizer.generate_notes(chunk_summaries, summarizer)

    print("✅ Notes generation completed successfully.")
