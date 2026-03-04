from transformers import pipeline
import os

transcript_text = open("transcript.txt").read()


# 1️⃣ CHUNK THE TRANSCRIPT
def chunk_transcript(transcript, chunk_size=800):
    chunks = []
    words = transcript.split()

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


# 2️⃣ LOAD LLM (LOAD ONLY ONCE)
def load_llm():
    summarizer = pipeline("text2text-generation", model="google/flan-t5-large")
    return summarizer


# 3️⃣ SUMMARIZE EACH CHUNK
def chunk_summarizer(chunks, summarizer):
    chunk_summaries = []

    for chunk in chunks:
        prompt = f"""
        Summarize the following transcript section clearly:

        {chunk}
        """

        result = summarizer(prompt, max_length=300)
        chunk_summaries.append(result[0]["generated_text"])

    return chunk_summaries


# COMBINING SUMMARIES


def combined_summary(chunk_summaries, summarizer):
    combined_summary = " ".join(chunk_summaries)
    final_prompt = f"""
    Convert the following summarized transcript into structured notes:

    1. Title
    2. Section-wise headings
    3. Bullet points under each section
    4. Key concepts
    5. Conclusion

    Text:
    {combined_summary}
    """

    final_notes = summarizer(final_prompt)
    file_name = "Final_Notes.txt"  # You can specify a different path here
    directory_path = "notes"
    full_path = os.path.join(directory_path, file_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    try:
        # Open the file in write mode ('w') and assign it to a file object 'file'
        with open(file_name, "w", encoding="utf-8") as file:
            # Use the write() method to add content to the file
            file.write(final_notes)
        print(f"File '{full_path}' created and saved successfully.")
    except IOError as e:
        print(f"Error saving file: {e}")
