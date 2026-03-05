import os
import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

nltk.download("punkt")


# ----------------------------
# SMART CHUNKING
# ----------------------------
def smart_chunk(transcript, max_words=500):

    sentences = sent_tokenize(transcript)

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:

        words = len(sentence.split())

        if word_count + words > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

        current_chunk.append(sentence)
        word_count += words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ----------------------------
# CLEAN MODEL OUTPUT
# ----------------------------
def clean_output(text):

    remove_phrases = [
        "You are an expert academic assistant",
        "Convert the following lecture transcript into structured lecture notes",
        "Rules:",
        "Transcript:",
        "Lecture Notes:",
        "Final Structured Notes:",
    ]

    for phrase in remove_phrases:
        text = text.replace(phrase, "")

    return text.strip()


# ----------------------------
# PROMPT BUILDER
# ----------------------------
def build_prompt(chunk):

    prompt = f"""
You are an expert academic assistant.

Convert the following lecture transcript into structured lecture notes.

Rules:
- Remove filler speech
- Remove repetition
- Keep only key concepts
- Use clear section headings
- Use bullet points
- Use concise academic language

Transcript:
{chunk}

Lecture Notes:
"""

    return prompt


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_llm():

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    return tokenizer, model


# ----------------------------
# GENERATE TEXT
# ----------------------------
def generate_response(prompt, tokenizer, model, max_tokens=350):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.2, top_p=0.9, do_sample=True
    )

    # remove prompt tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    text = clean_output(text)

    return text


# ----------------------------
# SUMMARIZE CHUNKS
# ----------------------------
def summarize_chunks(chunks, tokenizer, model):

    notes = []

    for i, chunk in enumerate(chunks):

        print(f"Processing chunk {i+1}/{len(chunks)}")

        prompt = build_prompt(chunk)

        summary = generate_response(prompt, tokenizer, model)

        notes.append(summary)

    return notes


# ----------------------------
# FINAL MERGE
# ----------------------------
def refine_notes(chunk_notes, tokenizer, model):

    cleaned_notes = []

    for note in chunk_notes:
        cleaned_notes.append(clean_output(note))

    combined = "\n".join(cleaned_notes)

    prompt = f"""
Merge the following partial lecture notes into ONE final structured set of notes.

Rules:
- Remove duplicate information
- Merge similar sections
- Keep only one final version
- Use headings and bullet points
- Use concise academic language

Notes:
{combined}

Final Structured Notes:
"""

    final_notes = generate_response(prompt, tokenizer, model, max_tokens=600)

    return final_notes


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def transcript_to_notes(transcript, input_name):

    print("Chunking transcript...")

    chunks = smart_chunk(transcript)

    print(f"{len(chunks)} chunks created")

    tokenizer, model = load_llm()

    print("Generating chunk summaries...")

    chunk_notes = summarize_chunks(chunks, tokenizer, model)

    print("Merging into final notes...")

    final_notes = refine_notes(chunk_notes, tokenizer, model)

    os.makedirs("notes", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_name))[0]
    notes_path = os.path.join("notes", f"{base_name}_notes.txt")

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(final_notes)

    print(f"\nFinal notes saved at {notes_path}")

    return final_notes


# # ----------------------------
# # RUN SCRIPT
# # ----------------------------
# if __name__ == "__main__":

#     input_file = "videoTranscripts/test_video.txt"

#     with open(input_file, "r", encoding="utf-8") as f:
#         transcript = f.read()

#     notes = transcript_to_notes(transcript, input_file)

#     print("\nGenerated Notes:\n")
#     print(notes)
