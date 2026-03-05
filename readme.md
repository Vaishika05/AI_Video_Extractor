# 🎥 Video Lecture to Structured Notes Generator

This project converts **video lectures into clean, structured academic notes automatically** using modern AI models.

The pipeline performs the following steps:

1. Extract audio from a video
2. Convert speech to text
3. Process the transcript
4. Generate structured lecture notes using a Large Language Model

This allows students or researchers to quickly transform **long lecture videos into concise study notes**.

---

# ⚙️ How the System Works

### Step 1 — Audio Extraction

The video file is processed using **MoviePy** to extract the audio.

```
video.mp4 → audio.mp3
```

Output is stored in:

```
audios/
```

---

### Step 2 — Speech to Text

The extracted audio is transcribed using **OpenAI Whisper** via HuggingFace.

Model used:

```
openai/whisper-small
```

Features:

- Handles long audio
- Uses chunked processing
- Supports GPU acceleration

Output:

```
videoTranscripts/video_name.txt
```

---

### Step 3 — Transcript Processing

The transcript is split into **manageable chunks** using **NLTK sentence tokenization**.

This avoids exceeding the token limit of the language model.

Each chunk contains approximately:

```
~500 words
```

---

### Step 4 — Lecture Notes Generation

Each transcript chunk is processed using the language model:

```
mistralai/Mistral-7B-Instruct-v0.2
```

The model:

- Removes filler speech
- Removes repetition
- Extracts key concepts
- Generates bullet points
- Adds clear section headings

---

### Step 5 — Final Notes Refinement

All partial summaries are merged into a **single structured set of notes**.

The final notes include:

- Organized sections
- Bullet points
- Clean academic writing

Output:

```
notes/video_name_notes.txt
```

---

# 🧠 Models Used

| Task                  | Model               |
| --------------------- | ------------------- |
| Speech Recognition    | Whisper Small       |
| Text Generation       | Mistral 7B Instruct |
| Sentence Tokenization | NLTK Punkt          |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/VideoExtractor.git
cd VideoExtractor
```

Install dependencies:

```bash
pip install torch
pip install transformers
pip install moviepy
pip install librosa
pip install nltk
pip install accelerate
pip install sentencepiece
```

Download NLTK tokenizer:

```python
import nltk
nltk.download("punkt")
```

---

# ▶️ Running the Project

Place your video file in the project directory.

Example:

```
test_video.mp4
```

Run the pipeline:

```bash
python main.py
```

---

# 📂 Output Files

After execution, the following files will be created:

### Extracted Audio

```
audios/test_video.mp3
```

### Transcript

```
videoTranscripts/test_video.txt
```

### Final Notes

```
notes/test_video_notes.txt
```

---

# 💻 Hardware Requirements

Recommended:

| Component | Requirement                      |
| --------- | -------------------------------- |
| RAM       | 16GB                             |
| GPU       | NVIDIA GPU (optional but faster) |
| Storage   | ~10GB                            |

The system also works on CPU but will run slower.

---

# 📜 License

This project is intended for **educational and research purposes**.
