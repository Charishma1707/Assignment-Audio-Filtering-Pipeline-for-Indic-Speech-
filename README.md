# Indic Audio Quality Filtering Pipeline

A data quality pipeline for large-scale Indic speech datasets — built to filter audio before TTS/ASR model training.

It combines standard DSP metrics (SNR, clipping, silence ratio, RMS) with AI-based signals (Whisper ASR confidence, SpeechBrain language ID) into a Streamlit dashboard where you can tune thresholds and see KEEP / DISCARD decisions in real time.

---

## What problem does this solve?

Training TTS or ASR models on raw crowd-sourced audio is risky — recordings often have background noise, clipping, long silences, or mislabelled languages. Rather than eyeballing individual files, this pipeline scores every file across six metrics and lets you pick quality thresholds interactively. You can then export the filtered list as CSV/JSON or generate a PDF report.

---

## Architecture

```
indic-audio-pipeline/
├── app.py                   # Streamlit dashboard
├── backend/
│   ├── __init__.py
│   ├── metrics.py           # SNR, clipping, silence, RMS, ASR confidence, language ID
│   ├── processor.py         # Parallel processing with ProcessPoolExecutor
│   ├── spectrogram.py       # Mel-spectrogram generation
│   └── report.py            # PDF report (fpdf2 + matplotlib)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Metrics

| Metric | How it's computed | Good direction |
|---|---|---|
| SNR (dB) | signal power / noise variance | ↑ higher |
| Clipping ratio | samples ≥ 99% peak amplitude | ↓ lower |
| Silence ratio | librosa VAD — non-speech / total | ↓ lower |
| RMS energy | √mean(x²) | informational |
| ASR confidence | Whisper avg_logprob → exp() | ↑ higher |
| Language | SpeechBrain VoxLingua107 ECAPA | label match |

**Note on ASR confidence:** Whisper's logprob-based scores are naturally low for short clips or non-English audio. The default threshold in the dashboard is set to 0.1 rather than 0.4 to avoid mass-discarding valid files. Tune it up once you've seen the distribution for your dataset.

---

## Local setup

### Requirements
- Python 3.9 or newer
- ffmpeg (Whisper needs it for .flac decoding)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

### Install and run

```bash
git clone https://github.com/YOUR_USERNAME/indic-audio-pipeline.git
cd indic-audio-pipeline

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Download the dataset

The pipeline targets [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) (access is gated — request it on the HuggingFace page first):

```bash
huggingface-cli login
python setup_dataset.py test_data
```

This downloads a ~3GB subset (Hindi, Tamil, Telugu, Bengali) into `test_data/audios/`.

---

## Docker

Docker is worth using here because the dependencies (PyTorch, SpeechBrain, Whisper, ffmpeg) are heavy and version-sensitive. Packaging them in an image means anyone can run the pipeline without fighting with their local Python environment.

### Build the image

```bash
docker build -t indic-audio-qc .
```

### Run the container

```bash
docker run -p 8501:8501 \
  -v $(pwd)/test_data:/app/test_data \
  -v $(pwd)/reports:/app/reports \
  indic-audio-qc
```

The `-v` flags mount your local audio folder and reports folder into the container so data persists after the container exits.

Then open http://localhost:8501.

### With GPU (optional)

If you have an NVIDIA GPU and nvidia-container-toolkit installed:

```bash
docker run --gpus all -p 8501:8501 \
  -v $(pwd)/test_data:/app/test_data \
  indic-audio-qc
```

Whisper and SpeechBrain will automatically use the GPU when available.

---

## GitHub submission steps

1. Create a new repo on GitHub (keep it public if it's an internship submission).

2. From the project folder:

```bash
git init
git add .
git commit -m "initial commit — indic audio quality pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/indic-audio-pipeline.git
git push -u origin main
```

3. Add a `.gitignore` so you don't push model weights or large data files:

```
# .gitignore
venv/
__pycache__/
*.pyc
test_data/
reports/
*.pdf
.env
```

4. Make sure the repo has:
   - This README at the root
   - `requirements.txt`
   - `Dockerfile`
   - The `backend/` package with all four modules
   - `app.py`

5. Optional but recommended: add a `screenshots/` folder with a screenshot of the dashboard and reference it in the README so reviewers can see what it looks like without running it.

---

## Troubleshooting

**`ffmpeg was not found`** — Whisper needs ffmpeg to decode .flac. Install it and restart your terminal.

**All files are being discarded** — The most common reason is the ASR confidence threshold. Whisper returns low logprob-based confidence scores for short or non-English clips. Drop the slider to 0.05–0.1 and watch what happens to your KEEP count. Similarly, check the silence ratio distribution — if your files have long pauses the default 40% cap may be too strict.

**SpeechBrain import error (k2_fsa)** — This is an optional dependency that sometimes fails on Apple Silicon or older Python versions. Language detection will fall back to "unknown" gracefully.

**Dashboard shows no files after running** — Check that `test_data/audios/` exists and contains `.wav`, `.flac`, or `.mp3` files. Run `setup_dataset.py` or point the pipeline at a folder that has audio in it.

**Port conflict** — `streamlit run app.py --server.port 8502`

---

## Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) — IndicVoices dataset
- [OpenAI Whisper](https://github.com/openai/whisper) — ASR confidence scoring
- [SpeechBrain](https://speechbrain.github.io/) — language identification
- [Streamlit](https://streamlit.io/) — dashboard framework
