#  Indic Audio Quality Filtering Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org)

A scalable, multi-threaded data quality pipeline designed for large Indic speech datasets. This tool evaluates raw, crowd-sourced audio files and filters them based on strict quality thresholds before they are used for ASR (Automatic Speech Recognition) or TTS (Text-to-Speech) model training.

<div align="center">
  <img src="assets/Screenshot 2026-04-13 203630.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
  <em>Interactive Streamlit dashboard evaluating sub-second Indic audio clips.</em>
</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 210202.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>

</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 210427.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
 
</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 225023.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
 
</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 225029.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
  
</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 225036.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
 
</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 225048.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
 
</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 225054.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>

</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 225107.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
  
</div>

<div align="center">
  <img src="assets/Screenshot 2026-04-13 225113.png" width="800" alt="Streamlit Dashboard Interface showing micro-decimal SNR">
  <br>
 
</div>

---

##  The Problem It Solves
Training models on raw, unvetted audio is a primary cause of model hallucination. Crowd-sourced datasets (like `IndicVoices`) often contain heavy background noise, hardware clipping, long periods of dead air, or incorrect language labels. 

Instead of manually auditing thousands of hours of audio, this pipeline automatically scores every file across six distinct metrics. The interactive dashboard allows Data Engineers to dynamically adjust acceptance thresholds based on the unique distribution of the dataset.

---

##  Quick Start Guide (How to Run)

### 1. Prerequisites
* **Python 3.9+** installed.
* **FFmpeg** installed system-wide (Required by OpenAI Whisper to decode `.flac` and `.mp3` files).
  * *Windows:* Install via `winget install ffmpeg` or download from the official site and add to your system PATH.
  * *Mac:* `brew install ffmpeg` 
  * *Linux:* `sudo apt install ffmpeg` 

### 2. Installation
Clone the repository and set up your environment:
```bash
git clone https://github.com/yourusername/indic-audio-pipeline.git
cd indic-audio-pipeline

# Create and activate a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torchvision  # Required for Streamlit file-watchers
```

### 3. Prepare the Data
Create a directory named `test_data/audios/` in the root of the project and place your `.wav`, `.flac`, or `.mp3` files inside. Nested folders are supported.

```
indic-audio-pipeline/
  app.py
  backend/
  test_data/
    audios/
      sample1.wav
      sample2.flac
```

### 4. Execute the Pipeline
Run the Streamlit application. The app will automatically scan the `test_data` folder, process the metrics using multi-threading, and launch the UI in your browser.

```bash
streamlit run app.py
```

---

##  Scalability Approach
Processing thousands of audio files through AI models is computationally expensive. This pipeline is built for scale using the following architecture:

**Thread-Pooling over Process-Pooling**: The execution engine utilizes `concurrent.futures.ThreadPoolExecutor`. While Process Pools are typically standard for CPU-bound tasks, spawning multiple concurrent instances of PyTorch/Whisper models across separated memory spaces causes severe memory leaks and worker crashes on Windows environments. Threads share memory space, ensuring stable, cross-platform batch execution.

**Lazy Loading AI Models**: Whisper and SpeechBrain models are initialized globally but loaded lazily via singleton patterns. This prevents GPU/RAM bottlenecking during parallel worker initialization.

**Fault Tolerance**: If a single corrupted audio file crashes an AI model, the pipeline catches the exception, assigns a 0.0 score to that specific metric, and continues processing the rest of the batch without failing.

---

##  Metrics Used & Why
To ensure high-fidelity training data, we extract a blend of standard DSP (Digital Signal Processing) metrics and AI acoustic signals:

| Metric | Purpose | Why It Matters |
|--------|---------|----------------|
| **SNR (Signal-to-Noise Ratio)** | Measures voice strength vs background noise | Low SNR destroys TTS training quality |
| **Clipping Ratio** | Detects microphone distortion from being too close/loud | Distorted audio is unrecoverable for training |
| **Silence Ratio** | Identifies files with excessive dead air | Training on silence wastes compute resources |
| **RMS (Root Mean Square)** | Measures overall perceived loudness | Ensures volume consistency across dataset |
| **ASR Confidence (Whisper)** | Proxy for human intelligibility | Low confidence = mumbled/obscured speech |
| **Language ID (SpeechBrain)** | Ensures dataset language purity | Flags language switches or mislabeled data |

---

##  Known Data Edge Cases (Sub-Second Audio)
When processing heavily chunked datasets like IndicVoices (where files are frequently under 1.0 second long), standard AI models exhibit known fallback behaviors due to a lack of continuous acoustic data.

### 1. The "Bengali" (bn) Fallback Bias
The SpeechBrain language identification model requires ~3 to 5 seconds of continuous speech (vowels, consonants, rhythm) to accurately classify a language. When fed a 0.5-second clip, the model lacks sufficient data and its confidence scores collapse. The neural network essentially "gives up" and defaults to its baseline mathematical bias, which in this system manifests as classifying everything as Bengali (bn).

### 2. Whisper ASR Confidence Defaulting to 1.0
Similarly, when Whisper processes a sub-second clip, it often fails to generate the `avg_logprob` (average log-probability) metadata used to calculate confidence. The pipeline is built to handle this gracefully; however, users should be aware that sub-second clips may bypass ASR confidence checks.

### 3. Micro-Decimal SNR Values
Standard studio audio yields SNR values of 10.0 to 20.0 dB. Sub-second crowd-sourced clips lack a sustained voice signal versus a measurable noise floor. Consequently, SNR calculations yield microscopic values (e.g., 0.0015). The dashboard's filtering logic is specifically designed to handle these micro-decimal distributions to prevent mass-discarding of valid data.

---

##  Project Structure

```
indic-audio-pipeline/
  app.py                    # Streamlit dashboard
  backend/
    __init__.py
    metrics.py              # DSP and AI-based quality metrics
    processor.py            # Parallel processing engine
    spectrogram.py          # Mel-spectrogram generation
    report.py               # PDF report generation
  setup_dataset.py         # Download IndicVoices dataset
  create_sample_dataset.py # Create small sample for testing
  extract_audio.py         # Extract audio from parquet files
  requirements.txt         # Python dependencies
  README.md                # This file
```

---

##  Dataset Setup

### Full Dataset (3GB)
For the complete dataset with Hindi, Tamil, Telugu, and Bengali:

```bash
python setup_dataset.py ./test_data
```

### Sample Dataset (40MB)
For quick testing and development:

```bash
python create_sample_dataset.py ./test_data
```

---


---

##  Troubleshooting

**`ffmpeg was not found`** - Whisper needs ffmpeg to decode .flac files. Install it and restart your terminal.

**All files being discarded** - Check the ASR confidence threshold. Drop it to 0.05-0.1 for short clips. Also verify silence ratio isn't too strict for your dataset.

**Language detection showing "Bengali"** - This is expected for sub-second clips. The model needs 3+ seconds for accurate classification.

**Micro-decimal SNR values** - Normal for sub-second crowd-sourced audio. The dashboard handles these appropriately.

---

##  Acknowledgments

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) - IndicVoices dataset
- [OpenAI Whisper](https://github.com/openai/whisper) - ASR confidence scoring
- [SpeechBrain](https://speechbrain.github.io/) - Language identification
- [Streamlit](https://streamlit.io/) - Dashboard framework
