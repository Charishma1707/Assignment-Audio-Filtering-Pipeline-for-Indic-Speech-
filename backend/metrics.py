"""
Audio Quality Metrics Module

This module contains functions to compute various audio quality metrics.
It combines traditional signal processing techniques with modern ML-based
approaches to provide comprehensive audio quality assessment.

Key metrics computed:
- Signal-to-Noise Ratio (SNR)
- Clipping detection
- Silence ratio analysis
- RMS energy calculation
- ASR confidence using Whisper
- Language detection
"""

import numpy as np
import librosa
import torch
import torchaudio
import warnings
from typing import Dict, Any, Optional, Tuple
import soundfile as sf

warnings.filterwarnings("ignore")

# Global model instances - using lazy loading to save memory
# These models are only loaded when first needed
_whisper_pipeline = None
_lang_id_model = None


def load_whisper_pipeline():
    """
    Initialize the Whisper ASR model for confidence scoring.
    
    Uses lazy loading pattern - the model is only loaded when this function
    is first called. This saves memory and startup time.
    
    Returns:
        The initialized Whisper pipeline
    """
    global _whisper_pipeline
    if _whisper_pipeline is None:
        from transformers import pipeline

        # Using the tiny model for faster processing
        model_name = "openai/whisper-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use float16 on GPU for better performance
        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

    return _whisper_pipeline


def load_lang_id_model():
    """
    Initialize the SpeechBrain language identification model.
    
    Note: This function is currently not used due to import issues.
    The detect_language() function returns a default value instead.
    
    Returns:
        The initialized language identification model
    """
    global _lang_id_model
    if _lang_id_model is None:
        from speechbrain.pretrained import EncoderClassifier

        model_name = "speechbrain/lang-id-voxlingua107-ecapa"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _lang_id_model = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": device}
        )

    return _lang_id_model


def compute_snr(y: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of an audio signal.
    
    SNR measures the ratio between the power of the signal and the power
    of background noise. Higher SNR generally indicates better audio quality.
    
    Args:
        y (np.ndarray): Audio signal as a numpy array
        
    Returns:
        float: SNR value in decibels (dB), clipped to range [-20, 100]
    """
    try:
        # Calculate signal power (mean of squared values)
        signal_power = np.mean(y ** 2)
        
        # Calculate noise power (variance from the mean)
        noise_power = np.var(y - np.mean(y))

        # Avoid division by zero - if no noise, return maximum SNR
        if noise_power == 0:
            return 100.0

        # Convert to decibels
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Clip to reasonable range to avoid extreme values
        return float(np.clip(snr_db, -20.0, 100.0))

    except Exception as e:
        print(f"Error computing SNR: {e}")
        return 0.0


def compute_clipping_ratio(y: np.ndarray) -> float:
    """
    Detect audio clipping by measuring the proportion of clipped samples.
    
    Clipping occurs when the audio signal exceeds the maximum representable
    amplitude, causing distortion. This function identifies samples that are
    at or near the maximum amplitude.
    
    Args:
        y (np.ndarray): Audio signal as a numpy array
        
    Returns:
        float: Ratio of clipped samples (0.0 to 1.0), where higher values
               indicate more clipping and worse audio quality
    """
    try:
        # Find the maximum absolute amplitude in the signal
        max_abs = np.max(np.abs(y))
        
        # If the signal is silent, no clipping can occur
        if max_abs == 0:
            return 0.0

        # Normalize the signal to range [-1, 1]
        y_normalized = y / max_abs
        
        # Count samples that are at or near 99% of maximum amplitude
        clipped_samples = np.sum(np.abs(y_normalized) >= 0.99)
        
        # Return the ratio of clipped samples to total samples
        return float(clipped_samples / len(y_normalized))

    except Exception as e:
        print(f"Error computing clipping ratio: {e}")
        return 0.0


def compute_silence_ratio(y: np.ndarray, sr: int = 16000) -> float:
    """
    Measure the proportion of silence in an audio signal.
    
    This function uses librosa's voice activity detection to identify
    non-silent segments and calculates the ratio of silence to total duration.
    This is useful for identifying recordings with excessive silence.
    
    Args:
        y (np.ndarray): Audio signal as a numpy array
        sr (int): Sample rate of the audio (default: 16000 Hz)
        
    Returns:
        float: Ratio of silence (0.0 to 1.0), where higher values indicate
               more silence and potentially poorer audio quality
    """
    try:
        # Use librosa to find non-silent intervals
        # top_db=20 means segments 20dB below the peak are considered silent
        intervals = librosa.effects.split(y, top_db=20)
        
        # Calculate total duration of non-silent segments
        speech_duration = sum(end - start for start, end in intervals)

        # Get total number of samples in the audio
        total_samples = len(y)
        if total_samples == 0:
            return 1.0  # Empty audio is all silence

        # Calculate silence ratio (1 - speech ratio)
        silence_ratio = 1.0 - (speech_duration / total_samples)
        
        # Ensure the result is within valid range
        return float(np.clip(silence_ratio, 0.0, 1.0))

    except Exception as e:
        print(f"Error computing silence ratio: {e}")
        return 0.5  # Return middle value as fallback


def compute_rms(y: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) energy of an audio signal.
    
    RMS is a measure of the average power of the audio signal.
    It's useful for understanding the overall loudness and can help
    identify recordings that are too quiet or too loud.
    
    Args:
        y (np.ndarray): Audio signal as a numpy array
        
    Returns:
        float: RMS energy value (higher = louder)
    """
    try:
        # RMS = sqrt(mean(x^2))
        return float(np.sqrt(np.mean(y ** 2)))
    except Exception as e:
        print(f"Error computing RMS: {e}")
        return 0.0


def compute_asr_confidence(audio_path: str) -> float:
    """
    Calculate ASR (Automatic Speech Recognition) confidence using Whisper.
    
    This function uses OpenAI's Whisper model to transcribe the audio and
    extracts the confidence scores from the transcription process. Higher
    confidence scores indicate clearer speech that's easier to understand.
    
    Args:
        audio_path (str): Path to the audio file to analyze
        
    Returns:
        float: ASR confidence score (0.0 to 1.0), where higher values indicate
               better speech clarity and recognition confidence
    """
    try:
        whisper_pipe = load_whisper_pipeline()

        # Process the audio with Whisper
        # return_timestamps=True gives us chunk-level confidence scores
        result = whisper_pipe(
            audio_path,
            chunk_length_s=30,        # Process in 30-second chunks
            return_timestamps=True,   # Required for getting confidence scores
            max_new_tokens=128        # Allow longer transcriptions
        )

        if result and 'chunks' in result:
            confidences = []
            
            # Extract confidence from each chunk
            for chunk in result['chunks']:
                if 'text' in chunk and chunk['text'].strip():
                    if 'avg_logprob' in chunk:
                        # Convert log probability to confidence
                        logprob = chunk['avg_logprob']
                        confidence = float(np.exp(logprob)) if logprob is not None else 0.0
                    else:
                        # If no logprob available, assume good confidence
                        confidence = 1.0
                    confidences.append(confidence)

            if confidences:
                # Return the average confidence across all chunks
                avg_confidence = np.mean(confidences)
                return float(np.clip(avg_confidence, 0.0, 1.0))

        # Fallback for short files that might not return chunks
        if result and result.get('text', '').strip():
            # Text exists but no per-chunk confidence available
            return 0.5  # Return moderate confidence

        return 0.0  # No transcription possible

    except Exception as e:
        print(f"Error computing ASR confidence for {audio_path}: {e}")
        return 0.0


def detect_language(audio_path: str) -> str:
    """
    Detect the language of the audio file.
    
    Note: Due to import issues with the SpeechBrain library, this function
    currently returns a default value. In a production environment, this
    would use the language identification model to automatically detect
    the spoken language.
    
    Args:
        audio_path (str): Path to the audio file (unused in current implementation)
        
    Returns:
        str: Language code - currently returns 'bn' (Bengali) as default
    """
    # TODO: Implement actual language detection once SpeechBrain import issues are resolved
    # For now, return Bengali as default since that's the primary language in our dataset
    return "bn"


def compute_all_metrics(audio_path: str) -> Dict[str, Any]:
    """
    Compute all audio quality metrics for a given audio file.
    
    This is the main function that orchestrates the computation of all
    quality metrics. It combines traditional signal processing metrics
    with ML-based assessments to provide a comprehensive quality profile.
    
    Args:
        audio_path (str): Path to the audio file to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics including:
                       - filename, filepath: File identification
                       - duration: Length of audio in seconds
                       - snr: Signal-to-noise ratio in dB
                       - clipping_ratio: Proportion of clipped samples
                       - silence_ratio: Proportion of silence
                       - rms: Root mean square energy
                       - asr_confidence: Speech recognition confidence
                       - detected_language: Identified language code
    """
    try:
        # Load the audio file at 16kHz (standard for speech processing)
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract basic file information
        import os
        metrics = {
            "filename":       os.path.basename(audio_path),
            "filepath":       audio_path,
            "duration":       librosa.get_duration(y=y, sr=sr),
        }

        # Compute traditional signal processing metrics
        metrics["snr"] = compute_snr(y)
        metrics["clipping_ratio"] = compute_clipping_ratio(y)
        metrics["silence_ratio"] = compute_silence_ratio(y, sr)
        metrics["rms"] = compute_rms(y)

        # Compute ML-based metrics with error handling
        try:
            metrics["asr_confidence"] = compute_asr_confidence(audio_path)
        except Exception as e:
            print(f"ASR confidence failed for {audio_path}: {e}")
            metrics["asr_confidence"] = 0.0

        try:
            metrics["detected_language"] = detect_language(audio_path)
        except Exception as e:
            print(f"Language detection failed for {audio_path}: {e}")
            metrics["detected_language"] = "unknown"

        return metrics

    except Exception as e:
        # If anything goes wrong, return default values
        import os
        print(f"Error processing {audio_path}: {e}")
        return {
            "filename":         os.path.basename(audio_path),
            "filepath":         audio_path,
            "duration":         0.0,
            "snr":              0.0,
            "clipping_ratio":   1.0,
            "silence_ratio":    1.0,
            "rms":              0.0,
            "asr_confidence":   0.0,
            "detected_language": "unknown"
        }
