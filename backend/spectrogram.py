"""
Spectrogram Generation Module

This module handles the creation of spectrograms from audio files.
Spectrograms are visual representations of audio that show frequency
content over time, which is useful for audio analysis and quality assessment.

The module provides two main functions:
1. generate_spectrogram() - Creates and saves spectrogram images
2. generate_spectrogram_for_display() - Returns spectrogram data for web display
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


def generate_spectrogram(audio_path: str, output_path: str) -> Optional[str]:
    """
    Create and save a Mel-spectrogram image from an audio file.
    
    This function loads an audio file, computes its Mel-spectrogram,
    and saves the visualization as an image file. The spectrogram
    shows how the frequency content of the audio changes over time.
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str): Path where the spectrogram image will be saved
        
    Returns:
        Optional[str]: Path to the saved spectrogram image, or None if failed
    """
    try:
        # Load the audio file at 16kHz (standard for speech processing)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Compute the Mel-spectrogram
        # Mel scale approximates human hearing perception
        S = librosa.feature.melspectrogram(
            y=y,                    # Audio signal
            sr=sr,                  # Sample rate
            n_mels=128,            # Number of Mel frequency bins
            fmax=8000,             # Maximum frequency to display (8kHz)
            n_fft=2048,             # FFT window size
            hop_length=512          # Hop size between windows
        )
        
        # Convert power spectrogram to decibel scale for better visualization
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Create a figure for the spectrogram
        plt.figure(figsize=(12, 8))
        
        # Display the spectrogram using librosa's helper function
        librosa.display.specshow(
            S_db,                   # Spectrogram data in dB
            sr=sr,                  # Sample rate
            x_axis='time',          # X-axis shows time
            y_axis='mel',           # Y-axis shows Mel frequency
            cmap='viridis'          # Color scheme
        )
        
        # Add color bar to show decibel scale
        plt.colorbar(format='%+2.0f dB')
        
        # Add title with filename for identification
        plt.title(f'Mel-Spectrogram\n{os.path.basename(audio_path)}')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the spectrogram as an image file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Close the figure to free memory
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error generating spectrogram for {audio_path}: {e}")
        return None


def generate_spectrogram_for_display(audio_path: str) -> Optional[np.ndarray]:
    """
    Generate spectrogram data for real-time display in web applications.
    
    This function is optimized for web display (e.g., Streamlit) where
    we need the spectrogram data as a numpy array rather than a saved
    image file. This allows for faster updates and interactivity.
    
    Args:
        audio_path (str): Path to the input audio file
        
    Returns:
        Optional[np.ndarray]: Spectrogram data as a 2D numpy array in dB scale,
                             or None if processing failed
    """
    try:
        # Load the audio file at 16kHz
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Compute the Mel-spectrogram with same parameters as image generation
        S = librosa.feature.melspectrogram(
            y=y,                    # Audio signal
            sr=sr,                  # Sample rate
            n_mels=128,            # Number of Mel frequency bins
            fmax=8000,             # Maximum frequency (8kHz for speech)
            n_fft=2048,             # FFT window size
            hop_length=512          # Hop size between windows
        )
        
        # Convert to decibel scale for consistent visualization
        S_db = librosa.power_to_db(S, ref=np.max)
        
        return S_db
        
    except Exception as e:
        print(f"Error generating spectrogram data for {audio_path}: {e}")
        return None
