"""
Audio Processing Module

This module handles the main audio processing pipeline for quality assessment.
It's designed to process multiple audio files in parallel, making it efficient
for large datasets.

The main function run_pipeline() orchestrates the entire process:
1. Discover audio files in the specified directory
2. Process them in parallel using ThreadPoolExecutor
3. Collect results and generate summary statistics
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import time
import glob

from .metrics import compute_all_metrics


def process_single_file(filepath: str) -> Dict[str, Any]:
    """
    Process a single audio file and compute all quality metrics.
    
    This function serves as a wrapper around compute_all_metrics() but adds
    error handling to ensure the pipeline continues even if individual files
    fail to process.
    
    Args:
        filepath (str): Full path to the audio file to process
        
    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics or default
                       values if processing fails
    """
    try:
        # Try to compute all metrics for this file
        return compute_all_metrics(filepath)
    except Exception as e:
        # If anything goes wrong, return default values so the pipeline doesn't crash
        print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return {
            "filename":          os.path.basename(filepath),
            "filepath":          filepath,
            "error":             str(e),
            "duration":          0.0,
            "snr":               0.0,
            "clipping_ratio":    1.0,
            "silence_ratio":     1.0,
            "rms":               0.0,
            "asr_confidence":    0.0,
            "detected_language": "unknown"
        }


def discover_audio_files(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    Find all audio files in a directory and its subdirectories.
    
    This function searches recursively through the specified folder and returns
    a sorted list of all audio files matching the supported extensions.
    
    Args:
        folder_path (str): Path to the folder to search for audio files
        extensions (List[str], optional): List of file extensions to include.
                                         Defaults to common audio formats.
        
    Returns:
        List[str]: Sorted list of full paths to audio files
    """
    # Default to common audio formats if none specified
    if extensions is None:
        extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']

    audio_files = []
    
    # Search for each extension recursively
    for ext in extensions:
        # Create pattern like "folder/**/*.wav"
        pattern = os.path.join(folder_path, '**', ext)
        found_files = glob.glob(pattern, recursive=True)
        audio_files.extend(found_files)

    # Remove duplicates and sort the results
    return sorted(list(set(audio_files)))


def run_pipeline(audio_dir: str) -> pd.DataFrame:
    """
    Main function to run the complete audio quality assessment pipeline.
    
    This is the entry point for processing all audio files in a directory.
    It handles file discovery, parallel processing, and result aggregation.
    
    Args:
        audio_dir (str): Path to the directory containing audio files
        
    Returns:
        pd.DataFrame: DataFrame containing quality metrics for all processed files
                     Returns empty DataFrame if no files are found or processed
    """
    # Validate the input directory
    if not os.path.exists(audio_dir):
        raise ValueError(f"Directory does not exist: {audio_dir}")

    if not os.path.isdir(audio_dir):
        raise ValueError(f"Path is not a directory: {audio_dir}")

    print(f"Searching for audio files in {audio_dir}...")
    audio_files = discover_audio_files(audio_dir)

    if not audio_files:
        print("No audio files found in the specified directory.")
        return pd.DataFrame()

    print(f"Found {len(audio_files)} audio files to process")

    # Determine optimal number of worker threads
    # We cap at 4 to avoid memory issues with ML models and don't exceed file count
    max_workers = min(max(1, os.cpu_count() - 1), len(audio_files), 4)
    print(f"Processing with {max_workers} parallel workers")

    # Track timing and results
    start_time = time.time()
    results = []
    failed_files = []

    # Process files in parallel using ThreadPoolExecutor
    # Note: We use ThreadPool instead of ProcessPool to avoid reloading ML models
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(process_single_file, fp): fp
            for fp in audio_files
        }

        # Show progress bar and collect results
        with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    if "error" in result:
                        failed_files.append(filepath)
                        print(f"\nFailed: {os.path.basename(filepath)} - {result['error']}")
                    else:
                        results.append(result)
                except Exception as e:
                    failed_files.append(filepath)
                    print(f"\nException: {os.path.basename(filepath)} - {str(e)}")
                pbar.update(1)

    # Calculate processing time
    processing_time = time.time() - start_time

    if not results:
        print("No files were successfully processed.")
        return pd.DataFrame()

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Define desired column order for consistency
    column_order = [
        'filename', 'filepath', 'duration', 'snr', 'clipping_ratio',
        'silence_ratio', 'rms', 'asr_confidence', 'detected_language'
    ]
    
    # Ensure all expected columns exist with appropriate defaults
    for col in column_order:
        if col not in df.columns:
            df[col] = '' if col in ('filename', 'filepath', 'detected_language') else 0.0

    # Reorder columns: main metrics first, then any extra columns
    existing_cols = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + extra_cols]

    # Print processing summary
    total_files = len(audio_files)
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files found:      {total_files}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed:                 {len(failed_files)}")
    print(f"Success rate:           {len(results)/total_files*100:.1f}%")
    print(f"Processing time:        {processing_time:.2f}s  "
          f"({processing_time/total_files:.2f}s per file)")
    print(f"Total audio duration:   {df['duration'].sum()/3600:.2f} hours")
    print(f"Average file duration:  {df['duration'].mean():.2f}s")

    # Show failed files (limited to avoid spam)
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:10]:
            print(f"  - {os.path.basename(f)}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more")

    return df
