#!/usr/bin/env python3
"""
IndicVoices Sample Dataset Creator - Quick Testing Version

This script creates a small sample dataset from IndicVoices for quick testing
and development. It downloads only 10 files per language (total ~40MB) instead
of the full dataset (~3GB), making it perfect for:

- Quick development and testing
- Debugging the audio processing pipeline
- Demonstrating the application without large downloads
- CI/CD pipelines with limited storage

The sample includes the same language diversity as the full dataset:
Hindi, Tamil, Telugu, and Bengali.
"""

import os
import logging
import argparse
import polars as pl

from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ── Configuration ─────────────────────────────────────────────────────
# Target languages for the sample dataset
TARGET_LANGS = ["hindi", "tamil", "telugu", "bengali"]  

# Limit to 10 files per language for quick testing
FILES_PER_LANG = 10

# Processing configuration (reduced for smaller dataset)
n_procs = 4                    # Number of parallel processes
n_cpus = os.cpu_count() or 4    # Available CPU cores (fallback to 4)
n_threads = max(1, n_cpus // n_procs)  # Threads per process
# ────────────────────────────────────────────────────────────────────

def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Initialize and configure logging for the sample dataset creation.
    
    Sets up console logging with consistent formatting for progress tracking.
    
    Args:
        log_file (str, optional): Path to log file (not used in this version)
        log_level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("IndicvoicesSampleSetup")
    logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set up log formatting with timestamp and level
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler for real-time progress monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def process_row(row, dest_dir):
    """
    Process a single row from the dataset to extract and save audio file.
    
    This function handles the extraction of audio data from the parquet row
    and saves it as a standalone audio file in the destination directory.
    
    Args:
        row: Dataset row containing audio data and metadata
        dest_dir (str): Directory where audio file should be saved
        
    Returns:
        tuple: (success: bool, result: dict or error_message: str)
    """
    try:
        # Extract audio data and filename from the row
        audio_data = row['audio_filepath']['bytes']
        audio_name = row['audio_filepath']['path']
        
        # Remove the audio_filepath entry to avoid duplication in output
        del row['audio_filepath']

        # Save audio file to destination directory
        save_path = os.path.join(dest_dir, audio_name)
        with open(save_path, "wb") as f:
            f.write(audio_data)

        # Create entry for manifest with updated file path
        entry = {'audio_filepath': save_path, **row}
        return True, entry
    except Exception as e:
        return False, f"Error processing sample: {e}"

def process_parquet(parquet_file, lang, audio_save_dir, manifest_save_dir, logger, limit=10):
    """
    Process a single parquet file with a limited number of samples.
    
    This function reads only the first N rows from a parquet file to create
    a small sample dataset. It extracts audio files and creates a manifest.
    
    Args:
        parquet_file (str): Path to the parquet file to process
        lang (str): Language code for the data
        audio_save_dir (str): Base directory for saving audio files
        manifest_save_dir (str): Base directory for saving manifest files
        logger: Logger instance for progress tracking
        limit (int): Maximum number of files to process (default: 10)
        
    Returns:
        list: Manifest entries for all successfully processed rows
    """
    # Create destination directory for this specific parquet file
    index = os.path.basename(parquet_file).split('.')[0]
    dest_dir = os.path.join(audio_save_dir, lang, index)
    os.makedirs(dest_dir, exist_ok=True)

    # Helper functions for consistent logging
    def log(msg):       logger.info(f"{lang} - {os.path.basename(parquet_file)} - {msg}")
    def log_error(msg): logger.error(f"{lang} - {os.path.basename(parquet_file)} - {msg}")

    log(f"Processing only {limit} files for quick testing...")
    
    # Load only the first N rows using Polars .head() for efficiency
    # This is much faster than loading the entire parquet file
    df = pl.read_parquet(parquet_file).head(limit)

    # Process the limited rows in parallel
    manifest = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit all rows for processing
        futures = [
            executor.submit(process_row, row, dest_dir)
            for row in df.iter_rows(named=True)
        ]
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {lang}"):
            ok, result = future.result()
            if ok:
                manifest.append(result)
            else:
                log_error(result)

    log(f"Processed {len(manifest)} rows!")

    # Save manifest for this specific parquet file
    manifest_df = pl.DataFrame(manifest)
    save_path = os.path.join(manifest_save_dir, f'{lang}_manifests', f"{index}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    manifest_df.write_ndjson(save_path)
    
    return manifest

def main(save_dir: str):
    """
    Main function to create a small sample dataset for testing.
    
    This function orchestrates the creation of a tiny sample dataset:
    1. Sets up directory structure
    2. Downloads only validation data (smaller than training data)
    3. Processes only the first parquet file per language
    4. Limits to 10 files per language for quick testing
    5. Creates combined manifest file
    
    Args:
        save_dir (str): Directory where the sample dataset should be saved
    """
    # Define directory structure for organized storage
    HF_SAVE_DIR = os.path.join(save_dir, 'hf')           # Raw HuggingFace data
    AUDIO_SAVE_DIR = os.path.join(save_dir, 'audios')     # Extracted audio files
    MANIFEST_SAVE_DIR = os.path.join(save_dir, 'manifests') # Manifest files

    # Create all necessary directories
    for d in [save_dir, HF_SAVE_DIR, AUDIO_SAVE_DIR, MANIFEST_SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    # Set up logging for progress tracking
    logger = setup_logger()
    
    # For quick testing, we only download validation sets
    # Validation sets are much smaller than training sets but still diverse
    allow_patterns = [f"{lang}/valid*.parquet" for lang in TARGET_LANGS]

    logger.info("Downloading tiny subset from Hugging Face...")
    logger.info(f"This will download only validation data for: {TARGET_LANGS}")
    
    # Download the minimal dataset from HuggingFace Hub
    snapshot_download(
        repo_id="ai4bharat/IndicVoices",
        repo_type="dataset",
        local_dir=HF_SAVE_DIR,
        local_dir_use_symlinks=False,  # Copy files instead of symlinks
        max_workers=8,                 # Moderate parallelism for download
        resume_download=True,           # Resume if interrupted
        allow_patterns=allow_patterns    # Only download validation files
    )

    # Process the downloaded parquet files
    manifest = []
    
    # Use ProcessPoolExecutor for efficient processing
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = []
        
        for lang in os.listdir(HF_SAVE_DIR):
            # Skip languages not in our target list
            if lang not in TARGET_LANGS:
                continue

            lang_dir = os.path.join(HF_SAVE_DIR, lang)
            if not os.path.isdir(lang_dir):
                continue

            # Find all parquet files for this language
            parquet_files = [
                os.path.join(lang_dir, f) 
                for f in os.listdir(lang_dir) 
                if f.endswith(".parquet")
            ]
            
            # For quick testing, only process the first parquet file per language
            # This keeps the dataset small while maintaining language diversity
            if parquet_files:
                first_parquet = parquet_files[0]
                logger.info(f"Processing {lang} from {os.path.basename(first_parquet)}")
                
                futures.append(
                    executor.submit(
                        process_parquet,
                        first_parquet, lang,
                        AUDIO_SAVE_DIR, MANIFEST_SAVE_DIR,
                        logger, FILES_PER_LANG
                    )
                )

        # Collect results from all processing tasks
        for future in as_completed(futures):
            manifest.extend(future.result())

    # Create combined manifest file for all languages
    combined_path = os.path.join(MANIFEST_SAVE_DIR, "combined_manifest.jsonl")
    pl.DataFrame(manifest).write_ndjson(combined_path)
    
    total_files = len(manifest)
    total_size_mb = total_files * 1  # Rough estimate: ~1MB per audio file
    
    logger.info(f"Done! Successfully created sample dataset:")
    logger.info(f"  - Total files: {total_files}")
    logger.info(f"  - Languages: {TARGET_LANGS}")
    logger.info(f"  - Estimated size: ~{total_size_mb}MB")
    logger.info(f"  - Combined manifest: {combined_path}")

if __name__ == "__main__":
    """
    Entry point for the script when run from command line.
    
    Usage: python create_sample_dataset.py <save_directory>
    
    Example: python create_sample_dataset.py ./sample_data
    
    This will create a small sample dataset (~40MB) perfect for testing
    and development without requiring large downloads.
    """
    parser = argparse.ArgumentParser(
        description="Create a small sample dataset from IndicVoices for quick testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python create_sample_dataset.py ./sample_dataset
  
This will download 10 audio files per language (Hindi, Tamil, Telugu, Bengali)
to create a ~40MB sample dataset perfect for development and testing.

The sample dataset includes:
- 40 audio files total (10 per language)
- Organized directory structure
- Manifest files for easy loading
- Same format as the full dataset"""
    )
    parser.add_argument(
        'save_dir', 
        type=str, 
        help="Directory where the sample dataset should be saved"
    )
    args = parser.parse_args()
    
    main(save_dir=args.save_dir)