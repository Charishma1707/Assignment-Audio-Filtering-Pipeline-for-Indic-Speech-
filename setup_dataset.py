#!/usr/bin/env python3
"""
IndicVoices Dataset Processing Script - Limited Size Version

This script downloads and processes the IndicVoices dataset from HuggingFace,
selecting only specific languages to keep the total size around 3GB.
This makes it manageable for development and testing while maintaining
diversity across major Indian languages.

The script:
1. Downloads selected language data from HuggingFace Hub
2. Extracts audio files from parquet format
3. Creates organized directory structure
4. Generates manifest files for each language
5. Creates a combined manifest for all languages

Target languages: Hindi, Tamil, Telugu, Bengali (~3GB total)
"""

import os
import logging
import argparse
import polars as pl

from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


# ── Configuration ─────────────────────────────────────────────────────
# Target languages to download (selected to keep size ~3GB)
TARGET_LANGS = ["hindi", "tamil", "telugu", "bengali"]

# Processing configuration
n_procs = 8                    # Number of parallel processes
n_cpus = os.cpu_count() - 1     # Available CPU cores
n_threads = n_cpus // n_procs     # Threads per process
# ────────────────────────────────────────────────────────────────────────────


def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Initialize and configure logging for the dataset processing script.
    
    Sets up both console and file logging with consistent formatting.
    
    Args:
        log_file (str, optional): Path to log file. If None, only console logging.
        log_level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("IndicvoicesSetup")
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

    # File handler for persistent logging
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

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
        entry = {
            'audio_filepath': save_path,
            **row  # Include all other metadata
        }

        return True, entry
    except Exception as e:
        return False, f"Error processing sample: {e}"


def process_parquet(parquet_file, lang, audio_save_dir, manifest_save_dir, logger):
    """
    Process a single parquet file containing audio data.
    
    This function reads a parquet file, extracts audio files from each row,
    saves them to the appropriate directory structure, and creates a manifest
    file tracking all processed entries.
    
    Args:
        parquet_file (str): Path to the parquet file to process
        lang (str): Language code for the data
        audio_save_dir (str): Base directory for saving audio files
        manifest_save_dir (str): Base directory for saving manifest files
        logger: Logger instance for progress tracking
        
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

    log("Processing...")
    
    # Load the parquet file using Polars for efficient reading
    df = pl.read_parquet(parquet_file)

    # Process all rows in parallel using ThreadPoolExecutor
    manifest = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Submit all rows for processing
        futures = [
            executor.submit(process_row, row, dest_dir)
            for row in df.iter_rows(named=True)
        ]
        
        # Collect results with progress bar
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing {lang}/{os.path.basename(parquet_file)}"
        ):
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

    log(f"Saved manifest to {save_path}")
    return manifest


def main(save_dir: str):
    """
    Main function to orchestrate the dataset download and processing.
    
    This function handles the complete workflow:
    1. Sets up directory structure
    2. Downloads selected language data from HuggingFace
    3. Processes all parquet files to extract audio
    4. Creates individual and combined manifest files
    
    Args:
        save_dir (str): Base directory where all data should be saved
    """
    # Define directory structure for organized storage
    HF_SAVE_DIR = os.path.join(save_dir, 'hf')           # Raw HuggingFace data
    AUDIO_SAVE_DIR = os.path.join(save_dir, 'audios')     # Extracted audio files
    MANIFEST_SAVE_DIR = os.path.join(save_dir, 'manifests') # Manifest files

    # Create all necessary directories
    for d in [save_dir, HF_SAVE_DIR, AUDIO_SAVE_DIR, MANIFEST_SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    # Set up logging
    log_file = os.path.join(save_dir, 'setup.log')
    if os.path.exists(log_file):
        os.remove(log_file)  # Start fresh log file
    logger = setup_logger(log_file=log_file)
    logger.info(f"Target languages: {TARGET_LANGS}")
    logger.info(f"Created directories under: {save_dir}")

    # Build download patterns to only get target languages and limited data
    allow_patterns = []
    for lang in TARGET_LANGS:
        allow_patterns.append(f"{lang}/valid*.parquet")     # Validation sets
        allow_patterns.append(f"{lang}/train-00000*.parquet") # First training split

    logger.info(f"Downloading IndicVoices for: {TARGET_LANGS}")
    logger.info(f"Allow patterns: {allow_patterns}")

    # Download dataset from HuggingFace Hub
    snapshot_download(
        repo_id="ai4bharat/IndicVoices",
        repo_type="dataset",
        local_dir=HF_SAVE_DIR,
        local_dir_use_symlinks=False,  # Copy files instead of symlinks
        max_workers=32,               # High parallelism for download
        resume_download=True,         # Resume if interrupted
        allow_patterns=allow_patterns # Only download target files
    )
    logger.info(f"Download complete → {HF_SAVE_DIR}")

    # Process all downloaded parquet files
    logger.info("Processing parquet files...")
    manifest = []

    # Use ProcessPoolExecutor for CPU-intensive parquet processing
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
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
            if not parquet_files:
                continue

            logger.info(f"Submitting {lang} — {len(parquet_files)} parquet file(s)")
            
            # Submit all parquet files for parallel processing
            futures = [
                executor.submit(
                    process_parquet,
                    parquet_file, lang,
                    AUDIO_SAVE_DIR, MANIFEST_SAVE_DIR,
                    logger
                )
                for parquet_file in parquet_files
            ]
            
            # Collect results with progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{lang} results"):
                manifest.extend(future.result())

    # Create combined manifest for all languages
    manifest_df = pl.DataFrame(manifest)
    combined_path = os.path.join(MANIFEST_SAVE_DIR, "combined_manifest.jsonl")
    manifest_df.write_ndjson(combined_path)
    logger.info(f"Done! Combined manifest → {combined_path}")


if __name__ == "__main__":
    """
    Entry point for the script when run from command line.
    
    Usage: python setup_dataset.py <save_directory>
    
    Example: python setup_dataset.py ./indic_dataset
    """
    parser = argparse.ArgumentParser(
        description="Download and process IndicVoices dataset for selected languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python setup_dataset.py ./my_dataset
  
This will download Hindi, Tamil, Telugu, and Bengali data (~3GB total)
to the './my_dataset' directory."""
    )
    parser.add_argument(
        'save_dir', 
        type=str, 
        help="Directory where the dataset should be saved"
    )
    args = parser.parse_args()
    
    main(save_dir=args.save_dir)