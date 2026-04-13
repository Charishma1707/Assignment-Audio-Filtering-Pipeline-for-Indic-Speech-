"""
Audio Extraction Tool for IndicVoices Dataset

This script extracts audio files from downloaded IndicVoices parquet files
and saves them as individual audio files with proper directory structure.
It also creates manifest files to track the extracted data.

The script processes:
- All language folders in the downloaded data
- All parquet files within each language folder
- Extracts audio data row by row to manage memory efficiently
- Creates organized directory structure by language and parquet file
- Generates manifest files for each processed parquet file

Usage: python extract_audio.py
"""

import os
import polars as pl
from tqdm import tqdm


def extract_audio():
    """
    Main function to extract audio files from downloaded IndicVoices dataset.
    
    This function:
    1. Locates the downloaded HuggingFace data
    2. Creates output directories for audio files and manifests
    3. Processes each language folder and parquet file
    4. Extracts audio data row by row for memory efficiency
    5. Saves individual audio files and creates manifest files
    """
    # Define directory structure
    base_dir = "test_data"
    hf_dir = os.path.join(base_dir, "hf")           # Downloaded parquet files
    audio_dir = os.path.join(base_dir, "audios")     # Extracted audio files
    manifest_dir = os.path.join(base_dir, "manifests") # Manifest files

    # Create output directories if they don't exist
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    # Check if the downloaded data exists
    if not os.path.exists(hf_dir):
        print(f"Error: {hf_dir} not found.")
        print("Please run the dataset setup script first to download the data.")
        return

    print("Starting audio extraction from IndicVoices dataset...")
    print(f"Source directory: {hf_dir}")
    print(f"Output directory: {audio_dir}")

    # Process each language folder in the downloaded data
    for lang in os.listdir(hf_dir):
        lang_path = os.path.join(hf_dir, lang)
        if not os.path.isdir(lang_path):
            continue  # Skip non-directory items
            
        print(f"\n--- Processing Language: {lang.upper()} ---")
        
        # Find all parquet files for this language
        parquet_files = [f for f in os.listdir(lang_path) if f.endswith('.parquet')]
        
        if not parquet_files:
            print(f"No parquet files found for language: {lang}")
            continue
        
        # Process each parquet file
        for p_file in parquet_files:
            p_path = os.path.join(lang_path, p_file)
            index = p_file.split('.')[0]  # Extract filename without extension
            
            # Create a specific directory for these audio files
            dest_dir = os.path.join(audio_dir, lang, index)
            os.makedirs(dest_dir, exist_ok=True)
            
            print(f"Processing {p_file}...")
            
            try:
                # Read the parquet file using Polars for efficiency
                df = pl.read_parquet(p_path)
                print(f"Loaded {df.height} rows from {p_file}")
                
                manifest = []
                
                # Extract audio data row by row to manage memory efficiently
                # This approach prevents loading all audio data into memory at once
                for row in tqdm(df.iter_rows(named=True), 
                              total=df.height, 
                              desc=f"Extracting {p_file}"):
                    try:
                        # Extract audio data and filename from the row
                        audio_data = row['audio_filepath']['bytes']
                        audio_name = row['audio_filepath']['path']
                        
                        # Save the audio file to the destination directory
                        save_path = os.path.join(dest_dir, audio_name)
                        with open(save_path, "wb") as f:
                            f.write(audio_data)
                        
                        # Create manifest entry (remove audio_filepath to avoid duplication)
                        del row['audio_filepath']
                        entry = {'audio_filepath': save_path, **row}
                        manifest.append(entry)
                        
                    except Exception as e:
                        print(f"Error extracting file {audio_name}: {e}")
                        continue  # Continue with next file
                
                # Save manifest file to track what was extracted
                if manifest:
                    manifest_df = pl.DataFrame(manifest)
                    manifest_save_path = os.path.join(manifest_dir, f"{lang}_{index}_manifest.jsonl")
                    manifest_df.write_ndjson(manifest_save_path)
                    print(f"Saved manifest: {manifest_save_path}")
                    print(f"Successfully extracted {len(manifest)} audio files")
                else:
                    print(f"No audio files were extracted from {p_file}")
                
            except Exception as e:
                print(f"Error processing parquet file {p_file}: {e}")
                continue  # Continue with next file

    print("\n" + "="*60)
    print("AUDIO EXTRACTION COMPLETE!")
    print(f"Extracted audio files are available in: {audio_dir}")
    print(f"Manifest files are available in: {manifest_dir}")
    print("="*60)


if __name__ == "__main__":
    """
    Entry point for the script when run from command line.
    
    Usage: python extract_audio.py
    
    Make sure you have already downloaded the IndicVoices dataset
    using the setup_dataset.py script before running this extraction tool.
    """
    extract_audio()
