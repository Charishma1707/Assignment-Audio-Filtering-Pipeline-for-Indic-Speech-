"""
Backend Module for Audio Quality Pipeline

This package provides core audio processing functionality including
metrics computation, spectrogram generation, parallel processing, and reporting.

Author: Principal AI/Data Engineer
Purpose: Indic Audio Quality Filtering Pipeline
"""

__version__ = "1.0.0"
__author__ = "Principal AI/Data Engineer"

from .metrics import (
    compute_snr,
    compute_clipping_ratio,
    compute_silence_ratio,
    compute_rms,
    compute_asr_confidence,
    detect_language,
    compute_all_metrics
)

from .spectrogram import (
    generate_spectrogram,
    generate_spectrogram_for_display
)

from .processor import (
    process_single_file,
    run_pipeline
)

from .report import (
    generate_pdf_report
)

__all__ = [
    # Metrics functions
    'compute_snr',
    'compute_clipping_ratio', 
    'compute_silence_ratio',
    'compute_rms',
    'compute_asr_confidence',
    'detect_language',
    'compute_all_metrics',
    
    # Spectrogram functions
    'generate_spectrogram',
    'generate_spectrogram_for_display',
    
    # Processing functions
    'process_single_file',
    'run_pipeline',
    
    # Report functions
    'generate_pdf_report'
]
