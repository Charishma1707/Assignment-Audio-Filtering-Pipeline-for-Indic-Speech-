"""
PDF Report Generation Module

This module creates professional PDF reports for audio quality assessment results.
It uses fpdf2 to generate comprehensive reports with statistics, charts,
and recommendations based on the audio analysis.

The main function generate_pdf_report() creates a complete report including:
- Executive summary with key findings
- Dataset statistics and visualizations
- Quality metrics analysis
- Filtering decisions and recommendations
"""

import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime
from typing import Dict, Any


class QualityReportPDF(FPDF):
    """
    Custom PDF class for generating professional quality reports.
    
    This class extends FPDF to provide custom formatting for audio quality
    assessment reports, including headers, footers, and styled sections.
    """

    def __init__(self):
        """
        Initialize the PDF report with default settings.
        """
        super().__init__()
        # Enable automatic page breaks with 15mm margin
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """
        Create the header for each page with title and timestamp.
        """
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Indic Audio Quality Assessment Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """
        Create the footer with page number.
        """
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        """
        Create a styled chapter title with background color.
        
        Args:
            title (str): The chapter title text
        """
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)  # Light blue background
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.set_fill_color(255, 255, 255)   # Reset to white
        self.ln(5)

    def section_title(self, title):
        """
        Create a styled section title with blue text.
        
        Args:
            title (str): The section title text
        """
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 100, 200)  # Blue text
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)       # Reset to black
        self.ln(3)

    def metrics_table(self, df: pd.DataFrame):
        """
        Create a formatted table showing summary statistics for all metrics.
        
        Args:
            df (pd.DataFrame): DataFrame containing the audio metrics
        """
        # Table header
        self.set_font('Arial', 'B', 11)
        self.cell(40, 8, 'Metric',   1, 0, 'C')
        self.cell(40, 8, 'Mean',     1, 0, 'C')
        self.cell(40, 8, 'Std Dev',  1, 0, 'C')
        self.cell(40, 8, 'Range',    1, 0, 'C')
        self.ln(8)

        self.set_font('Arial', '', 10)

        # Define which metrics to display and their formatting
        metrics_to_show = [
            ('snr',            'SNR (dB)',       1),
            ('clipping_ratio', 'Clipping Ratio', 2),
            ('silence_ratio',  'Silence Ratio',  2),
            ('rms',            'RMS Energy',     3),
            ('duration',       'Duration (s)',   1),
            ('asr_confidence', 'ASR Confidence', 3),
        ]

        # Generate table rows for each metric
        for metric, label, decimals in metrics_to_show:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val  = df[metric].std()
                min_val  = df[metric].min()
                max_val  = df[metric].max()

                self.cell(40, 8, label,                                             1, 0, 'L')
                self.cell(40, 8, f'{mean_val:.{decimals}f}',                       1, 0, 'C')
                self.cell(40, 8, f'{std_val:.{decimals}f}',                        1, 0, 'C')
                self.cell(40, 8, f'[{min_val:.{decimals}f}, {max_val:.{decimals}f}]', 1, 0, 'C')
                self.ln(8)

        self.ln(10)

    def add_plot(self, fig, title: str, width: int = 180):
        """
        Add a matplotlib plot to the PDF report.
        
        Args:
            fig: Matplotlib figure object
            title (str): Title to display above the plot
            width (int): Width of the plot in the PDF (default: 180)
        """
        # Convert matplotlib figure to PNG in memory
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)

        # Add section title and image
        self.section_title(title)
        self.image(img_buffer, w=width)
        self.ln(10)

        # Close the figure to free memory
        plt.close(fig)


def generate_pdf_report(df: pd.DataFrame, output_path: str = "Quality_Report.pdf") -> str:
    """
    Generate a comprehensive PDF report for audio quality assessment results.
    
    This is the main function that creates a complete professional report
    including executive summary, statistics, visualizations, and recommendations.
    
    Args:
        df (pd.DataFrame): DataFrame containing audio metrics and analysis results
        output_path (str): Path where the PDF report will be saved
        
    Returns:
        str: Path to the generated PDF report
        
    Raises:
        Exception: If report generation fails
    """
    try:
        # Initialize PDF document
        pdf = QualityReportPDF()
        pdf.add_page()

        # ── Executive Summary ─────────────────────────────────────────────
        pdf.chapter_title('Executive Summary')

        # Calculate key statistics
        total_files = len(df)
        if 'Decision' in df.columns:
            keep_count      = len(df[df['Decision'] == 'KEEP'])
            discard_count   = len(df[df['Decision'] == 'DISCARD'])
            keep_percentage    = (keep_count    / total_files * 100) if total_files else 0
            discard_percentage = (discard_count / total_files * 100) if total_files else 0
        else:
            keep_count = discard_count = keep_percentage = discard_percentage = 0

        avg_snr      = df['snr'].mean()      if 'snr'      in df.columns else 0.0
        avg_duration = df['duration'].mean() if 'duration' in df.columns else 0.0

        # Create executive summary text
        summary_text = (
            f"This report presents analysis of {total_files} audio files "
            f"from Indic speech dataset.\n\n"
            f"Key Findings:\n"
            f"  Total files processed: {total_files}\n"
            f"  Files recommended to KEEP: {keep_count} ({keep_percentage:.1f}%)\n"
            f"  Files recommended to DISCARD: {discard_count} ({discard_percentage:.1f}%)\n"
            f"  Average Signal-to-Noise Ratio: {avg_snr:.1f} dB\n"
            f"  Average duration: {avg_duration:.1f} seconds\n\n"
            f"The quality assessment combines traditional DSP metrics and advanced AI-based "
            f"analysis including voice activity detection, ASR confidence scoring, and "
            f"language identification."
        )

        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, summary_text)
        pdf.ln(10)

        # ── Dataset Statistics ────────────────────────────────────────────
        pdf.chapter_title('Dataset Statistics')
        pdf.section_title('Summary Statistics')
        pdf.metrics_table(df)

        # Language distribution chart
        if 'detected_language' in df.columns:
            lang_counts = df['detected_language'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            lang_counts.plot(kind='bar', ax=ax)
            ax.set_title('Detected Languages in Dataset')
            ax.set_xlabel('Language')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.add_plot(fig, 'Language Distribution')

        # ── Quality Metrics ───────────────────────────────────────────────
        pdf.chapter_title('Quality Metrics Analysis')

        # SNR distribution histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['snr'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title('Signal-to-Noise Ratio (SNR) Distribution')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        pdf.add_plot(fig, 'SNR Distribution')

        # ASR confidence and clipping ratio side-by-side
        if 'asr_confidence' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # ASR confidence histogram
            ax1.hist(df['asr_confidence'], bins=20, alpha=0.7, color='green', edgecolor='black')
            ax1.set_title('ASR Confidence Distribution')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

            # Clipping ratio histogram
            ax2.hist(df['clipping_ratio'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_title('Clipping Ratio Distribution')
            ax2.set_xlabel('Clipping Ratio')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.add_plot(fig, 'ASR Confidence & Clipping Ratio')

        # ── Decision Analysis ─────────────────────────────────────────────
        if 'Decision' in df.columns:
            pdf.section_title('Filtering Decisions')

            # Pie chart for keep/discard decisions
            fig, ax = plt.subplots(figsize=(8, 8))
            decision_counts = df['Decision'].value_counts()
            colors = ['lightgreen', 'lightcoral']
            ax.pie(
                decision_counts.values,
                labels=decision_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax.set_title('Keep vs Discard Decisions')
            pdf.add_plot(fig, 'Filtering Decisions')

        # ── Recommendations ───────────────────────────────────────────────
        pdf.chapter_title('Recommendations')

        # Determine quality and noise level labels
        quality_label = (
            'good'     if keep_percentage > 70 else
            'moderate' if keep_percentage > 50 else
            'poor'
        )
        noise_label = (
            'high'     if avg_snr >= 25 else
            'moderate' if avg_snr >= 15 else
            'low'
        )

        # Generate recommendations based on analysis
        recommendations = (
            f"Based on the analysis, the following recommendations are provided:\n\n"
            f"1. Dataset Quality: The dataset shows {quality_label} overall quality "
            f"with {keep_percentage:.1f}% of files meeting quality criteria.\n\n"
            f"2. Noise Levels: Average SNR of {avg_snr:.1f} dB indicates {noise_label} noise levels.\n\n"
            f"3. Processing Recommendations:\n"
            f"   - Consider audio enhancement for files with SNR < 15 dB\n"
            f"   - Investigate files with high clipping ratios for potential distortion\n"
            f"   - Review files with low ASR confidence for speech clarity issues\n\n"
            f"4. Quality Thresholds: Current thresholds can be adjusted based on "
            f"specific use case requirements."
        )

        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, recommendations)

        # ── Save PDF ──────────────────────────────────────────────────────
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save the PDF file
        pdf.output(output_path)
        print(f"Quality report saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        # Re-raise the exception so the calling function can handle it
        raise
