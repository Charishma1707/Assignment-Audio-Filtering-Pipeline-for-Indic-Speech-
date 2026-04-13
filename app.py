"""
Streamlit Dashboard for Audio Quality Assessment

This is the main web application interface for the audio quality filtering pipeline.
It provides an interactive dashboard where users can:
- Run the audio processing pipeline on a folder of audio files
- Adjust quality thresholds using interactive sliders
- View detailed analysis results with charts and visualizations
- Export results in various formats (CSV, JSON, PDF)
- Inspect individual files with spectrogram visualization

The application uses Streamlit for the web interface and Plotly for interactive charts.

To run: streamlit run app.py
"""

import os
import time
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.processor import run_pipeline
from backend.report import generate_pdf_report
from backend.spectrogram import generate_spectrogram_for_display

# ── Page Configuration ─────────────────────────────────────────────────────
# Set up the Streamlit page with title, icon, and layout preferences
st.set_page_config(
    page_title="Indic Audio QC",        # Browser tab title
    page_icon="🎙️",               # Favicon/icon
    layout="wide",                    # Use full screen width
    initial_sidebar_state="expanded",    # Show sidebar by default
)

# ── Custom CSS Styling ───────────────────────────────────────────────────
# Apply custom dark theme and styling to make the interface more professional
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

/* Main text styling */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background theme */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a6080;
    margin-bottom: 0.5rem;
}

/* Page title styling */
.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0;
}
.page-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #3d8ef0;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

/* Statistics cards styling */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
/* Color-coded top borders for different stat cards */
.stat-card.blue::before  { background: #3d8ef0; }
.stat-card.green::before { background: #22c55e; }
.stat-card.red::before   { background: #ef4444; }
.stat-card.amber::before { background: #f59e0b; }

.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5a6080;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: #ffffff;
    line-height: 1;
}
.stat-sub {
    font-size: 0.75rem;
    color: #5a6080;
    margin-top: 0.3rem;
}

/* ── section headings ── */
.section-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d8ef0;
    border-left: 3px solid #3d8ef0;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
}

/* ── metric row ── */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.metric-pill {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.metric-pill .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #5a6080;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-pill .val {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #e8eaf0;
}

/* ── keep/discard badge ── */
.badge-keep    { color: #22c55e; font-weight: 600; }
.badge-discard { color: #ef4444; font-weight: 600; }

/* ── dataframe tweaks ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── info / warning / error boxes ── */
.stAlert { border-radius: 8px; }

/* ── buttons ── */
.stButton > button {
    background: #3d8ef0;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
}
.stDownloadButton > button {
    background: #13151c;
    border: 1px solid #1e2130;
    color: #e8eaf0;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

/* hide streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plotly Configuration ─────────────────────────────────────────────────
# Define consistent dark theme for all Plotly charts
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",      # Transparent background
    plot_bgcolor="#13151c",                # Dark plot background
    font=dict(family="DM Sans, sans-serif", color="#9aa0b8", size=12),
    xaxis=dict(gridcolor="#1e2130", linecolor="#1e2130", zerolinecolor="#1e2130"),
    yaxis=dict(gridcolor="#1e2130", linecolor="#1e2130", zerolinecolor="#1e2130"),
    margin=dict(l=40, r=20, t=50, b=40),    # Chart margins
)


# ── Helper Functions ─────────────────────────────────────────────────────

def apply_filtering_logic(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Apply quality filtering logic to determine which files to keep or discard.
    
    This function evaluates each audio file against the specified quality thresholds
    and assigns a decision (KEEP/DISCARD) along with reasons for any failures.
    
    Args:
        df (pd.DataFrame): DataFrame containing audio metrics
        thresholds (dict): Dictionary with quality threshold values
        
    Returns:
        pd.DataFrame: DataFrame with added 'Decision' and 'Reason' columns
    """
    if df.empty:
        return df

    # Create a copy to avoid modifying the original
    df = df.copy()
    df["Decision"] = "KEEP"
    df["Reason"] = ""

    # Check each file against quality thresholds
    for idx, row in df.iterrows():
        reasons = []

        # Check SNR threshold
        if row["snr"] < thresholds["snr_min"]:
            reasons.append(f"SNR {row['snr']:.1f} dB < {thresholds['snr_min']:.1f}")

        # Check clipping ratio threshold
        if row["clipping_ratio"] > thresholds["clipping_max"]:
            reasons.append(f"Clipping {row['clipping_ratio']:.1%} > {thresholds['clipping_max']:.1%}")

        # Check silence ratio threshold
        if row["silence_ratio"] > thresholds["silence_max"]:
            reasons.append(f"Silence {row['silence_ratio']:.1%} > {thresholds['silence_max']:.1%}")

        # Check ASR confidence threshold (if available)
        if "asr_confidence" in df.columns and row["asr_confidence"] < thresholds["asr_min"]:
            reasons.append(f"ASR conf {row['asr_confidence']:.2f} < {thresholds['asr_min']:.2f}")

        # Assign decision based on failed checks
        if reasons:
            df.at[idx, "Decision"] = "DISCARD"
            df.at[idx, "Reason"] = "; ".join(reasons)
        else:
            df.at[idx, "Reason"] = "passed"

    return df


def stat_cards(df: pd.DataFrame):
    """
    Render the main statistics cards showing dataset overview.
    
    These four cards provide a quick summary of the dataset:
    - Total number of files processed
    - Number and percentage of files to keep
    - Number and percentage of files to discard
    - Total audio hours in the dataset
    
    Args:
        df (pd.DataFrame): DataFrame containing processed audio metrics
    """
    total = len(df)
    keep_n = len(df[df["Decision"] == "KEEP"]) if "Decision" in df.columns else 0
    discard_n = len(df[df["Decision"] == "DISCARD"]) if "Decision" in df.columns else 0
    total_hours = df["duration"].sum() / 3600 if "duration" in df.columns else 0.0

    # Create HTML for stat cards with custom styling
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card blue">
        <div class="stat-label">Total files</div>
        <div class="stat-value">{total:,}</div>
        <div class="stat-sub">in dataset</div>
      </div>
      <div class="stat-card green">
        <div class="stat-label">Keep</div>
        <div class="stat-value">{keep_n:,}</div>
        <div class="stat-sub">{keep_n/total*100:.1f}% of total</div>
      </div>
      <div class="stat-card red">
        <div class="stat-label">Discard</div>
        <div class="stat-value">{discard_n:,}</div>
        <div class="stat-sub">{discard_n/total*100:.1f}% of total</div>
      </div>
      <div class="stat-card amber">
        <div class="stat-label">Audio hours</div>
        <div class="stat-value">{total_hours:.1f}</div>
        <div class="stat-sub">kept: {df[df['Decision']=='KEEP']['duration'].sum()/3600:.1f} h</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def metric_strip(df: pd.DataFrame):
    """
    Render the secondary metrics row showing average values.
    
    This displays key quality metrics in a horizontal strip format:
    - Average SNR (Signal-to-Noise Ratio)
    - Average clipping ratio
    - Average silence ratio
    - Average RMS energy
    - Average ASR confidence
    - Number of unique languages detected
    
    Args:
        df (pd.DataFrame): DataFrame containing processed audio metrics
    """
    # Calculate average values for key metrics
    avg_snr = df["snr"].mean() if "snr" in df.columns else 0.0
    avg_clip = df["clipping_ratio"].mean() if "clipping_ratio" in df.columns else 0.0
    avg_sil = df["silence_ratio"].mean() if "silence_ratio" in df.columns else 0.0
    avg_rms = df["rms"].mean() if "rms" in df.columns else 0.0
    avg_asr = df["asr_confidence"].mean() if "asr_confidence" in df.columns else 0.0
    n_lang = df["detected_language"].nunique() if "detected_language" in df.columns else 0

    # Create HTML for metric pills with custom styling
    st.markdown(f"""
    <div class="metric-strip">
      <div class="metric-pill"><span class="label">Avg SNR</span>      <span class="val">{avg_snr:.1f} dB</span></div>
      <div class="metric-pill"><span class="label">Avg Clipping</span> <span class="val">{avg_clip:.2%}</span></div>
      <div class="metric-pill"><span class="label">Avg Silence</span>  <span class="val">{avg_sil:.2%}</span></div>
      <div class="metric-pill"><span class="label">Avg RMS</span>      <span class="val">{avg_rms:.4f}</span></div>
      <div class="metric-pill"><span class="label">Avg ASR Conf</span> <span class="val">{avg_asr:.3f}</span></div>
      <div class="metric-pill"><span class="label">Languages</span>    <span class="val">{n_lang}</span></div>
    </div>
    """, unsafe_allow_html=True)


def scatter_snr_asr(df: pd.DataFrame) -> go.Figure:
    color_map = {"KEEP": "#22c55e", "DISCARD": "#ef4444"}
    decisions  = df["Decision"].tolist() if "Decision" in df.columns else ["KEEP"] * len(df)
    colors     = [color_map.get(d, "#9aa0b8") for d in decisions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["snr"],
        y=df["asr_confidence"] if "asr_confidence" in df.columns else [0] * len(df),
        mode="markers",
        marker=dict(color=colors, size=7, opacity=0.8, line=dict(width=0.5, color="#0d0f14")),
        text=df["filename"],
        customdata=decisions,
        hovertemplate="<b>%{text}</b><br>SNR: %{x:.1f} dB<br>ASR conf: %{y:.3f}<br>%{customdata}<extra></extra>",
    ))
    # Legend proxies
    for label, clr in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=clr, size=9), name=label, showlegend=True,
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title="SNR vs ASR Confidence",
        xaxis_title="Signal-to-Noise Ratio (dB)",
        yaxis_title="ASR Confidence",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2130"),
        height=420,
    )
    return fig


def histogram_metric(df: pd.DataFrame, col: str, label: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col], nbinsx=30,
        marker_color=color, opacity=0.85,
        marker_line=dict(width=0.5, color="#0d0f14"),
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, title=label, xaxis_title=label, yaxis_title="Count", height=300)
    return fig


def bar_languages(df: pd.DataFrame) -> go.Figure:
    counts = df["detected_language"].value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color="#3d8ef0", opacity=0.9,
        text=counts.values, textposition="auto",
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, title="Language Distribution", xaxis_title="Language", yaxis_title="Files", height=350)
    return fig


def pie_decisions(df: pd.DataFrame) -> go.Figure:
    counts = df["Decision"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=["#22c55e", "#ef4444"], line=dict(color="#0d0f14", width=2)),
        textfont=dict(family="DM Sans", size=13),
        hole=0.45,
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, title="Keep / Discard Split", height=350, showlegend=True)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── header ──
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
      <p class="page-title">Indic Audio QC</p>
      <p class="page-subtitle">▸ QUALITY FILTERING PIPELINE</p>
    </div>
    """, unsafe_allow_html=True)

    # ── sidebar ──
    with st.sidebar:
        st.markdown("### Threshold Controls")

        snr_min = st.slider(
            "Min SNR (dB)", 0.0, 50.0, 10.0, 0.5,
            help="Discard files with SNR below this value",
        )
        clipping_max = st.slider(
            "Max Clipping Ratio (%)", 0.0, 50.0, 10.0, 0.5,
            help="Discard files where more than X% of samples are clipped",
        )
        silence_max = st.slider(
            "Max Silence Ratio (%)", 0.0, 100.0, 60.0, 1.0,
            help="Discard files where more than X% of the audio is silence",
        )
        asr_min = st.slider(
            "Min ASR Confidence", 0.0, 1.0, 0.1, 0.01,
            help="Discard files where Whisper confidence is below this value. "
                 "Whisper logprob-based scores tend to be low — keep this threshold modest.",
        )

        st.markdown("---")
        st.markdown("### Run Pipeline")

        if "processed_data" not in st.session_state:
            st.session_state.processed_data = pd.DataFrame()

        audio_dir = st.text_input("Audio folder", value="test_data/sample_audios")

        run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)
        if run_btn:
            if not os.path.isdir(audio_dir):
                st.error(f"Folder not found: {audio_dir}")
            else:
                with st.spinner("Processing — this takes a while for large datasets…"):
                    try:
                        df_raw = run_pipeline(audio_dir)
                        if df_raw.empty:
                            st.error("No audio files found or all failed.")
                        else:
                            st.session_state.processed_data = df_raw
                            st.session_state.pop("last_thresholds", None)
                            st.success(f"Done — {len(df_raw)} files processed.")
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

    # ── nothing processed yet ──
    if st.session_state.processed_data.empty:
        st.info("Set an audio folder in the sidebar and click **Run Pipeline** to start.")
        return

    # ── apply / re-apply thresholds ──
    thresholds = dict(
        snr_min=snr_min,
        clipping_max=clipping_max / 100.0,
        silence_max=silence_max  / 100.0,
        asr_min=asr_min,
    )
    if st.session_state.get("last_thresholds") != thresholds:
        st.session_state.processed_data = apply_filtering_logic(
            st.session_state.processed_data, thresholds
        )
        st.session_state.last_thresholds = thresholds.copy()

    df = st.session_state.processed_data

    # ── KPI cards ──
    stat_cards(df)
    metric_strip(df)

    # ── charts row 1 ──
    st.markdown('<p class="section-heading">Quality Metrics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(scatter_snr_asr(df), use_container_width=True)
    with c2:
        if "detected_language" in df.columns:
            st.plotly_chart(bar_languages(df), use_container_width=True)

    # ── charts row 2 — histograms ──
    st.markdown('<p class="section-heading">Metric Distributions</p>', unsafe_allow_html=True)

    metric_defs = [
        ("snr",            "SNR (dB)",         "#3d8ef0"),
        ("clipping_ratio", "Clipping Ratio",   "#f59e0b"),
        ("silence_ratio",  "Silence Ratio",    "#a855f7"),
        ("rms",            "RMS Energy",       "#06b6d4"),
        ("asr_confidence", "ASR Confidence",   "#22c55e"),
        ("duration",       "Duration (s)",     "#ec4899"),
    ]

    cols = st.columns(3)
    for i, (col_name, label, clr) in enumerate(metric_defs):
        if col_name in df.columns:
            with cols[i % 3]:
                st.plotly_chart(histogram_metric(df, col_name, label, clr), use_container_width=True)

    # ── keep / discard pie ──
    if "Decision" in df.columns:
        st.markdown('<p class="section-heading">Decision Analysis</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(pie_decisions(df), use_container_width=True)
        with c2:
            # Breakdown of discard reasons
            discarded = df[df["Decision"] == "DISCARD"]
            if not discarded.empty and "Reason" in discarded.columns:
                reason_counts = {}
                for reason_str in discarded["Reason"]:
                    for part in str(reason_str).split(";"):
                        tag = part.strip().split(" ")[0]  # e.g. "SNR", "Clipping"
                        reason_counts[tag] = reason_counts.get(tag, 0) + 1

                if reason_counts:
                    r_df = pd.DataFrame(reason_counts.items(), columns=["Reason", "Count"]).sort_values("Count", ascending=True)
                    fig_r = go.Figure(go.Bar(
                        y=r_df["Reason"], x=r_df["Count"],
                        orientation="h",
                        marker_color="#ef4444", opacity=0.85,
                        text=r_df["Count"], textposition="auto",
                    ))
                    fig_r.update_layout(**_PLOTLY_LAYOUT, title="Why Files Were Discarded", height=300)
                    st.plotly_chart(fig_r, use_container_width=True)

    # ── results table ──
    st.markdown('<p class="section-heading">Results Table</p>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        decision_filter = st.selectbox("Decision", ["All", "KEEP", "DISCARD"])
    with fc2:
        lang_opts = ["All"]
        if "detected_language" in df.columns:
            lang_opts += sorted(df["detected_language"].dropna().unique().tolist())
        lang_filter = st.selectbox("Language", lang_opts)
    with fc3:
        snr_floor = st.number_input("Min SNR shown", value=float(df["snr"].min()), step=1.0)

    view = df.copy()
    if decision_filter != "All":
        view = view[view["Decision"] == decision_filter]
    if lang_filter != "All" and "detected_language" in df.columns:
        view = view[view["detected_language"] == lang_filter]
    view = view[view["snr"] >= snr_floor]

    display_cols = [c for c in [
        "filename", "duration", "snr", "clipping_ratio", "silence_ratio",
        "rms", "asr_confidence", "detected_language", "Decision", "Reason",
    ] if c in view.columns]

    st.dataframe(view[display_cols], use_container_width=True, height=380)

    # ── spectrogram deep-dive ──
    if "Decision" in df.columns:
        discarded = df[df["Decision"] == "DISCARD"]
        if not discarded.empty:
            st.markdown('<p class="section-heading">Spectrogram Deep-Dive</p>', unsafe_allow_html=True)
            st.caption("Pick a discarded file to inspect its Mel-spectrogram.")

            selected = st.selectbox("File", discarded["filename"].tolist(), label_visibility="collapsed")
            if selected:
                row = df[df["filename"] == selected].iloc[0]
                col_m, col_s = st.columns([1, 2])

                with col_m:
                    st.markdown("**Metrics**")
                    pretty = {
                        "SNR":         f"{row.get('snr', 0):.2f} dB",
                        "Clipping":    f"{row.get('clipping_ratio', 0):.2%}",
                        "Silence":     f"{row.get('silence_ratio', 0):.2%}",
                        "RMS":         f"{row.get('rms', 0):.5f}",
                        "ASR Conf":    f"{row.get('asr_confidence', 0):.3f}",
                        "Language":    row.get("detected_language", "—"),
                        "Duration":    f"{row.get('duration', 0):.2f} s",
                        "Reason":      row.get("Reason", "—"),
                    }
                    for k, v in pretty.items():
                        st.markdown(f"<div class='metric-pill'><span class='label'>{k}</span><span class='val' style='font-size:0.9rem'>{v}</span></div>", unsafe_allow_html=True)

                with col_s:
                    st.markdown("**Mel-Spectrogram**")
                    fpath = row.get("filepath", row.get("audio_filepath", None))
                    if fpath and os.path.exists(str(fpath)):
                        spec = generate_spectrogram_for_display(str(fpath))
                        if spec is not None:
                            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#13151c")
                            ax.set_facecolor("#13151c")
                            ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
                            ax.set_title(selected, color="#9aa0b8", fontsize=10, pad=8)
                            ax.set_xlabel("Time frames", color="#5a6080")
                            ax.set_ylabel("Mel bins",    color="#5a6080")
                            ax.tick_params(colors="#5a6080")
                            for spine in ax.spines.values():
                                spine.set_edgecolor("#1e2130")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.warning("Could not generate spectrogram.")
                    else:
                        st.warning("Audio file not found on disk — spectrogram unavailable.")

    # ── export ──
    st.markdown('<p class="section-heading">Export</p>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)

    with e1:
        st.download_button(
            "↓ Download CSV",
            data=view.to_csv(index=False),
            file_name="audio_quality_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "↓ Download JSON",
            data=view.to_json(orient="records", indent=2),
            file_name="audio_quality_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with e3:
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Building report…"):
                try:
                    os.makedirs("reports", exist_ok=True)
                    pdf_path = generate_pdf_report(df, "reports/Quality_Report.pdf")
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "↓ Download PDF",
                            data=f.read(),
                            file_name="Quality_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")


if __name__ == "__main__":
    main()
