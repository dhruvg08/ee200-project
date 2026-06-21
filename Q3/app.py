import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.signal as signal
from fingerprint import SongDatabase, load_audio, compute_spectrogram, get_constellation, TARGET_SR

# Page configurations
st.set_page_config(
    page_title="Zapptain America - Music Identifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern premium look
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1d2e;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e3456 !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #38bdf8;
    }
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load database
@st.cache_resource
def get_db():
    db_path = os.path.join(os.path.dirname(__file__), "songs_db.pkl")
    if os.path.exists(db_path):
        return SongDatabase.load(db_path)
    return None

db = get_db()

# Sidebar info
with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8;'>🎵 Sonic Fingerprinting</h2>", unsafe_allow_html=True)
    st.write("An implementation of the Shazam constellation matching algorithm.")
    if db:
        st.success(f"Database loaded: {len(db.songs_list)} songs indexed.")
    else:
        st.error("No database loaded! Please make sure songs_db.pkl exists.")
        
    st.markdown("---")
    st.markdown("### Parameters")
    st.info(f"Sample Rate: {TARGET_SR} Hz\nWindow size: 512 samples\nOverlap: 384 samples")

# Header
st.markdown("<div class='header-title'>Zapptain America 🇺🇸</div>", unsafe_allow_html=True)
st.write("EE200: Signals, Systems & Networks - Audio Fingerprinting and Identification System")

tab1, tab2 = st.tabs(["🔍 Single-Clip Search", "📦 Batch Processing"])

# Tab 1: Single Clip Mode
with tab1:
    st.header("Search a Single Audio Clip")
    uploaded_file = st.file_uploader("Upload query clip (.mp3 or .wav)", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        if not db:
            st.error("Cannot perform search: Database is not loaded.")
        else:
            with st.spinner("Analyzing and matching clip..."):
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Load and match
                    audio, _ = load_audio(tmp_path)
                    prediction, votes, offsets, q_peaks = db.match_clip(audio)
                    
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("### Match Result")
                        if prediction:
                            st.metric(label="Predicted Song", value=prediction)
                            st.metric(label="Match Score (Votes)", value=votes)
                        else:
                            st.warning("No match found in database.")
                    
                    # Plotting
                    with col2:
                        st.markdown("### Visualizations")
                        
                  # Generate figures
try:
    f_arr, t_arr, Sxx = compute_spectrogram(audio)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    plt.style.use('dark_background')
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    img1 = ax1.pcolormesh(t_arr, f_arr, Sxx_db, shading='gouraud', cmap='viridis')
    ax1.set_title("Query Spectrogram")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")
    fig1.colorbar(img1, ax=ax1, label="Magnitude (dB)")
    
    ax2.pcolormesh(t_arr, f_arr, Sxx_db, shading='gouraud', cmap='viridis', alpha=0.2)
    if q_peaks:
        peak_times = [t_arr[min(p[0], len(t_arr)-1)] for p in q_peaks]
        peak_freqs = [f_arr[min(p[1], len(f_arr)-1)] for p in q_peaks]
        ax2.scatter(peak_times, peak_freqs, color='#38bdf8', s=12, label="Peaks")
    ax2.set_title(f"Constellation Map ({len(q_peaks)} peaks)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    
    st.pyplot(fig1)
    plt.close(fig1)
    
    if prediction and offsets is not None and len(offsets) > 0:
        fig2, ax = plt.subplots(figsize=(10, 4))
        min_off, max_off = offsets.min(), offsets.max()
        bins = np.arange(min_off - 1, max_off + 2, 1)
        ax.hist(offsets, bins=bins, color='#a855f7', edgecolor='black', alpha=0.8)
        ax.set_title(f"Offset Histogram for Match: {prediction}")
        ax.set_xlabel("Time Offset (bins)")
        ax.set_ylabel("Count")
        st.pyplot(fig2)
        plt.close(fig2)

except Exception as e:
    st.warning(f"Visualization error (match result above is still valid): {e}")
               
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

# Tab 2: Batch Processing Mode
with tab2:
    st.header("Batch Process Queries")
    st.write("Upload a ZIP archive containing multiple query files. The system will identify each clip and generate `results.csv`.")
    
    zip_file = st.file_uploader("Upload ZIP of queries", type=["zip"])
    
    if zip_file is not None:
        if not db:
            st.error("Cannot perform batch search: Database is not loaded.")
        else:
            with st.spinner("Processing batch archive..."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = os.path.join(tmp_dir, "queries.zip")
                    with open(zip_path, "wb") as f:
                        f.write(zip_file.read())
                        
                    # Extract zip
                    extract_dir = os.path.join(tmp_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                        
                    # Find all audio files (recursively or directly)
                    audio_files = []
                    for root, _, files in os.walk(extract_dir):
                        for f in files:
                            if f.lower().endswith(('.mp3', '.wav')) and not f.startswith('.'):
                                audio_files.append(os.path.join(root, f))
                                
                    audio_files.sort(key=lambda x: os.path.basename(x))
                    
                    if not audio_files:
                        st.warning("No .mp3 or .wav files found in the uploaded ZIP.")
                    else:
                        st.info(f"Processing {len(audio_files)} files...")
                        
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, filepath in enumerate(audio_files):
                            filename = os.path.basename(filepath)
                            try:
                                audio, _ = load_audio(filepath)
                                prediction, votes, _, _ = db.match_clip(audio)
                                if not prediction:
                                    prediction = ""
                                results.append({"filename": filename, "prediction": prediction})
                            except Exception as e:
                                results.append({"filename": filename, "prediction": ""})
                            progress_bar.progress((idx + 1) / len(audio_files))
                            
                        # Convert to DataFrame
                        df = pd.DataFrame(results)
                        st.success("Batch processing complete!")
                        
                        # Display table
                        st.dataframe(df, use_container_width=True)
                        
                        # CSV Download button
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="Download results.csv",
                            data=csv_data,
                            file_name="results.csv",
                            mime="text/csv"
                        )
