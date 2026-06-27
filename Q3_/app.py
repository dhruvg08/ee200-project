
import os
import gc
import io
import zipfile
import tempfile

import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav_io
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fingerprint import (
    SongDatabase, load_audio, compute_spectrogram,
    get_constellation, TARGET_SR,
)


st.set_page_config(
    page_title="Zapptain America — Music Identifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .main { background-color: #0f111a; color: #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap;
        background-color: #1a1d2e; border-radius: 8px 8px 0 0;
        gap: 1px; padding: 10px; color: #94a3b8;
        font-weight: 600; border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e3456 !important; color: #38bdf8 !important;
    }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #38bdf8; }
    .header-title {
        font-family: 'Outfit', sans-serif; font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3rem; margin-bottom: 0.5rem;
    }
    .match-box {
        background: #1a1d2e; border: 1px solid #38bdf8;
        border-radius: 10px; padding: 16px; margin: 8px 0;
    }
    .match-title { color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px; }
    .match-value { color: #38bdf8; font-size: 1.5rem; font-weight: 700;
                   word-break: break-word; }
    .section-divider { border-top: 1px solid #2e3456; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


def _add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN at the given SNR (dB). Lower value = more noise."""
    pwr = np.mean(audio.astype(np.float64) ** 2)
    if pwr < 1e-10:
        return audio.copy()
    noise_pwr = pwr / (10 ** (snr_db / 10.0))
    rng = np.random.default_rng()
    noise = rng.normal(0.0, np.sqrt(noise_pwr), len(audio))
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def _pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones (phase vocoder)."""
    return librosa.effects.pitch_shift(
        audio, sr=sr, n_steps=n_steps, n_fft=512
    ).astype(np.float32)


def _audio_to_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Convert float32 audio to 16-bit WAV bytes for st.audio()."""
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_io.write(buf, sr, int16)
    buf.seek(0)
    return buf.read()


def _render_spectrogram_constellation(audio, q_peaks, label=""):
    """Render spectrogram + constellation side-by-side."""
    try:
        f_arr, t_arr, Sxx = compute_spectrogram(audio)
        Sxx_db = 10.0 * np.log10(Sxx + 1e-10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#1a1d2e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1d2e")

        title1 = "Spectrogram" + (" — " + label if label else "")
        im = ax1.pcolormesh(t_arr, f_arr, Sxx_db, shading="auto", cmap="viridis")
        ax1.set_title(title1, color="white")
        ax1.set_ylabel("Frequency (Hz)", color="white")
        ax1.set_xlabel("Time (s)", color="white")
        ax1.tick_params(colors="white")
        cb = fig.colorbar(im, ax=ax1)
        cb.set_label("Power (dB)", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        ax2.pcolormesh(t_arr, f_arr, Sxx_db, shading="auto", cmap="viridis", alpha=0.25)
        if q_peaks:
            pt = [t_arr[min(p[0], len(t_arr) - 1)] for p in q_peaks]
            pf = [f_arr[min(p[1], len(f_arr) - 1)] for p in q_peaks]
            ax2.scatter(pt, pf, color="#38bdf8", s=12, label="Peaks")
            ax2.legend(facecolor="#1a1d2e", labelcolor="white")
        ax2.set_title(f"Constellation ({len(q_peaks)} peaks)", color="white")
        ax2.set_ylabel("Frequency (Hz)", color="white")
        ax2.set_xlabel("Time (s)", color="white")
        ax2.tick_params(colors="white")

        st.pyplot(fig)
        plt.close(fig)
        del f_arr, t_arr, Sxx, Sxx_db, fig
    except Exception as exc:
        st.warning(f"Visualisation error: {exc}")
    finally:
        plt.close("all")
        gc.collect()


def _render_offset_histogram(prediction, offsets):
    """Render the time-offset alignment histogram."""
    if not prediction or offsets is None or len(offsets) == 0:
        return
    try:
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#1a1d2e")
        ax.set_facecolor("#1a1d2e")
        lo, hi = int(offsets.min()), int(offsets.max())
        off_range = hi - lo
        n_bins = min(200, off_range + 3) if off_range > 0 else 3
        bins = np.linspace(lo - 1, hi + 1, n_bins + 1)
        ax.hist(offsets, bins=bins, color="#a855f7", edgecolor="black", alpha=0.8)
        ax.set_title(f"Offset Histogram — {prediction}", color="white")
        ax.set_xlabel("Time Offset (bins)", color="white")
        ax.set_ylabel("Vote Count", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)
        plt.close(fig)
        del fig
    except Exception as exc:
        st.warning(f"Histogram error: {exc}")
    finally:
        plt.close("all")
        gc.collect()


def _show_match_result(prediction, votes):
    """Display the styled identification result card."""
    if prediction:
        st.markdown(
            f"<div class='match-box'>"
            f"<div class='match-title'>🎵 Predicted Song</div>"
            f"<div class='match-value'>{prediction}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.metric("Match Score (Votes)", votes)
    else:
        st.warning("No match found in the database.")


def _sweep_bar_chart(x_vals, votes, preds, baseline, xlabel, title):
    """Bar chart of votes across a sweep, coloured by correctness."""
    colors = ["#38bdf8" if p == baseline else "#ef4444" for p in preds]
    width = 0.8 * (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#1a1d2e")
    ax.set_facecolor("#1a1d2e")
    ax.bar(x_vals, votes, color=colors, edgecolor="black", width=width)
    ax.axhline(y=5, color="#f59e0b", linestyle="--", linewidth=1.2,
               label="Min-votes threshold (5)")
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel("Votes", color="white")
    ax.set_title(title + "   [blue = correct   red = wrong / no-match]",
                 color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1d2e", labelcolor="white")
    st.pyplot(fig)
    plt.close(fig)
    del fig
    gc.collect()


@st.cache_resource
def get_db():
    try:
        db_path = os.path.join(os.path.dirname(__file__), "songs_db.pkl")
        if os.path.exists(db_path):
            return SongDatabase.load(db_path)
        return None
    except Exception as exc:
        st.error(f"Database load failed: {exc}")
        return None


db = get_db()


with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8;'>🎵 Sonic Fingerprinting</h2>",
                unsafe_allow_html=True)
    st.write("Shazam-style constellation matching algorithm.")

    if db:
        st.success(f"Database loaded: **{len(db.songs_list)}** songs indexed.")
        with st.expander(f"📚 View indexed songs ({len(db.songs_list)})"):
            for i, song in enumerate(db.songs_list, 1):
                st.markdown(f"**{i}.** {song}")
    else:
        st.error(
            "No database found!  \n"
            "Run `python build_db.py --songs_dir songs`  \n"
            "then restart the app."
        )

    st.markdown("---")
    st.markdown("### Parameters")
    st.info(
        f"**Sample Rate:** {TARGET_SR} Hz  \n"
        "**Window:** 512 samples  \n"
        "**Overlap:** 384 samples  \n"
        "**Fan-out:** 15 peaks/anchor  \n"
        "**Max peaks:** 200 / clip"
    )


st.markdown("<div class='header-title'>Zapptain America 🇺🇸</div>",
            unsafe_allow_html=True)
st.write("EE200: Signals, Systems & Networks — Audio Fingerprinting & Identification")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single-Clip Search",
    "📦 Batch Processing",
    "🎙️ Live Recording",
    "🔬 Robustness Testing",
])


with tab1:
    st.header("Search a Single Audio Clip")
    uploaded_file = st.file_uploader(
        "Upload query clip (.mp3 or .wav)", type=["mp3", "wav"], key="tab1_uploader"
    )

    if uploaded_file is not None:
        if not db:
            st.error("Cannot search: database not loaded.")
        else:
            tmp_path = None
            audio = None
            try:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                gc.collect()

                with st.spinner("Analysing and matching …"):
                    audio, _ = load_audio(tmp_path)
                    prediction, votes, offsets, q_peaks = db.match_clip(audio)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("### Match Result")
                    _show_match_result(prediction, votes)
                with col2:
                    st.markdown("### Visualisations")

                _render_spectrogram_constellation(audio, q_peaks, label="Query")
                _render_offset_histogram(prediction, offsets)

            except Exception as exc:
                st.error(f"Error processing file: {exc}")
            finally:
                if audio is not None:
                    del audio
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                gc.collect()


with tab2:
    st.header("Batch Process Queries")
    st.write(
        "Upload a ZIP of query clips. The system identifies each one and "
        "produces `results.csv` with columns `filename`, `prediction`."
    )
    zip_file = st.file_uploader("Upload ZIP of queries", type=["zip"],
                                key="tab2_uploader")

    if zip_file is not None:
        if not db:
            st.error("Cannot search: database not loaded.")
        else:
            with st.spinner("Processing batch archive …"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_path = os.path.join(tmp_dir, "queries.zip")
                    with open(zip_path, "wb") as zf:
                        zf.write(zip_file.getvalue())

                    extract_dir = os.path.join(tmp_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(zip_path, "r") as zref:
                        zref.extractall(extract_dir)

                    audio_files = []
                    for root, _, files in os.walk(extract_dir):
                        for fname in sorted(files):
                            if (fname.lower().endswith((".mp3", ".wav"))
                                    and not fname.startswith(".")
                                    and "__MACOSX" not in root):
                                audio_files.append(os.path.join(root, fname))
                    audio_files.sort(key=lambda x: os.path.basename(x))

                    if not audio_files:
                        st.warning("No .mp3 or .wav files found in the ZIP.")
                    else:
                        st.info(f"Processing {len(audio_files)} file(s) …")
                        results = []
                        bar = st.progress(0)
                        for idx, fpath in enumerate(audio_files):
                            fname = os.path.basename(fpath)
                            audio = None
                            try:
                                audio, _ = load_audio(fpath)
                                pred, _, _, _ = db.match_clip(audio)
                                results.append({"filename": fname,
                                                "prediction": pred or ""})
                            except Exception:
                                results.append({"filename": fname,
                                                "prediction": ""})
                            finally:
                                if audio is not None:
                                    del audio
                                gc.collect()
                            bar.progress((idx + 1) / len(audio_files))

                        df = pd.DataFrame(results)
                        st.success("Batch processing complete!")
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "⬇️ Download results.csv",
                            df.to_csv(index=False),
                            file_name="results.csv",
                            mime="text/csv",
                        )


with tab3:
    st.header("🎙️ Record & Identify")
    st.write(
        "Hold your device near the music and record **5–10 seconds**. "
        "The fingerprinter identifies the song from constellation hashes alone — "
        "no waveform comparison needed."
    )

    if not db:
        st.error("Cannot identify: database is not loaded.")
    else:
        audio_value = st.audio_input(
            "🎤  Click the microphone icon to start / stop recording",
            key="mic_recorder",
        )

        if audio_value is not None:
            st.markdown("#### Playback — what was recorded")
            st.audio(audio_value, format="audio/wav")

            tmp_path = None
            rec_audio = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_value.getvalue())
                    tmp_path = tmp.name
                gc.collect()

                with st.spinner("Fingerprinting your recording …"):
                    rec_audio, _ = load_audio(tmp_path)
                    prediction, votes, offsets, q_peaks = db.match_clip(rec_audio)

                st.markdown("#### Recorded Waveform")
                t_wave = np.linspace(0, len(rec_audio) / TARGET_SR, len(rec_audio))
                fig_w, ax_w = plt.subplots(figsize=(10, 2))
                fig_w.patch.set_facecolor("#1a1d2e")
                ax_w.set_facecolor("#1a1d2e")
                ax_w.plot(t_wave, rec_audio, color="#38bdf8",
                          linewidth=0.4, alpha=0.9)
                ax_w.set_xlabel("Time (s)", color="white")
                ax_w.set_ylabel("Amplitude", color="white")
                ax_w.set_title("Recorded Waveform", color="white")
                ax_w.tick_params(colors="white")
                st.pyplot(fig_w)
                plt.close(fig_w)
                del fig_w, t_wave

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("#### Identification Result")
                    _show_match_result(prediction, votes)
                    duration_s = len(rec_audio) / TARGET_SR
                    st.caption(
                        f"Duration: {duration_s:.1f} s  |  "
                        f"Peaks found: {len(q_peaks)}"
                    )
                with col2:
                    st.markdown("#### Tips for best results")
                    st.info(
                        "- Record **5–10 seconds** of the song playing.  \n"
                        "- Hold the mic **close** to the speaker.  \n"
                        "- Keep the room **quiet** (no talking).  \n"
                        "- The song must be **in the indexed database**."
                    )

                st.markdown("#### Spectrogram & Constellation")
                _render_spectrogram_constellation(rec_audio, q_peaks,
                                                  label="Recording")
                _render_offset_histogram(prediction, offsets)

            except Exception as exc:
                st.error(f"Error processing recording: {exc}")
            finally:
                if rec_audio is not None:
                    del rec_audio
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                gc.collect()


with tab4:
    st.header("🔬 Robustness Testing")
    st.write(
        "Upload a known clip and probe two types of degradation that "
        "**Q3A** explicitly asks about: additive noise and pitch shifts."
    )

    if not db:
        st.error("Cannot test: database is not loaded.")
    else:
        rb_file = st.file_uploader(
            "Upload a known query clip (.mp3 or .wav)",
            type=["mp3", "wav"],
            key="robustness_uploader",
        )

        if rb_file is not None:
            tmp_path = None
            clean_audio = None
            try:
                suffix = os.path.splitext(rb_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(rb_file.getvalue())
                    tmp_path = tmp.name
                gc.collect()

                clean_audio, _ = load_audio(tmp_path)

                with st.spinner("Running baseline identification …"):
                    base_pred, base_votes, _, _ = db.match_clip(clean_audio)

                bcol1, bcol2 = st.columns([2, 1])
                with bcol1:
                    st.markdown("**Baseline result (clean audio):**")
                    _show_match_result(base_pred, base_votes)
                with bcol2:
                    st.caption("Clean audio playback:")
                    st.audio(_audio_to_bytes(clean_audio, TARGET_SR),
                             format="audio/wav")

                st.markdown("<div class='section-divider'></div>",
                            unsafe_allow_html=True)

                st.subheader("A. Additive White Gaussian Noise (AWGN)")
                st.write(
                    "Lower SNR = more noise. Noise uniformly raises the spectrogram "
                    "floor; dominant signal peaks still survive — until the SNR is so "
                    "low that noise peaks outnumber real ones and wrong hashes are "
                    "generated. This is why the Shazam paper claims robustness well "
                    "below 10 dB SNR in practice."
                )

                nc1, nc2 = st.columns([3, 1])
                with nc1:
                    snr_db = st.slider("Signal-to-Noise Ratio (dB)",
                                       0, 40, 20, step=5, key="snr_slider")
                with nc2:
                    run_noise_sweep = st.button("📊 Full Sweep (0–40 dB)",
                                                key="noise_sweep")

                noisy = _add_noise(clean_audio, snr_db)
                with st.spinner(f"Testing at SNR = {snr_db} dB …"):
                    n_pred, n_votes, n_offsets, _ = db.match_clip(noisy)

                rc1, rc2, rc3 = st.columns([1, 1, 2])
                with rc1:
                    _show_match_result(n_pred, n_votes)
                with rc2:
                    st.caption("Listen to the noisy version:")
                    st.audio(_audio_to_bytes(noisy, TARGET_SR), format="audio/wav")
                with rc3:
                    _render_offset_histogram(n_pred, n_offsets)
                del noisy

                if run_noise_sweep:
                    snr_levels = [40, 30, 20, 15, 10, 5, 0]
                    sw_v, sw_p = [], []
                    prog = st.progress(0, text="Running noise sweep …")
                    for i, lvl in enumerate(snr_levels):
                        a = _add_noise(clean_audio, lvl)
                        p, v, _, _ = db.match_clip(a)
                        sw_v.append(v)
                        sw_p.append(p or "—")
                        del a
                        prog.progress((i + 1) / len(snr_levels),
                                      text=f"SNR = {lvl} dB done")
                    gc.collect()
                    _sweep_bar_chart(
                        snr_levels, sw_v, sw_p, base_pred,
                        xlabel="SNR (dB)",
                        title="Votes vs. SNR",
                    )
                    with st.expander("Raw sweep data"):
                        st.dataframe(pd.DataFrame({
                            "SNR (dB)": snr_levels,
                            "Prediction": sw_p,
                            "Votes": sw_v,
                        }))

                st.markdown("<div class='section-divider'></div>",
                            unsafe_allow_html=True)

                st.subheader("B. Pitch Shift Robustness")
                st.write(
                    "**Why does a small pitch shift break the identifier?** "
                    "Hashes encode *absolute* frequency-bin indices. A +1 semitone "
                    "shift moves every frequency up by ~6 %, landing in different "
                    "bins — so the hashes no longer match the database, even though "
                    "the song sounds identical to a human listener. "
                    "**Fix:** store pitch-invariant features (e.g. log-frequency "
                    "bins, chroma, or relative pitch intervals) instead of absolute "
                    "bin indices."
                )

                pc1, pc2 = st.columns([3, 1])
                with pc1:
                    n_steps = st.slider("Pitch Shift (semitones)",
                                        -6, 6, 0, step=1, key="pitch_slider")
                with pc2:
                    run_pitch_sweep = st.button("📊 Full Sweep (−6 to +6)",
                                                key="pitch_sweep")

                if n_steps != 0:
                    with st.spinner(
                        f"Shifting by {n_steps:+d} semitone(s) via phase vocoder …"
                    ):
                        shifted = _pitch_shift(clean_audio, TARGET_SR, n_steps)
                        p_pred, p_votes, p_offsets, _ = db.match_clip(shifted)

                    sc1, sc2, sc3 = st.columns([1, 1, 2])
                    with sc1:
                        _show_match_result(p_pred, p_votes)
                    with sc2:
                        st.caption("Listen to the pitch-shifted version:")
                        st.audio(_audio_to_bytes(shifted, TARGET_SR),
                                 format="audio/wav")
                    with sc3:
                        _render_offset_histogram(p_pred, p_offsets)
                    del shifted
                    gc.collect()
                else:
                    st.info("Move the slider away from 0 to apply a pitch shift.")

                if run_pitch_sweep:
                    semi_range = list(range(-6, 7))
                    sw_v2, sw_p2 = [], []
                    prog2 = st.progress(0, text="Running pitch sweep …")
                    for i, n in enumerate(semi_range):
                        if n == 0:
                            prd, vts, _, _ = db.match_clip(clean_audio)
                        else:
                            a = _pitch_shift(clean_audio, TARGET_SR, n)
                            prd, vts, _, _ = db.match_clip(a)
                            del a
                        sw_v2.append(vts)
                        sw_p2.append(prd or "—")
                        prog2.progress((i + 1) / len(semi_range),
                                       text=f"{n:+d} semitone(s) done")
                    gc.collect()
                    _sweep_bar_chart(
                        semi_range, sw_v2, sw_p2, base_pred,
                        xlabel="Pitch Shift (semitones)",
                        title="Votes vs. Pitch Shift",
                    )
                    with st.expander("Raw sweep data"):
                        st.dataframe(pd.DataFrame({
                            "Semitones": semi_range,
                            "Prediction": sw_p2,
                            "Votes": sw_v2,
                        }))

            except Exception as exc:
                st.error(f"Error in robustness testing: {exc}")
            finally:
                if clean_audio is not None:
                    del clean_audio
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                plt.close("all")
                gc.collect()
