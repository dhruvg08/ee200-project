import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from fingerprint import SongDatabase, load_audio, compute_spectrogram, get_constellation, TARGET_SR

def create_query_clip(audio, duration_s=10, start_s=30, sr=TARGET_SR):
    """Extracts a segment of the audio file to use as a query clip."""
    start_idx = int(start_s * sr)
    end_idx = int((start_s + duration_s) * sr)
    if end_idx > len(audio):
        # Fallback to end of audio
        end_idx = len(audio)
        start_idx = max(0, end_idx - int(duration_s * sr))
    return audio[start_idx:end_idx]

def add_noise(audio, snr_db):
    """Adds white Gaussian noise to the audio to achieve the desired SNR in dB."""
    sig_power = np.mean(audio ** 2)
    if sig_power == 0:
        return audio
    # SNR = sig_power / noise_power
    # noise_power = sig_power / (10 ** (snr_db / 10))
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def pitch_shift_resample(audio, factor):
    """Shifts pitch and stretches time by resampling the audio signal by a factor."""
    num_samples = int(len(audio) * factor)
    return signal.resample(audio, num_samples)

def run_experiments():
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load a sample song
    songs_dir = "data/songs_db"
    song_files = [f for f in os.listdir(songs_dir) if f.endswith(".mp3")]
    if not song_files:
        print("No songs found for experiments.")
        return
    
    sample_song_file = song_files[0]
    sample_song_name = os.path.splitext(sample_song_file)[0]
    print(f"Using '{sample_song_name}' for window and robustness experiments.")
    
    full_audio, _ = load_audio(os.path.join(songs_dir, sample_song_file))
    query_clip = create_query_clip(full_audio, duration_s=10, start_s=40)
    
    # --- EXPERIMENT 1: WINDOW LENGTH RESOLUTION (TIME VS FREQUENCY) ---
    print("Running Experiment 1: Window Length tradeoff...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Short Window
    nperseg_short = 128
    noverlap_short = 96
    f_s, t_s, Sxx_s = signal.spectrogram(query_clip, fs=TARGET_SR, nperseg=nperseg_short, noverlap=noverlap_short)
    Sxx_s_db = 10 * np.log10(np.abs(Sxx_s) + 1e-10)
    ax1.pcolormesh(t_s, f_s, Sxx_s_db, shading='gouraud', cmap='viridis')
    ax1.set_title(f"Short Window (nperseg={nperseg_short}): High Time, Low Freq Resolution")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")
    
    # Long Window
    nperseg_long = 1024
    noverlap_long = 768
    f_l, t_l, Sxx_l = signal.spectrogram(query_clip, fs=TARGET_SR, nperseg=nperseg_long, noverlap=noverlap_long)
    Sxx_l_db = 10 * np.log10(np.abs(Sxx_l) + 1e-10)
    ax2.pcolormesh(t_l, f_l, Sxx_l_db, shading='gouraud', cmap='viridis')
    ax2.set_title(f"Long Window (nperseg={nperseg_long}): Low Time, High Freq Resolution")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "window_tradeoff.png"))
    plt.close(fig)
    print("Saved window_tradeoff.png")
    
    # Load database for testing matching
    db_file = "data/songs_db.pkl"
    if not os.path.exists(db_file):
        print("Database pkl not found. Cannot run matching experiments.")
        return
    db_paired = SongDatabase.load(db_file)
    
    # Create single-peak database for comparison
    print("Building temporary single-peak database for comparison...")
    db_single = SongDatabase(mode='single')
    for f in song_files[:10]: # Index first 10 songs to save time
        s_name = os.path.splitext(f)[0]
        db_single.index_song(s_name, os.path.join(songs_dir, f))
        
    # Choose a song in the subset
    test_song_name = db_single.songs_list[0]
    print(f"Testing matching robustness on song: {test_song_name}")
    test_song_path = os.path.join(songs_dir, test_song_name + ".mp3")
    test_audio, _ = load_audio(test_song_path)
    test_query = create_query_clip(test_audio, duration_s=8, start_s=35)
    
    # --- EXPERIMENT 2: PAIRED HASHES VS SINGLE PEAKS (UNDER NOISE) ---
    print("Running Experiment 2: Paired vs Single peaks under noise...")
    snr_levels = [20, 10, 5, 0, -5, -10]
    paired_scores = []
    single_scores = []
    
    for snr in snr_levels:
        noisy_q = add_noise(test_query, snr)
        
        # Match using Paired database
        matched_song_p, score_p, _, _ = db_paired.match_clip(noisy_q)
        paired_scores.append(score_p if matched_song_p == test_song_name else 0)
        
        # Match using Single database
        matched_song_s, score_s, _, _ = db_single.match_clip(noisy_q)
        single_scores.append(score_s if matched_song_s == test_song_name else 0)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_levels, paired_scores, marker='o', color='blue', label='Paired Hashes (Target Zone)')
    ax.plot(snr_levels, single_scores, marker='s', color='orange', label='Single Peaks')
    ax.set_title("Matching Confidence (Max Votes) vs. SNR")
    ax.set_xlabel("Signal-to-Noise Ratio (SNR) in dB")
    ax.set_ylabel("Maximum Histogram Votes (Match Confidence)")
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(plots_dir, "paired_vs_single_noise.png"))
    plt.close(fig)
    print("Saved paired_vs_single_noise.png")
    
    # --- EXPERIMENT 3: PITCH SHIFT & TIME STRETCH ---
    print("Running Experiment 3: Pitch shift/Time stretch...")
    shift_factors = [0.95, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.05]
    match_scores_shift = []
    correct_matches = []
    
    for factor in shift_factors:
        if factor == 1.0:
            shifted_q = test_query
        else:
            shifted_q = pitch_shift_resample(test_query, factor)
            
        matched_song, score, _, _ = db_paired.match_clip(shifted_q)
        match_scores_shift.append(score)
        correct_matches.append(1 if matched_song == test_song_name else 0)
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Resampling/Pitch-Shift Factor')
    ax1.set_ylabel('Match Score (Votes)', color=color)
    ax1.plot(shift_factors, match_scores_shift, marker='o', color=color, label='Score')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Correct Identification (1=Yes, 0=No)', color=color)
    ax2.step(shift_factors, correct_matches, where='mid', color=color, linestyle='--', label='Correct')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.title("Identification Robustness vs. Resampling (Pitch/Time Shift)")
    fig.tight_layout()  
    fig.savefig(os.path.join(plots_dir, "pitch_shift_robustness.png"))
    plt.close(fig)
    print("Saved pitch_shift_robustness.png")
    
    # --- EXPERIMENT 4: PLOT OFFSET HISTOGRAM FOR TRUE MATCH ---
    print("Generating offset histogram for a true match...")
    matched_song, score, offsets, _ = db_paired.match_clip(test_query)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(offsets, bins=np.arange(min(offsets)-2, max(offsets)+3, 1), color='purple', edgecolor='black')
    ax.set_title(f"Offset Histogram for True Match: {test_song_name}")
    ax.set_xlabel("Time Offset (bins)")
    ax.set_ylabel("Match Count")
    fig.savefig(os.path.join(plots_dir, "offset_histogram_match.png"))
    plt.close(fig)
    print("Saved offset_histogram_match.png")
    
    print("All experiments run and plots generated.")

if __name__ == "__main__":
    run_experiments()
