import os
import subprocess
import pickle
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TARGET_SR = 8000        # Downsample to 8 kHz (frequency range up to 4 kHz)
NPERSEG = 512           # STFT window size (64 ms)
NOVERLAP = 384          # STFT overlap (48 ms overlap, 16 ms hop size)
NEIGHBORHOOD_SIZE = 15  # 2D peak finding neighborhood size
AMPLITUDE_THRESHOLD = 0.01  # Minimum amplitude threshold for peaks (relative to mean)
FAN_VALUE = 3           # Number of pairings per peak
MIN_DELTA_T = 1        # Min time gap between paired peaks
MAX_DELTA_T = 60       # Max time gap between paired peaks (about 1 second)

def load_audio(file_path, target_sr=TARGET_SR):
    """Loads an MP3 or WAV audio file and resamples it to target_sr using ffmpeg."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    cmd = [
        'ffmpeg', '-y', '-i', file_path,
        '-f', 's16le', '-ac', '1', '-ar', str(target_sr), '-'
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stdout, _ = process.communicate()
    
    if len(stdout) == 0:
        raise ValueError(f"Failed to decode audio file using ffmpeg: {file_path}")
        
    audio = np.frombuffer(stdout, dtype=np.int16).astype(np.float32)
    # Normalize audio amplitude
    if np.max(np.abs(audio)) > 0:
        audio /= np.max(np.abs(audio))
    return audio, target_sr

def compute_spectrogram(audio, sr=TARGET_SR, nperseg=NPERSEG, noverlap=NOVERLAP):
    """Computes the magnitude spectrogram of the audio signal."""
    f, t, Sxx = signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    # Use magnitude spectrogram
    Sxx = np.abs(Sxx)
    return f, t, Sxx

def get_constellation(Sxx, neighborhood_size=NEIGHBORHOOD_SIZE, min_amp_rel=AMPLITUDE_THRESHOLD):
    """Extracts local peaks from the spectrogram using a 2D local maximum filter."""
    if Sxx.size == 0:
        return []
    
    # 2D local maximum filter
    data_max = ndimage.maximum_filter(Sxx, size=neighborhood_size)
    maxima = (Sxx == data_max)
    
    # Filter out weak peaks (absolute threshold based on spectrogram mean)
    mean_val = np.mean(Sxx)
    threshold = mean_val * min_amp_rel
    background = (Sxx < threshold)
    maxima = maxima & (~background)
    
    # Find indices
    f_indices, t_indices = np.where(maxima)
    
    # Return as list of (time_bin, freq_bin) sorted by time_bin
    peaks = list(zip(t_indices, f_indices))
    peaks.sort(key=lambda x: x[0])
    return peaks

def generate_paired_hashes(peaks, fan_value=FAN_VALUE, min_delta_t=MIN_DELTA_T, max_delta_t=MAX_DELTA_T):
    """Generates hashes by pairing nearby peaks."""
    hashes = []
    n_peaks = len(peaks)
    for i in range(n_peaks):
        t_A, f_A = peaks[i]
        count = 0
        for j in range(i + 1, n_peaks):
            t_B, f_B = peaks[j]
            delta_t = t_B - t_A
            if delta_t < min_delta_t:
                continue
            if delta_t > max_delta_t:
                break
            
            # Hash format: (freq_A, freq_B, delta_t)
            h = (int(f_A), int(f_B), int(delta_t))
            hashes.append((h, int(t_A)))
            
            count += 1
            if count >= fan_value:
                break
    return hashes

def generate_single_peaks(peaks):
    """Generates single peak 'hashes' for Q3A comparison."""
    # Hash is just (freq_bin), mapped to time_bin
    return [((int(f_bin),), int(t_bin)) for t_bin, f_bin in peaks]

class SongDatabase:
    def __init__(self, mode='paired'):
        """mode can be 'paired' or 'single'."""
        self.mode = mode
        self.database = {}  # hash -> list of (song_name, t_song)
        self.songs_list = []
        
    def index_song(self, song_name, audio_path):
        """Indexes a song and adds its hashes to the database."""
        try:
            audio, _ = load_audio(file_path=audio_path)
            _, _, Sxx = compute_spectrogram(audio)
            peaks = get_constellation(Sxx)
            
            if self.mode == 'paired':
                hashes = generate_paired_hashes(peaks)
            else:
                hashes = generate_single_peaks(peaks)
                
            for h, t_song in hashes:
                if h not in self.database:
                    self.database[h] = []
                self.database[h].append((song_name, t_song))
                
            if song_name not in self.songs_list:
                self.songs_list.append(song_name)
            print(f"Indexed song: {song_name} ({len(peaks)} peaks, {len(hashes)} hashes)")
        except Exception as e:
            print(f"Error indexing {song_name}: {e}")
            
    def match_clip(self, clip_audio):
        """Matches a query audio clip against the indexed database."""
        _, _, Sxx = compute_spectrogram(clip_audio)
        peaks = get_constellation(Sxx)
        
        if self.mode == 'paired':
            query_hashes = generate_paired_hashes(peaks)
        else:
            query_hashes = generate_single_peaks(peaks)
            
        # Matches: song_name -> list of (t_song - t_query) offsets
        matches = {}
        for h, t_q in query_hashes:
            if h in self.database:
                for song_name, t_d in self.database[h]:
                    offset = t_d - t_q
                    if song_name not in matches:
                        matches[song_name] = []
                    matches[song_name].append(offset)
                    
        # Find best match
        best_song = None
        max_votes = 0
        best_offsets = []
        
        for song_name, offsets in matches.items():
            # Find the peak count in the offset histogram
            if len(offsets) == 0:
                continue
            # Use binning with tolerance of 1 time bin
            offsets = np.array(offsets)
            min_off, max_off = offsets.min(), offsets.max()
            bins = np.arange(min_off - 1, max_off + 2, 1)
            hist, bin_edges = np.histogram(offsets, bins=bins)
            
            song_max_votes = hist.max()
            if song_max_votes > max_votes:
                max_votes = song_max_votes
                best_song = song_name
                best_offsets = offsets
                
        # Return best song name, match score (votes), and all offsets for plotting
        return best_song, max_votes, best_offsets, peaks

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Helper function to plot spectrogram and constellation
def plot_spec_and_constellation(audio, song_name="Song"):
    f, t, Sxx = compute_spectrogram(audio)
    peaks = get_constellation(Sxx)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spectrogram
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    img = ax1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax1.set_title(f"Spectrogram of {song_name}")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")
    fig.colorbar(img, ax=ax1, label="Magnitude (dB)")
    
    # Constellation
    ax2.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma', alpha=0.3)
    peak_times = [t[p[0]] for p in peaks]
    peak_freqs = [f[p[1]] for p in peaks]
    ax2.scatter(peak_times, peak_freqs, color='cyan', s=10, label="Peaks")
    ax2.set_title(f"Constellation Map of {song_name} ({len(peaks)} peaks)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test script locally
    print("Audio Fingerprinting engine written. Mode configs:")
    print(f"SR={TARGET_SR}, NPERSEG={NPERSEG}, OVERLAP={NOVERLAP}")
