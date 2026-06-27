import numpy as np
import librosa
import pickle
import scipy.signal as signal
from scipy.ndimage import maximum_filter
from collections import defaultdict


TARGET_SR    = 8_000   
N_FFT        = 512     
HOP_LENGTH   = 128     
NEIGHBORHOOD = 20      
AMP_MIN_DB   = -60.0  
NUM_PEAKS    = 200  
FAN_OUT      = 15 
MIN_DT       = 1 
MAX_DT       = 200 


def load_audio(path: str, sr: int = TARGET_SR):
    """Load an audio file, convert to mono, resample to *sr*.

    Returns
    -------
    audio : np.ndarray  shape (N,)   mono float32 signal
    sr    : int         actual sample rate used
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio, sr


# ── Spectrogram ──────────────────────────────────────────────────────────────

def compute_spectrogram(audio: np.ndarray, sr: int = TARGET_SR):
    """Short-Time Fourier Transform → power spectrogram.

    Uses a Hann window of N_FFT samples with HOP_LENGTH samples hop.

    Returns
    -------
    f   : np.ndarray  frequency axis [Hz]        shape (F,)
    t   : np.ndarray  time axis [s]              shape (T,)
    Sxx : np.ndarray  power spectral density     shape (F, T)
    """
    f, t, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
    )
    return f, t, Sxx


# ── Constellation (peak picking) ─────────────────────────────────────────────

def get_constellation(
    f: np.ndarray,
    t: np.ndarray,
    Sxx: np.ndarray,
    amp_min_db: float = AMP_MIN_DB,
    max_peaks: int = NUM_PEAKS,
):
    """Extract the strongest local-maxima from a power spectrogram.

    A neighbourhood of NEIGHBORHOOD × NEIGHBORHOOD bins is used for the
    maximum filter so that peaks are well-separated.

    Returns
    -------
    peaks : list[tuple[int, int]]
        (t_bin_index, f_bin_index) sorted by time.
        Convention: p[0] = time index, p[1] = freq index.
        This matches the visualisation code in app.py:
            peak_times = [t_arr[p[0]] for p in q_peaks]
            peak_freqs = [f_arr[p[1]] for p in q_peaks]
    """
    Sxx_db = 10.0 * np.log10(Sxx + 1e-10)   # power → dB

    local_max = maximum_filter(
        Sxx_db,
        size=NEIGHBORHOOD,
        mode="constant",
        cval=Sxx_db.min() - 1,
    )
    is_peak = (Sxx_db == local_max) & (Sxx_db > amp_min_db)

    # Sxx.shape = (F, T) → np.where returns (freq_indices, time_indices)
    f_idx, t_idx = np.where(is_peak)

    if len(f_idx) == 0:
        return []

    # --- keep the strongest max_peaks ----------------------------------------
    mags = Sxx_db[f_idx, t_idx]
    if len(mags) > max_peaks:
        keep  = np.argpartition(mags, -max_peaks)[-max_peaks:]
        f_idx = f_idx[keep]
        t_idx = t_idx[keep]

    order = np.argsort(t_idx)
    peaks = list(zip(t_idx[order].tolist(), f_idx[order].tolist()))
    return peaks


# ── Hashing ──────────────────────────────────────────────────────────────────

def generate_hashes(peaks, fan_out: int = FAN_OUT):
    """Pair each anchor peak with its next *fan_out* peaks within [MIN_DT, MAX_DT].

    The hash key is a plain tuple (f1, f2, dt) so it is stable across
    pickle/unpickle in different Python processes.

    Returns
    -------
    hashes : list[tuple[tuple[int, int, int], int]]
        Each entry is ((f1, f2, dt), anchor_t_bin).
    """
    hashes = []
    n = len(peaks)
    for i, (t1, f1) in enumerate(peaks):
        for j in range(1, fan_out + 1):
            if i + j >= n:
                break
            t2, f2 = peaks[i + j]
            dt = t2 - t1
            if dt < MIN_DT:
                continue
            if dt > MAX_DT:
                # peaks are time-sorted; no later pair will satisfy MAX_DT
                break
            hashes.append(((f1, f2, dt), t1))
    return hashes


# ── Database ─────────────────────────────────────────────────────────────────

class SongDatabase:
    """In-memory audio fingerprint database.

    Attributes
    ----------
    songs_list : list[str]
        Maps song_id (int) → song name (str, filename without extension).
    db : defaultdict[tuple, list]
        Maps (f1, f2, dt) → [(song_id, anchor_t_bin), ...].
    """

    def __init__(self):
        self.songs_list: list = []
        self.db: dict = defaultdict(list)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_song(self, song_name: str, audio: np.ndarray, sr: int = TARGET_SR) -> int:
        """Fingerprint *audio* and insert it into the database.

        Parameters
        ----------
        song_name : str   label stored in songs_list (filename without extension)
        audio     : np.ndarray   mono float32 signal at *sr* Hz
        sr        : int          sample rate

        Returns
        -------
        n_hashes : int   number of hashes generated (useful for progress reporting)
        """
        song_id = len(self.songs_list)
        self.songs_list.append(song_name)

        f, t, Sxx = compute_spectrogram(audio, sr)
        peaks     = get_constellation(f, t, Sxx)
        hashes    = generate_hashes(peaks)

        for key, t_bin in hashes:
            self.db[key].append((song_id, t_bin))

        return len(hashes)

    # ── Matching ──────────────────────────────────────────────────────────────

    def match_clip(
        self,
        audio: np.ndarray,
        sr: int = TARGET_SR,
        min_votes: int = 8,
        min_margin_ratio: float = 1.3,
    ):
        
        f, t, Sxx = compute_spectrogram(audio, sr)
        q_peaks   = get_constellation(f, t, Sxx)
        hashes    = generate_hashes(q_peaks)

        votes: dict = defaultdict(lambda: defaultdict(int))

        for key, q_t in hashes:
            if key in self.db:
                for song_id, db_t in self.db[key]:
                    votes[song_id][db_t - q_t] += 1

        if not votes:
            return None, 0, None, q_peaks

        peak_counts = {sid: max(off_counts.values())
                       for sid, off_counts in votes.items()}
        ranked = sorted(peak_counts.items(), key=lambda kv: kv[1], reverse=True)

        best_id, best_votes = ranked[0]
        runner_up_votes = ranked[1][1] if len(ranked) > 1 else 0

        if best_votes < min_votes:
            return None, best_votes, None, q_peaks

        if runner_up_votes > 0 and best_votes < runner_up_votes * min_margin_ratio:
            return None, best_votes, None, q_peaks

        # Collect all (offset × count) for the winning song for the histogram
        all_offsets: list = []
        for off, cnt in votes[best_id].items():
            all_offsets.extend([off] * cnt)

        return (
            self.songs_list[best_id],
            best_votes,
            np.array(all_offsets, dtype=np.int64),
            q_peaks,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the database to *path*."""
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "SongDatabase":
        """Load a previously pickled database from *path*."""
        with open(path, "rb") as fh:
            return pickle.load(fh)
