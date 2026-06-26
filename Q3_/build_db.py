#!/usr/bin/env python3
"""
build_db.py — Index a folder of songs into songs_db.pkl

Run this ONCE locally (or in Streamlit Cloud's startup) before launching
the app.  The resulting songs_db.pkl should be committed / shipped alongside
the app so the deployed version works immediately.

Usage examples
--------------
    python build_db.py                          # scans ./songs/, writes ./songs_db.pkl
    python build_db.py --songs_dir data/songs   # custom input folder
    python build_db.py --output my_db.pkl       # custom output file

Important
---------
The song's filename WITHOUT its extension is the label that the identifier
will output.  Do NOT rename the files provided by the course.
"""

import os
import sys
import time
import argparse

from fingerprint import SongDatabase, load_audio

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")


def build(songs_dir: str, output: str) -> None:
    if not os.path.isdir(songs_dir):
        print(f"[ERROR] Directory not found: '{songs_dir}'")
        sys.exit(1)

    # Collect audio files (hidden files like .DS_Store are skipped)
    files = sorted(
        f for f in os.listdir(songs_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS) and not f.startswith(".")
    )

    if not files:
        print(f"[!] No supported audio files found in: {songs_dir}")
        print(f"    Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(files)} song(s) in '{songs_dir}'. Building database ...")
    print("-" * 65)

    db = SongDatabase()
    t0 = time.time()

    for i, filename in enumerate(files, 1):
        song_name = os.path.splitext(filename)[0]   # label = name without ext
        filepath  = os.path.join(songs_dir, filename)
        t_song    = time.time()
        try:
            audio, sr = load_audio(filepath)
            n_hashes  = db.add_song(song_name, audio, sr)
            elapsed   = time.time() - t_song
            print(
                f"  [{i:>3}/{len(files)}]  {song_name:<45}"
                f"  {n_hashes:>7,} hashes  ({elapsed:.1f}s)"
            )
        except Exception as exc:
            print(f"  [{i:>3}/{len(files)}]  ERROR — {filename}: {exc}")

    print("-" * 65)
    db.save(output)

    total_hashes   = sum(len(v) for v in db.db.values())
    total_keys     = len(db.db)
    total_time     = time.time() - t0

    print(f"\n✓  Database saved → {output}")
    print(f"   Songs    : {len(db.songs_list)}")
    print(f"   Hash keys: {total_keys:,}")
    print(f"   Entries  : {total_hashes:,}")
    print(f"   Time     : {total_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the audio fingerprint database from a folder of songs."
    )
    parser.add_argument(
        "--songs_dir",
        default="songs",
        help="Folder containing song files (default: songs/)",
    )
    parser.add_argument(
        "--output",
        default="songs_db.pkl",
        help="Output pickle file (default: songs_db.pkl)",
    )
    args = parser.parse_args()
    build(args.songs_dir, args.output)
