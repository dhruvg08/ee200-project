import os
import time
from fingerprint import SongDatabase

def main():
    songs_dir = "data/songs_db"
    db_file = "data/songs_db.pkl"
    
    print("Initializing song database in 'paired' mode...")
    db = SongDatabase(mode='paired')
    
    files = [f for f in os.listdir(songs_dir) if f.endswith(".mp3")]
    files.sort()
    
    print(f"Found {len(files)} songs in {songs_dir}. Starting indexing...")
    start_time = time.time()
    
    for idx, filename in enumerate(files):
        song_name = os.path.splitext(filename)[0]
        song_path = os.path.join(songs_dir, filename)
        print(f"[{idx+1}/{len(files)}] Indexing: {song_name}")
        db.index_song(song_name, song_path)
        
    print(f"Indexing complete in {time.time() - start_time:.2f} seconds.")
    print(f"Saving database to {db_file}...")
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    db.save(db_file)
    print("Database saved successfully!")

if __name__ == "__main__":
    main()
