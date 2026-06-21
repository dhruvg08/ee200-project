import os
import sys
import csv
import argparse
from fingerprint import SongDatabase, load_audio

def main():
    parser = argparse.ArgumentParser(description="Batch music identification tool")
    parser.add_argument("--query_dir", required=True, help="Directory containing query audio clips")
    parser.add_argument("--output", default="results.csv", help="Path to save results CSV (default: results.csv)")
    parser.add_argument("--db", default="data/songs_db.pkl", help="Path to database pickle file")
    
    # Handle direct positional arguments if passed (mookit evaluations sometimes pass just positional args)
    # If the user runs `python3 batch.py query_dir output_file`
    args, unknown = parser.parse_known_args()
    
    query_dir = args.query_dir
    output_file = args.output
    db_file = args.db
    
    # If standard args aren't parsed but positional are provided, fall back
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
        query_dir = sys.argv[1]
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
            
    print(f"Loading database from {db_file}...")
    if not os.path.exists(db_file):
        print(f"Database file not found: {db_file}. Please run build_db.py first.")
        sys.exit(1)
        
    db = SongDatabase.load(db_file)
    print("Database loaded successfully.")
    
    if not os.path.exists(query_dir):
        print(f"Query directory not found: {query_dir}")
        sys.exit(1)
        
    query_files = [f for f in os.listdir(query_dir) if f.lower().endswith(('.mp3', '.wav'))]
    query_files.sort()
    
    print(f"Found {len(query_files)} query files. Starting matching...")
    
    results = []
    for filename in query_files:
        filepath = os.path.join(query_dir, filename)
        print(f"Processing query: {filename}...")
        try:
            audio, _ = load_audio(filepath)
            prediction, votes, _, _ = db.match_clip(audio)
            
            # If no prediction or score too low, return empty or best guess
            if not prediction:
                prediction = ""
                
            results.append((filename, prediction))
            print(f"-> Predicted: {prediction} (score: {votes})")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append((filename, ""))
            
    # Write to CSV
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction"])
        for row in results:
            writer.writerow(row)
            
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
