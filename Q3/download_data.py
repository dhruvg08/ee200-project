import os
import zipfile
import requests

def download_file_from_google_drive(file_id, destination):
    print(f"Downloading file with ID {file_id} from Google Drive...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # Try downloading
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    save_response_content(response, destination)
    print("Download completed.")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def main():
    file_id = "1U4fbfBa34NkJvLwu9coiF8VTztxNYVyj"
    zip_dest = "project_data.zip"
    extract_dir = "data"
    
    # Create extraction directory if not exists
    os.makedirs(extract_dir, exist_ok=True)
    
    # Download
    download_file_from_google_drive(file_id, zip_dest)
    
    # Extract
    print(f"Extracting {zip_dest} to {extract_dir}...")
    with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed successfully.")
    
    # Clean up zip file
    if os.path.exists(zip_dest):
        os.remove(zip_dest)
        print("Cleaned up temporary zip file.")

if __name__ == "__main__":
    main()
