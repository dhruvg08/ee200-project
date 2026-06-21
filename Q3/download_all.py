import os
import requests

songs = {
    "1py5PxQ2-NsczhJoARK-dX4fana8iotXQ": "A Day In The Life.mp3",
    "1L9SL6wjfE7rtyFhkkCk-0wYhKs0AFS2A": "A Hard Day's Night.mp3",
    "1gAWM8oCZAb1UjXN8UWcxPZ_G0N8cQYni": "Across The Universe.mp3",
    "1cT35ShMp7ur9bZgm88KUVcIL2O2rXjih": "Back In The U.S.S.R..mp3",
    "1CVUME6OvZ_n62e7_JLrjU85a3y6_ZLMW": "Blackbird.mp3",
    "1-FbYD8LBlG-YVuw0yxBJ76v9AiBv1nzv": "Bohemian Rhapsody.mp3",
    "1xB476UTdo3dzUe5DNoTd6OfF1M9KsdNk": "Can't Buy Me Love.mp3",
    "15fvnUmi2kIU-W4I1HedUYgTfnC2awpBU": "Crazy Little Thing Called Love.mp3",
    "19M2s4X1Z9aOV-u2KUPW2IR9FDeBdbGDQ": "Day Tripper.mp3",
    "1GS4kZh2OVxogXgpLkvnVGc--uC6oA4ku": "Don't Stop Me Now.mp3",
    "1Jj07UVd8N4iBNmoNDc5N_wSy9bdqmpMr": "Drive My Car.mp3",
    "1RAHG5snwFOe4fvx7DrK5zGIh0vL3lSDk": "Eight Days A Week.mp3",
    "1SblFwcoip0AIb8JRvGlxUf_jkN27DBeU": "Eleanor Rigby.mp3",
    "1xJOuVSSF20vlNp2E72-q3C3AMII4hQFh": "Get Back.mp3",
    "1aqDhY8bVzwK-pJChgd5AWXOihQlrY2qk": "Hello, Goodbye.mp3",
    "1zMFQ0gPZiI7OLaV1HIlFz9AoPDX5dpYt": "Help!.mp3",
    "1z9yYDrov6d4_FrTxGT_FZm6XO66Siobu": "Helter Skelter.mp3",
    "12C4-teeWMDccmiihZtamcFAcRiRMqMJ4": "Hey Jude.mp3",
    "1Xh-gtH5-QiiF3JyKFg0vLCr9DJbApSii": "I Am The Walrus.mp3",
    "1dka88rQhPvEaJNiCUfJunYinaxKR7Ol6": "I Saw Her Standing There.mp3",
    "1dMbnVUKvFtxffPh-v_9RfAcbjxxZ7-1m": "I Want It All.mp3",
    "1MvYOkXoeuwLJOlPO2mKOT5WG4FqJwTtf": "I Want To Hold Your Hand.mp3",
    "1HYbIdaft4klwAccdWXW0dBYPJvZ29p28": "I'll Follow The Sun.mp3",
    "1mxRJToXfi8NTQNsZji6Gii7h85nS8EEm": "I've Got A Feeling.mp3",
    "16GurU1pDaEbgCjUVapjrkGbhsNhSDZNS": "In My Life.mp3",
    "1TK570obz8rMUKwBfxOK46qJXKTPQ_QQC": "Killer Queen.mp3",
    "1T6pMsTi9TMz-_rGef9Z4pY9Z5R0OviFM": "Let It Be.mp3",
    "1X83x8_09gAGWbcPNSwtuWwr28gLJ14_Z": "Love Me Do.mp3",
    "1-viCk2LOFWRsSS6uZ5GZ1zvl6fLPrrgN": "Lucy In The Sky With Diamonds.mp3",
    "1mPcEQZ7ccF3wQS-Uji97hzqj_Rweg7pJ": "Never Gonna Give You Up.mp3",
    "1tVU7L50YdKiasn2SXAIJIiia5EQwGIG4": "Norwegian Wood (This Bird Has Flown).mp3",
    "12ddCCz6pY2IGxTwIX95TMuGzKUEXUMEC": "Penny Lane.mp3",
    "17Bg2ZVoZ_oHfywU29GUNHbpuRSIVQhE5": "Radio Ga Ga.mp3",
    "1-4u9xvqiLnS_V2IzrwGYvVitCbf97s1p": "Revolution.mp3",
    "1IlRXdFQJyMYqp5bzaN7_6vdGQztckVvy": "Sgt. Pepper's Lonely Hearts Club Band.mp3",
    "131i7mMQ1cg9a-bk-CYHj1l5lZ0Gnk0IN": "She Said She Said.mp3",
    "1va3srczJSv-i95B--fO9aQtoa6xKdaMY": "Somebody To Love.mp3",
    "1XeFr3bUc7-z44vOPQgcqIo2yplfeCjMy": "Something.mp3",
    "1MDaBxHeB1msDI1zPpANOUZwIKT76X1o2": "Taxman.mp3",
    "1SBdzWbbekSX4IwqrUj1pttqctGus8lrp": "The Long And Winding Road.mp3",
    "1Ag515CqCtMUZuARXbfTPMtDVj6Vsj3cf": "Two Of Us.mp3",
    "1BGZ4aP57m430m8AAtPRvr7w3gXNMZi7J": "Under Pressure.mp3",
    "1vgAV4cXzIarUujCGmJ7PF21dhItILEJg": "We Are The Champions.mp3",
    "1QV12_zyfqJBadKhe6jW-iVjm2K0-Awfn": "We Can Work It Out.mp3",
    "1SmlrIme4iL4dYIQpigwU43BECOeoy_SH": "We Will Rock You.mp3",
    "1IO3RZLp0yttGzI5vb7QwG9o7RlZ08gP5": "While My Guitar Gently Weeps.mp3",
    "1cSPLp8OCNSEb8a_rugg4Cb4A_J-2Ocyw": "With A Little Help From My Friends.mp3",
    "1GkP3higlrc8ZctPj4Odnjjh40T-AXQwy": "Within You Without You.mp3",
    "1IPJl_pecKzWuc__kiUi6FTcYjCrPygfC": "Yesterday.mp3",
    "1F1RbGa2MhUDow92_hoA9SnITKutVvl9r": "You Really Got A Hold On Me.mp3"
}

os.makedirs("data/songs_db", exist_ok=True)
session = requests.Session()

def download_file(file_id, dest):
    print(f"Downloading {dest}...")
    URL = "https://docs.google.com/uc?export=download"
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check for confirm token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    with open(dest, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

for file_id, filename in songs.items():
    dest_path = os.path.join("data/songs_db", filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        print(f"{filename} already exists. Skipping.")
    else:
        try:
            download_file(file_id, dest_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

print("All downloads finished!")
