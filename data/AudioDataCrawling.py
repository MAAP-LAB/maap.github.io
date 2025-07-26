import os
import requests
from dataclasses import dataclass,field
from typing import Dict

@dataclass
class JamendoMusicConfig:
    client_id:str="1f75034b"
    save_path:str = './downloads'
    download:bool = False  # Set to True to download tracks, False to save to CSV
    offset_start:int = 0
    offset_end:int = 1000000
    step:int=200 # Set to step number of download, Max is 200  
    num_threads:int=16
    tags:Dict[str, list] = field(
        default_factory=
            lambda:
                {
                "genre": [
                    "pucnk",
                    "electronic",
                    "soundtrack",
                    "ambient",
                    "rnb",
                    "dance",
                    "lounge",
                    "house",
                    "trance",
                    "progressive",
                    "classical",
                    "rap",
                    "techno",
                    "indie",
                    "rock",
                    "pop",
                    "jazz",
                    "hiphop",
                    "metal",
                    "blues",
                    "reggae",
                    "country",
                    "folk",
                ],
                "mood/theme": [
                    "dream",
                    "emotional",
                    "film",
                    "energetic",
                    "inspiring",
                    "love",
                    "melancholic",
                    "relaxing",
                    "sad",
                    "romantic",
                    "hopeful",
                    "motivational",
                    "happy",
                    "sport",
                    "children",
                    "trailer",
                    "joyful",
                    "christmas",
                    "epic",
                    "motivational",
                    "dark",
                    "scifi",
                    "festive"
                ],
                "instrument": [
                    "synthesizer",
                    "drums",
                    "strings",
                    "guitar",
                    "piano",
                    "saxophone",
                    "beat",
                    "violin",
                    "bell",
                    "percussion",
                    "choir",
                    "pad",
                    "flute",
                    "electricguitar",
                    "keyboard",
                    "horn",
                    "guitar",
                    "bongo",
                    "accordion",
                    "bass",
                    "clavier"
                ]
                }
        )

class JamendoMusic(JamendoMusicConfig):
    def __init__(self, args: JamendoMusicConfig):
        # args의 속성들을 현재 인스턴스에 복사
        for field_name in args.__dataclass_fields__:
            setattr(self, field_name, getattr(args, field_name))
        """
        Initialize the Jamendo Music API client.
        """
        if self.save_path is None or isinstance(self.save_path, str) is False:
            raise ValueError("save_path must be a valid string path.")
        
        self.base_url = "https://api.jamendo.com"
        self.genre = None  # Initialize genre to None

    def clean_name(self, name):
        """
        Clean the track name by removing special characters.
        """
        return name \
                .rstrip()\
                .lstrip()\
                .replace('/', '')\
                .replace('\\', '')\
                .replace(':', '')\
                .replace('?', '')\
                .replace('*', '')\
                .replace('"', '')\
                .replace('<', '')\
                .replace('>', '')\
                .replace('|', '')\
                .replace(' ','')\
                .replace('_', '')\
                .replace(';', '')\
                .replace('~', '')\
                .replace('`', '')\
                .replace('.', '')\
                .replace('!', '')\
                .replace('@', '')\
                .replace('#', '')\
                .replace('$', '')\
                .replace('%', '')\
                .replace('^', '')\
                .replace('(', '')\
                .replace(')','')\
                .replace('"', '')\
                .replace('\'', '')\
                .replace('+', '')\
                .replace('=', '')\
                .replace('{', '')\
                .replace('}', '')\
                .replace('[', '')\
                .replace(']', '')\
                .replace(',', '')\
                .replace('`', '')
    
    def setPath(self,genre="pop"):
        """
        Set the save path for downloaded files.
        """
        if self.save_path is None or isinstance(self.save_path, str) is False:
            raise ValueError("save_path must be a valid string path.")
        self.genre = genre
        os.makedirs(f"{self.save_path}", exist_ok=True)  # Ensure the save path exists
            
    def search_tracks(self,tags=[]):
        """
        search for a genre on Jamendo Music.
        """
        url = f"{self.base_url}/v3.0/tracks"
        params = {
            'client_id': self.client_id,
            'format': 'json',
            'offset': self.offset_start,
            'limit': self.step,  # Limit the number of results
            'tags': tags,  # Tags to filter tracks
            # 'order': 'popularity',  # Order by popularity
            # 'instrumental': '1',  # Only instrumental tracks
        }

        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            response.raise_for_status()

    def to_csv(self,track,filename='jamendo_tracks.csv'):
        """
        Save the tracks to a CSV file.
        """
        import csv

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        
        if not track:
            print("No tracks to save.")
            return
        
        with open(filename, "a") as file:
            writer = csv.writer(file)
            # Write header if the file is empty
            if file.tell() == 0:  # Check if the file is empty
                writer.writerow(['name','duration','artist_name','waveform','tag'])

            for item in track:
                if item['audiodownload_allowed']:
                    item['tag'] = self.genre  # Add genre to the item
                    name = self.clean_name(item['name'])

                    # Ensure all keys are present in the item
                    row = [name] + [item.get(key, '') for key in ['duration','artist_name','waveform','tag']]
                    writer.writerow(row)

    def to_csv_batch(self,tracks,filename='jamendo_tracks.csv'):
        """
        Save a batch of tracks to a CSV file.
        """
        import csv
        from concurrent.futures import ThreadPoolExecutor

        if not tracks:
            print("No tracks to save.")
            return

        with open(filename, "a") as file:
            writer = csv.writer(file)
            # Write header if the file is empty
            if file.tell() == 0:  # Check if the file is empty
                writer.writerow(['name','duration','artist_name','waveform','tag','audio'])

            def write_track(item):
                if item['audiodownload_allowed']:
                    item['tag'] = self.genre  # Add genre to the item
                    name = self.clean_name(item['name'])
                    audio = f"{self.save_path}/{self.genre}_{name}.wav"  # Construct the file path
                    # Ensure all keys are present in the item
                    row = [name] + [item.get(key, '') for key in ['duration','artist_name','waveform','tag']] + [audio]
                    writer.writerow(row)

            with ThreadPoolExecutor(self.num_threads) as e:
                return list(e.map(write_track, tracks))
            

    def download_url(self,data,file_extension='wav'):
        """
        Download a file from a given URL.
        """
        url =data['audiodownload']
        permission= data['audiodownload_allowed']
        artist_name = data['artist_name']
        album_name = '_'.join(data['album_name'].rstrip().lstrip().split(' '))

        name = self.clean_name(data['name'])

        if permission is False:
            return f"Permission denied for downloading {name} by {artist_name} from album {album_name}."
        
        print(f"Downloading {name} by {artist_name} from album {album_name}...")

        response = requests.get(url, stream=True)
        
        if response.status_code == 200:

            if self.genre is not None: file = f"{self.save_path}/{self.genre}_{name}.{file_extension}"
            else: raise ValueError("Genre must be set before downloading tracks.")

            with open(file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return file
        else:
            response.raise_for_status()

    def download_batch(self,tracks,file_extension='wav'):
        """
        Download a batch of tracks.
        """
        import functools
        from concurrent.futures import ThreadPoolExecutor

        downloader = functools.partial(
            self.download_url, file_extension=file_extension
        )
        with ThreadPoolExecutor(self.num_threads) as e:
            return list(e.map(downloader, tracks))

def main(args):
    jamendo = JamendoMusic(args)
    for tag, genre_list in jamendo.tags.items():
        for idx,genre in enumerate(genre_list):
            jamendo.setPath(genre=genre)
            for i in range(jamendo.offset_start,jamendo.offset_end, jamendo.step):
                jamendo_tracks = jamendo.search_tracks(tags=[genre])
                if len(jamendo_tracks) == 0:
                    print(f"No more tracks found at offset {i}.")
                    break
                print(f"Found {len(jamendo_tracks)} tracks at offset {i}.",genre_list[idx])
                
                if jamendo.download:
                    jamendo.download_batch(jamendo_tracks, file_extension='wav')
                    print(f"Downloaded {len(jamendo_tracks)} tracks for genre {genre} at offset {i}.")
                else:
                    metadata = f"./downloads/metadata.csv"
                    jamendo.to_csv_batch(jamendo_tracks, filename=metadata)
                    print(f"Tracks saved to {metadata}.")
    return

if __name__ == "__main__":
    from argparse_dataclass import ArgumentParser
    parser = ArgumentParser(JamendoMusicConfig)
    args = parser.parse_args()
    main(args)