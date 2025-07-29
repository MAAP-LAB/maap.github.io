import os
import requests
from dataclasses import dataclass,field
from typing import Dict

@dataclass
class JamendoMusicConfig:
    client_id:str="1f75034b"
    save_path:str = 'data/downloads'
    download:bool = False  # Set to True to download tracks, False to save to CSV
    offset_start:int = 0
    offset_end:int = 1000000
    step:int=200 # Set to step number of download, Max is 200  
    num_threads:int=16
    
    vocalinstrumental:str='' #{'vocal', 'instrumental'}
    gender:str='' #{'male', 'female'}
    speed:str='high' #{'verylow', 'low', 'medium', 'high', 'veryhigh'}
    lang:str='en'
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

class JamendoMusic:
    def __init__(self, config: JamendoMusicConfig):
        self.config = config
        """
        Initialize the Jamendo Music API client.
        """
        if self.config.save_path is None or isinstance(self.config.save_path, str) is False:
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
        if self.config.save_path is None or isinstance(self.config.save_path, str) is False:
            raise ValueError("save_path must be a valid string path.")
        self.genre = genre
        os.makedirs(f"{self.config.save_path}", exist_ok=True)  # Ensure the save path exists
            
    def search_tracks(self,offset,tags=[]):
        """
        search for a genre on Jamendo Music.
        """
        url = f"{self.base_url}/v3.0/tracks"
        params = {
            'client_id': self.config.client_id,
            'format': 'json',
            'offset': offset,
            'limit': self.config.step,
            'tags': tags,
            'vocalinstrumental':self.config.vocalinstrumental,
            'gender': self.config.gender,
            'speed' : self.config.speed,
            'lang' : self.config.lang
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
        
        with open(filename, "a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write header if the file is empty
            if file.tell() == 0:  # Check if the file is empty
                writer.writerow(['name','duration','artist_name','waveform','tag','speed','gender','vocalinstrumental','lang','audio'])
            
            def write_track(item):
                if item['audiodownload_allowed']:
                    item['tag'] = self.genre  # Add genre to the item
                    item['speed'] = self.config.speed
                    item['gender'] = self.config.gender
                    item['vocalinstrumental'] = self.config.vocalinstrumental
                    item['lang'] = self.config.lang

                    name = self.clean_name(item['name'])
                    audio = f"{self.config.save_path}/{self.genre}_{name}.wav" # Construct the file path
                    # Ensure all keys are present in the item
                    row = [name] + [item.get(key, '') for key in ['duration','artist_name','waveform','tag','speed','gender','vocalinstrumental','lang']] + [audio]
                    writer.writerow(row)

            with ThreadPoolExecutor(self.config.num_threads) as e:
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

        if permission:
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
        with ThreadPoolExecutor(self.config.num_threads) as e:
            return list(e.map(downloader, tracks))
        
def main(config:JamendoMusicConfig):
    jamendo = JamendoMusic(config)
    for tag, genre_list in config.tags.items():
        for idx,genre in enumerate(genre_list):
            jamendo.setPath(genre=genre)
            for i in range(config.offset_start,config.offset_end,config.step):
                jamendo_tracks = jamendo.search_tracks(offset=i,tags=[genre])
                if len(jamendo_tracks) == 0:
                    print(f"No more tracks found at offset {i}.")
                    break
                print(f"Found {len(jamendo_tracks)} tracks at offset {i}.",genre_list[idx])
                
                if config.download:
                    jamendo.download_batch(jamendo_tracks, file_extension='wav')
                    print(f"Downloaded {len(jamendo_tracks)} tracks for genre {genre} at offset {i}.")
                else:
                    metadata = os.path.join(config.save_path,"metadata.csv")
                    jamendo.to_csv_batch(jamendo_tracks, filename=metadata)
                    print(f"Tracks saved to {metadata}.")
    return

if __name__ == "__main__":
    from argparse_dataclass import ArgumentParser
    parser = ArgumentParser(JamendoMusicConfig)
    config = parser.parse_args()
    main(config)