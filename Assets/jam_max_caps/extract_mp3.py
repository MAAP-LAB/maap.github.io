import os
import soundfile as sf
from datasets import load_dataset

# Google Drive 경로
DRIVE_PATH = "/content/drive/MyDrive"
CACHE_DIR = os.path.join(DRIVE_PATH, "huggingface_cache")  # HF Dataset Cache
OUTPUT_DIR = os.path.join(DRIVE_PATH, "JamendoMax_MP3")    # MP3 저장 경로

def read_jamendo_max():
    # Google Drive를 cache_dir로 지정하여 캐싱
    dataset = load_dataset(
        "amaai-lab/JamendoMaxCaps", 
        data_dir="data",
        cache_dir=CACHE_DIR
    )
    return dataset

def save_mp3_files(dataset, output_dir=OUTPUT_DIR):
    for example in dataset['train']:
        audio = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        path = example['audio']['path']

        # 원본 폴더 구조 유지
        output_path = os.path.join(output_dir, path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sf.write(output_path, audio, sample_rate, format='MP3')
        print(f"Saved file: {output_path}")

if __name__ == "__main__":
    dataset = read_jamendo_max()
    save_mp3_files(dataset)
