import os

# Hugging Face 캐시 전부 Google Drive로
os.environ["HF_HOME"] = "/content/drive/MyDrive/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/content/drive/MyDrive/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/drive/MyDrive/huggingface_cache"

from datasets import load_dataset
import soundfile as sf

DRIVE_PATH = "/content/drive/MyDrive"
OUTPUT_DIR = os.path.join(DRIVE_PATH, "JamendoMax_MP3")

def read_jamendo_max():
    dataset = load_dataset(
        "amaai-lab/JamendoMaxCaps", 
        data_dir="data",
        cache_dir=os.environ["HF_HOME"]
    )
    return dataset

def save_mp3_files(dataset, output_dir=OUTPUT_DIR):
    for example in dataset['train']:
        audio = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        path = example['audio']['path']

        output_path = os.path.join(output_dir, path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sf.write(output_path, audio, sample_rate, format='MP3')
        print(f"Saved file: {output_path}")

if __name__ == "__main__":
    dataset = read_jamendo_max()
    save_mp3_files(dataset)
