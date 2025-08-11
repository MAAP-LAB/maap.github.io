import os
import soundfile as sf
from datasets import load_dataset

# 구글 드라이브 내에 캐시 파일을 저장할 폴더 경로를 지정합니다.
# 폴더는 미리 만들어두거나, 코드를 통해 생성할 수 있습니다.
CACHE_DIR = "/content/drive/MyDrive/huggingface_cache" 
os.makedirs(CACHE_DIR, exist_ok=True)

def read_jamendo_max():
    """
    JamendoMaxCaps 데이터셋을 구글 드라이브를 캐시로 사용하여 로드합니다.
    """
    # data_dir 인자는 Hugging Face Hub에서 직접 로드할 때는 필요하지 않을 수 있습니다.
    # 만약 특정 데이터 파일만 로드해야 하는 경우가 아니라면 제거해도 무방합니다.
    dataset = load_dataset("amaai-lab/JamendoMaxCaps", cache_dir=CACHE_DIR)
    return dataset

def save_mp3_files(dataset, output_dir_name="mp3_files"):
    """
    데이터셋의 오디오를 MP3 파일로 구글 드라이브에 저장합니다.
    """
    # 저장할 경로도 구글 드라이브로 지정합니다.
    output_dir = f"/content/drive/MyDrive/{output_dir_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"오디오 파일 저장 위치: {output_dir}")

    # 실제 파일 저장을 위해 주석을 해제하고 사용하세요.
    for example in dataset['train']:
        audio = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        # 파일 이름만 추출하여 새로운 경로 생성
        file_name = os.path.basename(example['audio']['path'])
        output_path = os.path.join(output_dir, file_name)
        
        # Hugging Face datasets는 기본적으로 오디오를 로드할 때 디코딩하므로
        # mp3로 다시 저장하려면 soundfile과 같은 라이브러리가 필요합니다.
        sf.write(output_path, audio, sample_rate)
        print(f"저장 완료: {output_path}")

if __name__ == "__main__":
    # 1. 데이터셋 로드 (캐시 경로: 구글 드라이브)
    jamendo_dataset = read_jamendo_max()
    print("데이터셋 정보:")
    print(jamendo_dataset)

    # 2. 오디오 파일 저장 (저장 경로: 구글 드라이브)
    # save_mp3_files(jamendo_dataset)