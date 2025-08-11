import os, gc
import soundfile as sf
from datasets import load_dataset

DRIVE = "/content/drive/MyDrive"
CACHE = os.path.join(DRIVE, "huggingface_cache")
TMP   = os.path.join(DRIVE, "tmp")
OUT   = os.path.join(DRIVE, "JamendoMax_MP3")

os.makedirs(CACHE, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# 캐시/임시 전부 Drive로
os.environ["HF_HOME"] = CACHE
os.environ["HF_DATASETS_CACHE"] = CACHE
os.environ["HF_HUB_CACHE"] = CACHE
os.environ["TRANSFORMERS_CACHE"] = CACHE
os.environ["TMPDIR"] = TMP

def read_jamendo_max_streaming():
    # 스트리밍 모드: 대용량 캐시 없이 바로바로 읽어서 처리
    return load_dataset(
        "amaai-lab/JamendoMaxCaps",
        data_dir="data",
        streaming=True
    )

def save_mp3_files_streaming(dataset, output_dir=OUT):
    # 스트리밍은 iterable이라 enumerate 가능
    train_iter = dataset["train"]

    for i, example in enumerate(train_iter, 1):
        # 원본 경로 유지
        rel_path = example["audio"]["path"]  # 예: "split/folder/file.mp3" 같은 구조
        out_path = os.path.join(output_dir, rel_path)

        # 이미 만들었으면 건너뛰기(Resume)
        if os.path.exists(out_path):
            if i % 200 == 0:
                print(f"[skip] {i}: {out_path}")
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 오디오 배열은 접근 시점에 개별 다운로드/디코딩됨
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]

        # MP3로 저장
        sf.write(out_path, audio, sr, format="MP3")

        # 메모리 즉시 반환
        del audio
        gc.collect()

        if i % 50 == 0:
            print(f"[saved] {i}: {out_path}")

if __name__ == "__main__":
    ds = read_jamendo_max_streaming()
    save_mp3_files_streaming(ds)
