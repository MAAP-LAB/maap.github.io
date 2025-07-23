import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import librosa
from dataclasses import dataclass

@dataclass
class DataVisConfig:
    DATA_PATH = "./downloads/metadata.csv"   # 👉 여기에 너의 CSV 파일 경로 입력
    OUT_PATH = "."
    WAVEFORM_COL = "waveform"
    AUDIO_COL = "audio"

class DataVis(DataVisConfig):
    def __init__(self):
        super().__init__()
    
        # CSV 로딩
        self.df = pd.read_csv(self.DATA_PATH)
        print("✅ CSV 로드 완료:", self.df.shape)

    def vis_duration(self):
        # ---------------------------------------------------
        # 1. duration 분포 분석
        # ---------------------------------------------------
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df['duration'], bins=50, kde=True)
        plt.title('Duration Distribution')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUT_PATH,'vis_duration_distribution.png'))
        print("📊 duration 분포 그래프 저장됨")

        # 이상치 확인
        print("\n📌 Duration 통계:")
        print(self.df['duration'].describe())
    
    def vis_tag(self):
        # ---------------------------------------------------
        # 2. tag 분석 (빈도 높은 상위 태그 시각화)
        # ---------------------------------------------------
        all_tags = self.df['tag'].dropna().astype(str).str.split(',').explode()
        tag_counts = all_tags.value_counts()

        plt.figure(figsize=(10, 6))
        sns.barplot(y=tag_counts.head(20).index, x=tag_counts.head(20).values)
        plt.title("Top 20 Tags")
        plt.xlabel("Count")
        plt.ylabel("Tag")
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUT_PATH,'vis_top_20_tags.png'))
        print("📊 상위 20개 태그 그래프 저장됨")

        # 태그 수 출력
        print(f"\n📌 전체 고유 태그 수: {len(tag_counts)}")

    def vis_waveform(self):
        # ---------------------------------------------------
        # waveform 길이 분석 - "peaks" 배열의 길이
        # ---------------------------------------------------
        def extract_peaks_length(waveform_str):
            try:
                waveform_dict = json.loads(waveform_str)
                return len(waveform_dict.get("peaks", []))
            except Exception as e:
                return np.nan

        self.df['waveform_length'] = self.df['waveform'].apply(extract_peaks_length)

        plt.figure(figsize=(8, 4))
        sns.histplot(self.df['waveform_length'].dropna(), bins=50)
        plt.title('Waveform (peaks) Length Distribution')
        plt.xlabel('Number of Peaks')
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUT_PATH,'vis_waveform_peaks_length_distribution.png'))
        print("📊 waveform peaks 길이 분포 그래프 저장됨")

        print("\n📌 waveform peaks 길이 통계:")
        print(self.df['waveform_length'].describe())

    def vis_sample_saveform(self):
        def extract_peaks_array(waveform_str):
            try:
                waveform_dict = json.loads(waveform_str)
                return waveform_dict.get("peaks", [])
            except Exception as e:
                return []

        # 가장 긴 waveform 기준 시각화
        sample_idx = self.df['waveform_length'].idxmax()
        sample_peaks = extract_peaks_array(self.df.loc[sample_idx, 'waveform'])

        plt.figure(figsize=(12, 3))
        plt.plot(sample_peaks[:2000])  # 너무 길면 일부만
        plt.title(f"Waveform Peaks of Sample (id={sample_idx})")
        plt.xlabel("Sample Index")
        plt.ylabel("Peak Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUT_PATH,"vis_sample_waveform_peaks.png"))
        print("📊 waveform peaks 시각화 저장됨")

        print(f"\n📌 Sample waveform peaks 값:")
        print(f"Max: {np.max(sample_peaks):.2f}, Min: {np.min(sample_peaks):.2f}, Mean: {np.mean(sample_peaks):.2f}, Std: {np.std(sample_peaks):.2f}")

    
    def vis_artist_group(self):
        # ---------------------------------------------------
        # 5. artist_name별 곡 수
        # ---------------------------------------------------
        artist_counts = self.df['artist_name'].value_counts()

        plt.figure(figsize=(10, 4))
        sns.barplot(x=artist_counts.head(10).values, y=artist_counts.head(10).index)
        plt.title('Top 10 Artists by Track Count')
        plt.xlabel('Number of Tracks')
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUT_PATH,"vis_top_artists.png"))
        print("📊 Top artist 그래프 저장됨")

        # ---------------------------------------------------
        # 6. 결측 및 이상치 확인
        # ---------------------------------------------------
        print("\n📌 결측치 확인:")
        print(self.df.isnull().sum())

        print("\n📌 duration == 0 인 샘플 수:", (self.df['duration'] == 0).sum())
        print("📌 waveform_length == 0 인 샘플 수:", (self.df['waveform'] == 0).sum())

        # ---------------------------------------------------
        # 완료 메시지
        # ---------------------------------------------------
        print("\n✅ 모델 입력 분석 스크립트 완료! 그래프들은 현재 디렉토리에 저장됨.")

    @staticmethod
    def vis_log_mel_spectrogram(input_features):
        from transformers import WhisperFeatureExtractor
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        plt.figure(figsize=(8,4))
        librosa.display.specshow(
            np.asarray(input_features[0]),
            x_axis="time",
            y_axis="mel",
            sr=feature_extractor.sampling_rate,
            hop_length=feature_extractor.hop_length,
        )
        plt.savefig('./vis_log_mel_spectrogram')

if __name__=="__main__":
    from argparse_dataclass import ArgumentParser
    parser = ArgumentParser(DataVisConfig)
    args = parser.parse_args()

    vis = DataVis()
    vis.vis_duration()
    vis.vis_tag()
    vis.vis_artist_group()
    vis.vis_waveform()
    vis.vis_sample_saveform()