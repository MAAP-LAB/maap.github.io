import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
from datasets import load_dataset, Audio
from functools import partial
from dataclasses import dataclass,field
from typing import List
from transformers import WhisperFeatureExtractor


@dataclass
class DataLoaderConfig:
    folder_path:str='./downloads/metadata.csv'
    num_proc:int = 16
    split:str='train'
    batch_size:int=16
    is_process:bool=True

    sampling_rate:int = 16000
    use_processors:List[str]= field(default_factory=lambda: ['filter_duration','map_prepare_dataset','map_log_mel_stectrogram'])
    max_duration:int = 300
    min_duration:int = 200
    feature_extractor=WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    out_vis:str='./'
    
class DataLoader(DataLoaderConfig):
    def __init__(self, args: DataLoaderConfig):
        for field_name in args.__dataclass_fields__:
            setattr(self, field_name, getattr(args, field_name))
        
        filter_processor_fn = {
            'filter_duration':partial(self.duration_filter,min_duration=self.min_duration,max_duration=self.max_duration),
            'filter_sampling_rate': partial(self.sampling_rate_filter,sampling_rate=self.sampling_rate)
        }

        map_processor_fn = {
            'map_prepare_dataset':partial(self.prepare_dataset,feature_extractor=self.feature_extractor),
            'map_log_mel_spectrogram':partial(self.vis_log_mel_spectrogram,feature_extractor=self.feature_extractor)
        }
        
        self.preprocessors = [filter_processor_fn.get(fn) for fn in self.use_processors if fn.split('_')[0] == 'filter']
        self.postprocessors = [map_processor_fn.get(fn) for fn in self.use_processors if fn.split('_')[0] == 'map']

    def load(self):
        dataset = load_dataset('csv', data_files=self.folder_path, split=self.split,num_proc=self.num_proc)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        return dataset
    
    @staticmethod
    def duration_filter(example, min_duration=1.0, max_duration=None):
        durations = example["duration"]
        return [d >= min_duration and (max_duration is None or d <= max_duration)for d in durations]
    
    @staticmethod
    def sampling_rate_filter(example,sampling_rate):
        return [sr for sr in example["sampling_rate"] if sr == sampling_rate]
    
    @staticmethod
    def prepare_dataset(example,feature_extractor):

        audio_arrays = [audio["array"] for audio in example["audio"]]
        sampling_rate = example["audio"][0]["sampling_rate"]  # 동일한 샘플링 레이트 가정

        features = feature_extractor(audio_arrays,
                                    sampling_rate=sampling_rate,
                                    padding=True,
                                    return_attention_mask=True,
                                    return_tensors="np")  # or "pt"

        return features
    
    @staticmethod
    def vis_log_mel_spectrogram(example,feature_extractor):
        input_features = [input_feature for input_feature in example["input_features"]]
        
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(
            np.asarray(input_features[0]),
            x_axis="time",
            y_axis="mel",
            sr=feature_extractor.sampling_rate,
            hop_length=feature_extractor.hop_length,
        )
        plt.savefig('./log_mel_spectrogram.png')
        
def load_dataset_hf(args:DataLoaderConfig):
    """
    Load the dataset from a CSV file and return it as a Hugging Face dataset.
    """
    dataloader=DataLoader(args)
    batch_size=dataloader.batch_size
    num_proc=dataloader.num_proc
    is_process=dataloader.is_process

    dataset = dataloader.load()
    if is_process:
        for preprocessor in dataloader.preprocessors:
            dataset = dataset.filter(preprocessor,batched=True,batch_size=batch_size, num_proc=num_proc)
        
        for postprocessor in dataloader.postprocessors:
            dataset = dataset.map(postprocessor,batched=True,batch_size=batch_size, num_proc=num_proc)
    
    return dataset

def read_csv_with_pandas(csv_path):
    """
    Reads a CSV file using pandas and returns a DataFrame.
    """
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    from argparse_dataclass import ArgumentParser
    parser = ArgumentParser(DataLoaderConfig)
    args = parser.parse_args()
    dataset = load_dataset_hf(args)