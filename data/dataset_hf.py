import pandas as pd
import librosa
from datasets import load_dataset, Audio
from functools import partial
from dataclasses import dataclass, field
from typing import List
from transformers import WhisperFeatureExtractor


@dataclass
class AudioDataLoaderConfig:
    metadata_csv_path: str = 'data/downloads/metadata.csv'
    split: str = 'train[:4]'
    batch_size: int = 16
    num_proc: int = 16
    enable_preprocessing: bool = True

    sampling_rate: int = 16000
    min_duration: int = 200
    max_duration: int = 300

    processors: List[str] = field(default_factory=lambda: ['filter_duration', 'map_prepare_dataset'])
    output_dir: str = './'
    feature_extractor: WhisperFeatureExtractor = field(
        default_factory=lambda: WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    )


class AudioDatasetLoader:
    def __init__(self, config: AudioDataLoaderConfig):
        self.config = config
        self._filter_registry = {
            'filter_duration': partial(self._filter_by_duration,
                                       min_duration=config.min_duration,
                                       max_duration=config.max_duration),
            'filter_sampling_rate': partial(self._filter_by_sampling_rate,
                                            sampling_rate=config.sampling_rate)
        }

        self._map_registry = {
            'map_prepare_dataset': partial(self._apply_padding,
                                           feature_extractor=config.feature_extractor)
        }

    def load(self):
        dataset = load_dataset(
            'csv',
            data_files=self.config.metadata_csv_path,
            split=self.config.split,
            num_proc=self.config.num_proc
        )

        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))
        return self._apply_processors(dataset) if self.config.enable_preprocessing else dataset

    def _apply_processors(self, dataset):
        for processor_name in self.config.processors:
            if processor_name.startswith('filter'):
                filter_fn = self._filter_registry.get(processor_name)
                if filter_fn:
                    dataset = dataset.filter(
                        filter_fn,
                        batched=True,
                        batch_size=self.config.batch_size,
                        num_proc=self.config.num_proc
                    )
            elif processor_name.startswith('map'):
                map_fn = self._map_registry.get(processor_name)
                if map_fn:
                    dataset = dataset.map(
                        map_fn,
                        batched=True,
                        batch_size=self.config.batch_size,
                        num_proc=self.config.num_proc
                    )
        return dataset

    @staticmethod
    def _filter_by_duration(batch, min_duration=1.0, max_duration=None):
        return [
            min_duration <= d <= max_duration if max_duration else d >= min_duration
            for d in batch["duration"]
        ]

    @staticmethod
    def _filter_by_sampling_rate(batch, sampling_rate):
        return [sr == sampling_rate for sr in batch["sampling_rate"]]

    @staticmethod
    def _apply_padding(batch, feature_extractor):
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"]
        return feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            padding=True,
        )


def load_dataset_hf(config: AudioDataLoaderConfig):
    """
    Load and optionally preprocess an audio dataset using Hugging Face's datasets library.
    """
    loader = AudioDatasetLoader(config)
    return loader.load()

if __name__ == "__main__":
    from argparse_dataclass import ArgumentParser
    parser = ArgumentParser(AudioDataLoaderConfig)
    config = parser.parse_args()
    dataset = load_dataset_hf(config)
    print(dataset)
    