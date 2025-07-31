import pytest
from data.dataset_hf import AudioDataLoaderConfig,AudioDatasetLoader,load_dataset_hf
from datasets import Dataset
import numpy as np


@pytest.fixture
def dummy_config(tmp_path):
    return AudioDataLoaderConfig(
        metadata_csv_path="tests/test_metadata.csv",
        sampling_rate=16000,
        min_duration=1.0,
        max_duration=10.0,
        enable_preprocessing=True,
        output_dir=tmp_path
    )


@pytest.fixture
def dummy_dataset():
    data = {
        "audio": [
            {"array": np.random.rand(16000 * 3), "sampling_rate": 16000},
            {"array": np.random.rand(16000 * 5), "sampling_rate": 16000},
            {"array": np.random.rand(16000 * 20), "sampling_rate": 16000}
        ],
        "duration": [3.0, 5.0, 20.0],
        "sampling_rate": [16000, 16000, 16000]
    }
    return Dataset.from_dict(data)


def test_filter_by_duration(dummy_dataset):
    loader = AudioDatasetLoader(AudioDataLoaderConfig())
    filtered = loader._filter_by_duration(dummy_dataset, min_duration=4.0, max_duration=10.0)
    assert filtered == [False, True, False]


def test_filter_by_sampling_rate(dummy_dataset):
    loader = AudioDatasetLoader(AudioDataLoaderConfig(sampling_rate=16000))
    filtered = loader._filter_by_sampling_rate(dummy_dataset, sampling_rate=16000)
    assert all(filtered)


def test_apply_padding(dummy_dataset):
    config = AudioDataLoaderConfig()
    loader = AudioDatasetLoader(config)
    batch = {"audio": dummy_dataset["audio"]}
    features = loader._apply_padding(batch, feature_extractor=config.feature_extractor)

    assert "input_features" in features
    assert features["input_features"].shape[0] == len(batch["audio"])
    assert isinstance(features["input_features"], np.ndarray)

def test_dataloader():
    split="train[:5]"
    metadata_csv_path="data/downloads/metadata.csv"
    config = AudioDataLoaderConfig(metadata_csv_path=metadata_csv_path,split=split)
    dataset = load_dataset_hf(config)

    assert len(dataset) == 5