from datasets import load_dataset, Audio
from functools import partial
from dataclasses import dataclass,field
from typing import List
import pandas as pd

@dataclass
class DataLoaderConfig:
    folder_path:str='./downloads/metadata.csv'
    num_proc:int = 16
    split:str='train'
    batch_size:int=16
    is_process:bool=False
    statistics=False

    sampling_rate : int = 16000
    use_processors : List[str]= field(default_factory=lambda: ['filter_duration'])
    max_duration : int = 300
    min_duration : int = 200

class DataLoader(DataLoaderConfig):
    def __init__(self, args: DataLoaderConfig):
        for field_name in args.__dataclass_fields__:
            setattr(self, field_name, getattr(args, field_name))
        
        filter_processor_fn = {
            'filter_duration':partial(self.duration_filter,min_duration=self.min_duration,max_duration=self.max_duration),
            'filter_sampling_rate': partial(self.sampling_rate_filter,sampling_rate=self.sampling_rate)
        }
        map_processor_fn = {

        }
        
        self.processors = [filter_processor_fn.get(fn) for fn in self.use_processors if fn.split('_')[0] == 'filter']
        self.postprocessors = [map_processor_fn.get(fn) for fn in self.use_processors if fn.split('_')[0] == 'map']
    
    def load(self,statistics=True):
        dataset = load_dataset('csv', data_files=self.folder_path, split=self.split,num_proc=self.num_proc)
        if statistics:
            dataset = dataset.remove_columns(['audio'])
            dataset = pd.DataFrame(dataset)
        else:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        return dataset
    
    @staticmethod
    def duration_filter(example, min_duration=1.0, max_duration=None):
        durations = example["duration"]
        return [d >= min_duration and (max_duration is None or d <= max_duration)for d in durations]
    
    @staticmethod
    def sampling_rate_filter(example,sampling_rate):
        return [sr for sr in example["sampling_rate"] if sr == sampling_rate]


def load_dataset_hf(args:DataLoaderConfig):
    """
    Load the dataset from a CSV file and return it as a Hugging Face dataset.
    """
    dataloader=DataLoader(args)
    batch_size=dataloader.batch_size
    num_proc=dataloader.num_proc
    is_process=dataloader.is_process
    statistics=dataloader.statistics

    if statistics:
        dataset = dataloader.load(statistics=True)
    else:
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
    print(dataset)
    # pd_dataset = pd.DataFrame(dataset)  # Print the first entry of the dataset
    # print(pd_dataset['tag'].describe())
    # csv_file = read_csv_with_pandas(folder_path)
    # print(csv_file.describe())