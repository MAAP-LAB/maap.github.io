from datasets import load_dataset, Audio
from functools import partial
from dataclasses import dataclass,field
from typing import List
import pandas as pd

@dataclass
class ProcessorConfig:
    use_processors : List[str] = field(default_factory=list)
    min_duration : int = 200
    max_duration : int = 300
    sampling_rate : int = 16000

class Processor(ProcessorConfig):
    def __init__(self):
        super().__init__()
        processor_map = {
            'filter_duration':partial(self.duration_filter,min_duration=self.min_duration,max_duration=self.max_duration),
            'filter_sampling_rate': partial(self.sampling_rate_filter,sampling_rate=self.sampling_rate)
        }
        self.use_processors=['filter_duration']
        self.preprocessors = [processor_map.get(fn) for fn in self.use_processors]
        self.postprocessors = []
    
    @staticmethod
    def duration_filter(example, min_duration=1.0, max_duration=None):
        durations = example["duration"]
        return [d >= min_duration and (max_duration is None or d <= max_duration)for d in durations]
    
    @staticmethod
    def sampling_rate_filter(example,sampling_rate):
        return [sr for sr in example["sampling_rate"] if sr == sampling_rate]


class DataLoader():
    def __init__(self, folder_path='./downloads/metadata.csv', sampling_rate=16000 ,split='train', num_proc=8):
        self.folder_path = folder_path
        self.split = split  # Default split
        self.num_proc = num_proc  # Default number of processes
        self.sampling_rate=sampling_rate
    
    def load(self,statistics=True):
        dataset = load_dataset('csv', data_files=self.folder_path, split=self.split,num_proc=self.num_proc)
        if statistics:
            dataset = dataset.remove_columns(['audio'])
            dataset = pd.DataFrame(dataset)
        else:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        return dataset

def load_dataset_hf(folder_path='./downloads/metadata.csv', batch_size=16, split='train', num_proc=8, process=True ,statistics=False):
    """
    Load the dataset from a CSV file and return it as a Hugging Face dataset.
    """
    dataloader = DataLoader(folder_path=folder_path, split=split, num_proc=num_proc)
    processor = Processor()
    
    if statistics:
        dataset = dataloader.load(statistics=True)
    else:
        dataset = dataloader.load()
        if process:
            for preprocessor in processor.preprocessors:
                dataset = dataset.filter(preprocessor,batched=True,batch_size=batch_size, num_proc=num_proc)
            
            for postprocessor in processor.postprocessors:
                dataset = dataset.map(postprocessor,batched=True,batch_size=batch_size, num_proc=num_proc)
    
    return dataset

def read_csv_with_pandas(csv_path):
    """
    Reads a CSV file using pandas and returns a DataFrame.
    """
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    folder_path = './downloads/metadata.csv'
    dataset = load_dataset_hf(folder_path=folder_path,split="train[:10]",num_proc=16,statistics=False)
    print(dataset)
    # pd_dataset = pd.DataFrame(dataset)  # Print the first entry of the dataset
    # print(pd_dataset['tag'].describe())
    # csv_file = read_csv_with_pandas(folder_path)
    # print(csv_file.describe())