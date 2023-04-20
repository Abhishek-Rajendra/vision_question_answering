import os
from typing import Dict
import torch

from torch.utils.data import Dataset
from datasets import load_dataset


class DaquarDataset(Dataset):
    def __init__(self, config: Dict, split: str):
        self._config = config
        self._split = split
        self.dataset = load_dataset(
            "csv",
            data_files={
                split: os.path.join(self._config["data"]["dataset_folder"], self._config["data"][f"{split}_dataset"])
            }
        )    
        with open(os.path.join(self._config["data"]["dataset_folder"], self._config["data"]["answer_space"])) as f:
            self.answer_space = f.read().splitlines()
        self.dataset = self.dataset.map(
            lambda examples: {
                'label': [
                    self.answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
                    for ans in examples[self._config["data"]["answer_col"]]
                ]
            },
            batched=True
        )
    
    def __len__(self):
        return self.dataset[self._split].num_rows
    
    def __getitem__(self, index):
        example = self.dataset[self._split][index]
        question = example[self._config["data"]["question_col"]]
        image_id = example[self._config["data"]["image_col"]]
        label = example["label"]
        
        return question, image_id, label
