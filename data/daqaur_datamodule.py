import os
from typing import Dict

from datasets import load_dataset
from torch.utils.data import DataLoader


import pytorch_lightning as pl

from data.daquar_dataset import DaquarDataset


class DaquarDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        # download data if needed
        pass
    
    def setup(self, stage=None) -> None:
        self.train_dataset = DaquarDataset(self.config, "train")
        self.test_dataset = DaquarDataset(self.config, "eval")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["num_workers"],
            shuffle=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"]["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn
        )