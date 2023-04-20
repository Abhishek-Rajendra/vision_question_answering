import os
from typing import Dict

from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor
import lightning as L

from data.daquar_dataset import DaquarDataset


class DaquarDataModule(L.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self._config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._config["model"]["text_encoder"])
        self.preprocessor = AutoFeatureExtractor.from_pretrained(
            self._config["model"]["image_encoder"])

    def prepare_data(self):
        # download data if needed
        pass

    def setup(self, stage=None) -> None:
        self.train_dataset = DaquarDataset(self._config, "train")
        self.test_dataset = DaquarDataset(self._config, "eval")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._config["training"]["batch_size"],
            num_workers=self._config["training"]["num_workers"],
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self._config["training"]["batch_size"],
            num_workers=self._config["training"]["num_workers"],
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        """ This function overrides defaul batch collate function and aggregates 
        data of the batch into a single graph tensor """

        texts = [i[0] for i in batch]
        image_ids = [i[1] for i in batch]
        labels = [i[2] for i in batch]

        images = [
            Image.open(
                os.path.join(
                    self._config["data"]["dataset_folder"],
                    self._config["data"]["images_folder"],
                    image_id + ".png"
                )
            ).convert('RGB') for image_id in image_ids
        ]

        encoded_text = self.tokenizer(
            text=texts,
            padding=self._config["tokenizer"]["padding"],
            max_length=self._config["tokenizer"]["max_length"],
            truncation=self._config["tokenizer"]["truncation"],
            return_tensors='pt',
            return_token_type_ids=self._config["tokenizer"]["return_token_type_ids"],
            return_attention_mask=self._config["tokenizer"]["return_attention_mask"],
        )

        processed_image = self.preprocessor(
            images=images,
            return_tensors="pt",
        )

        return encoded_text['input_ids'].squeeze().long(), encoded_text['token_type_ids'].squeeze(), \
            encoded_text['attention_mask'].squeeze(), processed_image['pixel_values'].squeeze(), \
            torch.tensor(labels, dtype=torch.int64)
