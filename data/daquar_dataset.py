import os
from typing import Dict
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoFeatureExtractor

class DaquarDataset(Dataset):
    def __init__(self, config: Dict, split: str):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["model"]["image_encoder"])
        self.dataset = load_dataset(
            "csv",
            data_files={
                split: os.path.join(config["data"]["dataset_folder"], config["data"][f"{split}_dataset"])
            }
        )    
        with open(os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"])) as f:
            self.answer_space = f.read().splitlines()
        self.dataset = self.dataset.map(
            lambda examples: {
                'label': [
                    self.answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
                    for ans in examples[config["data"]["answer_col"]]
                ]
            },
            batched=True
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        question = example[self.config["data"]["question_col"]]
        image_id = example[self.config["data"]["image_col"]]
        image = Image.open(
                    os.path.join(
                        self.config["data"]["dataset_folder"],
                        self.config["data"]["images_folder"], 
                        image_id + ".png"
                    )
                ).convert('RGB')
        label = example['label']
        
        encoded_text = self.tokenizer(
            text=question,
            padding=self.config["tokenizer"]["padding"],
            max_length=self.config["tokenizer"]["max_length"],
            truncation=self.config["tokenizer"]["truncation"],
            return_tensors='pt',
            return_token_type_ids=self.config["tokenizer"]["return_token_type_ids"],
            return_attention_mask=self.config["tokenizer"]["return_attention_mask"],
        )
        
        processed_image = self.preprocessor(
            images=[image],
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
            "pixel_values": processed_image['pixel_values'].squeeze(),
            "labels": torch.tensor(label, dtype=torch.int64)
        }
