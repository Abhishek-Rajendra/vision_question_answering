
import argparse
from typing import Text

import yaml

from data.daqaur_datamodule import DaquarDataModule
from model import MultimodalVQAModel
import pytorch_lightning as pl
import os


def main(config_path: Text) -> None:

    print(os.getcwd())
    # Load configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    # Initialize data module
    data_module = DaquarDataModule(config)

    data_module.setup()

    num_labels = len(data_module.train_dataset.answer_space)

    # Initialize model
    model = MultimodalVQAModel(
        num_labels=num_labels,
        intermediate_dims=config["model"]["intermediate_dims"],
        dropout=config["model"]["dropout"],
        pretrained_text_name=config["model"]["text_encoder"],
        pretrained_image_name=config["model"]["image_encoder"],
    )

    # Initialize trainer
    trainer = pl.Trainer(
        gpus=config["training"]["gpus"],
        max_epochs=config["training"]["max_epochs"],
        progress_bar_refresh_rate=config["training"]["progress_bar_refresh_rate"],
        **config["trainer"]
    )

    # Train the model
    trainer.fit(model, data_module)

    # Evaluate the model on the validation dataset
    trainer.validate(model, datamodule=data_module.val_dataloader())


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)