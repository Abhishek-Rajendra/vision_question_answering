
import argparse
import json
import logging
import os
import numpy as np

from typing import Text

import yaml
import torch
from data.daqaur_datamodule import DaquarDataModule
from model import MultimodalVQAModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def main(config_path: Text) -> None:

    print(os.getcwd())
    # Load configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        print('Cuda is available, make sure you are training on GPU')

    # device = config["trainer"]["devices"]
    # if device != -1:
    #     torch.cuda.set_device(device)
    # else:
    #     config["base"]["device"] = torch.device('cpu')

    # Initialize data module
    data_module = DaquarDataModule(config)

    data_module.setup()

    answer_space = data_module.train_dataset.answer_space

    # Initialize model
    model = MultimodalVQAModel(
        answer_space=answer_space,
        num_labels=len(answer_space),
        intermediate_dims=config["model"]["intermediate_dims"],
        dropout=config["model"]["dropout"],
        pretrained_text_name=config["model"]["text_encoder"],
        pretrained_image_name=config["model"]["image_encoder"],
    )

    # Set the directory name for checkpoints based on the model name
    checkpoint_dir = os.path.join(config["training"]["checkpoint"], config["model"]["name"])

    class SaveMetricsCallback(L.Callback):
        def on_train_end(self, trainer, pl_module):
            logging.info("Training complete")

            os.makedirs(config["metrics"]["metrics_folder"], exist_ok=True)

            # Save trainer metrics to file
            with open(config["metrics"]["metrics_folder"] + "/" + "trainer-" + config["model"]["name"] + ".json", "w") as f:
                callback_metrics = trainer.callback_metrics.copy()
                for key, value in callback_metrics.items():
                    if isinstance(value, torch.Tensor):
                        callback_metrics[key] = str(value.item())
                json.dump(callback_metrics, f)

            # Test the model on the test dataset
            test_results = trainer.validate(
                pl_module, datamodule=data_module, ckpt_path='best')

            # Save test metrics to file
            with open(config["metrics"]["metrics_folder"] + "/" + "test-" + config["model"]["name"] + ".json", "w") as f:
                test_metrics = test_results[0].copy()
                test_metrics["best_checkpoint_path"] = trainer.checkpoint_callback.best_model_path
                for key, value in test_metrics.items():
                    if isinstance(value, torch.Tensor):
                        test_metrics[key] = str(value.item())
                json.dump(test_metrics, f)

    # early_stopping_callback = EarlyStopping(
    #     monitor='val_loss',
    #     patience=config["training"]["early_stop_patience"]
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='checkpoint-{epoch}',
        save_top_k=config["training"]["save_top_k"],
        save_last=True
    )

    # Initialize TensorBoardLogger
    logger = TensorBoardLogger("tb_logs", name=config["model"]["name"])

    # Initialize trainer
    trainer = L.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[checkpoint_callback, SaveMetricsCallback()]
    )

    # Train the model
    trainer.fit(model, data_module)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    main(args.config)
