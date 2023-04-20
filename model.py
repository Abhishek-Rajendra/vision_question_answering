from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchmetrics import F1Score
from transformers import AutoModel
import lightning as L

from evaluate import WuPalmerScoreCalculator


class MultimodalVQAModel(L.LightningModule):
    def __init__(
            self,
            answer_space: List[str],
            num_labels: int,
            intermediate_dims: int,
            dropout: float,
            pretrained_text_name: str,
            pretrained_image_name: str):

        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        self.validation_step_outputs = []

        # Wu-Palmer score calculator
        self.wups_calculator = WuPalmerScoreCalculator(answer_space)

        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )

        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size +
                      self.image_encoder.config.hidden_size, intermediate_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(intermediate_dims, self.num_labels)

        self.criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text['pooler_output'],
                    encoded_image['pooler_output'],
                ],
                dim=1
            )
        )
        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }

        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss

        return out

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch
        outputs = self(input_ids, pixel_values,
                       attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch
        outputs = self(input_ids, pixel_values,
                       attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        outputs["labels"] = labels
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append(outputs)

        return loss

    def on_validation_epoch_end(self) -> None:
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        metrics = self.wups_calculator.compute_metrics((logits, labels))
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()  # clear the list for next epoch

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
