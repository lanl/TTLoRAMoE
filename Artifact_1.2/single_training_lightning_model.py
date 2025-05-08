import time

import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import sys

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.model = model
        self.data = config["dataset_name"]
        if self.data == "socialiqa" or self.data =="sick" or self.data == "cb"in self.data or self.data == "mnli":
            self.num_classes = 3
        elif self.data == "cosmosqa" or self.data == "hellaswag":
            self.num_classes = 4
        elif self.data == "csqa":
            self.num_classes = 5
        else:
            self.num_classes = 2
        # print(self.num_classes)
        
        self.val_f1= torchmetrics.F1Score(task="multiclass",num_classes=self.num_classes, average = 'micro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass",num_classes=self.num_classes, average = 'micro')
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"])
        self.log("train_loss", outputs["loss"],  prog_bar=True)
        return outputs["loss"]  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"])
        self.log("val_loss", outputs["loss"], prog_bar=False)
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.val_acc(predicted_labels, batch["label"])
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"])
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.test_acc(predicted_labels, batch["label"])
        self.log("accuracy", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
