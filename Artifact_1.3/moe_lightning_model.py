import time
import sys
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
from moe_wrapper import MoEsparseRouting, SharedState

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.model = model
        self.data = config["dataset_name"]
        self.experts_trainable = config["experts_trainable"]
        
        '''Choose the max number of classes that are being used in classification task'''
        self.num_classes = 5 

        self.val_f1= torchmetrics.F1Score(task="multiclass",num_classes=self.num_classes, average = 'micro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass",num_classes=self.num_classes, average = 'micro')
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask, labels, expert_label=None):
        if isinstance(self.model, MoEsparseRouting):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, expert_label=expert_label)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["label"], expert_label=batch["expert_label"])
        total_loss = outputs["loss"] + SharedState.router_loss_weight * SharedState.routerloss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("NaN/Inf detected in loss:", total_loss)
        self.log("train_loss", outputs["loss"], prog_bar=True)
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.log("router_loss", SharedState.routerloss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        self.log("router_acc", SharedState.router_accuracy, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["label"], expert_label=batch["expert_label"])
        total_loss = outputs["loss"] + SharedState.router_loss_weight * SharedState.routerloss
        self.log("val_loss", outputs["loss"], prog_bar=False)
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.val_acc(predicted_labels, batch["label"])
        self.log("router_loss", SharedState.routerloss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("router_acc", SharedState.router_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["label"], expert_label=batch["expert_label"])
        total_loss = outputs["loss"] + SharedState.router_loss_weight * SharedState.routerloss
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.test_acc(predicted_labels, batch["label"])
        self.log("router_loss", SharedState.routerloss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        self.log("accuracy", self.test_acc, prog_bar=True)
        self.log("router_acc", SharedState.router_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        # Collect all trainable parameters from experts dictionary
        if self.experts_trainable:
            expert_params = []
            for expert_name, layers in self.model.experts.items():
                for layer, attention_types in layers.items():
                    for attention_type, tt_cores in attention_types.items():
                        if isinstance(tt_cores, dict):
                            for core_name, param in tt_cores.items():
                                if param.requires_grad:
                                    expert_params.append(param)
                        else:
                            if param.requires_grad:  # Ensure we only collect trainable params
                                expert_params.append(param)
            # Define optimizer with both model and expert parameters
            optimizer = torch.optim.Adam(
                [{"params": self.parameters(), "lr": self.learning_rate},  # Model params
                {"params": expert_params, "lr": self.learning_rate}]  # Expert params
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
