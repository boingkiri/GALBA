import torch
import torch.nn as nn
import torchmetrics
import sys


from .classification.classification_model import ClassificationTargetModel

class TargetModelContainer(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.task = cfg.target_model.task 
        self.model_name = cfg.target_model.name

        super().__init__()
        if self.task == "classification":
            num_classes = cfg.dataset.num_classes
            # self.model = ClassificationTargetModel(self.cfg, self.model_name, num_classes=num_classes)
            self.model = ClassificationTargetModel(self.cfg)
            self.metrics = torch.nn.ModuleDict(
                {
                    "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
                }
            )
        else:
            raise ValueError(f"Task {self.task} is not supported.")


    def forward(self, x):
        return self.model.forward(x)
    
    def forward_with_features(self, x):
        return self.model.forward_with_features(x)
    
    def loss_function(self, preds, targets):
        """
        Define the loss function for the classification task.
        This can be cross-entropy loss or any other suitable loss function.
        """
        # return F.cross_entropy(preds, targets)
        return self.model.loss_function(preds, targets)

    def measure_metrics(self, preds, targets, inputs=None):
        result_dict = {}

        if self.task == "classification":
             for metric_name, metric in self.metrics.items():
                result_dict[metric_name] = metric(preds, targets)
             return result_dict

        else:
            raise ValueError(f"Task {self.task} is not supported.")
