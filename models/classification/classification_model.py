import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# from vit_pytorch import ViT
from .vit import ViT
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from typing import Tuple


class ClassificationTargetModel(nn.Module):
    # def __init__(self, model_name='resnet18', num_classes=10, *args, **kwargs):
    def __init__(self, cfg, *args, **kwargs):
        super(ClassificationTargetModel, self).__init__()
        self.cfg = cfg
        model_name = cfg.target_model.name
        num_classes = cfg.dataset.num_classes
        if model_name == 'resnet18':
            self.model = ResNet18(num_classes)
        elif model_name == 'resnet34':
            self.model = ResNet34(num_classes)
        elif model_name == 'resnet50':
            self.model = ResNet50(num_classes)
        elif model_name == 'resnet101':
            self.model = ResNet101(num_classes)
        elif model_name == 'resnet152':
            self.model = ResNet152(num_classes)
        elif model_name == 'vit_b':
            self.model = ViT(
                image_size=224,
                patch_size=16,
                num_classes=num_classes,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072,
                dropout=0.1,
                emb_dropout=0.1
            )
            self.torchvision_resize = torchvision.transforms.Resize((224, 224))
        elif model_name == "vit":
            image_size = kwargs.get('image_size', 224)
            patch_size = kwargs.get('patch_size', 16)
            num_classes = num_classes
            dim = kwargs.get('dim', 768)
            depth = kwargs.get('depth', 12)
            heads = kwargs.get('heads', 12)
            mlp_dim = kwargs.get('mlp_dim', 3072)
            dropout = kwargs.get('dropout', 0.1)
            emb_dropout = kwargs.get('emb_dropout', 0.1)
            self.model = ViT(
                image_size=image_size,
                patch_size=patch_size,
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                emb_dropout=emb_dropout
            )
            self.torchvision_resize = torchvision.transforms.Resize((image_size, image_size))
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        self.model_name = model_name
        self.loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=cfg.train.get('label_smoothing', 0.0))
        if cfg.train.get('label_smoothing', 0.0) > 0:
            print("label smoothing is set: ", cfg.train.get('label_smoothing', 0.0))

    def forward(self, x):
        self.model = self.model.to(x.device)
        if "resnet" in self.model_name:
            out, _ = self.model(x)
        elif "vit" in self.model_name:
            x = self.torchvision_resize(x) if hasattr(self, 'torchvision_resize') else x
            out = self.model(x)
        else:
            raise ValueError(f"Model {self.model_name} is not supported for forward pass.")
        return out
    
    def forward_with_features(self, x):
        out, features = self.model(x)
        return out, features

    def loss_function(self, preds, targets):
        """
        Define the loss function for the classification task.
        This can be cross-entropy loss or any other suitable loss function.
        """
        none_loss = self.loss(preds, targets)
        mean_loss = none_loss.mean()

        output_dict = {
            "loss": mean_loss,
            "logging":{
                "cross_entropy_loss": mean_loss
            },
            "aux": {
                "loss_none": none_loss,
                "loss_mean": mean_loss
            }
        }

        return output_dict

        # return self.model.target_model_loss(preds, targets)


    