import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchmetrics import MeanMetric

class EmbeddingModel(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_layers_to_unfreeze=25):
        super(EmbeddingModel, self).__init__()
        
        # Load pre-trained EfficientNet-B7
        base_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

        # Freeze layers except for the last `num_layers_to_unfreeze`
        total_layers = len(list(base_model.features.children()))
        for i, layer in enumerate(base_model.features.children()):
            if i < total_layers - num_layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.base_model = nn.Sequential(*base_model.features)
        self.pooling = nn.AdaptiveAvgPool2d(1)  # Equivalent to 'avg' pooling in Keras
        self.flatten = nn.Flatten()
        
        # Fully connected layers (same as Keras model)
        self.fc = nn.Sequential(
            nn.Linear(2560, 512),  # EfficientNet-B7 has 2560 output features
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128)  # Final embedding layer
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class DistanceLayer(nn.Module):
    """Computes squared Euclidean distances between anchor-positive and anchor-negative pairs."""
    def __init__(self):
        super(DistanceLayer, self).__init__()

    def forward(self, anchor, positive, negative):
        ap_distance = torch.sum((anchor - positive) ** 2, dim=-1)  # Squared L2 distance
        an_distance = torch.sum((anchor - negative) ** 2, dim=-1)
        return ap_distance, an_distance

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model):
        super(SiameseNetwork, self).__init__()
        self.embedding = embedding_model
        self.distance_layer = DistanceLayer()

    def forward(self, anchor, positive, negative):
        anchor_emb = self.embedding(anchor)  
        positive_emb = self.embedding(positive)
        negative_emb = self.embedding(negative)

        return self.distance_layer(anchor_emb, positive_emb, negative_emb)

class SiameseModel(nn.Module):
    def __init__(self, siamese_net, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_net = siamese_net
        self.margin = margin
        self.loss_tracker = MeanMetric()
        self.accuracy_tracker = MeanMetric()

    def forward(self, inputs):
        return self.siamese_net(inputs)

    def compute_loss(self, data):
        anchor, positive, negative = data
        ap_distance, an_distance = self.siamese_net(anchor, positive, negative)

        loss = ap_distance - an_distance
        loss = F.relu(loss + self.margin)  # max(0, ap_distance - an_distance + margin)
        return loss.mean()

    def compute_accuracy(self, data):
        anchor, positive, negative = data
        ap_distance, an_distance = self.siamese_net(anchor, positive, negative)
        accuracy = torch.mean((ap_distance < an_distance).float())
        return accuracy
