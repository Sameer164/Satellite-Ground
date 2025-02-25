import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import MaxPool1d
from torch import einsum
import math
import random
import numpy as np

class StreetFeatureExtractor(nn.Module):
    def __init__(self, backbone="res18"):
        '''
        CNN for extracting street image feature
        outputSize: output feature size
        Return => extracted features of size #outputSize
        '''
        super(StreetFeatureExtractor, self).__init__()
        if backbone == "res18":
            bb = models.resnet18(pretrained=True)
            modules=list(bb.children())[:-1]
            self.dimAfterBB = 512 #feature dims after backbone
            self.featureExtractor=nn.Sequential(*modules)
        elif backbone == "vgg16":
            bb = models.vgg16(pretrained=True)
            self.dimAfterBB = 4096 #feature dims after backbone
            classifier = bb.classifier
            classifier = list(classifier)[:4]
            bb.classifier = nn.Sequential(*classifier)
            self.featureExtractor = bb
        elif backbone == "res34":
            bb = models.resnet34(pretrained=True)
            modules=list(bb.children())[:-1]
            self.dimAfterBB = 512 #feature dims after backbone
            self.featureExtractor=nn.Sequential(*modules)
        elif backbone == "res50":
            bb = models.resnet50(pretrained=True)
            modules=list(bb.children())[:-1]
            self.dimAfterBB = 2048 #feature dims after backbone
            self.featureExtractor=nn.Sequential(*modules)
        else:
            RuntimeError(f"not implemented this backbone {backbone}")

    def forward(self,x):
        x = self.featureExtractor(x)

        x = x.reshape(-1, self.dimAfterBB)

        return x

class SatelliteFeatureExtractor(nn.Module):
    def __init__(self, inputChannel = 6, backbone="res18"):
        '''
        CNN for extracting satellite image feature
        inputChannel: number of channels input image
        outputSize: output feature size
        Return => extracted features of size #outputSize
        '''
        super(SatelliteFeatureExtractor, self).__init__()
        if backbone == "res18":
            bb = models.resnet18(pretrained=True)
            self.dimAfterBB = 512 #feature dims after backbone
        elif backbone == "vgg16":
            bb = models.vgg16(pretrained=True)
            self.dimAfterBB = 4096 #feature dims after backbone
            classifier = bb.classifier
            classifier = list(classifier)[:4]
            bb.classifier = nn.Sequential(*classifier)
            self.featureExtractor = bb
        elif backbone == "res34":
            bb = models.resnet34(pretrained=True)
            self.dimAfterBB = 512 #feature dims after backbone
        elif backbone == "res50":
            bb = models.resnet50(pretrained=True)
            self.dimAfterBB = 2048 #feature dims after backbone
        else:
            RuntimeError(f"not implemented this backbone {backbone}")

        if backbone != 'vgg16':
            if inputChannel != 3:
                modules=list(bb.children())[1:-1]
                modules.insert(0, nn.Conv2d(6,64,7,stride=2,padding=3,bias=False))
            else:
                modules=list(bb.children())[:-1]
            self.featureExtractor=nn.Sequential(*modules)


    def forward(self,x):
        x = self.featureExtractor(x)
        x = x.reshape(-1, self.dimAfterBB)

        return x

class CrossViewMatcher(nn.Module):
    def __init__(self, feature_dims=4096, backbone="vgg16"):
        super(CrossViewMatcher, self).__init__()
        
        # Feature extractors for both views
        self.street_encoder = StreetFeatureExtractor(backbone=backbone)
        self.satellite_encoder = SatelliteFeatureExtractor(backbone=backbone, inputChannel=3)
        
        # Projection heads to map features to common space
        self.street_projector = nn.Sequential(
            nn.Linear(feature_dims, feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims, feature_dims // 2),
            nn.LayerNorm(feature_dims // 2)
        )
        
        self.satellite_projector = nn.Sequential(
            nn.Linear(feature_dims, feature_dims),
            nn.ReLU(),
            nn.Linear(feature_dims, feature_dims // 2),
            nn.LayerNorm(feature_dims // 2)
        )

    def forward(self, street_img, satellite_img):
        # Extract features
        street_features = self.street_encoder(street_img)
        satellite_features = self.satellite_encoder(satellite_img)
        
        # Project to common space
        street_embedding = self.street_projector(street_features)
        satellite_embedding = self.satellite_projector(satellite_features)
        
        # Normalize embeddings
        street_embedding = F.normalize(street_embedding, p=2, dim=1)
        satellite_embedding = F.normalize(satellite_embedding, p=2, dim=1)
        
        return street_embedding, satellite_embedding

if __name__ == "__main__":
    model_street = StreetFeatureExtractor(backbone="vgg16")

    print(model_street)

    feat = torch.rand((8, 3, 320, 180))

    print(model_street(feat).shape)


