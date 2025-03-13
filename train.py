import argparse
import os
import numpy as np
import math
import itertools
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CrossViewMatcher, SatelliteFeatureExtractor, StreetFeatureExtractor
from dataset import ImageDataset, SingleImageDataset
from SMTL import softMarginTripletLoss

SEQUENCE_SIZE = 7

def ValidateOne(distArray, topK):
    acc = 0.0
    dataAmount = 0.0
    for i in range(distArray.shape[0]):
        groundTruths = distArray[i,i]
        pred = torch.sum(distArray[:,i] < groundTruths)
        if pred < topK:
            acc += 1.0
        dataAmount += 1.0
    return acc / dataAmount

def ValidateAll(streetFeatures, satelliteFeatures):
    distArray = 2 - 2 * torch.matmul(satelliteFeatures, torch.transpose(streetFeatures, 0, 1))
    topOnePercent = int(distArray.shape[0] * 0.01) + 1
    valAcc = torch.zeros((1, topOnePercent))
    for i in range(topOnePercent):
        valAcc[0,i] = ValidateOne(distArray, i)
    
    return valAcc


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        '''
        linear decay LR scheduler
        n_epochs: number of total training epochs
        offset: train start epochs
        decay_start_epoch: epoch start decay
        '''
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def save_model(savePath, transMixer, sateFeature, strFeature, epoch):
    modelFolder = os.path.join(savePath, f"epoch_{epoch}")
    os.makedirs(modelFolder)
    torch.save(transMixer.state_dict(), os.path.join(modelFolder, f'trans_{epoch}.pth'))
    torch.save(sateFeature.state_dict(), os.path.join(modelFolder, f'SFE_{epoch}.pth'))
    torch.save(strFeature.state_dict(), os.path.join(modelFolder, f'GFE_{epoch}.pth'))
    # torch.save(HPEstimator.state_dict(), os.path.join(modelFolder, f'HPE_{epoch}.pth'))


def InferOnce(grdFE, satFE, transMixer, batch, device, noMask):
    grdImgs = batch["street"].to(device)
    sateImgs = batch["satellite"].to(device)

    numSeqInBatch = grdImgs.shape[0]

    #street view featuer extraction
    grdImgs = grdImgs.view(grdImgs.shape[0]*grdImgs.shape[1],\
        grdImgs.shape[2],grdImgs.shape[3], grdImgs.shape[4])

    grdFeature = grdFE(grdImgs)
    grdFeature = grdFeature.view(numSeqInBatch, SEQUENCE_SIZE, -1)

    #satellite view feature extraction
    sateImgs = sateImgs.view(sateImgs.shape[0], sateImgs.shape[1]*sateImgs.shape[2],\
        sateImgs.shape[3], sateImgs.shape[4])
    sateFeature = satFE(sateImgs)
    sateFeature = sateFeature.view(numSeqInBatch, -1)
    # print(sateFeature.shape)
   

    if not noMask:
        grdMixedFeature = transMixer(grdFeature, mask=True, masked_range = [0,6], max_masked=opt.max_masked)
    else:
        grdMixedFeature = transMixer(grdFeature, mask=False, masked_range = [0,6])
    grdGlobalFeature = grdMixedFeature.permute(0,2,1)
    grdGlobalLatent = F.avg_pool1d(grdGlobalFeature, grdGlobalFeature.shape[2]).squeeze(2)

    return sateFeature, grdGlobalLatent


def contrastive_loss(street_emb, sat_emb, temperature=0.07):
    # Normalized features
    street_emb = F.normalize(street_emb, dim=1)
    sat_emb = F.normalize(sat_emb, dim=1)
    
    # Similarity matrix
    logits = torch.mm(street_emb, sat_emb.t()) / temperature
    
    # Labels are on diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Calculate loss
    loss = F.cross_entropy(logits, labels)
    return loss

def validate_model(model, val_loader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    # Lists to store all embeddings
    all_street_embeddings = []
    all_satellite_embeddings = []
    
    with torch.no_grad():
        for batch in val_loader:
            street_imgs = batch["street"].to(device)
            satellite_imgs = batch["satellite"].to(device)
            
            street_embeddings, satellite_embeddings = model(street_imgs, satellite_imgs)
            
            all_street_embeddings.append(street_embeddings)
            all_satellite_embeddings.append(satellite_embeddings)
            
    # Concatenate all embeddings
    street_embeddings = torch.cat(all_street_embeddings, dim=0)
    satellite_embeddings = torch.cat(all_satellite_embeddings, dim=0)
    
    # Compute similarity matrix
    similarity = torch.mm(street_embeddings, satellite_embeddings.t())
    
    # Get top-k predictions
    _, pred_top5 = similarity.topk(5, dim=1)
    
    # Ground truth labels (diagonal indices)
    labels = torch.arange(similarity.size(0), device=device)
    
    # Calculate top-1 and top-5 accuracy
    correct_top1 += (pred_top5[:, 0] == labels).sum().item()
    correct_top5 += sum([1 for i, p in enumerate(pred_top5) if labels[i] in p])
    total += similarity.size(0)
    
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    
    print(f"Validation Results:")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    
    return top1_accuracy, top5_accuracy

class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.07, triplet_weight=10.0, contrastive_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.triplet_weight = triplet_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, street_embeddings, satellite_embeddings):
        # InfoNCE/Contrastive Loss
        logits = torch.mm(street_embeddings, satellite_embeddings.t()) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        # Soft Margin Triplet Loss (from SMTL.py in codebase)
        dists = 2 - 2 * torch.matmul(street_embeddings, satellite_embeddings.t())
        pos_dists = torch.diag(dists)
        N = len(pos_dists)
        diag_ids = torch.arange(N, device=street_embeddings.device)

        # Match from satellite to street
        triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
        loss_s2p = torch.log(1 + torch.exp(self.triplet_weight * triplet_dist_s2p))
        loss_s2p[diag_ids, diag_ids] = 0
        loss_s2p = loss_s2p.sum() / (N * (N - 1))

        # Match from street to satellite
        triplet_dist_p2s = pos_dists - dists
        loss_p2s = torch.log(1 + torch.exp(self.triplet_weight * triplet_dist_p2s))
        loss_p2s[diag_ids, diag_ids] = 0
        loss_p2s = loss_p2s.sum() / (N * (N - 1))

        # Combine losses
        triplet_loss = (loss_s2p + loss_p2s) / 2.0
        total_loss = self.contrastive_weight * contrastive_loss + triplet_loss

        return total_loss, {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item(),
            'triplet': triplet_loss.item()
        }

def train_single_image_matcher(model, train_loader, val_loader, optimizer, device, epochs=100, save_dir='checkpoints'):
    criterion = CombinedLoss(
        temperature=0.07,
        triplet_weight=10.0,
        contrastive_weight=1.0
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    best_top1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'contrastive': 0, 'triplet': 0}
        
        for batch in tqdm(train_loader):
            street_imgs = batch["street"].to(device)
            satellite_imgs = batch["satellite"].to(device)
            
            optimizer.zero_grad()
            street_embeddings, satellite_embeddings = model(street_imgs, satellite_imgs)
            
            loss, loss_components = criterion(street_embeddings, satellite_embeddings)
            loss.backward()
            optimizer.step()
            
            for k in epoch_losses:
                epoch_losses[k] += loss_components[k]
        
        # Calculate average losses
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        
        # Print metrics
        print(f"\nEpoch {epoch+1}")
        for k, v in avg_losses.items():
            print(f"Avg {k.capitalize()} Loss: {v:.4f}")
        
        # Validate and save model
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    street_imgs = batch["street"].to(device)
                    satellite_imgs = batch["satellite"].to(device)
                    street_embeddings, satellite_embeddings = model(street_imgs, satellite_imgs)
                    loss, _ = criterion(street_embeddings, satellite_embeddings)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Compute accuracy metrics
            top1, top5, top10 = compute_accuracy(model, val_loader, device)
            print(f"Top-1: {top1:.2f}%, Top-5: {top5:.2f}%, Top-10: {top10:.2f}%")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'top1': top1,
                'top5': top5,
                'top10': top10
            }
            
            # Save best model based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(checkpoint, os.path.join(save_dir, 'best_model_loss.pth'))
            
            # Save best model based on top1 accuracy
            if top1 > best_top1:
                best_top1 = top1
                torch.save(checkpoint, os.path.join(save_dir, 'best_model_accuracy.pth'))
            
            # Save latest model
            torch.save(checkpoint, os.path.join(save_dir, 'latest_model.pth'))
            
            # Save epoch checkpoint
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def compute_accuracy(model, loader, device):
    model.eval()
    all_street_embeddings = []
    all_satellite_embeddings = []
    
    with torch.no_grad():
        for batch in loader:
            street_imgs = batch["street"].to(device)
            satellite_imgs = batch["satellite"].to(device)
            street_embeddings, satellite_embeddings = model(street_imgs, satellite_imgs)
            all_street_embeddings.append(street_embeddings)
            all_satellite_embeddings.append(satellite_embeddings)
    
    street_embeddings = torch.cat(all_street_embeddings, dim=0)
    satellite_embeddings = torch.cat(all_satellite_embeddings, dim=0)
    
    similarity = torch.mm(street_embeddings, satellite_embeddings.t())
    
    # Calculate top-k accuracy
    _, indices = similarity.topk(10, dim=1)
    correct_at_k = torch.arange(similarity.shape[0], device=device).unsqueeze(1) == indices
    top1 = correct_at_k[:, 0].float().mean().item() * 100
    top5 = correct_at_k[:, :5].any(dim=1).float().mean().item() * 100
    top10 = correct_at_k[:, :10].any(dim=1).float().mean().item() * 100
    
    return top1, top5, top10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feature_dims", type=int, default=4096)
    parser.add_argument("--backbone", type=str, default="vgg16")
    parser.add_argument("--use_contrastive", action="store_true")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CrossViewMatcher(feature_dims=args.feature_dims, backbone=args.backbone)
    model = model.to(device)

    # Create datasets and dataloaders
    train_dataset = SingleImageDataset(mode='train')
    val_dataset = SingleImageDataset(mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Train model
    train_single_image_matcher(model, train_loader, val_loader, optimizer, 
                             device, epochs=args.epochs, 
                             use_contrastive=args.use_contrastive)