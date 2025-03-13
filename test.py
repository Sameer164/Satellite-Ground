import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from model import CrossViewMatcher
from dataset import SingleImageDataset
import numpy as np
import os
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def compute_metrics(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute comprehensive retrieval metrics
    """
    num_queries = similarity_matrix.shape[0]
    labels = torch.arange(num_queries, device=similarity_matrix.device)
    
    # Sort similarities in descending order
    _, sorted_indices = similarity_matrix.sort(dim=1, descending=True)
    
    # Find rank of correct match for each query
    correct_indices = (sorted_indices == labels.view(-1, 1)).nonzero()[:, 1]
    
    # Compute Recall@K
    recall = {}
    for k in k_values:
        recall[k] = (correct_indices < k).float().mean().item() * 100
    
    # Compute MRR (Mean Reciprocal Rank)
    mrr = (1.0 / (correct_indices + 1.0)).mean().item()
    
    # Compute localization accuracy (within top K)
    loc_acc = {}
    for k in k_values:
        loc_acc[k] = (correct_indices < k).float().sum().item()
    
    return recall, mrr, loc_acc

def visualize_results(similarity_matrix, dataset, save_dir, num_examples=5):
    """
    Visualize some example matches
    """
    os.makedirs(save_dir, exist_ok=True)
    
    _, indices = similarity_matrix.topk(5, dim=1)
    for i in range(num_examples):
        query = dataset[i]['street']
        top_matches = [dataset[idx.item()]['satellite'] for idx in indices[i]]
        
        # Create visualization...
        plt.figure(figsize=(15, 3))
        plt.subplot(1, 6, 1)
        plt.imshow(query.permute(1, 2, 0))
        plt.title('Query')
        
        for j, match in enumerate(top_matches):
            plt.subplot(1, 6, j+2)
            plt.imshow(match.permute(1, 2, 0))
            plt.title(f'Match {j+1}')
        
        plt.savefig(os.path.join(save_dir, f'example_{i}.png'))
        plt.close()

def test(model, test_loader, device, save_dir='test_results'):
    """
    Comprehensive testing function
    """
    model.eval()
    all_street_embeddings = []
    all_satellite_embeddings = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            street_imgs = batch['street'].to(device)
            satellite_imgs = batch['satellite'].to(device)
            
            # Get embeddings
            street_embedding, satellite_embedding = model(street_imgs, satellite_imgs)
            
            all_street_embeddings.append(street_embedding.cpu())
            all_satellite_embeddings.append(satellite_embedding.cpu())
    
    # Concatenate all embeddings
    street_embeddings = torch.cat(all_street_embeddings, dim=0)
    satellite_embeddings = torch.cat(all_satellite_embeddings, dim=0)
    
    print("Computing similarity matrix...")
    similarity = torch.mm(street_embeddings, satellite_embeddings.t())
    
    # Compute metrics
    recall, mrr, loc_acc = compute_metrics(similarity)
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results
    print("\nTest Results:")
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Number of test samples: {len(test_loader.dataset)}\n\n")
        
        print(f"Number of test samples: {len(test_loader.dataset)}")
        
        for k in recall.keys():
            line = f"Recall@{k}: {recall[k]:.2f}%"
            print(line)
            f.write(line + '\n')
        
        line = f"Mean Reciprocal Rank: {mrr:.4f}"
        print(line)
        f.write(line + '\n')
        
        for k in loc_acc.keys():
            line = f"Localization Accuracy@{k}: {loc_acc[k]}/{len(test_loader.dataset)}"
            print(line)
            f.write(line + '\n')
    
    # Visualize some results
    visualize_results(similarity, test_loader.dataset, 
                     os.path.join(save_dir, 'visualizations'))
    
    return recall, mrr, loc_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_dims", type=int, default=2048)  # Updated for ResNet50
    parser.add_argument("--backbone", type=str, default="res50")
    parser.add_argument("--save_dir", type=str, default="test_results")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = CrossViewMatcher(feature_dims=args.feature_dims, backbone=args.backbone)
    
    # Load saved model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create test dataset and dataloader
    test_dataset = SingleImageDataset(mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Run test
    recall, mrr, loc_acc = test(model, test_loader, device, args.save_dir)


