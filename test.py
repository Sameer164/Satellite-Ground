import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from model import CrossViewMatcher
from dataset import SingleImageDataset
import numpy as np

def compute_metrics(similarity_matrix):
    """
    Compute retrieval metrics:
    - Recall@K (R@K): percentage of queries where correct match is in top K
    - Mean Reciprocal Rank (MRR)
    """
    num_queries = similarity_matrix.shape[0]
    
    # Ground truth matches are along the diagonal
    labels = torch.arange(num_queries, device=similarity_matrix.device)
    
    # Sort similarities in descending order
    _, sorted_indices = similarity_matrix.sort(dim=1, descending=True)
    
    # Find rank of correct match for each query
    correct_indices = (sorted_indices == labels.view(-1, 1)).nonzero()[:, 1]
    
    # Compute Recall@K
    recall = {}
    for k in [1, 5, 10]:
        recall[k] = (correct_indices < k).float().mean().item() * 100
    
    # Compute MRR
    mrr = (1.0 / (correct_indices + 1.0)).mean().item()
    
    return recall, mrr

def test(model, test_loader, device):
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
    # Compute similarity matrix
    similarity = torch.mm(street_embeddings, satellite_embeddings.t())
    
    # Compute metrics
    recall, mrr = compute_metrics(similarity)
    
    print("\nTest Results:")
    print(f"Recall@1: {recall[1]:.2f}%")
    print(f"Recall@5: {recall[5]:.2f}%")
    print(f"Recall@10: {recall[10]:.2f}%")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    
    return recall, mrr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_dims", type=int, default=4096)
    parser.add_argument("--backbone", type=str, default="vgg16")
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
    recall, mrr = test(model, test_loader, device)


