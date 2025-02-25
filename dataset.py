import glob
import random
import os
import json
import math
import time
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True

def LatLngToPixel(lat, lng, centerLat, centerLng, zoom):
    x, y = LatLngToGlobalPixel(lat, lng, zoom)
    cx, cy = LatLngToGlobalPixel(centerLat, centerLng, zoom)
    return x - cx, y - cy

def LatLngToGlobalPixel(lat, lng, zoom):
    siny = math.sin(lat * math.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)

    return [(256 * (0.5 + lng / 360.0)) * (2 ** zoom), (256 * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)))*(2 ** zoom)]

class SingleImageDataset(Dataset):
    def __init__(self, root="./CVUSA_subset", transforms_street=None, transforms_sat=None, mode='train', split_ratio=0.8):
        self.root = root
        self.mode = mode
        
        # Default transforms if none provided
        if transforms_street is None:
            self.transforms_street = transforms.Compose([
                transforms.Resize((512, 384)),
                transforms.ColorJitter(0.2, 0.2, 0.2) if mode == 'train' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transforms_street = transforms.Compose(transforms_street)
            
        if transforms_sat is None:
            self.transforms_sat = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transforms_sat = transforms.Compose(transforms_sat)
        
        # Get all satellite images
        sat_path = os.path.join(root, "bingmap")
        street_path = os.path.join(root, "streetview")
        
        # Get all image pairs
        self.pairs = self._get_pairs(sat_path, street_path)
        
        # Split dataset
        total_pairs = len(self.pairs)
        if mode == 'train':
            self.pairs = self.pairs[:int(total_pairs * split_ratio)]
        elif mode == 'val':
            self.pairs = self.pairs[int(total_pairs * split_ratio):int(total_pairs * 0.9)]
        elif mode == 'test':
            self.pairs = self.pairs[int(total_pairs * 0.9):]
            
        print(f"Loaded {len(self.pairs)} pairs for {mode}")
        
    def _get_pairs(self, sat_path, street_path):
        pairs = []
        
        # Get all satellite images
        sat_images = glob.glob(os.path.join(sat_path, "input*.png"))
        
        for sat_img in sat_images:
            # Extract the number from satellite image name
            number = re.search(r'input(\d+)\.png', sat_img).group(1)
            # Construct corresponding street view image path
            street_img = os.path.join(street_path, f"{number}.jpg")
            
            # Check if both images exist
            if os.path.exists(street_img):
                pairs.append((sat_img, street_img))
        
        return sorted(pairs)  # Sort to ensure deterministic ordering
        
    def __getitem__(self, index):
        sat_path, street_path = self.pairs[index]
        
        # Load and transform images
        try:
            satellite_img = Image.open(sat_path).convert('RGB')
            street_img = Image.open(street_path).convert('RGB')
            
            satellite_img = self.transforms_sat(satellite_img)
            street_img = self.transforms_street(street_img)
            
            return {
                "street": street_img,
                "satellite": satellite_img,
                "idx": index  # Useful for debugging
            }
            
        except Exception as e:
            print(f"Error loading images: {sat_path} or {street_path}")
            print(f"Error: {str(e)}")
            # Return the first item as a fallback
            return self.__getitem__(0)
        
    def __len__(self):
        return len(self.pairs)

class ImageDataset(Dataset):
    def __init__(self, root="dataset/json", transforms_street=[transforms.ToTensor(),],transforms_sat=[transforms.ToTensor(),], sequence_size = 7, mode='train', zoom=20):
        self.zoom = zoom
        self.transforms_street = transforms.Compose(transforms_street)
        self.transforms_sat = transforms.Compose(transforms_sat)
        self.seqence_size = sequence_size
        self.mode = mode

        if mode == "train" or mode == "val" or "dev":
            self.year = "2019"
        else:
            raise RuntimeError("no such mode")
        
        self.json_files = sorted(glob.glob(os.path.join(root, self.year+"_JSON") + '/*.json'), key=lambda x:int(x.split("/")[-1].split(".json")[0]))
        if self.year == "2019":
            if mode == "train":
                self.json_files = self.json_files[:int(len(self.json_files)*0.8)]
            elif mode == "val":
                self.json_files = self.json_files[int(len(self.json_files)*0.8+1):]
            elif mode == "dev1":
                self.json_files = self.json_files[:int(len(self.json_files)*0.05)]
            elif mode == "dev2":
                self.json_files = self.json_files[int(len(self.json_files)*0.99):]

        self.val_center = []
        if mode == "val" or mode == "dev1" or mode == "dev2":
            for i in self.json_files:
                f = open(i, 'r')
                meta_data = json.load(f)#load json
                center_lat, center_lon = meta_data["center"]
                self.val_center.append([center_lat, center_lon])
                f.close()

    def get_sat_center(self, idx):
        if len(self.val_center) > 0:
            return self.val_center[idx]

    def __getitem__(self, index):
        f = open(self.json_files[index])#open json
        meta_data = json.load(f)#load json
        center_lat, center_lon = meta_data["center"]
        f.close()

        street_images = []
        sate_imgs = []

        dir_sate_img = meta_data["satellite_views"][str(self.zoom)]
        dir_sate_img = dir_sate_img.split("\\")[1:]
        dir_sate_img = "/".join(dir_sate_img)
        sate_img = self.transforms_sat(Image.open(os.path.join("dataset/satellite", dir_sate_img)))
        sate_imgs.append(sate_img)

        sate_imgs = torch.stack(tuple(sate_imgs), 0)

        all_street_views = meta_data["street_views"]
        if len(all_street_views.keys()) > self.seqence_size:#if one sequence >7 random drop some
            if self.mode == "train":
                for d in range(len(all_street_views.keys()) - self.seqence_size):
                    all_street_views.pop(random.choice(list(all_street_views.keys())))
            else:
                for d in range(len(all_street_views.keys()) - self.seqence_size):
                    all_street_views.pop(list(all_street_views.keys())[-1])

        if len(all_street_views.keys()) < 7:
            print(self.json_files[index])

        for k in sorted(all_street_views.keys()):
            v = all_street_views[k]
            px, py = LatLngToPixel(v["lat"],v["lon"],center_lat, center_lon,20)
            dir_img = v["name"]
            dir_img = dir_img.split("\\")[1:]
            dir_img = "/".join(dir_img)
            dir_img = os.path.join("dataset/street", os.path.join(str(self.year)+"_street", dir_img))
            img = self.transforms_street(Image.open(dir_img))

            street_images.append(img)

        #stack to torch tensors on dim=0
        street_images = torch.stack(tuple(street_images), 0)
        return {"street":street_images, "satellite":sate_imgs}


    def __len__(self):
        return len(self.json_files)

if __name__ == "__main__":
    # Test the dataset
    dataset = SingleImageDataset(root="CVUSA_subset", mode='train')
    print(f"Total training pairs: {len(dataset)}")
    
    # Test loading an item
    sample = dataset[0]
    print(f"Street image shape: {sample['street'].shape}")
    print(f"Satellite image shape: {sample['satellite'].shape}")
    
    # Create dataloaders
    train_dataset = SingleImageDataset(root="CVUSA_subset", mode='train')
    val_dataset = SingleImageDataset(root="CVUSA_subset", mode='val')
    test_dataset = SingleImageDataset(root="CVUSA_subset", mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Test batch loading
    for batch in train_loader:
        print(f"Batch shapes:")
        print(f"Street: {batch['street'].shape}")
        print(f"Satellite: {batch['satellite'].shape}")
        break
