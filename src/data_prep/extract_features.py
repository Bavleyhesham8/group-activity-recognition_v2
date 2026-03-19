import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import yaml

class CropDataset(Dataset):
    def __init__(self, crop_files, transform=None):
        self.crop_files = crop_files
        self.transform = transform

    def __len__(self):
        return len(self.crop_files)

    def __getitem__(self, idx):
        img_path = self.crop_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)

def extract_features(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup ResNet50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity() # Remove classifier
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    crops_dir = Path(config['data']['output_base']) / "crops"
    features_dir = Path(config['data']['output_base']) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    crop_files = list(crops_dir.glob("**/*.jpg"))
    dataset = CropDataset(crop_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['training']['stage1']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    print(f"Extracting features from {len(crop_files)} crops...")
    with torch.no_grad():
        for imgs, paths in tqdm(dataloader):
            imgs = imgs.to(device)
            features = resnet(imgs).cpu().numpy()
            
            for feat, path in zip(features, paths):
                save_name = Path(path).stem + ".npy"
                # Keep structure if needed, or save flat
                # For this project, we usually save crops in frame folders
                frame_id = Path(path).parent.name
                video_id = Path(path).parent.parent.name
                
                target_dir = features_dir / video_id / frame_id
                target_dir.mkdir(parents=True, exist_ok=True)
                np.save(target_dir / save_name, feat)

if __name__ == "__main__":
    extract_features("configs/config.yaml")
