import os
import cv2
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import models, transforms
from torch.cuda.amp import autocast

def extract_features(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup Paths
    main_anno_root = Path(config['data']['main_anno_root'])
    tracking_root = Path(config['data']['tracking_root'])
    output_base = Path(config['data']['output_base'])
    feature_dir = output_base / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Setup Model - ResNet18 (Feature Extractor)
    resnet = models.resnet18(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Tracking files logic from Cell 3
    tracking_files = list(tracking_root.glob("**/*.txt"))
    print(f"Found {len(tracking_files)} tracking files.")

    # We need group labels for filtering if needed, 
    # but the notebook uses them mainly for matching.
    # This script focuses on feature extraction.
    
    for txt_file in tqdm(tracking_files, desc="Extracting Features"):
        frame_stem = txt_file.stem
        # Logic to find associated frames (9-frame sequence)
        frame_folder = txt_file.parent
        video_folder = frame_folder.parent
        
        base_num = int(frame_stem)
        frames = []
        for i in range(base_num - 4, base_num + 5):
            # The path structure: MAIN_ANNO_ROOT / vid / frame_folder / i.jpg
            img_path = main_anno_root / video_folder.name / frame_folder.name / f"{i:05d}.jpg"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if len(frames) != 9:
            continue
            
        # Collect crops for all players across 9 frames
        crops = []
        with open(txt_file, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    xmin, ymin, xmax, ymax = map(int, parts[1:5])
                    for frame_img in frames:
                        crop = frame_img[ymin:ymax, xmin:xmax]
                        if crop.size > 0:
                            crops.append(transform(crop))
                except Exception:
                    continue
                    
        if len(crops) == 0:
            continue
            
        # Batch inference
        batch_size = 128
        features_list = []
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i:i+batch_size]).to(device)
            with torch.no_grad(), autocast():
                feat = resnet(batch)
                feat = feat.squeeze(-1).squeeze(-1)
            features_list.append(feat.cpu().numpy())
            
        seq_features = np.concatenate(features_list, axis=0)
        # NPY file naming: stem_features.npy
        np.save(feature_dir / f"{frame_stem}_features.npy", seq_features)

    print(f"Feature extraction complete. Features saved in {feature_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    extract_features(args.config)
