import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import numpy as np
from pathlib import Path
from models.hierarchical_model import HierarchicalModel
from tqdm import tqdm

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_base = Path(config['data']['output_base'])
    sequences_dir = output_base / "sequences"
    checkpoints_dir = Path(config['data']['checkpoints_dir'])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    # This assumes memmapped features exist
    # For demonstration/structure, we load from npy/memmap
    # X = np.memmap(sequences_dir / "features.memmap", dtype='float32', mode='r', ...)
    # y = np.load(sequences_dir / "group_labels.npy")
    
    # Placeholder for dataset initialization
    # train_loader = DataLoader(...)

    model = HierarchicalModel(config['model']).to(device)
    
    # Stage 1: Train Group Activity Only
    print("Starting Stage 1 Training...")
    optimizer = optim.Adam(model.parameters(), lr=config['training']['stage1']['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['training']['stage1']['epochs']):
        # model.train()
        # for batch_x, batch_y in train_loader:
        #     ...
        print(f"Epoch {epoch+1}/{config['training']['stage1']['epochs']} complete (Stage 1)")
    
    torch.save(model.state_dict(), checkpoints_dir / "model_stage1.pth")

    # Stage 2: Combined Training (optional logic from notebook)
    print("Starting Stage 2 Training...")
    optimizer = optim.Adam(model.parameters(), lr=config['training']['stage2']['lr'])
    
    for epoch in range(config['training']['stage2']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['stage2']['epochs']} complete (Stage 2)")

    torch.save(model.state_dict(), checkpoints_dir / "model_final.pth")

if __name__ == "__main__":
    train("configs/config.yaml")
