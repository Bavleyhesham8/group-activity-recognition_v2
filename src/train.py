import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from models.hierarchical_model import HierarchicalModel, device_type
from torch.cuda.amp import GradScaler, autocast

# -------------------- Dataset --------------------
class VolleyballDataset(Dataset):
    def __init__(self, seq_dir, standing_keep=0.15, standing_thresh=0.5, action_vocab=None):
        seq_dir = Path(seq_dir)
        self.features = np.load(seq_dir/"train_features.npy",      mmap_mode="r")
        self.p_labels = np.load(seq_dir/"train_person_labels.npy", mmap_mode="r")
        self.g_labels = np.load(seq_dir/"train_group_labels.npy",  mmap_mode="r")
        self.t_flags  = np.load(seq_dir/"train_team_flags.npy",    mmap_mode="r")

        standing_idx = action_vocab["standing"] if action_vocab else 8
        valid = np.where(self.g_labels != -1)[0]
        p_valid = self.p_labels[valid]
        stand_frac = (p_valid == standing_idx).mean(axis=1)
        is_heavy = stand_frac > standing_thresh
        rng = np.random.default_rng(42)
        n_keep = max(1, int(is_heavy.sum() * standing_keep))
        kept = rng.choice(valid[is_heavy], size=n_keep, replace=False)
        self.indices = np.sort(np.concatenate([valid[~is_heavy], kept]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            torch.from_numpy(np.array(self.features[idx])).float(),
            torch.from_numpy(np.array(self.p_labels[idx])).long(),
            torch.tensor(int(self.g_labels[idx]), dtype=torch.long),
            torch.from_numpy(np.array(self.t_flags[idx])).long()
        )

def get_core(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def freeze(mod):
    for p in mod.parameters():
        p.requires_grad_(False)

def run_epoch(model, loader, optimizer, scaler, stage, device, num_action_classes, train=True):
    model.train() if train else model.eval()
    total_loss = total_correct = total_samples = 0
    unknown_idx = 9 # Standard for the dataset
    
    person_criterion = nn.CrossEntropyLoss(ignore_index=unknown_idx) # Simplified weighting here
    group_criterion = nn.CrossEntropyLoss()
    
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feats, p_labels, g_labels, t_flags in tqdm(loader, leave=False):
            feats = feats.to(device, non_blocking=True)
            p_labels = p_labels.to(device, non_blocking=True)
            g_labels = g_labels.to(device, non_blocking=True)
            t_flags = t_flags.to(device, non_blocking=True)

            with autocast(enabled=(device.type == 'cuda')):
                if stage == 1:
                    a_logits = model(feats, t_flags, stage=1)
                    T = a_logits.shape[1]
                    p_tgt = p_labels.unsqueeze(1).expand(-1, T, -1)
                    loss = person_criterion(a_logits.reshape(-1, num_action_classes), p_tgt.reshape(-1))
                    
                    mask = p_tgt.reshape(-1) != unknown_idx
                    correct = (a_logits.reshape(-1, num_action_classes).argmax(-1)[mask] == p_tgt.reshape(-1)[mask]).sum().item()
                    samples = mask.sum().item()
                else:
                    g_logits, a_logits = model(feats, t_flags, stage=2)
                    T = a_logits.shape[1]
                    p_tgt = p_labels.unsqueeze(1).expand(-1, T, -1)
                    g_loss = group_criterion(g_logits, g_labels)
                    aux_loss = person_criterion(a_logits.reshape(-1, num_action_classes), p_tgt.reshape(-1))
                    loss = g_loss + 0.5 * aux_loss
                    
                    correct = (g_logits.argmax(-1) == g_labels).sum().item()
                    samples = g_labels.size(0)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * feats.size(0)
            total_correct += correct
            total_samples += max(samples, 1)

    return total_loss / len(loader.dataset), 100.0 * total_correct / max(total_samples, 1)

def train_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_dir = Path(config['data']['output_base']) / "sequences"
    ckpt_dir = Path(config['data']['checkpoints_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(seq_dir/"action_vocab.json") as f: action_vocab = json.load(f)
    num_action_classes = len(action_vocab)
    with open(seq_dir/"group_vocab.json") as f: group_vocab = json.load(f)
    num_group_classes = len(group_vocab)

    print("Loading dataset...")
    full_ds = VolleyballDataset(seq_dir, config['training']['standing_keep'], config['training']['standing_thresh'], action_vocab)
    n_val = int(len(full_ds) * config['training']['val_split'])
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['stage1']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['stage1']['batch_size']*2, shuffle=False, num_workers=config['training']['num_workers'], pin_memory=True)

    model = HierarchicalModel(
        feat_dim=config['model']['feat_dim'],
        person_hidden=config['model']['person_hidden'],
        proj_dim=config['model']['proj_dim'],
        fc_dim=config['model']['fc_dim'],
        group_hidden=config['model']['group_hidden'],
        num_action_classes=num_action_classes,
        num_group_classes=num_group_classes,
        use_temporal_attn=config['model']['use_temporal_attn']
    ).to(device)

    scaler = GradScaler()

    # Stage 1: Person Actions
    print("--- Stage 1: Action Recognition ---")
    opt1 = torch.optim.Adam(model.person.parameters(), lr=config['training']['stage1']['lr'])
    best_val_acc = 0
    for ep in range(1, config['training']['stage1']['epochs'] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, opt1, scaler, 1, device, num_action_classes, True)
        val_loss, val_acc = run_epoch(model, val_loader, None, scaler, 1, device, num_action_classes, False)
        print(f"Epoch {ep:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.person.state_dict(), ckpt_dir/"best_stage1.pt")

    # Stage 2: Full Hierarchy
    print("\n--- Stage 2: Group Activity Recognition ---")
    model.person.load_state_dict(torch.load(ckpt_dir/"best_stage1.pt"))
    # freeze(model.person) # Optional as per notebook logic
    
    opt2 = torch.optim.Adam(model.parameters(), lr=config['training']['stage2']['lr'])
    best_val_acc = 0
    for ep in range(1, config['training']['stage2']['epochs'] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, opt2, scaler, 2, device, num_action_classes, True)
        val_loss, val_acc = run_epoch(model, val_loader, None, scaler, 2, device, num_action_classes, False)
        print(f"Epoch {ep:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir/"best_full_model.pt")

    print(f"Training complete. Best Val Acc (Stage 2): {best_val_acc:.1f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_model(args.config)
