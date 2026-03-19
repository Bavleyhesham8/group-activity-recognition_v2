import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from models.hierarchical_model import HierarchicalModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# -------------------- Dataset (minimal for eval) --------------------
class VolleyballEvalDataset(Dataset):
    def __init__(self, seq_dir):
        seq_dir = Path(seq_dir)
        self.features = np.load(seq_dir/"train_features.npy",      mmap_mode="r")
        self.p_labels = np.load(seq_dir/"train_person_labels.npy", mmap_mode="r")
        self.g_labels = np.load(seq_dir/"train_group_labels.npy",  mmap_mode="r")
        self.t_flags  = np.load(seq_dir/"train_team_flags.npy",    mmap_mode="r")
        self.indices = np.where(self.g_labels != -1)[0]

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

def evaluate_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_dir = Path(config['data']['output_base']) / "sequences"
    reports_dir = Path(config['data']['checkpoints_dir']) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(seq_dir/"action_vocab.json") as f: action_vocab = json.load(f)
    with open(seq_dir/"group_vocab.json") as f: group_vocab = json.load(f)
    num_action_classes = len(action_vocab)
    num_group_classes = len(group_vocab)
    inv_group_vocab = {v: k for k, v in group_vocab.items()}

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

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle both full dict or state_only
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    print("Loading test dataset...")
    ds = VolleyballEvalDataset(seq_dir)
    loader = DataLoader(ds, batch_size=config['training']['stage2']['batch_size'], shuffle=False)

    all_preds = []
    all_labels = []
    print("Running Inference...")
    with torch.no_grad():
        for feats, _, g_labels, t_flags in tqdm(loader):
            g_logits, _ = model(feats.to(device), t_flags.to(device), stage=2)
            preds = g_logits.argmax(-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(g_labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Classification Report
    target_names = [inv_group_vocab[i] for i in range(num_group_classes)]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(report)
    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png")
    print(f"Saved confusion matrix to {reports_dir / 'confusion_matrix.png'}")

    # Per-class accuracy breakdown
    breakdown_path = reports_dir / "per_class_breakdown.txt"
    with open(breakdown_path, "w") as f:
        f.write(f"{'Class':15s} | {'Acc':>7} | {'N':>5}\n")
        f.write("-" * 32 + "\n")
        for i in range(num_group_classes):
            mask = (y_true == i)
            if not mask.any(): continue
            acc = (y_pred[mask] == i).sum() / mask.sum()
            line = f"{inv_group_vocab[i]:15s} | {100*acc:>6.1f}% | {mask.sum():>5d}"
            f.write(line + "\n")
            print(line)

    print(f"Evaluation complete. Reports saved in {reports_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt", default="outputs/checkpoints/best_full_model.pt")
    args = parser.parse_args()
    evaluate_model(args.config, args.ckpt)
