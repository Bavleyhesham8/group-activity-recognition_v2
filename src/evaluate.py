import torch
import yaml
import numpy as np
from pathlib import Path
from models.hierarchical_model import HierarchicalModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(config_path, model_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalModel(config['model']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data and labels
    # Load label mapping
    output_base = Path(config['data']['output_base'])
    sequences_dir = output_base / "sequences"
    with open(sequences_dir / "label_map.yaml", 'r') as f:
        label_map = yaml.safe_load(f)
    
    idx_to_group = {v: k for k, v in label_map['group_to_idx'].items()}
    target_names = [idx_to_group[i] for i in range(len(idx_to_group))]

    print("Running evaluation...")
    y_true = []
    y_pred = []

    # Placeholder for evaluation loop
    # with torch.no_grad():
    #     for batch_x, batch_y in test_loader:
    #         logits, _ = model(batch_x.to(device))
    #         preds = torch.argmax(logits, dim=1)
    #         y_true.extend(batch_y.numpy())
    #         y_pred.extend(preds.cpu().numpy())

    # Generate reports
    # report = classification_report(y_true, y_pred, target_names=target_names)
    # print(report)

    # Confusion Matrix
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig('results/confusion_matrix.png')
    # print("Saved confusion matrix to results/confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="outputs/checkpoints/model_final.pth")
    args = parser.parse_args()
    
    evaluate(args.config, args.model)
