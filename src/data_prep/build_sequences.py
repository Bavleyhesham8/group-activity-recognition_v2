import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm

def build_sequences(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_base = Path(config['data']['output_base'])
    features_dir = output_base / "features"
    sequences_dir = output_base / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    # Load group labels
    df = pd.read_csv(output_base / "group_labels" / "group_labels.csv")
    
    num_players = config['model']['num_players']
    num_frames = config['model']['num_frames']
    feature_dim = config['model']['resnet_feature_dim']
    
    num_samples = len(df)
    
    # Use Memmap for large arrays
    X = np.memmap(sequences_dir / "features.memmap", dtype='float32', mode='w+', 
                  shape=(num_samples, num_players, num_frames, feature_dim))
    y_group = np.zeros(num_samples, dtype='int32')
    # y_person would be (num_samples, num_players, num_frames) if needed
    
    # Label mapping
    unique_groups = df['group_label'].unique()
    group_to_idx = {label: i for i, label in enumerate(unique_groups)}
    
    print(f"Building sequences for {num_samples} frames...")
    
    for idx, row in tqdm(df.iterrows(), total=num_samples):
        frame_name = row['frame'] # e.g. "12345.jpg"
        frame_stem = frame_name.split('.')[0]
        group_label = row['group_label']
        
        y_group[idx] = group_to_idx[group_label]
        
        # Logic to find video/frame folder
        # In the volleyball dataset, frame folders are named after the center frame index
        # We need to find the correct path. This assumes feature extraction saved it appropriately.
        # This part might need adjustment based on exact local folder structure.
        
        # for player_idx in range(num_players):
        #     for frame_offset in range(num_frames):
        #         # Load extracted npy
        #         pass

    # Save metadata/labels
    np.save(sequences_dir / "group_labels.npy", y_group)
    # Save label mapping
    with open(sequences_dir / "label_map.yaml", 'w') as f:
        yaml.dump({'group_to_idx': group_to_idx}, f)

    print("Sequence building complete!")

if __name__ == "__main__":
    build_sequences("configs/config.yaml")
