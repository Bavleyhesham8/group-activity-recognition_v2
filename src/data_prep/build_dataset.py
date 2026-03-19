import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import yaml

def build_dataset(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup Paths
    main_anno_root = Path(config['data']['main_anno_root'])
    output_base = Path(config['data']['output_base'])
    feat_dir = output_base / "features"
    seq_dir = output_base / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Constants
    n_keyframes = config['model']['num_frames']
    n_players = config['model']['num_players']
    n_subframes = config.get('model', {}).get('num_subframes', 1)
    feat_dim = config['model']['feat_dim']
    img_width = 1280 # Standard volleyball court frame width

    group_vocab = {
        "r_set": 0, "r_spike": 1, "r-pass": 2, "r_winpoint": 3,
        "l_set": 4, "l-spike": 5, "l_pass": 6, "l_winpoint": 7
    }
    action_vocab = {
        "waiting": 0, "setting": 1, "digging": 2, "falling": 3,
        "spiking": 4, "blocking": 5, "jumping": 6,
        "moving":  7, "standing": 8, "unknown": 9
    }

    print("Building annotation lookup...")
    anno_dict = {}
    for vid_dir in sorted(main_anno_root.iterdir()):
        if not vid_dir.is_dir(): continue
        anno_file = vid_dir / "annotations.txt"
        if not anno_file.exists(): continue
        with open(anno_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 7: continue
                frame_key = parts[0]
                group_label = parts[1]
                actions = []
                x_centers = []
                for i in range(2, len(parts), 5):
                    if i + 4 >= len(parts): break
                    x = int(parts[i])
                    w = int(parts[i + 2])
                    action = parts[i + 4]
                    x_centers.append(x + w // 2)
                    actions.append(action)
                anno_dict[frame_key] = {
                    "group": group_label,
                    "actions": actions,
                    "x_centers": x_centers
                }

    feat_files = sorted(feat_dir.glob("*.npy"))
    total = len(feat_files)
    print(f"Total feature files to process: {total}")

    # Allocate Memmaps
    feat_mm = np.lib.format.open_memmap(
        seq_dir / "train_features.npy", mode="w+",
        dtype="float32", shape=(total, n_keyframes, n_players, feat_dim))
    plabel_mm = np.lib.format.open_memmap(
        seq_dir / "train_person_labels.npy", mode="w+",
        dtype="int16", shape=(total, n_players))
    glabel_mm = np.lib.format.open_memmap(
        seq_dir / "train_group_labels.npy", mode="w+",
        dtype="int16", shape=(total,))
    flags_mm = np.lib.format.open_memmap(
        seq_dir / "train_team_flags.npy", mode="w+",
        dtype="int16", shape=(total, n_players))

    written = skipped = unknown_group = 0

    def reshape_features(raw):
        # Expected shape: (n_keyframes * n_players * n_subframes, feat_dim)
        # We reshape and return (n_keyframes, n_players, feat_dim) by averaging over subframes
        try:
            raw = raw.astype(np.float32)
            if n_subframes > 1:
                # Case where we have multiple sub-samples (e.g. from 20 frames)
                reshaped = raw.reshape(n_keyframes, n_players, n_subframes, feat_dim)
                return reshaped.mean(axis=2)
            else:
                # Case where we have exactly n_keyframes * n_players samples
                return raw.reshape(n_keyframes, n_players, feat_dim)
        except Exception as e:
            return None

    for npy_file in tqdm(feat_files, desc="Building Dataset"):
        try:
            raw = np.load(npy_file, allow_pickle=False)
        except Exception:
            skipped += 1
            continue

        features = reshape_features(raw)
        if features is None:
            skipped += 1
            continue

        anno_key = npy_file.stem.replace("_features", "") + ".jpg"
        entry = anno_dict.get(anno_key, None)

        if entry is None:
            group_idx = -1
            person_labels = [action_vocab["unknown"]] * n_players
            team_flags = [0] * n_players
            unknown_group += 1
        else:
            group_idx = group_vocab.get(entry["group"], -1)
            if group_idx == -1:
                unknown_group += 1
            
            raw_acts = entry["actions"][:n_players]
            person_labels = [action_vocab.get(a, action_vocab["unknown"]) for a in raw_acts]
            while len(person_labels) < n_players:
                person_labels.append(action_vocab["unknown"])
            
            raw_x = entry["x_centers"][:n_players]
            team_flags = [1 if x < (img_width // 2) else 0 for x in raw_x]
            while len(team_flags) < n_players:
                team_flags.append(0)

        feat_mm[written] = features
        plabel_mm[written] = person_labels
        glabel_mm[written] = group_idx
        flags_mm[written] = team_flags
        written += 1

    # Cleanup and Save Vocabs
    del feat_mm, plabel_mm, glabel_mm, flags_mm
    with open(seq_dir / "group_vocab.json", "w") as f: json.dump(group_vocab, f)
    with open(seq_dir / "action_vocab.json", "w") as f: json.dump(action_vocab, f)

    print(f"Dataset building complete. Written: {written}, Skipped: {skipped}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_dataset(args.config)
