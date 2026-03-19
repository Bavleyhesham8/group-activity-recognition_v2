# Volleyball Group Activity Recognition (CVPR 2016)

A professional PyTorch implementation of the hierarchical temporal model for group activity recognition in volleyball videos, based on the CVPR 2016 paper.

## 🚀 Project Structure
```text
volleyball-group-activity-recognition-cvpr2016/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── group-activity-recognition-training.ipynb     # Training and visualization
│   └── feature-extraction-resnet.ipynb              # ResNet feature extraction logic
├── src/
│   ├── data_prep/
│   │   ├── extract_features.py                       # Layer-wise feature extraction
│   │   └── build_sequences.py                       # Sequence aggregation (memmap)
│   ├── models/
│   │   └── hierarchical_model.py                    # PersonLSTM + GroupLSTM architecture
│   ├── train.py                                     # Stage 1 and Stage 2 training loops
│   └── evaluate.py                                  # Inference and metrics reporting
├── configs/
│   └── config.yaml                                  # Hyperparameters and paths
├── processed/                                       # Intermediate features/sequences (.gitignore)
├── outputs/                                         # Trained models and log files (.gitignore)
└── data/                                            # Raw dataset (volleyball clips) (.gitignore)
```

## 🛠️ Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/volleyball-group-activity-recognition.git
    cd volleyball-group-activity-recognition
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Dataset**:
    Follow the original CVPR 2016 dataset instructions to download volleyball clips and tracking annotations.

## 📈 Usage
### 1. Feature Extraction
Extract ResNet50 features from player crops:
```bash
python src/data_prep/extract_features.py
```

### 2. Sequence Building
Process extracted features into memmapped sequences for efficient training:
```bash
python src/data_prep/build_sequences.py
```

### 3. Training
Train the hierarchical model (Stage 1 and Stage 2):
```bash
python src/train.py
```

### 4. Evaluation
Evaluate the model on the test set:
```bash
python src/evaluate.py --model outputs/checkpoints/model_final.pth
```

## 📊 Methodology
This implementation follows a two-stage hierarchical approach:
1.  **Person Level**: Individual player temporal dynamics are captured using a `PersonLSTM` with temporal attention.
2.  **Group Level**: A `TwoTeamGroupLSTM` aggregates players from both sides of the net to classify the overall group activity.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
