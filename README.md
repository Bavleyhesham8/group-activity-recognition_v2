# Volleyball Group Activity Recognition (CVPR 2016)

A professional PyTorch implementation of the hierarchical temporal model for group activity recognition in volleyball videos, based on the CVPR 2016 paper.

**Author:** bavley hesham ibrahim  
**Instructor:** dr.Mostafa S. Ibrahim

## 🖼️ Demos
![Demo 1](assests/demo_105655_frame_105655_pred_2_true_2.jpg)
![Demo 2](assests/demo_116285_frame_116285_pred_2_true_2.jpg)
![Demo 3](assests/demo_21835_frame_21835_pred_2_true_2.jpg)

## 🚀 Project Structure
```text
group-activity-recognition/
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
│   │   └── build_dataset.py                        # Dataset aggregation (memmap)
│   ├── models/
│   │   └── hierarchical_model.py                    # PersonLSTM + GroupLSTM architecture
│   ├── train.py                                     # Stage 1 and Stage 2 training loops
│   └── evaluate.py                                  # Inference and metrics reporting
├── configs/
│   └── config.yaml                                  # Hyperparameters and paths
├── assests/                                         # Project assets and plots
├── processed/                                       # Intermediate features/sequences (.gitignore)
├── outputs/                                         # Trained models and log files (.gitignore)
└── data/                                            # Raw dataset (volleyball clips) (.gitignore)
```

## 📊 Performance
Our model achieves **70.0% accuracy** on the multi-class dataset, significantly outperforming the baseline hierarchical model (51.1%).

### Classification Report
![Classification Report](assests/111.png)

### Training Plots & Confusion Matrix
![Confusion Matrix and Epochs](assests/confusion_epoch_47.png)

### Comparison with Paper Baselines
| Method | Accuracy (%) |
| :--- | :---: |
| B1-Image Classification | 46.7 |
| B2-Person Classification | 33.1 |
| B3-Fine-tuned Person Classification | 35.2 |
| B4-Temporal Model with Image Features | 37.4 |
| B5-Temporal Model with Person Features | 45.9 |
| B6-Our Two-stage Model without LSTM 1 | 48.8 |
| B7-Our Two-stage Model without LSTM 2 | 49.7 |
| **Hierarchical Model (Paper)** | **51.1** |
| **Our Optimized Implementation** | **70.0** |

## 🛠️ Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Bavleyhesham8/group-activity-recognition_v2.git
    cd group-activity-recognition_v2
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Dataset**:
    Follow the original CVPR 2016 dataset instructions to download volleyball clips and tracking annotations.

## 📈 Usage
### 1. Feature Extraction
Extract ResNet18 features from player crops:
```bash
python src/data_prep/extract_features.py
```

### 2. Dataset Building
Process extracted features into memmapped datasets for efficient training:
```bash
python src/data_prep/build_dataset.py
```

### 3. Training
Train the hierarchical model (Stage 1 and Stage 2):
```bash
python src/train.py
```

### 4. Evaluation
Evaluate the model on the test set:
```bash
python src/evaluate.py --ckpt outputs/checkpoints/best_full_model.pt
```

## 📜 Methodology
This implementation follows a two-stage hierarchical approach with several enhancements:
1.  **Feature Extraction**: Uses a pre-trained ResNet18 for high-quality player feature representation.
2.  **Person Level**: Individual player temporal dynamics are captured using a `PersonLSTM`.
3.  **Group Level**: A `TwoTeamGroupLSTM` aggregates players from both sides of the net to classify the overall group activity.
4.  **Training Strategy**: Two-stage training with Stage 1 focusing on action recognition and Stage 2 on group activity.

## 🙏 Acknowledgements
Special thanks to **Dr.Mostafa S. Ibrahim** for his guidance and the original [CVPR 2016 paper](assests/ibrahim-cvpr16.pdf).

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
