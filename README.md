# Network Intrusion Detection: Traditional ML with Cross-Dataset Evaluation

## 📋 Table of Contents

- Overview
- Installation
- Project Structure
- Usage
- Results
- Generated Outputs
- Technical Details
- Interpretation
- Limitations
- Future Work
- References

## 🎯 Overview

This project implements a complete network intrusion detection pipeline using traditional machine learning models, with a strong focus on cross-dataset evaluation between UNSW-NB15 and CICIDS2017. We engineer a set of universal features from both datasets so that models can be trained and evaluated in a common feature space.

### Key Highlights

- **Datasets**: UNSW-NB15 and CICIDS2017
- **Models**: Random Forest, Linear SVM (calibrated), Decision Tree
- **Features**: 27 universal features (after removing constants)
- **Experiments**: Intra-dataset and Cross-dataset
- **Analysis**: Full metrics, visualizations, and a markdown report suited for articles

## 🚀 Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM (recommended for CICIDS2017)
- 10GB+ disk space

### Dependencies

```bash
pip install -r requirements.txt
```

### Main Libraries

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML models and evaluation
- `matplotlib` - Visualizations
- `seaborn` - Statistical graphics
- `tabulate` - Nicely formatted tables

## 📁 Project Structure

```
cybersecuraty/
├── data/
│   └── raw/
│       ├── UNSW-NB15/          # UNSW-NB15 dataset
│       └── CICIDS2017/         # CICIDS2017 dataset
├── src/
│   ├── data_corrected.py       # Loading, processing, and feature engineering
│   ├── experiments.py          # Intra- and cross-dataset experiments
│   ├── models.py               # RF, SVM, Decision Tree
│   ├── eval.py                 # Metrics and evaluation helpers
│   ├── utils.py                # Utilities
│   └── kaggle_dl.py            # Optional dataset downloader
├── outputs/                    # Baseline experiment outputs
├── outputs_article_analysis/   # Full article-oriented analysis
├── main.py                     # Main script
├── run_article_analysis.py     # Standalone article analysis script
└── README.md                   # This file
```

## 🎮 Usage

### Basic Execution

```bash
# Run all experiments
python main.py

# Run only the article analysis
python main.py --run_article_analysis true

# Use custom dataset folders
python main.py --unsw_dir /path/to/unsw --cicids_dir /path/to/cicids
```

### Available Parameters

```bash
python main.py --help
```

- `--unsw_dir`: Path to the UNSW-NB15 dataset directory
- `--cicids_dir`: Path to the CICIDS2017 dataset directory
- `--gpu`: Use GPU for XGBoost (if enabled in your environment)
- `--out_dir`: Output directory
- `--auto_download`: Automatically download datasets
- `--use_unsw_train_test`: Use official UNSW train/test splits
- `--use_cicids_traffic_labelling`: Use CICIDS TrafficLabelling sources
- `--run_article_analysis`: Run the full article-oriented analysis

### Article-Oriented Analysis Only

```bash
# Run only the article analysis pipeline
python run_article_analysis.py
```

## 📊 Results

### Intra-Dataset Performance

| Dataset    | Model         | Accuracy | F1-Score | ROC-AUC |
| ---------- | ------------- | -------- | -------- | ------- |
| UNSW-NB15  | Random Forest | 87.8%    | 90.3%    | 98.2%   |
| UNSW-NB15  | SVM           | 74.4%    | 78.0%    | 84.7%   |
| UNSW-NB15  | Decision Tree | 86.3%    | 89.1%    | 96.9%   |
| CICIDS2017 | Random Forest | 83.8%    | 58.3%    | 86.4%   |
| CICIDS2017 | SVM           | 77.8%    | 36.2%    | 90.9%   |
| CICIDS2017 | Decision Tree | 86.6%    | 69.3%    | 78.1%   |

### Cross-Dataset Performance

| Transfer      | Model         | F1-Score |
| ------------- | ------------- | -------- |
| UNSW → CICIDS | Random Forest | 4.3%     |
| UNSW → CICIDS | SVM           | 43.1%    |
| UNSW → CICIDS | Decision Tree | 39.5%    |
| CICIDS → UNSW | Random Forest | 0.0%     |
| CICIDS → UNSW | SVM           | 73.0%    |
| CICIDS → UNSW | Decision Tree | 2.3%     |

### Key Findings

1. **Random Forest** delivers the best intra-dataset performance.
2. **SVM** transfers best across datasets (cross-dataset).
3. **Decision Tree** surfaces overfitting most clearly:
   - **UNSW**: Controlled overfitting (4.5% accuracy gap)
   - **CICIDS**: High overfitting (12.1% accuracy gap)
4. **Cross-dataset transfer** is challenging, especially from CICIDS → UNSW.
5. **Universal features** (27) enable apples-to-apples comparison.
6. **Overfitting** is more severe in imbalanced datasets (CICIDS).

## 📁 Generated Outputs

### outputs/

- JSON metrics for each experiment
- Confusion matrices (PNG)
- Cross-validation results

### outputs_article_analysis/

- **CSV Tables**:
  - `dataset_comparison.csv` - Dataset comparison summary
  - `performance_comparison.csv` - Model performance
  - `feature_analysis.csv` - Feature distributions and stats
- **PNG Figures**:
  - `class_distribution.png` - Class distribution
  - `performance_comparison.png` - Performance comparison
  - `cross_dataset_transfer_rf.png` - RF cross-dataset transfer
  - `cross_dataset_transfer_svm.png` - SVM cross-dataset transfer
  - `cross_dataset_transfer_dt.png` - Decision Tree cross-dataset transfer
  - `decision_tree_unsw.png` - Decision Tree visualization (UNSW)
  - `decision_tree_cicids.png` - Decision Tree visualization (CICIDS)
  - `decision_tree_overfitting_analysis.png` - Train vs Test (DT)
- **Report**:
  - `article_report.md` - Full article-style report

## 🔧 Technical Details

### Universal Features (27)

After removing constant features, we use a compact, comparable set including:

- **Basics**: duration, total_bytes, total_pkts, mean_pkt_size
- **Forward/Backward**: fwd_bytes, bwd_bytes, fwd_pkts, bwd_pkts
- **Timing**: iat_std
- **Flags**: syn_count, ack_count, flag_sum
- **Network**: ttl, loss, win
- **Statistics**: pkt_size_std, fwd_bwd_ratio
- **Log**: Log transforms of key features

### Data Processing

1. **Loading**: Official UNSW train/test; CICIDS split 70/30.
2. **Cleaning**: Remove infinities and NaNs; fix BOM/odd columns.
3. **Feature Engineering**: 39 raw signals → 27 universal features.
4. **Normalization**: MinMaxScaler fitted on combined data, then applied.
5. **Alignment**: Same feature space for both datasets to enable cross-eval.

### Models

- **Random Forest**: n_estimators=100, max_depth=10, regularized
- **SVM**: LinearSVC + CalibratedClassifierCV (C=1.0), tuned for speed
- **Decision Tree**: max_depth=10, min_samples_split=20, regularized
- **Validation**: 5-fold Stratified K-Fold for robust metrics

## 📈 Interpretation

### Intra-Dataset

- **UNSW**: More balanced; overall better metrics.
- **CICIDS**: Imbalanced; harder to detect attacks reliably.

### Cross-Dataset

- **SVM**: Best generalization across domains.
- **Random Forest**: Prone to learning dataset-specific artifacts.
- **Decision Tree**: Overfits visibly, especially on CICIDS.
- **Direction matters**: CICIDS → UNSW is harder than UNSW → CICIDS.

### Overfitting Analysis

- **UNSW**: Overfitting is well controlled across models.
- **CICIDS**: Decision Tree shows notable overfitting (12.1% accuracy gap).
- **Random Forest**: Best balance between performance and generalization.
- **SVM**: Most robust in cross-dataset transfer.

## 🚨 Limitations

1. **Cross-dataset transfer** is inherently difficult.
2. **Universal features** may discard dataset-specific signals.
3. **Class imbalance** impacts performance and calibration.
4. **Runtime** can be long for large datasets.

## 🔮 Future Work

1. **Domain Adaptation**: Techniques to improve transferability.
2. **Feature Engineering**: Dataset-specific and semantic features.
3. **Ensembles**: Blending models for stability.
4. **Deep Learning**: Comparative baselines.
5. **Real-time Detection**: Optimizations for production.

## 📚 References

- UNSW-NB15: Moustafa, N., & Slay, J. (2015)
- CICIDS2017: Sharafaldin, I., et al. (2018)
- Cross-dataset evaluation in IDS: recent literature

Note: This project is for academic research in network intrusion detection. Results are valid for the specific datasets and configurations used here.
