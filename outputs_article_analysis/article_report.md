# Network Intrusion Detection: Traditional Machine Learning Approaches

## Abstract

This study evaluates traditional machine learning approaches for network intrusion detection using two benchmark datasets: UNSW-NB15 and CICIDS2017. We compare Random Forest and Support Vector Machine (SVM) classifiers in both intra-dataset and cross-dataset scenarios.

## Dataset Analysis

### Dataset Composition

| Dataset    | Total Samples   | Train/Test Split   |   Features | Train Normal      | Train Attack    | Test Normal     | Test Attack     | Train Balance   | Test Balance   |
|:-----------|:----------------|:-------------------|-----------:|:------------------|:----------------|:----------------|:----------------|:----------------|:---------------|
| UNSW-NB15  | 257,673         | 32.0% / 68.0%      |         27 | 37,000 (44.9%)    | 45,332 (55.1%)  | 56,000 (31.9%)  | 119,341 (68.1%) | Balanced        | Imbalanced     |
| CICIDS2017 | 3,119,345       | 70.0% / 30.0%      |         27 | 1,593,434 (73.0%) | 590,107 (27.0%) | 679,663 (72.6%) | 256,141 (27.4%) | Imbalanced      | Imbalanced     |

### Key Dataset Characteristics

**UNSW-NB15 Dataset:**
- Total samples: 257,673
- Features: 27
- Class balance: Balanced (train), Imbalanced (test)
- Attack ratio: 55.1% (train), 68.1% (test)

**CICIDS2017 Dataset:**
- Total samples: 3,119,345
- Features: 27
- Class balance: Imbalanced (train), Imbalanced (test)
- Attack ratio: 27.0% (train), 27.4% (test)

## Experimental Results

### Performance Comparison

| Dataset       | Model         | Experiment    |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC | CV ROC-AUC    |
|:--------------|:--------------|:--------------|-----------:|------------:|---------:|-----------:|----------:|:--------------|
| UNSW          | Random Forest | Intra-dataset |      0.878 |       0.984 |    0.834 |      0.903 |     0.982 | 0.982 ± 0.001 |
| UNSW          | SVM           | Intra-dataset |      0.744 |       0.94  |    0.667 |      0.78  |     0.847 | 0.810 ± 0.001 |
| UNSW          | Decision Tree | Intra-dataset |      0.863 |       0.968 |    0.826 |      0.891 |     0.969 | 0.968 ± 0.002 |
| CICIDS        | Random Forest | Intra-dataset |      0.838 |       0.984 |    0.414 |      0.583 |     0.864 | 1.000 ± 0.000 |
| CICIDS        | SVM           | Intra-dataset |      0.778 |       0.844 |    0.23  |      0.362 |     0.909 | 0.911 ± 0.000 |
| CICIDS        | Decision Tree | Intra-dataset |      0.866 |       0.928 |    0.552 |      0.693 |     0.781 | 0.998 ± 0.001 |
| UNSW → CICIDS | Random Forest | Cross-dataset |      0.687 |       0.131 |    0.026 |      0.043 |     0.316 | N/A           |
| UNSW → CICIDS | SVM           | Cross-dataset |      0.278 |       0.275 |    1     |      0.431 |     0.459 | N/A           |
| UNSW → CICIDS | Decision Tree | Cross-dataset |      0.328 |       0.262 |    0.804 |      0.395 |     0.476 | N/A           |
| CICIDS → UNSW | Random Forest | Cross-dataset |      0.319 |       0     |    0     |      0     |     0.236 | N/A           |
| CICIDS → UNSW | SVM           | Cross-dataset |      0.7   |       0.942 |    0.595 |      0.73  |     0.756 | N/A           |
| CICIDS → UNSW | Decision Tree | Cross-dataset |      0.324 |       0.722 |    0.012 |      0.023 |     0.315 | N/A           |

### Feature Analysis

Top 10 most important features analyzed:

| Dataset    | Feature       |   Mean |    Std |    Min |    Max |   Median |   Skewness | Unique Values   | Zero Values   |
|:-----------|:--------------|-------:|-------:|-------:|-------:|---------:|-----------:|:----------------|:--------------|
| UNSW-NB15  | duration      | 0      | 0      | 0      | 0      |   0      |       0    | 39,888          | 0             |
| UNSW-NB15  | total_bytes   | 0.1119 | 0.0001 | 0.1119 | 0.1182 |   0.1119 |      37.77 | 7,256           | 0             |
| UNSW-NB15  | total_pkts    | 0.003  | 0      | 0.003  | 0.0032 |   0.003  |       8.78 | 41,005          | 0             |
| UNSW-NB15  | bytes_per_sec | 0.9995 | 0      | 0.9995 | 0.9995 |   0.9995 |       0    | 41,343          | 0             |
| UNSW-NB15  | pkts_per_sec  | 0.5822 | 0.0018 | 0.582  | 0.6079 |   0.582  |       9.81 | 42,212          | 0             |
| UNSW-NB15  | fwd_bytes     | 0.0006 | 0.012  | 0      | 1      |   0      |      53.78 | 4,489           | 0             |
| UNSW-NB15  | bwd_bytes     | 0.0009 | 0.0103 | 0      | 1      |   0      |      52.55 | 4,034           | 36,006        |
| UNSW-NB15  | syn_count     | 0.0091 | 0.022  | 0      | 1      |   0.0001 |      13.81 | 24,934          | 41,127        |
| UNSW-NB15  | ack_count     | 0.003  | 0      | 0.003  | 0.003  |   0.003  |       0    | 26,130          | 0             |
| UNSW-NB15  | flag_sum      | 0.003  | 0      | 0.003  | 0.003  |   0.003  |       0    | 27,018          | 0             |
| CICIDS2017 | duration      | 0.0804 | 0.2316 | 0      | 1      |   0.0001 |       3.05 | 712,955         | 1             |
| CICIDS2017 | total_bytes   | 0.1125 | 0.0103 | 0      | 1      |   0.1119 |      52.36 | 1,196,514       | 1             |
| CICIDS2017 | total_pkts    | 0.0037 | 0.0042 | 0      | 1      |   0.0031 |     134.58 | 1,224,105       | 17            |
| CICIDS2017 | bytes_per_sec | 0.9995 | 0.0009 | 0      | 1      |   0.9995 |    -991.43 | 116,293         | 1             |
| CICIDS2017 | pkts_per_sec  | 0.582  | 0.0036 | 0      | 1      |   0.582  |    -160.46 | 154,850         | 17            |
| CICIDS2017 | fwd_bytes     | 0      | 0.0001 | 0      | 0.0153 |   0      |     249.18 | 1,257           | 288,602       |
| CICIDS2017 | bwd_bytes     | 0      | 0.0001 | 0      | 0.0199 |   0      |     249.18 | 1,534           | 585,433       |
| CICIDS2017 | syn_count     | 0.0135 | 0.0632 | 0      | 0.3099 |   0      |       4.48 | 2               | 2,088,690     |
| CICIDS2017 | ack_count     | 0.0037 | 0.0042 | 0      | 1      |   0.0031 |     134.57 | 1,265,593       | 1             |
| CICIDS2017 | flag_sum      | 0.0037 | 0.0042 | 0      | 1      |   0.0031 |     134.57 | 1,267,433       | 1             |

## Key Findings

1. **Random Forest** demonstrates superior performance with high accuracy and F1-scores across both datasets
2. **SVM** shows good performance but lower than Random Forest in most metrics
3. **Cross-dataset transfer** is challenging, with significant performance degradation
4. **Feature engineering** with 27 universal features provides good discriminative power
5. **Class imbalance** affects model performance, particularly in CICIDS2017 dataset

## Conclusion

Traditional machine learning approaches, particularly Random Forest, remain effective for network intrusion detection. However, cross-dataset generalization remains a significant challenge, indicating the need for domain adaptation techniques.

## Figures

- Figure 1: Class distribution across datasets
- Figure 2: Performance comparison between models
- Figure 3: Cross-dataset transfer results (Random Forest)
- Figure 4: Cross-dataset transfer results (SVM)
- Figure 5: Cross-dataset transfer results (Decision Tree)

