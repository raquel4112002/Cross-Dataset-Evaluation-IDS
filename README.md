# Network Intrusion Detection: Traditional ML vs Cross-Dataset Evaluation

Este projeto implementa um sistema completo de detecção de intrusões em redes usando modelos de machine learning tradicionais (Random Forest e SVM) com avaliação cross-dataset entre os datasets UNSW-NB15 e CICIDS2017.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Uso](#uso)
- [Resultados](#resultados)
- [Arquivos Gerados](#arquivos-gerados)
- [Contribuição](#contribuição)

## 🎯 Visão Geral

O projeto compara a performance de modelos tradicionais de machine learning (Random Forest e SVM) para detecção de intrusões em redes, com foco especial na transferência cross-dataset. Utiliza features universais extraídas de ambos os datasets para permitir comparação direta.

### Características Principais

- **Datasets**: UNSW-NB15 e CICIDS2017
- **Modelos**: Random Forest, SVM e Decision Tree
- **Features**: 27 features universais (após remoção de constantes)
- **Experimentos**: Intra-dataset e Cross-dataset
- **Análise**: Métricas completas, visualizações e relatórios para artigos

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- 8GB+ RAM (recomendado para CICIDS2017)
- 10GB+ espaço em disco

### Dependências

```bash
pip install -r requirements.txt
```

### Dependências Principais

- `pandas` - Manipulação de dados
- `numpy` - Computação numérica
- `scikit-learn` - Modelos de ML
- `matplotlib` - Visualizações
- `seaborn` - Gráficos estatísticos
- `tabulate` - Formatação de tabelas

## 📁 Estrutura do Projeto

```
cybersecuraty/
├── data/
│   └── raw/
│       ├── UNSW-NB15/          # Dataset UNSW-NB15
│       └── CICIDS2017/         # Dataset CICIDS2017
├── src/
│   ├── data_corrected.py       # Carregamento e processamento de dados
│   ├── data.py                 # Features universais e alinhamento
│   ├── experiments.py          # Experimentos intra e cross-dataset
│   ├── models.py               # Definição dos modelos (RF, SVM)
│   ├── eval.py                 # Avaliação de métricas
│   ├── utils.py                # Utilitários
│   └── kaggle_dl.py           # Download automático de datasets
├── outputs/                    # Resultados dos experimentos
├── outputs_article_analysis/   # Análise completa para artigo
├── main.py                     # Script principal
├── run_article_analysis.py     # Análise detalhada para artigo
├── test_cross.py              # Teste de experimentos cross-dataset
└── README.md                  # Este arquivo
```

## 🎮 Uso

### Execução Básica

```bash
# Executar todos os experimentos
python main.py

# Executar apenas análise para artigo
python main.py --run_article_analysis true

# Usar datasets personalizados
python main.py --unsw_dir /path/to/unsw --cicids_dir /path/to/cicids
```

### Parâmetros Disponíveis

```bash
python main.py --help
```

- `--unsw_dir`: Diretório do dataset UNSW-NB15
- `--cicids_dir`: Diretório do dataset CICIDS2017
- `--gpu`: Usar GPU para XGBoost (se disponível)
- `--out_dir`: Diretório de saída
- `--auto_download`: Download automático dos datasets
- `--use_unsw_train_test`: Usar splits oficiais do UNSW
- `--use_cicids_traffic_labelling`: Usar TrafficLabelling do CICIDS
- `--run_article_analysis`: Executar análise completa para artigo

### Análise Detalhada para Artigo

```bash
# Executar apenas a análise para artigo
python run_article_analysis.py
```

## 📊 Resultados

### Performance Intra-Dataset

| Dataset    | Modelo        | Accuracy | F1-Score | ROC-AUC |
| ---------- | ------------- | -------- | -------- | ------- |
| UNSW-NB15  | Random Forest | 87.8%    | 90.3%    | 98.2%   |
| UNSW-NB15  | SVM           | 74.4%    | 78.0%    | 84.7%   |
| UNSW-NB15  | Decision Tree | 86.3%    | 89.1%    | 96.9%   |
| CICIDS2017 | Random Forest | 83.8%    | 58.3%    | 86.4%   |
| CICIDS2017 | SVM           | 77.8%    | 36.2%    | 90.9%   |
| CICIDS2017 | Decision Tree | 86.6%    | 69.3%    | 78.1%   |

### Performance Cross-Dataset

| Transferência | Modelo        | F1-Score |
| ------------- | ------------- | -------- |
| UNSW → CICIDS | Random Forest | 4.3%     |
| UNSW → CICIDS | SVM           | 43.1%    |
| UNSW → CICIDS | Decision Tree | 39.5%    |
| CICIDS → UNSW | Random Forest | 0.0%     |
| CICIDS → UNSW | SVM           | 73.0%    |
| CICIDS → UNSW | Decision Tree | 2.3%     |

### Principais Descobertas

1. **Random Forest** tem melhor performance intra-dataset
2. **SVM** tem melhor transferência cross-dataset
3. **Decision Tree** mostra overfitting mais claramente:
   - **UNSW**: Overfitting controlado (gap 4.5% accuracy)
   - **CICIDS**: Overfitting ALTO (gap 12.1% accuracy)
4. **Cross-dataset transfer** é desafiador, especialmente CICIDS → UNSW
5. **Features universais** (27) são eficazes para comparação
6. **Overfitting** é mais severo em datasets desequilibrados (CICIDS)

## 📁 Arquivos Gerados

### outputs/

- Métricas JSON de cada experimento
- Matrizes de confusão (PNG)
- Resultados de validação cruzada

### outputs_article_analysis/

- **Tabelas CSV**:
  - `dataset_comparison.csv` - Comparação dos datasets
  - `performance_comparison.csv` - Performance dos modelos
  - `feature_analysis.csv` - Análise das features
- **Gráficos PNG**:
  - `class_distribution.png` - Distribuição de classes
  - `performance_comparison.png` - Comparação de performance
  - `cross_dataset_transfer_rf.png` - Transferência RF
  - `cross_dataset_transfer_svm.png` - Transferência SVM
  - `cross_dataset_transfer_dt.png` - Transferência Decision Tree
  - `decision_tree_unsw.png` - Visualização Decision Tree UNSW
  - `decision_tree_cicids.png` - Visualização Decision Tree CICIDS
  - `decision_tree_overfitting_analysis.png` - Análise de Overfitting
- **Relatório**:
  - `article_report.md` - Relatório completo para artigo

## 🔧 Detalhes Técnicos

### Features Universais (27)

Após remoção de features constantes, o sistema utiliza:

- **Básicas**: duration, total_bytes, total_pkts, mean_pkt_size
- **Forward/Backward**: fwd_bytes, bwd_bytes, fwd_pkts, bwd_pkts
- **Timing**: iat_std
- **Flags**: syn_count, ack_count, flag_sum
- **Rede**: ttl, loss, win
- **Estatísticas**: pkt_size_std, fwd_bwd_ratio
- **Log**: Transformações logarítmicas das features principais

### Processamento de Dados

1. **Carregamento**: UNSW usa splits oficiais, CICIDS usa 70/30
2. **Limpeza**: Remoção de valores infinitos e NaN
3. **Features**: Engenharia de 39 features → 27 universais
4. **Normalização**: MinMaxScaler aplicado aos dados combinados
5. **Alinhamento**: Mesmo espaço de features para cross-dataset

### Modelos

- **Random Forest**: n_estimators=100, max_depth=10, regularizado
- **SVM**: LinearSVC + CalibratedClassifierCV, C=1.0, otimizado
- **Decision Tree**: max_depth=10, min_samples_split=20, regularizado
- **Validação**: 5-fold StratifiedKFold para métricas robustas

## 📈 Interpretação dos Resultados

### Intra-Dataset

- **UNSW**: Dataset mais equilibrado, melhor performance geral
- **CICIDS**: Dataset desequilibrado, desafio para detecção de ataques

### Cross-Dataset

- **SVM superior**: Melhor generalização entre datasets
- **Random Forest limitado**: Overfitting aos padrões específicos
- **Decision Tree**: Overfitting severo, especialmente em CICIDS
- **Direção importa**: CICIDS → UNSW mais difícil que UNSW → CICIDS

### Análise de Overfitting

- **UNSW**: Todos os modelos controlam overfitting bem
- **CICIDS**: Decision Tree mostra overfitting severo (12.1% gap accuracy)
- **Random Forest**: Melhor balance entre performance e generalização
- **SVM**: Mais robusto para cross-dataset transfer

## 🚨 Limitações

1. **Cross-dataset transfer** é naturalmente desafiador
2. **Features universais** podem perder informação específica
3. **Desequilíbrio de classes** afeta performance
4. **Tempo de execução** longo para datasets grandes

## 🔮 Trabalho Futuro

1. **Domain Adaptation**: Técnicas para melhorar transferência
2. **Feature Engineering**: Features mais específicas por dataset
3. **Ensemble Methods**: Combinação de modelos
4. **Deep Learning**: Redes neurais para comparação
5. **Real-time Detection**: Otimização para produção

## 📚 Referências

- UNSW-NB15: Moustafa, N., & Slay, J. (2015)
- CICIDS2017: Sharafaldin, I., et al. (2018)
- Cross-dataset evaluation em IDS: Literatura recente

**Nota**: Este projeto foi desenvolvido para pesquisa acadêmica em detecção de intrusões em redes. Os resultados são válidos para os datasets e configurações específicas utilizadas.
