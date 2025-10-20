from __future__ import annotations
import os, glob
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_unsw_correct(unsw_dir: str, use_train_test_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega UNSW-NB15 usando os splits oficiais de treino/teste.
    
    Args:
        unsw_dir: Diretório com os dados UNSW
        use_train_test_split: Se True, usa train/test oficiais. Se False, concatena tudo.
    
    Returns:
        Tuple[DataFrame, DataFrame]: (train_data, test_data) ou (all_data, None)
    """
    if use_train_test_split:
        # Usar splits oficiais
        train_path = os.path.join(unsw_dir, "UNSW_NB15_training-set.csv")
        test_path = os.path.join(unsw_dir, "UNSW_NB15_testing-set.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Train/test files not found in {unsw_dir}")
        
        print(f"Loading UNSW training set: {train_path}")
        train_df = pd.read_csv(train_path, encoding='latin1')
        
        print(f"Loading UNSW testing set: {test_path}")
        test_df = pd.read_csv(test_path, encoding='latin1')
        
        print(f"UNSW Train shape: {train_df.shape}")
        print(f"UNSW Test shape: {test_df.shape}")
        
        return train_df, test_df
    else:
        # Concatena todos os CSVs (método anterior)
        paths = sorted(glob.glob(os.path.join(unsw_dir, "UNSW-NB15_*.csv")))
        if not paths:
            raise FileNotFoundError(f"No UNSW CSV files found in {unsw_dir}")
        
        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p, low_memory=False, encoding='latin1')
                if df.empty: continue
                dfs.append(df)
            except Exception:
                continue
        
        if not dfs: 
            raise RuntimeError("No readable UNSW CSVs.")
        
        all_df = pd.concat(dfs, ignore_index=True)
        print(f"UNSW All data shape: {all_df.shape}")
        return all_df, None

def load_cicids_correct(cic_dir: str, use_traffic_labelling: bool = True) -> pd.DataFrame:
    """
    Carrega CICIDS2017. Por padrão usa TrafficLabelling (mais completo).
    
    Args:
        cic_dir: Diretório com os dados CICIDS
        use_traffic_labelling: Se True, usa TrafficLabelling. Se False, usa MachineLearningCVE.
    
    Returns:
        DataFrame: Dados concatenados
    """
    if use_traffic_labelling:
        subdir = "TrafficLabelling"
        print("Using CICIDS TrafficLabelling (more complete)")
    else:
        subdir = "MachineLearningCVE"
        print("Using CICIDS MachineLearningCVE")
    
    paths = sorted(glob.glob(os.path.join(cic_dir, subdir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {os.path.join(cic_dir, subdir)}")
    
    print(f"Found {len(paths)} CICIDS files in {subdir}")
    
    dfs = []
    for p in paths:
        df = None
        for kwargs in (
            dict(low_memory=False),
            dict(low_memory=False, encoding="latin1"),
            dict(low_memory=False, engine="python"),
            dict(low_memory=False, encoding="latin1", engine="python", on_bad_lines="skip"),
        ):
            try:
                df = pd.read_csv(p, **kwargs)
                break
            except Exception:
                df = None
        
        if df is None or df.empty:
            print(f"Warning: Could not read {p}")
            continue
        
        # Limpar nomes de colunas
        df.columns = [str(c).strip() for c in df.columns]
        dfs.append(df)
        print(f"Loaded {p}: {df.shape}")
    
    if not dfs:
        raise RuntimeError(f"No readable CICIDS2017 CSVs in {subdir}")
    
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"CICIDS Total shape: {all_df.shape}")
    return all_df

def process_unsw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa dados UNSW para criar labels binárias e limpar features.
    """
    # Detectar coluna de label
    label_col = None
    for c in ["label", "Label", "attack_cat", "Label"]:
        if c in df.columns:
            label_col = c
            break
    
    if label_col is None:
        raise ValueError("No label column found in UNSW data")
    
    print(f"Using label column: {label_col}")
    
    # Criar label binária
    if "attack_cat" in df.columns:
        # UNSW tem attack_cat
        attack_mask = df["attack_cat"].astype(str).str.lower() != "normal"
    else:
        # UNSW tem label binária
        attack_mask = df[label_col].astype(int) == 1
    
    y = attack_mask.astype(int).rename("target")
    
    # Remover colunas não numéricas e de label
    drop_cols = [c for c in ["id", "ID", "attack_cat", "Label", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    
    # Limpar dados
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Combinar features e target
    result = pd.concat([X, y], axis=1)
    
    print(f"Processed UNSW: {result.shape}, Target distribution: {result['target'].value_counts().to_dict()}")
    return result

def process_cicids_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa dados CICIDS para criar labels binárias e limpar features.
    """
    # Detectar coluna de label
    label_col = None
    for c in ["Label", "label", "Attack", "Attack_type"]:
        if c in df.columns:
            label_col = c
            break
    
    if label_col is None:
        raise ValueError("No label column found in CICIDS data")
    
    print(f"Using label column: {label_col}")
    
    # Criar label binária: BENIGN=0, ataque=1
    lab = df[label_col].astype(str).str.strip()
    y = (~lab.str.contains("BENIGN", case=False)).astype(int).rename("target")
    
    # Manter apenas colunas numéricas
    X = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Combinar features e target
    result = pd.concat([X, y], axis=1)
    
    print(f"Processed CICIDS: {result.shape}, Target distribution: {result['target'].value_counts().to_dict()}")
    return result

def sample_data(df: pd.DataFrame, sample_per_class: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Faz amostragem balanceada dos dados.
    """
    if sample_per_class is None:
        return df
    
    parts = []
    for target_val, group in df.groupby("target"):
        if len(group) <= sample_per_class:
            parts.append(group.sample(len(group), random_state=random_state))
        else:
            parts.append(group.sample(sample_per_class, random_state=random_state))
    
    result = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"Sampled data: {result.shape}, Target distribution: {result['target'].value_counts().to_dict()}")
    return result

def engineer_universal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features universais que podem ser aplicadas a ambos os datasets.
    """
    result = df.copy()
    
    # Features básicas de rede
    if 'duration' in result.columns:
        result['duration_log'] = np.log1p(result['duration'])
    
    if 'total_bytes' in result.columns:
        result['total_bytes_log'] = np.log1p(result['total_bytes'])
    
    if 'total_pkts' in result.columns:
        result['total_pkts_log'] = np.log1p(result['total_pkts'])
    
    # Features de forward/backward
    if 'fwd_bytes' in result.columns and 'bwd_bytes' in result.columns:
        result['fwd_bwd_ratio'] = result['fwd_bytes'] / (result['bwd_bytes'] + 1)
        result['fwd_bwd_ratio_log'] = np.log1p(result['fwd_bwd_ratio'])
    
    if 'fwd_pkts' in result.columns and 'bwd_pkts' in result.columns:
        result['fwd_bwd_pkts_ratio'] = result['fwd_pkts'] / (result['bwd_pkts'] + 1)
    
    # Features de timing
    if 'iat_std' in result.columns:
        result['iat_std_log'] = np.log1p(result['iat_std'])
    
    # Features de flags
    flag_cols = ['syn_count', 'ack_count', 'fin_count', 'rst_count', 'psh_count', 'urg_count']
    available_flags = [col for col in flag_cols if col in result.columns]
    if available_flags:
        result['flag_sum'] = result[available_flags].sum(axis=1)
        result['flag_sum_log'] = np.log1p(result['flag_sum'])
    
    # Features de tamanho de pacote
    if 'mean_pkt_size' in result.columns:
        result['mean_pkt_size_log'] = np.log1p(result['mean_pkt_size'])
    
    # Limpar valores infinitos e NaN
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return result

def align_features(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Alinha features entre dois datasets, removendo features constantes e aplicando scaling.
    """
    Xa_raw = df_a.drop(columns=["target"]).copy()
    Xb_raw = df_b.drop(columns=["target"]).copy()
    
    # Limpar dados e remover colunas problemáticas
    Xa_raw = Xa_raw.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xb_raw = Xb_raw.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Remover colunas com caracteres especiais ou que não são features
    problematic_cols_a = [col for col in Xa_raw.columns if 'ï»¿' in col or col.lower() in ['id', 'index']]
    problematic_cols_b = [col for col in Xb_raw.columns if 'ï»¿' in col or col.lower() in ['id', 'index']]
    
    if problematic_cols_a:
        Xa_raw = Xa_raw.drop(columns=problematic_cols_a)
    if problematic_cols_b:
        Xb_raw = Xb_raw.drop(columns=problematic_cols_b)
    
    # Criar features universais
    Xa = engineer_universal_features(Xa_raw)
    Xb = engineer_universal_features(Xb_raw)
    
    # Remover features constantes
    constant_features = []
    for col in Xa.columns:
        if Xa[col].nunique() <= 1 or Xb[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"Removendo {len(constant_features)} features constantes: {constant_features}")
        Xa = Xa.drop(columns=constant_features)
        Xb = Xb.drop(columns=constant_features)
    
    feats = Xa.columns
    print(f"Usando {len(feats)} features após remoção de constantes")
    
    # Combinar dados para fit do scaler
    X_combined = pd.concat([Xa, Xb], ignore_index=True)
    scaler = MinMaxScaler()
    X_combined_scaled = scaler.fit_transform(X_combined[feats].astype(float))
    
    # Separar novamente
    Xa_scaled = X_combined_scaled[:len(Xa)]
    Xb_scaled = X_combined_scaled[len(Xa):]
    
    # Aplicar scaling
    Xa[feats] = Xa_scaled
    Xb[feats] = Xb_scaled
    
    # Reconstruir dataframes com target
    A = pd.concat([Xa, df_a["target"].reset_index(drop=True)], axis=1)
    B = pd.concat([Xb, df_b["target"].reset_index(drop=True)], axis=1)
    
    return A, B, scaler

def load_and_process_datasets(unsw_dir: str, cicids_dir: str, 
                            use_unsw_train_test: bool = True,
                            use_cicids_traffic_labelling: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Carrega e processa ambos os datasets de forma correta com splits apropriados.
    
    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, DataFrame, MinMaxScaler]: 
        (unsw_train, unsw_test, cicids_train, cicids_test, scaler)
    """
    print("=== LOADING UNSW-NB15 ===")
    if use_unsw_train_test:
        # Usar splits oficiais: training set para treino, testing set para teste
        unsw_train_raw, unsw_test_raw = load_unsw_correct(unsw_dir, use_train_test_split=True)
        print(f"UNSW Train: {unsw_train_raw.shape}, Test: {unsw_test_raw.shape}")
        
        unsw_train_processed = process_unsw_data(unsw_train_raw)
        unsw_test_processed = process_unsw_data(unsw_test_raw)
    else:
        # Se não usar splits oficiais, dividir manualmente
        unsw_raw, _ = load_unsw_correct(unsw_dir, use_train_test_split=False)
        unsw_processed = process_unsw_data(unsw_raw)
        
        # Split 70/30
        n = len(unsw_processed)
        train_size = int(0.7 * n)
        unsw_train_processed = unsw_processed.iloc[:train_size].reset_index(drop=True)
        unsw_test_processed = unsw_processed.iloc[train_size:].reset_index(drop=True)
    
    print("\n=== LOADING CICIDS2017 ===")
    cicids_raw = load_cicids_correct(cicids_dir, use_cicids_traffic_labelling)
    cicids_processed = process_cicids_data(cicids_raw)
    
    # Split CICIDS 70/30
    n = len(cicids_processed)
    train_size = int(0.7 * n)
    cicids_train_processed = cicids_processed.iloc[:train_size].reset_index(drop=True)
    cicids_test_processed = cicids_processed.iloc[train_size:].reset_index(drop=True)
    
    print(f"CICIDS Train: {cicids_train_processed.shape}, Test: {cicids_test_processed.shape}")
    
    print("\n=== ALIGNING FEATURES ===")
    # Alinhar features usando dados de treino para fit do scaler
    unsw_train_aligned, cicids_train_aligned, scaler = align_features(unsw_train_processed, cicids_train_processed)
    
    # Para os dados de teste, usar as mesmas features que foram selecionadas no treino
    # Primeiro, processar os dados de teste da mesma forma
    unsw_test_features = engineer_universal_features(unsw_test_processed.drop(columns=["target"]))
    cicids_test_features = engineer_universal_features(cicids_test_processed.drop(columns=["target"]))
    
    # Usar apenas as features que foram mantidas no treino
    train_features = unsw_train_aligned.drop(columns=["target"]).columns
    unsw_test_features = unsw_test_features[train_features]
    cicids_test_features = cicids_test_features[train_features]
    
    # Aplicar o mesmo scaler aos dados de teste
    unsw_test_scaled = scaler.transform(unsw_test_features.astype(float))
    cicids_test_scaled = scaler.transform(cicids_test_features.astype(float))
    
    unsw_test_aligned = pd.concat([
        pd.DataFrame(unsw_test_scaled, columns=train_features),
        unsw_test_processed["target"].reset_index(drop=True)
    ], axis=1)
    
    cicids_test_aligned = pd.concat([
        pd.DataFrame(cicids_test_scaled, columns=train_features),
        cicids_test_processed["target"].reset_index(drop=True)
    ], axis=1)
    
    return unsw_train_aligned, unsw_test_aligned, cicids_train_aligned, cicids_test_aligned, scaler
