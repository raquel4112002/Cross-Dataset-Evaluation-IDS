from __future__ import annotations
import os, numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from .data_corrected import load_and_process_datasets
from .models import build_rf, build_svm, build_xgb, build_decision_tree
from .eval import evaluate_binary
from .utils import ensure_dir

def _make_model(name: str, gpu: bool):
    name = name.lower()
    if name == "rf":
        return build_rf()
    if name == "svm":
        return build_svm()
    if name == "xgb":
        return build_xgb(tree_method=("gpu_hist" if gpu else "hist"))
    if name == "dt" or name == "decision_tree":
        return build_decision_tree()
    raise ValueError(f"Unknown model: {name}")

def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        import numpy as np
        return 1/(1+np.exp(-s))    # logistic to [0,1]
    preds = model.predict(X)
    return preds.astype(float)

def run_intra(df_train, df_test, model_name: str, out_dir: str, gpu: bool):
    """
    Executa experimento intra-dataset usando splits corretos de treino/teste.
    
    Args:
        df_train: DataFrame de treino
        df_test: DataFrame de teste
        model_name: Nome do modelo ('rf', 'svm', 'xgb')
        out_dir: Diretório de saída
        gpu: Se usar GPU para XGBoost
    """
    X_train = df_train.drop(columns=["target"]).values
    y_train = df_train["target"].values
    X_test = df_test.drop(columns=["target"]).values
    y_test = df_test["target"].values
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Validação cruzada no conjunto de treino
    model = _make_model(model_name, gpu)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Treinar modelo final
    model.fit(X_train, y_train)
    yprob = _predict_proba(model, X_test)
    yhat = (yprob >= 0.5).astype(int)
    ensure_dir(out_dir)
    
    # Salvar métricas incluindo CV
    metrics = evaluate_binary(y_test, yprob, yhat, out_dir, f"intra_{model_name}")
    metrics["cv_roc_auc_mean"] = cv_scores.mean()
    metrics["cv_roc_auc_std"] = cv_scores.std()
    
    # Salvar métricas atualizadas
    from .utils import save_json
    save_json(metrics, os.path.join(out_dir, f"intra_{model_name}_metrics.json"))
    
    return metrics

def run_cross(df_train, df_test, model_name: str, out_dir: str, gpu: bool):
    Xtr = df_train.drop(columns=["target"]).values
    ytr = df_train["target"].values
    Xte = df_test.drop(columns=["target"]).values
    yte = df_test["target"].values
    model = _make_model(model_name, gpu)
    model.fit(Xtr, ytr)
    yprob = _predict_proba(model, Xte)
    yhat = (yprob >= 0.5).astype(int)
    ensure_dir(out_dir)
    return evaluate_binary(yte, yprob, yhat, out_dir, f"cross_{model_name}")
