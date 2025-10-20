from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def build_rf(n_estimators: int = 100, max_depth: int = 10,
             min_samples_split: int = 20, min_samples_leaf: int = 10,
             max_features: str = "sqrt", n_jobs: int = -1, random_state: int = 42):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_jobs=n_jobs, random_state=random_state
    )

def build_svm(C: float = 1.0, max_iter: int = 10000, tol: float = 1e-4):
    """
    Constrói um modelo Linear Support Vector Classifier (LinearSVC) otimizado para velocidade.
    LinearSVC é muito mais rápido que SVC para kernels lineares em datasets grandes.
    Usa CalibratedClassifierCV para obter probabilidades.
    """
    # LinearSVC é muito mais rápido que SVC para kernel linear
    linear_svc = LinearSVC(C=C, max_iter=max_iter, tol=tol, random_state=42, dual="auto")
    
    # CalibratedClassifierCV permite obter probabilidades do LinearSVC
    # cv=3 é mais rápido que cv=5 mas ainda fornece boa calibração
    return CalibratedClassifierCV(linear_svc, method='sigmoid', cv=3)

def build_xgb(tree_method: str = "auto", n_estimators: int = 100,
              max_depth: int = 4, learning_rate: float = 0.05,
              subsample: float = 0.7, colsample_bytree: float = 0.7,
              reg_alpha: float = 0.1, reg_lambda: float = 1.0,
              n_jobs: int = -1, random_state: int = 42):
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")
    
    params = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
        reg_lambda=reg_lambda, n_jobs=n_jobs,
        random_state=random_state, objective="binary:logistic",
        tree_method=tree_method, eval_metric="auc"
    )
    return xgb.XGBClassifier(**params)

def build_decision_tree(max_depth: int = 10, min_samples_split: int = 20, 
                       min_samples_leaf: int = 10, max_features: str = "sqrt"):
    """Build Decision Tree model with regularization to prevent overfitting."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
