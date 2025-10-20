import argparse, os
from src.data_corrected import load_and_process_datasets
from src.experiments import run_intra, run_cross
from src.utils import ensure_dir

def parse_args():
    ap = argparse.ArgumentParser(description="IDS Cross-Dataset Evaluation")
    ap.add_argument("--unsw_dir", type=str, default="data/raw/UNSW-NB15",
                    help="Path to UNSW-NB15 CSVs (or will be created)")
    ap.add_argument("--cicids_dir", type=str, default="data/raw/CICIDS2017",
                    help="Path to CICIDS2017 CSVs (or will be created)")
    # Removido sample_per_class - agora usamos todos os dados disponíveis
    ap.add_argument("--gpu", type=lambda x: str(x).lower()=='true', default=False,
                    help="Use GPU for XGBoost if available")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--auto_download", type=lambda x: str(x).lower()=='true', default=False,
                    help="Auto-download datasets from KaggleHub")
    ap.add_argument("--use_unsw_train_test", type=lambda x: str(x).lower()=='true', default=True,
                    help="Use official UNSW train/test splits")
    ap.add_argument("--use_cicids_traffic_labelling", type=lambda x: str(x).lower()=='true', default=True,
                    help="Use CICIDS TrafficLabelling (more complete) instead of MachineLearningCVE")
    ap.add_argument("--run_article_analysis", type=lambda x: str(x).lower()=='true', default=True,
                    help="Run complete analysis for article (tables, graphs, report)")
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    if args.auto_download:
        try:
            from src.kaggle_dl import download_unsw, download_cicids
            os.makedirs(args.unsw_dir, exist_ok=True)
            os.makedirs(args.cicids_dir, exist_ok=True)
            print("[auto_download] Downloading UNSW-NB15 →", args.unsw_dir)
            download_unsw(args.unsw_dir)
            print("[auto_download] Downloading CICIDS2017 →", args.cicids_dir)
            download_cicids(args.cicids_dir)
            print("[auto_download] Done.")
        except Exception as e:
            print("[auto_download] Warning:", e)

    # Load data using correct train/test splits and proper CICIDS structure
    unsw_train, unsw_test, cicids_train, cicids_test, _ = load_and_process_datasets(
        unsw_dir=args.unsw_dir,
        cicids_dir=args.cicids_dir,
        use_unsw_train_test=args.use_unsw_train_test,
        use_cicids_traffic_labelling=args.use_cicids_traffic_labelling
    )

    print(f"\nDataset sizes:")
    print(f"UNSW Train: {unsw_train.shape}, Test: {unsw_test.shape}")
    print(f"CICIDS Train: {cicids_train.shape}, Test: {cicids_test.shape}")

    # Intra-dataset baselines
    for model in ["rf","svm"]:  # Removido xgb por enquanto
        print(f"\n=== {model.upper()} INTRA-DATASET ===")
        print("UNSW:")
        run_intra(unsw_train, unsw_test, model, os.path.join(args.out_dir, "UNSW"), args.gpu)
        print("CICIDS:")
        run_intra(cicids_train, cicids_test, model, os.path.join(args.out_dir, "CICIDS"), args.gpu)

    # Cross-dataset transfer
    for model in ["rf","svm"]:  # Removido xgb por enquanto
        print(f"\n=== {model.upper()} CROSS-DATASET ===")
        print("UNSW -> CICIDS:")
        run_cross(unsw_train, cicids_test, model, os.path.join(args.out_dir, "UNSW_to_CICIDS"), args.gpu)
        print("CICIDS -> UNSW:")
        run_cross(cicids_train, unsw_test, model, os.path.join(args.out_dir, "CICIDS_to_UNSW"), args.gpu)

    # Análise completa para artigo
    if args.run_article_analysis:
        print(f"\n=== EXECUTANDO ANÁLISE COMPLETA PARA ARTIGO ===")
        print("Executando run_article_analysis.py...")
        
        import subprocess
        result = subprocess.run(["python", "run_article_analysis.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Análise para artigo concluída com sucesso!")
            print("Arquivos gerados em: outputs_article_analysis/")
        else:
            print("❌ Erro na análise para artigo:")
            print(result.stderr)

if __name__ == "__main__":
    main()
