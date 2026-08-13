from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "all_tickets_processed_improved_v3.csv"
REPORT_DATA_DIR = ROOT_DIR / "reports" / "data"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"
MODEL_SELECTION_PATH = METRICS_DIR / "model_selection.json"
BASELINE_THRESHOLD_PATH = METRICS_DIR / "baseline_selected_threshold.json"
CNN_THRESHOLD_PATH = METRICS_DIR / "selected_threshold.json"
BASELINE_MODEL_PATH = ARTIFACT_DIR / "baseline.joblib"
CNN_MODEL_PATH = ARTIFACT_DIR / "cnn" / "textcnn.pt"
