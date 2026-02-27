from datetime import datetime

def create_metrics_document(model_name, accuracy, precision, recall, f1, roc_auc=None):
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "timestamp": datetime.utcnow(),
        "training_date": datetime.utcnow().strftime("%Y-%m-%d")
    }
