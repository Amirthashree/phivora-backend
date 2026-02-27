from datetime import datetime

def create_scan_document(input_data, verdict, confidence, final_score, model_breakdown, features, channel="api"):
    return {
        "input": input_data,
        "verdict": verdict,
        "confidence": confidence,
        "final_score": final_score,
        "model_breakdown": model_breakdown,
        "features_used": features,
        "channel": channel,
        "timestamp": datetime.utcnow(),
        "is_phishing": verdict == "PHISHING"
    }
