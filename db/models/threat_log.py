from datetime import datetime

def create_threat_document(input_data, verdict, confidence, model_breakdown, features, channel="api"):
    return {
        "input": input_data,
        "verdict": verdict,
        "confidence": confidence,
        "model_breakdown": model_breakdown,
        "features_used": features,
        "channel": channel,
        "timestamp": datetime.utcnow(),
        "reviewed": False,
        "severity": "HIGH" if confidence > 90 else "MEDIUM" if confidence > 70 else "LOW"
    }
