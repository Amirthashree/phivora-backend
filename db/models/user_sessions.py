from datetime import datetime

def create_session_document(session_id, channel="api"):
    return {
        "session_id": session_id,
        "channel": channel,
        "scan_count": 0,
        "phishing_detected": 0,
        "started_at": datetime.utcnow(),
        "last_active": datetime.utcnow()
    }
