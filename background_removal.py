"""One reusable session per explicit model, serialized across photo/video requests."""
import threading
from rembg import remove, new_session
_lock = threading.Lock()
_sessions = {}

def remove_background_bytes(data, model="u2netp"):
    with _lock:
        if model not in _sessions:
            _sessions[model] = new_session(model)
        return remove(data, session=_sessions[model])
