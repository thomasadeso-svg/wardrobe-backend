"""One reusable session per explicit model, serialized across photo/video requests."""
import threading
_lock = threading.Lock()
_sessions = {}

def remove_background_bytes(data, model="u2netp"):
    with _lock:
        # rembg imports pymatting/Numba, which can compile kernels on a cold start.
        # Load it only for image processing; health/outfit routes need no models.
        from rembg import remove, new_session

        if model not in _sessions:
            _sessions[model] = new_session(model)
        return remove(data, session=_sessions[model])
