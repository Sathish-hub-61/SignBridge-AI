import threading
import pyttsx3

_lock = threading.Lock()


def speak(text):
    """Speak in a background thread so the video does not freeze."""
    if not text:
        return

    def run():
        with _lock:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()