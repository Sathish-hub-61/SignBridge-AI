"""Mock recognizer implementation for development and testing."""


from app.schemas import RecognizedToken


class MockRecognizer:
    """Provides canned or rule-based token outputs for offline development."""

    def __init__(self) -> None:
        self.default_vocabulary = ["HELP", "PAIN", "DOCTOR", "YES", "NO", "THANK_YOU"]

    def recognize_window(self, frames_tensor: list) -> list[RecognizedToken]:
        """Return dummy recognized tokens."""
        return [RecognizedToken(token="HELP", confidence=0.92, timestamp_ms=0)]
