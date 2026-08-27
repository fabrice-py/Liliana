"""Exceptions métier de Liliana.

Chaque exception porte un message technique (``str(exc)``) et un message
``user_message`` compréhensible par l'utilisateur final, affiché tel quel dans
l'interface (cf. README section "Troubleshooting").
"""

from __future__ import annotations


class LilianaError(Exception):
    """Exception de base. Toutes les erreurs applicatives en héritent."""

    user_message = "Liliana encountered an unexpected error."

    def __init__(self, message: str = "", user_message: str | None = None) -> None:
        super().__init__(message or self.user_message)
        if user_message is not None:
            self.user_message = user_message


class ConfigurationError(LilianaError):
    user_message = "Liliana is not configured correctly. Check your .env file."


class LLMError(LilianaError):
    user_message = "Liliana cannot reach the local language model."


class LLMUnavailableError(LLMError):
    user_message = (
        "Liliana cannot reach Ollama. Start it with `ollama serve`, "
        "then pull a model with `ollama pull <model>`."
    )


class ModelNotFoundError(LLMError):
    user_message = (
        "The configured language model is not installed. "
        "Run `ollama pull <model>` and set LLM_MODEL in your .env file."
    )


class STTError(LilianaError):
    user_message = "Liliana could not transcribe your audio."


class STTUnavailableError(STTError):
    user_message = (
        "Speech-to-text is unavailable. Install it with "
        "`pip install faster-whisper`."
    )


class EmptyTranscriptionError(STTError):
    user_message = "Liliana did not hear anything. Please try speaking again."


class TTSError(LilianaError):
    user_message = "Liliana could not speak her answer, but you can still read it."


class TTSUnavailableError(TTSError):
    user_message = (
        "Text-to-speech is unavailable. Install Piper and download a voice "
        "(see `python scripts/download_voices.py`). Liliana will keep "
        "answering in text."
    )


class AudioError(LilianaError):
    user_message = (
        "Liliana cannot access the microphone. "
        "Please check your operating system's microphone permissions and "
        "allow access in your browser."
    )


class DatabaseError(LilianaError):
    user_message = "Liliana could not read or write her local database."
