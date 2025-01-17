from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

class ModelProvider(Enum):
    GROQ = "groq"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENAI = "openai"

@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    temperature: float = 0
    max_retries: int = 2 