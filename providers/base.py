# providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# --- Format netral internal ---
# Kita simpan history dalam bentuk netral, tiap provider menerjemahkan
# ke/dari format API-nya sendiri saat dipanggil.

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class ProviderResult:
    """Hasil satu panggilan model."""
    text: str = ""                       # jawaban teks (jika final)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_assistant_msg: object = None     # pesan assistant asli, untuk disimpan ke history

class LLMProvider(ABC):
    @abstractmethod
    def format_tools(self, raw_tools: list[dict]) -> list[dict]:
        """Konversi tools netral → format API provider."""

    @abstractmethod
    def build_system(self, system_prompt: str) -> None:
        """Setup system prompt sesuai konvensi provider."""

    @abstractmethod
    async def complete(self, history: list, tools: list) -> ProviderResult:
        """Panggil model sekali, kembalikan hasil ternormalisasi."""

    @abstractmethod
    def append_user(self, history: list, text: str) -> None: ...

    @abstractmethod
    def append_assistant(self, history: list, result: ProviderResult) -> None: ...

    @abstractmethod
    def append_tool_results(self, history: list, results: list[tuple[ToolCall, str]]) -> None: ...