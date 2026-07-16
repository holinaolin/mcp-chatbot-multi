# providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class ProviderResult:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    raw_assistant_msg: object = None

class LLMProvider(ABC):
    @abstractmethod
    def format_tools(self, raw_tools: list[dict]) -> list[dict]: ...

    @abstractmethod
    def build_system(self, system_prompt: str) -> None: ...

    @abstractmethod
    async def complete(self, history: list, tools: list) -> ProviderResult: ...

    @abstractmethod
    def append_user(self, history: list, text: str) -> None: ...

    @abstractmethod
    def append_assistant(self, history: list, result: ProviderResult) -> None: ...

    @abstractmethod
    def append_tool_results(self, history: list, results: list) -> None: ...

    @abstractmethod
    async def suggest_followups(self, history: list) -> list[str]:
        """Berdasarkan riwayat percakapan, hasilkan 2-3 pertanyaan lanjutan
        yang mungkin ingin ditanyakan user. Panggilan ringan & terpisah dari
        jawaban utama, supaya andal dan tidak bergantung format di jawaban."""
        ...


# --- Helper dipakai semua provider ---
SUGGEST_INSTRUCTION = (
    "Berdasarkan percakapan di atas, tuliskan TEPAT 3 pertanyaan lanjutan singkat "
    "yang kemungkinan ingin ditanyakan user berikutnya. Relevan dengan konteks dan "
    "hasil analisis terakhir. Jawab HANYA dengan 3 pertanyaan, satu per baris, "
    "tanpa nomor, tanpa tanda hubung, tanpa penjelasan apa pun."
)

def parse_suggestions(text: str) -> list[str]:
    lines = [l.strip("-•* \t").strip() for l in text.strip().splitlines()]
    return [l for l in lines if l][:3]