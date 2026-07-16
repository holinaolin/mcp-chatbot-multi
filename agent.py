# agent.py
from dotenv import load_dotenv
from mcp_client import MCPBridge
from providers.claude import ClaudeProvider
from providers.openai import OpenAIProvider

load_dotenv()

SYSTEM_PROMPT = (
    "Kamu asisten analisis data sensor yang membantu. "
    "Gunakan tool yang tersedia sesuai kebutuhan, lalu jelaskan hasilnya "
    "dengan bahasa yang mudah dipahami. Jangan menutup jawaban dengan basa-basi "
    "seperti 'Jika ada hal lain yang bisa saya bantu, beri tahu saya'.\n\n"
    "Untuk permintaan analisis yang TIDAK tersedia di tool khusus "
    "(mis. regresi custom, uji hipotesis, model ML yang perlu kamu rancang "
    "sendiri, transformasi data tertentu), gunakan tool 'execute_python': "
    "tulis kode Python-nya sendiri, jalankan, lalu jelaskan hasilnya. "
    "Data sensor tersedia di kode sebagai DataFrame `df` dengan kolom "
    "timestamp, sensor_A, sensor_B, sensor_C, sensor_D. "
    "Selalu gunakan print() di kode untuk menampilkan hasil."
)

MODELS = {
    "Claude Sonnet": lambda: ClaudeProvider(model="claude-sonnet-4-6"),
    "GPT-4o":        lambda: OpenAIProvider(model="gpt-4o"),
}

class ChatAgent:
    def __init__(self):
        self.bridge = MCPBridge()
        self.raw_tools = []
        self._provider_cache = {}

    async def start(self):
        await self.bridge.connect()
        self.raw_tools = await self.bridge.list_tools_raw()

    def _get_provider(self, model_name: str):
        if model_name not in self._provider_cache:
            provider = MODELS[model_name]()
            provider.build_system(SYSTEM_PROMPT)
            tools = provider.format_tools(self.raw_tools)
            self._provider_cache[model_name] = (provider, tools)
        return self._provider_cache[model_name]

    async def send(self, conv: dict, user_msg: str):
        """Proses satu giliran chat. Mengembalikan (teks_jawaban, list_saran)."""
        provider, tools = self._get_provider(conv["model_name"])
        history = conv["provider_history"]

        provider.append_user(history, user_msg)

        while True:
            result = await provider.complete(history, tools)
            provider.append_assistant(history, result)

            if not result.tool_calls:
                # jawaban utama selesai; minta saran lanjutan (panggilan terpisah)
                suggestions = await provider.suggest_followups(history)
                return result.text, suggestions

            executed = []
            for call in result.tool_calls:
                try:
                    output = await self.bridge.call_tool(call.name, call.arguments)
                except Exception as e:
                    output = f"Error: {e}"
                executed.append((call, output))
            provider.append_tool_results(history, executed)

    async def aclose(self):
        await self.bridge.aclose()