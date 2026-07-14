# agent.py
from dotenv import load_dotenv
from mcp_client import MCPBridge
from providers.claude import ClaudeProvider
from providers.openai import OpenAIProvider

load_dotenv()

SYSTEM_PROMPT = (
    "Kamu asisten yang membantu. "
    "Untuk pertanyaan cuaca, WAJIB gunakan tool 'get_weather', jangan menjawab dari pengetahuanmu sendiri. "
    "Untuk perhitungan matematika, WAJIB gunakan tool 'calculate'."
)

MODELS = {
    "Claude Sonnet": lambda: ClaudeProvider(model="claude-sonnet-4-6"),
    "GPT-4o":        lambda: OpenAIProvider(model="gpt-4o"),
}

class ChatAgent:
    def __init__(self):
        self.bridge = MCPBridge()
        self.raw_tools = []
        self._provider_cache = {}   # model_name -> instance provider (+ tools)

    async def start(self):
        await self.bridge.connect()
        self.raw_tools = await self.bridge.list_tools_raw()

    def _get_provider(self, model_name: str):
        """Cache satu provider per model (tools sudah diformat)."""
        if model_name not in self._provider_cache:
            provider = MODELS[model_name]()
            provider.build_system(SYSTEM_PROMPT)
            tools = provider.format_tools(self.raw_tools)
            self._provider_cache[model_name] = (provider, tools)
        return self._provider_cache[model_name]

    async def send(self, conv: dict, user_msg: str) -> str:
        """Proses satu giliran chat pada percakapan `conv`.
        Memutakhirkan conv['provider_history'] secara in-place."""
        provider, tools = self._get_provider(conv["model_name"])
        history = conv["provider_history"]

        provider.append_user(history, user_msg)

        while True:
            result = await provider.complete(history, tools)
            provider.append_assistant(history, result)

            if not result.tool_calls:
                return result.text

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