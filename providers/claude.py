# providers/claude.py
from anthropic import Anthropic
from providers.base import LLMProvider, ProviderResult, ToolCall

class ClaudeProvider(LLMProvider):
    def __init__(self, model="claude-sonnet-4-6"):
        self.client = Anthropic()
        self.model = model
        self.system_prompt = ""

    def format_tools(self, raw_tools):
        return [{
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["schema"],
        } for t in raw_tools]

    def build_system(self, system_prompt):
        self.system_prompt = system_prompt

    async def complete(self, history, tools) -> ProviderResult:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            tools=tools,
            messages=history,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in resp.content if b.type == "tool_use"
        ]
        # Simpan content sebagai list dict JSON-able (bukan objek SDK)
        serializable = []
        for b in resp.content:
            if b.type == "text":
                serializable.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                serializable.append({
                    "type": "tool_use", "id": b.id,
                    "name": b.name, "input": b.input,
                })
        return ProviderResult(text=text, tool_calls=calls, raw_assistant_msg=serializable)

    def append_user(self, history, text):
        history.append({"role": "user", "content": text})

    def append_assistant(self, history, result: ProviderResult):
        history.append({"role": "assistant", "content": result.raw_assistant_msg})

    def append_tool_results(self, history, results):
        blocks = [{
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": output,
        } for call, output in results]
        history.append({"role": "user", "content": blocks})