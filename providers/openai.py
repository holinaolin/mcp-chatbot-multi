# providers/openai.py
import json
from openai import OpenAI
from providers.base import (LLMProvider, ProviderResult, ToolCall,
                            SUGGEST_INSTRUCTION, parse_suggestions)

class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.system_prompt = ""

    def format_tools(self, raw_tools):
        out = []
        for t in raw_tools:
            schema = dict(t["schema"])
            schema.setdefault("additionalProperties", False)
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": schema,
                },
            })
        return out

    def build_system(self, system_prompt):
        self.system_prompt = system_prompt

    async def complete(self, history, tools):
        messages = [{"role": "system", "content": self.system_prompt}] + history
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        msg = resp.choices[0].message
        calls = []
        if msg.tool_calls:
            for c in msg.tool_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args))
        return ProviderResult(
            text=msg.content or "",
            tool_calls=calls,
            raw_assistant_msg=msg.model_dump(exclude_none=True),
        )

    async def suggest_followups(self, history):
        """Panggilan terpisah tanpa tools: minta 3 pertanyaan lanjutan."""
        try:
            messages = ([{"role": "system", "content": self.system_prompt}]
                        + history
                        + [{"role": "user", "content": SUGGEST_INSTRUCTION}])
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=256,
            )
            return parse_suggestions(resp.choices[0].message.content or "")
        except Exception:
            return []

    def append_user(self, history, text):
        history.append({"role": "user", "content": text})

    def append_assistant(self, history, result: ProviderResult):
        history.append(result.raw_assistant_msg)

    def append_tool_results(self, history, results):
        for call, output in results:
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            })