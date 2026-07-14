# mcp_client.py
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPBridge:
    def __init__(self, server_cmd="python", server_args=("server.py",)):
        self.params = StdioServerParameters(command=server_cmd, args=list(server_args))
        self.session: ClientSession | None = None
        self._stack = AsyncExitStack()

    async def connect(self):
        read, write = await self._stack.enter_async_context(stdio_client(self.params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def list_tools_raw(self) -> list[dict]:
        """Format netral: {name, description, schema}."""
        resp = await self.session.list_tools()
        return [{
            "name": t.name,
            "description": t.description or "",
            "schema": t.inputSchema or {"type": "object", "properties": {}},
        } for t in resp.tools]

    async def call_tool(self, name: str, args: dict) -> str:
        result = await self.session.call_tool(name, args)
        parts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
        return "\n".join(parts) if parts else str(result.content)

    async def aclose(self):
        await self._stack.aclose()