# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-tools")

def log(msg: str):
    with open("tool_calls.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluasi ekspresi matematika sederhana, mis. '3 * (4 + 2)'."""
    log(f"[calculate] expression={expression}")
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}
    def ev(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp): return ops[type(node.op)](ev(node.operand))
        raise ValueError("Ekspresi tidak diizinkan")
    return str(ev(ast.parse(expression, mode="eval").body))

@mcp.tool()
def get_weather(city: str) -> str:
    """Ambil cuaca (dummy) untuk sebuah kota."""
    log(f"[get_weather] city={city}")
    data = {"jakarta": "32°C, cerah berawan", "bandung": "24°C, hujan ringan"}
    return data.get(city.lower(), f"Data cuaca untuk {city} tidak tersedia.")

if __name__ == "__main__":
    mcp.run(transport="stdio")