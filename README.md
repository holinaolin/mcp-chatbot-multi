## ENV
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

## STEPS TO RUN PROJECT
```bash
1. mkdir mcp-chatbot-multi && cd mcp-chatbot-multi
2. python -m venv venv
3. source venv/Scripts/activate      # Git Bash / Windows
4. pip install anthropic openai "mcp[cli]" python-dotenv streamlit
5. streamlit run app.py
```