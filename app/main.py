from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import subprocess
import os
import re
import tempfile
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# In-memory recent code storage
recent_code = {"last_code": ""}

# Request model
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# Run Python code safely
def run_python_code(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmpfile:
            tmpfile.write(code)
            tmpfile_path = tmpfile.name

        result = subprocess.run(
            ["python3", tmpfile_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error:\n{result.stderr.strip()}"
    except Exception as e:
        return f"Exception: {str(e)}"

# Improved extract_code_block
def extract_code_block(text: str) -> str:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        if code.lower().startswith('python'):
            code_lines = code.splitlines()
            if len(code_lines) > 1 and code_lines[0].strip() == 'python':
