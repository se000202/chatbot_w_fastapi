from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
import math
import re
import ast

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# ---- LaTeX 후처리 함수들 ----

def auto_wrap_inline_latex(response: str) -> str:
    inline_latex_pattern = re.compile(r'(\\(?:frac|sqrt|sum|int|log|sin|cos|tan)[^$ \n]*)')

    def replacer(match):
        return f'$$ {match.group(1)} $$'

    response = inline_latex_pattern.sub(replacer, response)
    return response

def auto_wrap_list_latex(response: str) -> str:
    lines = response.split('\n')
    new_lines = []
    for line in lines:
        if re.match(r'^\s*[-*]\s', line) and re.search(r'\\(frac|sqrt|sum|int|log|sin|cos|tan)', line):
            content = line.strip()
            content_no_bullet = re.sub(r'^[-*]\s+', '', content)
            new_lines.append(f'- $$ {content_no_bullet} $$')
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

# ---- GPT 호출 ----

def get_chatbot_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# ---- 안전한 exec 처리 ----

def safe_exec_function(code_str: str) -> str:
    try:
        # AST 파싱
        tree = ast.parse(code_str)

        # 허용 노드
        allowed_nodes = (
            ast.Module, ast.Import, ast.ImportFrom,
            ast.Assign, ast.Expr, ast.Call, ast.Name,
            ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,  # Num <3.8 / Constant >=3.8
            ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Index, ast.Slice,
            ast.Attribute, ast.Compare, ast.If, ast.IfExp, ast.BoolOp,
            ast.And, ast.Or, ast.Not, ast.For, ast.While, ast.Break,
            ast.Continue, ast.Pass, ast.Return
        )

        # 허용된 Import는 math 만
        allowed_imports = {"math"}

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"금지된 노드 발견: {type(node).__name__}")

            # Import 제한
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name != "math":
                        raise ValueError(f"금지된 import 발견: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module != "math":
                    raise ValueError(f"금지된 import from 발견: {node.module}")

        # 안전한 실행 환경
        safe_globals = {
            "__builtins__": {},
            "math": math
        }
        safe_locals = {}

        # 실행
        exec(compile(tree, filename="<safe_exec>", mode="exec"), safe_globals, safe_locals)

        # 결과 찾기
        if "_result" in safe_locals:
            return f"계산 결과: {safe_locals['_result']}"
        else:
            return "✅ 코드 실행 완료 (결과는 별도 출력 없음)."

    except Exception as e:
        return f"코드 실행 중 오류 발생: {e}"

# ---- /chat endpoint ----

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    code_keywords = ["python 코드", "파이썬 코드", "python function", "python program"]

    if any(keyword in last_msg for keyword in code_keywords):
        system_prompt = [
            {"role": "system", "content": "You are an assistant that writes safe Python code to perform mathematical computations. "
                                          "The code must use only standard math functions (via 'import math'), and must not use any external modules. "
                                          "It must be written as executable Python code. "
                                          "If applicable, assign the final result to a variable named _result so that it can be read after execution. "
                                          "Do not use file I/O, OS operations, or network operations. Do not use exec, eval, compile, __import__, open, or any OS-related functions. "
                                          "Only generate pure Python math code."},
            {"role": "user", "content": last_msg}
        ]
        code = get_chatbot_response(system_prompt)
        result = safe_exec_function(code)
        return {"response": f"```\n{code}\n```\n\n{result}"}

    # Default prompt (강화됨)
    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, always surround it with $$...$$. "
                                      "Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math. "
                                      "If your output includes inline LaTeX expressions (\\frac, \\sqrt, \\sum, etc.) in lists or bullet points, also enclose the entire list item with $$...$$. "
                                      "If your output is normal text, do not use $$."}
    ]

    answer = get_chatbot_response(system_prompt_default + messages)
    answer = auto_wrap_inline_latex(answer)
    answer = auto_wrap_list_latex(answer)

    return {"response": answer}
