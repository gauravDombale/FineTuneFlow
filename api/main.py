from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import os
from mlx_lm import load, generate

app = FastAPI(title="Tool-Calling LLM Demo")

# Lazy load model
model = None
tokenizer = None

def get_model():
    global model, tokenizer
    if model is None:
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        # Try to load DPO adapter if exists, else SFT, else base
        if os.path.exists("adapters_dpo"):
            print("Loading DPO model")
            model, tokenizer = load(model_path, adapter_path="adapters_dpo")
        elif os.path.exists("adapters_sft"):
            print("Loading SFT model")
            model, tokenizer = load(model_path, adapter_path="adapters_sft")
        else:
            print("Loading Base model")
            model, tokenizer = load(model_path)
    return model, tokenizer

class ToolCallRequest(BaseModel):
    query: str
    tools: List[Dict[str, Any]]

@app.post("/tool_call")
async def call_tool(req: ToolCallRequest):
    m, t = get_model()
    
    # Construct system prompt with tools schema
    tools_str = json.dumps(req.tools, indent=4)
    system_prompt = f"You are a helpful assistant with access to the following functions. Use them if required -\n{tools_str}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.query}
    ]
    
    prompt = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(m, t, prompt=prompt, max_tokens=200)
    
    try:
        call_json = json.loads(out)
        return call_json
    except:
        return {"error": "Invalid JSON generated", "raw": out}
