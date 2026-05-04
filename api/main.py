from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import os
import time
from mlx_lm import load, generate

app = FastAPI(
    title="FineTuneFlow — Tool-Calling LLM Demo",
    description="A fine-tuned Qwen-0.5B model trained with LoRA + DPO to generate strict JSON tool calls.",
    version="1.0.0",
)

# Lazy load model
model = None
tokenizer = None
_adapter_used = "none"

def get_model():
    global model, tokenizer, _adapter_used
    if model is None:
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        if os.path.exists("adapters_dpo"):
            print("Loading DPO model")
            model, tokenizer = load(model_path, adapter_path="adapters_dpo")
            _adapter_used = "dpo"
        elif os.path.exists("adapters_sft"):
            print("Loading SFT model")
            model, tokenizer = load(model_path, adapter_path="adapters_sft")
            _adapter_used = "sft"
        else:
            print("Loading Base model")
            model, tokenizer = load(model_path)
            _adapter_used = "base"
    return model, tokenizer


class ToolCallRequest(BaseModel):
    query: str
    tools: List[Dict[str, Any]]


@app.get("/health")
async def health():
    """Health check and model status endpoint."""
    return {
        "status": "ok",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": _adapter_used if model is not None else "not_loaded",
    }


@app.post("/tool_call")
async def call_tool(req: ToolCallRequest):
    """
    Generate a JSON tool call given a user query and a list of available tool schemas.

    Returns a JSON object: `{"name": "...", "arguments": {...}}`
    """
    m, t = get_model()

    tools_str = json.dumps(req.tools, indent=4)
    system_prompt = (
        f"You are a helpful assistant with access to the following functions. "
        f"Use them if required -\n{tools_str}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.query},
    ]

    prompt = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    start = time.perf_counter()
    out = generate(m, t, prompt=prompt, max_tokens=200)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    try:
        call_json = json.loads(out)
        return JSONResponse(content={**call_json, "_meta": {"latency_ms": latency_ms, "adapter": _adapter_used}})
    except Exception:
        return JSONResponse(
            status_code=422,
            content={"error": "Model did not produce valid JSON", "raw": out, "_meta": {"latency_ms": latency_ms}}
        )
