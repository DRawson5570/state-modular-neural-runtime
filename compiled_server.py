import sys, os, time, uuid, json, argparse
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compiled_engine import CompiledInference

engine: Optional[CompiledInference] = None
_model_id: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class CompileRequest(BaseModel):
    id: str
    text: str

app = FastAPI(title="Compiled Inference Server — No Spoon")


def _make_id():
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": _model_id, "object": "model", "owned_by": "compiled-inference",
        "compiled": {
            "states": engine.stats.compiled_states,
            "tokens": engine.stats.compiled_tokens,
            "ram_mb": engine.stats.compiled_bytes / 1e6,
            "mode": engine.stats.mode,
        },
    }]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            return JSONResponse({"error": {"message": "No user message"}}, 400)
        user_msg = user_messages[-1].content
        model = req.model or _model_id
        resp_id = _make_id()

        if req.stream:
            async def stream_gen():
                for token in engine.chat(user_msg, max_tokens=req.max_tokens,
                                         temperature=req.temperature, top_p=req.top_p, stream=True):
                    chunk = {"id": resp_id, "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {"id": resp_id, "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": model,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_gen(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        response = engine.chat(user_msg, max_tokens=req.max_tokens,
                               temperature=req.temperature, top_p=req.top_p)
        return {
            "id": resp_id, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": engine.stats.last_tokens,
                      "total_tokens": engine.stats.last_tokens},
            "compiled": {"states": engine.stats.compiled_states, "tokens": engine.stats.compiled_tokens,
                         "tps": engine.stats.last_tps, "tool_calls": engine.stats.tool_calls},
        }
    except Exception as e:
        return JSONResponse({"error": {"message": str(e), "type": type(e).__name__}}, 500)


@app.post("/v1/compile")
async def compile_content(req: CompileRequest):
    try:
        state = engine.compile(req.id, req.text)
        return {"status": "compiled", "id": req.id, "tokens": state.n_tokens,
                "size_mb": state.size_bytes / 1e6}
    except Exception as e:
        return JSONResponse({"error": {"message": str(e)}}, 500)


@app.post("/v1/compile/file")
async def compile_file(req: Request):
    data = await req.json()
    path = data.get("path", "")
    if not path or not os.path.exists(path):
        return JSONResponse({"error": {"message": f"File not found: {path}"}}, 400)
    with open(path) as f:
        text = f.read()
    state = engine.compile(os.path.basename(path), text)
    return {"status": "compiled", "id": os.path.basename(path), "tokens": state.n_tokens,
            "size_mb": state.size_bytes / 1e6}


@app.delete("/v1/compiled")
async def clear_compiled():
    engine.clear()
    return {"status": "cleared"}


@app.get("/v1/compiled")
async def list_compiled():
    items = []
    for cid, state in engine.library.items():
        items.append({"id": cid, "tokens": state.n_tokens, "size_mb": state.size_bytes / 1e6})
    return {"items": items, "total_mb": sum(i["size_mb"] for i in items)}


@app.get("/v1/engine/stats")
async def get_stats():
    s = engine.stats
    return {"model": _model_id, "mode": s.mode, "turns": s.turns,
            "compiled_states": s.compiled_states, "compiled_tokens": s.compiled_tokens,
            "compiled_mb": s.compiled_bytes / 1e6, "tool_calls": s.tool_calls,
            "last_tps": s.last_tps, "gpu_vram_gb": s.gpu_vram_gb}


def main():
    global engine, _model_id
    parser = argparse.ArgumentParser(description="Compiled Inference Server — No Spoon")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hybrid", default="auto", choices=["auto", "gpu", "hybrid"])
    parser.add_argument("--system", default="", help="System prompt")
    args = parser.parse_args()

    _model_id = args.model.split("/")[-1].lower()
    print(f"Loading {args.model}...")
    engine = CompiledInference(args.model, hybrid=args.hybrid, system_prompt=args.system)
    print(f"Ready: {engine}")
    print(f"Server: http://{args.host}:{args.port}/v1")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
