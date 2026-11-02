"""RNAQwen-preview, served.

The RNA side (ERNIE-RNA -> FSQ -> code table) stays in `gguf/rna_prompt.py`.  The
language side is the merged Q4_K_M artifact already listening on llama-server.  This
process sits between them: it turns (sequence, question) into exactly the token ids
the GGUF was trained on, calls the backend, and returns the answer with the turn
boundary respected.

    python serve/server.py --port 8100 --backend http://127.0.0.1:8099

Endpoints:
    GET  /                      the tool page
    GET  /health                this service + the backend
    GET  /templates             the sequence templates the model was trained on
    POST /rna/ask               {sequence, question} -> answered
    POST /v1/chat/completions   OpenAI-shaped; a message containing <RNA> and a
                                sequence gets the code channel, anything else is
                                passed through untouched
"""
import argparse
import importlib.util
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
GG = ROOT / "gguf"
STATIC = Path(__file__).resolve().parent / "static"

TEMPLATES = [
    "What are the first 4 nucleotides of this RNA?",
    "What are the first 6 nucleotides of this RNA?",
    "What are the first 8 nucleotides of this RNA?",
    "What are the last 4 nucleotides of this RNA?",
    "What are the last 6 nucleotides of this RNA?",
    "What are the last 8 nucleotides of this RNA?",
    "What are the nucleotides at positions 1 through 6 of this RNA?",
    "What is the dinucleotide at positions 1 and 2 of this RNA?",
    "What is the reverse complement of the first 4 nucleotides?",
    "What is the reverse complement of the first 6 nucleotides?",
]
MAX_NT = 1024
# The answer turn ends at the first newline: answers are 4-8 characters of RNA and
# never contain one, so anything past it is the model replaying its own training
# transcript.  Without this the answer is still correct but the exact-match score
# is not (see reports/gguf_stop_eval.json).
STOP = ["\n", "<|im_end|>", "<|im_start|>"]

ARGS = None
RP = None
LOCK = threading.Lock()
STATE = {"loaded": False, "error": "", "load_s": 0.0}


def _load_rna_side() -> None:
    global RP
    t0 = time.time()
    try:
        spec = importlib.util.spec_from_file_location("rp", GG / "rna_prompt.py")
        rp = importlib.util.module_from_spec(spec)
        sys.modules["rp"] = rp
        spec.loader.exec_module(rp)
        rp.build("AUGCAUGC", "warmup")  # loads ERNIE-RNA + FSQ + code table once
        RP = rp
        STATE["loaded"] = True
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        STATE["error"] = f"{type(exc).__name__}: {exc}"
    STATE["load_s"] = time.time() - t0
    print(f"[rna] loaded={STATE['loaded']} {STATE['load_s']:.1f}s {STATE['error']}",
          flush=True)


def _post(url: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8"))


def _post_stream(url: str, body: dict, timeout: int = 600) -> StreamingResponse:
    """Proxy an SSE chat stream without buffering the whole answer."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    upstream = urllib.request.urlopen(req, timeout=timeout)

    def gen():
        try:
            for line in upstream:
                yield line
        finally:
            upstream.close()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def clean_answer(txt: str) -> str:
    """The model opens the answer turn with "A:"; the eval has always stripped it."""
    t = (txt or "").strip()
    if "A:" in t:
        t = t.split("A:")[-1].strip()
    return t


def normalise(seq: str) -> str:
    s = "".join(c for c in (seq or "").upper() if c.isalpha())
    s = s.replace("T", "U")
    return s[:MAX_NT]


def ask_backend(ids: list, max_new: int, temperature: float) -> dict:
    body = {"prompt": ids, "n_predict": max_new, "temperature": temperature,
            "cache_prompt": False, "stop": STOP}
    t0 = time.time()
    out = _post(f"{ARGS.backend}/completion", body)
    return dict(content=out.get("content") or "",
                stopped=out.get("stop"), predicted=out.get("tokens_predicted"),
                evaluated=out.get("tokens_evaluated"),
                ms=int((time.time() - t0) * 1000))


app = FastAPI(title="RNAQwen-preview")


def _probe(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as f:
            return json.loads(f.read().decode("utf-8")).get("status") == "ok", ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _health() -> dict:
    backend_ok, backend_err = _probe(ARGS.backend)
    chat_url = ARGS.chat_backend or ARGS.backend
    chat_ok, chat_err = (backend_ok, backend_err) if chat_url == ARGS.backend else _probe(chat_url)
    return dict(status="ok" if STATE["loaded"] and backend_ok and chat_ok else "degraded",
                rna_side=dict(loaded=STATE["loaded"], load_s=round(STATE["load_s"], 1),
                              error=STATE["error"]),
                backend=dict(url=ARGS.backend, ok=backend_ok, error=backend_err),
                chat_backend=dict(url=chat_url, ok=chat_ok, error=chat_err),
                model=ARGS.model, max_nt=MAX_NT)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return JSONResponse(_health())


@app.get("/templates")
def templates():
    return {"templates": TEMPLATES, "max_nt": MAX_NT}


@app.post("/rna/ask")
def rna_ask(payload: dict):
    seq = normalise(payload.get("sequence", ""))
    question = (payload.get("question") or "").strip()
    if not seq:
        raise HTTPException(400, "sequence is empty")
    if not question:
        question = TEMPLATES[0]
    if not STATE["loaded"]:
        raise HTTPException(503, f"RNA side not ready: {STATE['error'] or 'loading'}")
    with LOCK:
        ids, ncodes = RP.build(seq, question)
    out = ask_backend(ids, int(payload.get("max_new_tokens", 32)),
                      float(payload.get("temperature", 0.0)))
    return dict(answer=clean_answer(out["content"]), raw=out["content"],
                question=question, nt=len(seq), codes=ncodes, tokens=len(ids),
                ms=out["ms"], stopped=out["stopped"], predicted=out["predicted"],
                evaluated=out["evaluated"])


@app.post("/v1/chat/completions")
def chat_completions(payload: dict):
    """OpenAI-shaped.  A <RNA>...</RNA> block in the last user message is the only
    thing this service interprets; everything else goes to the backend as-is."""
    msgs = payload.get("messages") or []
    last = msgs[-1].get("content", "") if msgs else ""
    seq = ""
    if "<RNA>" in last:
        body = last.split("<RNA>", 1)[1]
        seq = normalise(body.split("</RNA>", 1)[0] if "</RNA>" in body else body)
    if seq:
        question = last.split("</RNA>", 1)[-1].strip() or TEMPLATES[0]
        ans = rna_ask(dict(sequence=seq, question=question,
                           max_new_tokens=payload.get("max_tokens", 32),
                           temperature=payload.get("temperature", 0.0)))
        return {"id": "rnaqwen-1", "object": "chat.completion", "model": ARGS.model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": ans["answer"]}}],
                "usage": {"prompt_tokens": ans["tokens"],
                          "completion_tokens": ans["predicted"] or 0,
                          "total_tokens": ans["tokens"] + (ans["predicted"] or 0)},
                "rnaqwen": {k: ans[k] for k in ("codes", "nt", "ms")}}
    # The backend runs with --reasoning on. Without this, chat turns come back as
    # reasoning_content with an empty content field (the model thinks, then hits
    # max_tokens before the answer). Clients can still opt in explicitly.
    kwargs = payload.setdefault("chat_template_kwargs", {})
    kwargs.setdefault("enable_thinking", False)
    chat_url = ARGS.chat_backend or ARGS.backend
    if payload.get("stream"):
        return _post_stream(f"{chat_url}/v1/chat/completions", payload)
    return JSONResponse(_post(f"{chat_url}/v1/chat/completions", payload))


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--backend", default="http://127.0.0.1:8099")
    ap.add_argument("--chat-backend", default=None,
                    help="general chat backend; defaults to --backend")
    ap.add_argument("--model", default="RNAQwen-preview")
    ap.add_argument("--no-load", action="store_true",
                    help="skip the RNA side (backend-only passthrough)")
    ARGS = ap.parse_args()
    if not ARGS.no_load:
        threading.Thread(target=_load_rna_side, daemon=True).start()
    import uvicorn
    print(f"[serve] http://{ARGS.host}:{ARGS.port}  backend {ARGS.backend}  "
          f"chat {ARGS.chat_backend or ARGS.backend}", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
