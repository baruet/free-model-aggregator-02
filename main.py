#!/usr/bin/env python3
"""free-model-aggregator-02: one server, two client wires, real free models.

The server serves the OpenAI Chat Completions API and the Anthropic Messages
API (JSON and server-sent events) and answers each of the trial's six prompts
with a real completion from a live free OpenRouter model, dialled through the
grader's gateway on Chat Completions. It never answers from its own text.

Model choice is by the house's published rating: for each prompt the server
orders the free models that meet the prompt's requirements by rating,
highest first, dials the first, and on a 429 or 5xx fails over to the next
(a 429 is retried once on the same model first). Every refusal is reported
on the response in the x-makefun-upstream-failures header so the failover
column can see it.

Standard library only. Run as `python3 -P -s main.py` (the trial's flags)
or `python3 main.py --list` to print the catalog and exit 0.
"""

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# The fourteen free models: the house's sealed copy of the live OpenRouter
# free catalog (brief, models/v1). Order is the brief's own order; it is the
# order the catalog lists them in. rating_milli is the house's quality
# rating, higher is stronger. `json` means the model supports JSON mode,
# `tools` means it takes tools.
# ---------------------------------------------------------------------------
MODELS = [
    {"id": "openrouter/baidu/ernie-5.5-flash-preview:free",
     "rating_milli": 1150, "context_length": 1024000,
     "tools": True, "vision": False, "json": True, "free": True},
    {"id": "openrouter/cohere/north-mini-code:free",
     "rating_milli": 1200, "context_length": 256000,
     "tools": True, "vision": False, "json": False, "free": True},
    {"id": "openrouter/dots-studio/dots-3-note-preview:free",
     "rating_milli": 1150, "context_length": 512000,
     "tools": True, "vision": True, "json": True, "free": True},
    {"id": "openrouter/google/gemma-4-26b-a4b-it:free",
     "rating_milli": 1150, "context_length": 262144,
     "tools": True, "vision": True, "json": True, "free": True},
    {"id": "openrouter/google/gemma-4-31b-it:free",
     "rating_milli": 1210, "context_length": 262144,
     "tools": True, "vision": True, "json": True, "free": True},
    {"id": "openrouter/inclusionai/ling-3.2-flash:free",
     "rating_milli": 1150, "context_length": 1048576,
     "tools": True, "vision": True, "json": True, "free": True},
    {"id": "openrouter/liquid/lfm-3-1.7b:free",
     "rating_milli": 1150, "context_length": 262144,
     "tools": True, "vision": False, "json": True, "free": True},
    {"id": "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     "rating_milli": 1150, "context_length": 256000,
     "tools": True, "vision": True, "json": False, "free": True},
    {"id": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
     "rating_milli": 1285, "context_length": 1000000,
     "tools": True, "vision": False, "json": False, "free": True},
    {"id": "openrouter/nvidia/nemotron-3.5-lightning:free",
     "rating_milli": 1221, "context_length": 1000000,
     "tools": True, "vision": False, "json": False, "free": True},
    {"id": "openrouter/nvidia/nemotron-3.5-super-vl:free",
     "rating_milli": 1303, "context_length": 1000000,
     "tools": True, "vision": True, "json": True, "free": True},
    {"id": "openrouter/particle-ai/particle-1:free",
     "rating_milli": 1150, "context_length": 1048576,
     "tools": True, "vision": True, "json": True, "free": True},
    {"id": "openrouter/recursive/flint-v1.0:free",
     "rating_milli": 1150, "context_length": 262144,
     "tools": True, "vision": False, "json": True, "free": True},
    {"id": "openrouter/thinkingmachines/inkling-2-small:free",
     "rating_milli": 1475, "context_length": 1048576,
     "tools": True, "vision": True, "json": False, "free": True},
]

PROVIDER_ID = "openrouter"
PROVIDER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_KEY_ENV = "OPENROUTER_API_KEY"

# A gateway call must answer inside the env's 30 s per-prompt cap; 25 s
# leaves room for the reply to travel back.
GATEWAY_TIMEOUT_S = 25
# At most this many gateway calls for one prompt (first tries + failovers).
MAX_ATTEMPTS_PER_PROMPT = 6
# The gateway refuses past 60 calls a trial; six prompts stay far under it.


def catalog_doc(models):
    """The catalog document: one provider, exactly the dialable free ids."""
    return {"providers": [{
        "id": PROVIDER_ID,
        "base_url": PROVIDER_BASE_URL,
        "key_env": PROVIDER_KEY_ENV,
        "wire": "openai",
        "models": [
            {"id": m["id"], "free": True,
             "context_length": m.get("context_length")}
            for m in models
        ],
    }]}


def catalog_bytes(models):
    """The catalog as bytes, one fixed spelling, so --list and the register
    body are byte for byte the same document."""
    return json.dumps(catalog_doc(models), separators=(",", ":")).encode("utf-8")


def empty_catalog_bytes():
    return b'{"providers":[]}'


def load_brief(path):
    """Parse the brief at MAKEFUN_PROMPT_FILE (a text file with two JSON
    lines embedded in it): the free-model list and the prompts manifest.
    Returns (models, prompts); either is None when the file or the line is
    absent. The models list is the house's table for this trial."""
    models = None
    prompts = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    continue
                schema = doc.get("schema", "")
                if schema.endswith("/models/v1") and isinstance(doc.get("models"), list):
                    models = doc["models"]
                elif schema.endswith("/prompts/v1") and isinstance(doc.get("prompts"), list):
                    prompts = doc["prompts"]
    except OSError:
        return None, None
    return models, prompts


def brief_models_or_embedded():
    """The model table for this trial: the brief's list when it is present
    and holds fourteen free ids, else the embedded house table (they agree)."""
    path = os.environ.get("MAKEFUN_PROMPT_FILE")
    if path:
        models, _ = load_brief(path)
        if models and len(models) == 14 and all(
                isinstance(m, dict) and m.get("free") and m.get("id")
                for m in models):
            return models
    return MODELS


def candidates_for(models, requires_json, requires_tools):
    """The models that meet a prompt's requirements, best rating first.
    Ties (the eight at 1150) break by id, so the order is fixed."""
    ok = []
    for m in models:
        if requires_json and not m.get("json"):
            continue
        if requires_tools and not m.get("tools"):
            continue
        ok.append(m)
    ok.sort(key=lambda m: (-int(m.get("rating_milli", 0)), m["id"]))
    return ok


def prompt_tokens_estimate(messages):
    """A lower-bound token estimate for usage fallbacks: ~1 token per 4
    characters of the request text. Only used when the gateway sends no
    usage at all; the wire check needs an input count >= 0."""
    chars = 0
    for msg in messages or []:
        content = msg.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chars += len(part.get("text", ""))
    return max(0, chars // 4)

# ---------------------------------------------------------------------------
# Gateway dialing
# ---------------------------------------------------------------------------

def _error_code(payload):
    """The error code of a gateway error body, for the refusal trail."""
    try:
        doc = json.loads(payload.decode("utf-8", "replace"))
        err = doc.get("error")
        if isinstance(err, dict):
            return str(err.get("code") or err.get("type") or err.get("message") or "error")
        if isinstance(err, str):
            return err
    except Exception:
        pass
    return "error"


def dial_gateway(body, stream):
    """POST one Chat Completions body to the gateway.

    Returns (status, payload_bytes, is_stream) for a 2xx answer, where
    payload_bytes is the whole body for JSON and the raw SSE bytes for a
    stream; or (status, error_bytes, False) for a refusal. A network error
    or a timeout is reported as (599, b'', False). The trial's dummy key is
    sent as the vendor's SDK sends it; the gateway swaps it for the
    entrant's own OpenRouter key, which never enters the sandbox."""
    base = os.environ.get("OPENAI_BASE_URL", "")
    key = os.environ.get("OPENAI_API_KEY", "")
    url = base.rstrip("/") + "/chat/completions"
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Accept", "text/event-stream" if stream else "application/json")
    try:
        with urllib.request.urlopen(req, timeout=GATEWAY_TIMEOUT_S) as resp:
            payload = resp.read()
            return resp.status, payload, bool(stream)
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        except Exception:
            payload = b""
        return exc.code, payload, False
    except Exception:
        return 599, b"", False


def serve_via_gateway(cc_body, models, requires_json, requires_tools):
    """Dial the gateway for one prompt, with retry + failover.

    cc_body is the Chat Completions body (model already "auto"; this
    function sets the dialled id). Returns a dict:
      {"ok": True, "model": <dialled id>, "status": 200,
       "payload": <bytes>, "stream": <bool>, "failures": [...]}
    or {"ok": False, "status": <last status>, "payload": <bytes>,
        "failures": [...]} when every candidate refused. A 429 is retried
    once on the same model before failing over; a 5xx fails over at once;
    any other 4xx (400/401/403) is not transient and ends the prompt."""
    cands = candidates_for(models, requires_json, requires_tools)
    failures = []
    attempts = 0
    last_status = 502
    last_payload = b""
    for model in cands:
        if attempts >= MAX_ATTEMPTS_PER_PROMPT:
            break
        model_id = model["id"]
        retried = False
        while True:
            if attempts >= MAX_ATTEMPTS_PER_PROMPT:
                break
            attempts += 1
            body = dict(cc_body)
            body["model"] = model_id
            status, payload, is_stream = dial_gateway(body, bool(cc_body.get("stream")))
            if status == 200:
                return {"ok": True, "model": model_id, "status": 200,
                        "payload": payload, "stream": is_stream,
                        "failures": failures}
            last_status, last_payload = status, payload
            failures.append({"status": int(status),
                             "code": _error_code(payload),
                             "model": model_id})
            if len(failures) > 8:
                failures = failures[:8]
            if status == 429 and not retried:
                retried = True
                continue
            break
    return {"ok": False, "status": last_status, "payload": last_payload,
            "failures": failures}


def failures_header(failures):
    """The x-makefun-upstream-failures header value, or None when there
    were no refusals to report."""
    if not failures:
        return None
    return json.dumps(failures[:8], separators=(",", ":"))


# ---------------------------------------------------------------------------
# Wire translation: Anthropic <-> Chat Completions
# ---------------------------------------------------------------------------

def anthropic_text_of(content):
    """The plain text of an Anthropic message content (a string, or a list
    of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def anthropic_to_cc(req):
    """Translate an Anthropic Messages request into a Chat Completions body.
    The system prompt becomes a leading system message; Anthropic tools
    become OpenAI function tools. The model is left as the request names
    it ("auto") for serve_via_gateway to replace."""
    messages = []
    system = req.get("system")
    if system is not None:
        text = anthropic_text_of(system) if not isinstance(system, str) else system
        if text:
            messages.append({"role": "system", "content": text})
    for msg in req.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({"type": "image_url", "image_url": {
                            "url": "data:%s;base64,%s" % (
                                src.get("media_type", "image/png"),
                                src.get("data", ""))}})
                    elif src.get("type") == "url":
                        parts.append({"type": "image_url", "image_url": {
                            "url": src.get("url", "")}})
            messages.append({"role": role, "content": parts})
        else:
            messages.append({"role": role, "content": ""})
    body = {
        "model": req.get("model", "auto"),
        "messages": messages,
        "max_tokens": req.get("max_tokens", 512),
    }
    if req.get("temperature") is not None:
        body["temperature"] = req["temperature"]
    if req.get("top_p") is not None:
        body["top_p"] = req["top_p"]
    if req.get("stop_sequences") is not None:
        body["stop"] = req["stop_sequences"]
    tools = req.get("tools")
    if tools:
        body["tools"] = [
            {"type": "function", "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            }} for t in tools if isinstance(t, dict)
        ]
    if req.get("stream"):
        body["stream"] = True
    return body


def openai_finish_to_anthropic(finish_reason):
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop",
    }.get(finish_reason, "end_turn")


def cc_response_to_anthropic(resp, model_id):
    """Translate a Chat Completions answer into an Anthropic Messages
    answer, naming the model really dialled."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    content = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                content.append({"type": "text", "text": part.get("text", "")})
    for call in message.get("tool_calls", []) or []:
        fn = (call or {}).get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        content.append({"type": "tool_use",
                        "id": call.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args})
    if not content:
        content.append({"type": "text", "text": ""})
    usage = resp.get("usage") or {}
    out = {
        "id": "msg_" + str(resp.get("id", "unknown")),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model_id,
        "stop_reason": openai_finish_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }
    return out


def sse_events(payload):
    """Split raw SSE bytes into (event_name, data_string) pairs."""
    text = payload.decode("utf-8", "replace")
    events = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        event = "message"
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        events.append((event, "\n".join(data_lines)))
    return events


def cc_stream_to_openai_events(payload, model_id, include_usage, prompt_tokens):
    """Rewrite an OpenAI SSE stream so every chunk names the model really
    dialled. Appends a usage chunk when the gateway sent none and the
    client asked for usage, then [DONE]. Returns the SSE bytes."""
    out = []
    saw_usage = False
    text_len = 0
    for event, data in sse_events(payload):
        if data.strip() == "[DONE]":
            continue
        try:
            doc = json.loads(data)
        except ValueError:
            continue
        if isinstance(doc, dict):
            doc["model"] = model_id
            if doc.get("usage"):
                saw_usage = True
            for choice in doc.get("choices", []) or []:
                delta = (choice or {}).get("delta", {}) or {}
                if isinstance(delta.get("content"), str):
                    text_len += len(delta["content"])
            out.append("data: " + json.dumps(doc, separators=(",", ":")))
    if include_usage and not saw_usage:
        usage_doc = {
            "id": "chatcmpl-usage",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model_id,
            "choices": [],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": max(1, text_len // 4),
                "total_tokens": int(prompt_tokens) + max(1, text_len // 4),
            },
        }
        out.append("data: " + json.dumps(usage_doc, separators=(",", ":")))
    out.append("data: [DONE]")
    return ("\n\n".join(out) + "\n\n").encode("utf-8")


def cc_stream_to_anthropic_events(payload, model_id, prompt_tokens):
    """Convert an OpenAI SSE stream into Anthropic SSE events (message_start,
    content_block_start, content_block_delta, content_block_stop,
    message_delta, message_stop), naming the model really dialled. The
    streamed prompts carry no tools, so the stream is text only; a tool
    call delta, should one ever appear, is folded into the text as its
    raw JSON so the event order stays valid."""
    out = []
    resp_id = None
    text_parts = []
    stop_reason = "end_turn"
    usage = None
    started = False
    for event, data in sse_events(payload):
        if data.strip() == "[DONE]":
            continue
        try:
            doc = json.loads(data)
        except ValueError:
            continue
        if not isinstance(doc, dict):
            continue
        if resp_id is None and doc.get("id"):
            resp_id = doc["id"]
        if doc.get("usage"):
            usage = doc["usage"]
        for choice in doc.get("choices", []) or []:
            delta = (choice or {}).get("delta", {}) or {}
            if choice.get("finish_reason"):
                stop_reason = openai_finish_to_anthropic(choice["finish_reason"])
            content = delta.get("content")
            if isinstance(content, str) and content:
                if not started:
                    started = True
                    out.append("event: message_start\ndata: " + json.dumps({
                        "type": "message_start",
                        "message": {
                            "id": "msg_" + str(resp_id or "unknown"),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model_id,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": int(prompt_tokens),
                                      "output_tokens": 0},
                        }}, separators=(",", ":")))
                    out.append("event: content_block_start\ndata: " + json.dumps({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }, separators=(",", ":")))
                text_parts.append(content)
                out.append("event: content_block_delta\ndata: " + json.dumps({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content},
                }, separators=(",", ":")))
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                raw = json.dumps(tool_calls, separators=(",", ":"))
                if not started:
                    started = True
                    out.append("event: message_start\ndata: " + json.dumps({
                        "type": "message_start",
                        "message": {
                            "id": "msg_" + str(resp_id or "unknown"),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model_id,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": int(prompt_tokens),
                                      "output_tokens": 0},
                        }}, separators=(",", ":")))
                    out.append("event: content_block_start\ndata: " + json.dumps({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }, separators=(",", ":")))
                text_parts.append(raw)
                out.append("event: content_block_delta\ndata: " + json.dumps({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": raw},
                }, separators=(",", ":")))
    if not started:
        out.append("event: message_start\ndata: " + json.dumps({
            "type": "message_start",
            "message": {
                "id": "msg_" + str(resp_id or "unknown"),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_id,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": int(prompt_tokens),
                          "output_tokens": 0},
            }}, separators=(",", ":")))
        out.append("event: content_block_start\ndata: " + json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }, separators=(",", ":")))
    out_tokens = 0
    if usage and usage.get("completion_tokens") is not None:
        out_tokens = int(usage["completion_tokens"])
    else:
        out_tokens = max(1, sum(len(t) for t in text_parts) // 4)
    out.append("event: content_block_stop\ndata: " + json.dumps({
        "type": "content_block_stop", "index": 0}, separators=(",", ":")))
    out.append("event: message_delta\ndata: " + json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": out_tokens},
    }, separators=(",", ":")))
    out.append("event: message_stop\ndata: " + json.dumps(
        {"type": "message_stop"}, separators=(",", ":")))
    return ("\n\n".join(out) + "\n\n").encode("utf-8")

# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

# The model table for this process, fixed at startup so every route and
# the register body agree on it.
SERVER_MODELS = None


class AggregatorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt, *args):
        pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return None

    def _send_json(self, status, doc, extra_headers=None):
        body = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                if value is not None:
                    self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status, content_type, body, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                if value is not None:
                    self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        if self.path == "/v1/models" or self.path.startswith("/v1/models?"):
            self._handle_models()
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found_error",
                                            "code": "not_found"}})

    def _handle_models(self):
        models = SERVER_MODELS or MODELS
        ids = [m["id"] for m in models]
        if self.headers.get("x-api-key"):
            # Anthropic wire.
            data = [{"id": mid, "type": "model",
                     "display_name": mid.split("/")[-1],
                     "created_at": "2026-11-14T00:00:00Z"} for mid in ids]
            doc = {"data": data, "has_more": False,
                   "first_id": ids[0] if ids else None,
                   "last_id": ids[-1] if ids else None}
        else:
            # OpenAI wire (Authorization: Bearer, or no auth at all).
            data = [{"id": mid, "object": "model", "created": 0,
                     "owned_by": "openrouter"} for mid in ids]
            doc = {"object": "list", "data": data}
        self._send_json(200, doc)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/messages":
            self._handle_messages()
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found_error",
                                            "code": "not_found"}})

    def _handle_chat_completions(self):
        req = self._read_json()
        if req is None:
            self._send_json(400, {"error": {"message": "invalid JSON",
                                            "type": "invalid_request_error",
                                            "code": "invalid_json"}})
            return
        models = SERVER_MODELS or MODELS
        requires_json = bool((req.get("response_format") or {}).get("type") in
                             ("json_object", "json_schema"))
        requires_tools = bool(req.get("tools"))
        cc_body = {
            "model": "auto",
            "messages": req.get("messages", []),
            "max_tokens": req.get("max_tokens") or req.get("max_completion_tokens") or 512,
        }
        for key in ("temperature", "top_p", "response_format", "tools",
                    "stream", "stream_options", "stop", "seed"):
            if req.get(key) is not None:
                cc_body[key] = req[key]
        if cc_body.get("stream"):
            cc_body["stream"] = True
        result = serve_via_gateway(cc_body, models, requires_json, requires_tools)
        extra = {"x-makefun-upstream-failures":
                 failures_header(result.get("failures"))}
        if not result["ok"]:
            self._send_bytes(result["status"] or 502, "application/json",
                             result["payload"] or b"{}", extra)
            return
        payload = result["payload"]
        if result["stream"]:
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            prompt_tokens = prompt_tokens_estimate(cc_body.get("messages"))
            events = cc_stream_to_openai_events(payload, result["model"],
                                                include_usage, prompt_tokens)
            self._send_bytes(200, "text/event-stream", events, extra)
        else:
            try:
                resp = json.loads(payload.decode("utf-8"))
            except ValueError:
                self._send_bytes(502, "application/json", b"{}", extra)
                return
            resp["model"] = result["model"]
            if not resp.get("usage"):
                text = ""
                for choice in resp.get("choices", []) or []:
                    msg = (choice or {}).get("message", {}) or {}
                    if isinstance(msg.get("content"), str):
                        text += msg["content"]
                resp["usage"] = {
                    "prompt_tokens": prompt_tokens_estimate(cc_body.get("messages")),
                    "completion_tokens": max(1, len(text) // 4),
                    "total_tokens": prompt_tokens_estimate(cc_body.get("messages")) + max(1, len(text) // 4),
                }
            self._send_json(200, resp, extra)

    def _handle_messages(self):
        req = self._read_json()
        if req is None:
            self._send_json(400, {"error": {"message": "invalid JSON",
                                            "type": "invalid_request_error",
                                            "code": "invalid_json"}})
            return
        models = SERVER_MODELS or MODELS
        requires_tools = bool(req.get("tools"))
        requires_json = False  # the Anthropic wire has no JSON mode
        cc_body = anthropic_to_cc(req)
        result = serve_via_gateway(cc_body, models, requires_json, requires_tools)
        extra = {"x-makefun-upstream-failures":
                 failures_header(result.get("failures"))}
        if not result["ok"]:
            self._send_bytes(result["status"] or 502, "application/json",
                             result["payload"] or b"{}", extra)
            return
        payload = result["payload"]
        prompt_tokens = prompt_tokens_estimate(cc_body.get("messages"))
        if result["stream"]:
            events = cc_stream_to_anthropic_events(payload, result["model"],
                                                   prompt_tokens)
            self._send_bytes(200, "text/event-stream", events, extra)
        else:
            try:
                resp = json.loads(payload.decode("utf-8"))
            except ValueError:
                self._send_bytes(502, "application/json", b"{}", extra)
                return
            anthropic = cc_response_to_anthropic(resp, result["model"])
            if anthropic["usage"]["output_tokens"] == 0:
                text = "".join(
                    b.get("text", "") for b in anthropic["content"]
                    if b.get("type") == "text")
                anthropic["usage"]["output_tokens"] = max(1, len(text) // 4)
                anthropic["usage"]["input_tokens"] = int(prompt_tokens)
            self._send_json(200, anthropic, extra)


# ---------------------------------------------------------------------------
# Lifecycle: --list, register, serve, poll /work, exit 0
# ---------------------------------------------------------------------------

def pick_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def http_json(method, url, body=None, headers=None, timeout=10):
    data = None
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read()
        except Exception:
            return exc.code, b""
    except Exception:
        return 599, b""


def register_with_env(env_url, port, models):
    """POST /register. Retries a few times; returns True on {"ok": true}."""
    url = env_url.rstrip("/") + "/register"
    body = {"protocol_version": 1, "port": port,
            "catalog": catalog_doc(models)}
    for _ in range(5):
        status, payload = http_json("POST", url, body, timeout=10)
        if status == 200:
            try:
                doc = json.loads(payload.decode("utf-8"))
                if doc.get("ok"):
                    return True
            except ValueError:
                pass
        time.sleep(0.5)
    return False


def work_is_done(env_url):
    status, payload = http_json("GET", env_url.rstrip("/") + "/work", timeout=5)
    if status != 200:
        return False
    try:
        return bool(json.loads(payload.decode("utf-8")).get("done"))
    except ValueError:
        return False


def main(argv):
    global SERVER_MODELS
    if "--list" in argv:
        path = os.environ.get("MAKEFUN_PROMPT_FILE")
        if not path:
            sys.stdout.write(empty_catalog_bytes().decode("utf-8") + "\n")
            return 0
        models = brief_models_or_embedded()
        sys.stdout.write(catalog_bytes(models).decode("utf-8") + "\n")
        return 0

    SERVER_MODELS = brief_models_or_embedded()
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    port = pick_free_port()
    server = ThreadingHTTPServer(("0.0.0.0", port), AggregatorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if env_url:
        register_with_env(env_url, port, SERVER_MODELS)
        # Poll /work until the env has driven every prompt, then exit 0.
        # A process still running at the 240 s wall scores 0, so never
        # wait for anything else.
        while True:
            if work_is_done(env_url):
                break
            time.sleep(0.1)
    else:
        # No env (a local smoke test): serve until killed.
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
