"""FastAPI app exposing an OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from dvllm.server.worker import EngineWorker, Job, SamplingConfig, _partial_suffix_len

logger = logging.getLogger(__name__)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


class APIError(Exception):
    def __init__(self, message: str, status: int = 400, code: str | None = None, param=None):
        super().__init__(message)
        self.message, self.status, self.code, self.param = message, status, code, param


def _err(message, status=400, code=None, param=None, type_="invalid_request_error"):
    return JSONResponse(
        {"error": {"message": message, "type": type_, "param": param, "code": code}},
        status_code=status,
    )


# ---------------------------------------------------------------------- #
# Request parsing
# ---------------------------------------------------------------------- #
def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return str(content)


def _normalize_messages(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise APIError("'messages' must be a non-empty array", param="messages")
    out = []
    for m in raw:
        if not isinstance(m, dict) or "role" not in m:
            raise APIError("each message must be an object with a 'role'", param="messages")
        role = m["role"]
        if role == "developer":
            role = "system"
        msg = {"role": role, "content": _content_to_text(m.get("content"))}
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = (tc or {}).get("function") or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                calls.append({"type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
            msg["tool_calls"] = calls
        if role == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def _num(body, key, default, cast=float, lo=None, hi=None):
    v = body.get(key)
    if v is None:
        return default
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise APIError(f"'{key}' must be a number", param=key)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise APIError(f"'{key}' out of range [{lo}, {hi}]", param=key)
    return v


# ---------------------------------------------------------------------- #
# Output parsing
# ---------------------------------------------------------------------- #
def split_reasoning(text: str, final: bool) -> tuple[str, str]:
    """Split generated text into (reasoning, content).

    While streaming (final=False) any trailing characters that might be the
    start of a <think>/</think> tag are held back.
    """
    stripped = text.lstrip()
    if not stripped.startswith(THINK_OPEN):
        if not final and stripped and THINK_OPEN.startswith(stripped):
            return "", ""  # could still become "<think>"
        # Some templates put <think> in the prompt; handle a bare </think>.
        if THINK_CLOSE in text:
            r, c = text.split(THINK_CLOSE, 1)
            return r.strip(), c.lstrip()
        if not final:
            text = text[: len(text) - _partial_suffix_len(text, [THINK_CLOSE])]
        return "", text
    body = stripped[len(THINK_OPEN):]
    if THINK_CLOSE in body:
        r, c = body.split(THINK_CLOSE, 1)
        return r.strip("\n"), c.lstrip()
    if not final:
        body = body[: len(body) - _partial_suffix_len(body, [THINK_CLOSE])]
    return body.lstrip("\n"), ""


def extract_tool_calls(content: str) -> tuple[str, list[dict]]:
    if TOOL_OPEN not in content:
        return content, []
    calls = []
    for raw in TOOL_CALL_RE.findall(content):
        try:
            obj = json.loads(raw)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if not isinstance(name, str):
                continue
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        except Exception:
            continue
    if not calls:
        return content, []
    return content[: content.index(TOOL_OPEN)].rstrip(), calls


# ---------------------------------------------------------------------- #
# App
# ---------------------------------------------------------------------- #
def create_app(
    worker: EngineWorker,
    served_model_name: str,
    max_context: int,
    default_max_tokens: int,
    api_key: str | None = None,
    device: str = "?",
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        worker.start()
        yield

    app = FastAPI(title="dvllm OpenAI-compatible server", lifespan=lifespan)
    tokenizer = worker.tokenizer
    created = int(time.time())
    stop_token_ids = {
        i
        for i in [tokenizer.eos_token_id, *tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])]
        if isinstance(i, int) and i >= 0
    }

    @app.exception_handler(APIError)
    async def _api_error(_, exc: APIError):
        return _err(exc.message, exc.status, exc.code, exc.param)

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1/"):
            if request.headers.get("authorization", "") != f"Bearer {api_key}":
                return _err("Invalid API key", 401, "invalid_api_key", type_="authentication_error")
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": served_model_name, "device": device, "queue": worker.queue_size()}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "dvllm",
                    "max_model_len": max_context,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise APIError("request body must be valid JSON")
        if not isinstance(body, dict):
            raise APIError("request body must be a JSON object")

        messages = _normalize_messages(body.get("messages"))
        tools = body.get("tools") or None
        if tools is not None and not isinstance(tools, list):
            tools = None
        if body.get("tool_choice") == "none":
            tools = None

        tmpl_kwargs = {"enable_thinking": False}
        extra = body.get("chat_template_kwargs")
        if isinstance(extra, dict):
            tmpl_kwargs.update(extra)
        thinking = bool(tmpl_kwargs.get("enable_thinking"))

        try:
            prompt = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
            )
        except Exception as e:
            if tools is None:
                raise APIError(f"failed to apply chat template: {e}")
            logger.warning("chat template failed with tools (%s); retrying without", e)
            tools = None
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
            )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        n_prompt = len(prompt_ids)
        budget = max_context - n_prompt
        if budget < 1:
            raise APIError(
                f"This model's maximum context length is {max_context} tokens, "
                f"but the prompt is {n_prompt} tokens.",
                code="context_length_exceeded",
                param="messages",
            )
        max_tokens = body.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = min(default_max_tokens, budget)
        else:
            max_tokens = _num({"max_tokens": max_tokens}, "max_tokens", 0, int, lo=1)
            if max_tokens > budget:
                # Clamp rather than fail, so agents with large defaults still work.
                max_tokens = budget

        # Qwen3 recommended defaults (non-thinking: 0.7/0.8, thinking: 0.6/0.95), top_k 20.
        sampling = SamplingConfig(
            temperature=_num(body, "temperature", 0.6 if thinking else 0.7, lo=0.0, hi=2.0),
            top_p=_num(body, "top_p", 0.95 if thinking else 0.8, lo=0.0, hi=1.0) or 1.0,
            top_k=_num(body, "top_k", 20, int, lo=-1),
            presence_penalty=_num(body, "presence_penalty", 0.0, lo=-2.0, hi=2.0),
            frequency_penalty=_num(body, "frequency_penalty", 0.0, lo=-2.0, hi=2.0),
        )
        if sampling.top_k < 0:
            sampling.top_k = 0

        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        stop = [s for s in (stop or []) if isinstance(s, str) and s][:8]

        loop = asyncio.get_running_loop()
        job = Job(
            prompt_ids=prompt_ids,
            max_tokens=int(max_tokens),
            sampling=sampling,
            stop=stop,
            stop_token_ids=stop_token_ids,
            loop=loop,
            events=asyncio.Queue(),
        )
        worker.submit(job)

        req_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = served_model_name
        stream = bool(body.get("stream"))

        def usage(stats):
            return {
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
            }

        if not stream:
            try:
                while True:
                    ev = await job.events.get()
                    if ev[0] == "error":
                        return _err(ev[1], 500, type_="server_error")
                    if ev[0] == "done":
                        break
            except asyncio.CancelledError:
                job.cancelled.set()
                raise
            _, text, finish_reason, stats = ev
            reasoning, content = split_reasoning(text, final=True)
            tool_calls = []
            if tools:
                content, tool_calls = extract_tool_calls(content)
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning_content"] = reasoning
            if tool_calls:
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}
                ],
                "usage": usage(stats),
                "system_fingerprint": None,
            }

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(delta, finish_reason=None, usage_obj=None, choices=True):
            obj = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "system_fingerprint": None,
                "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}]
                if choices
                else [],
            }
            if usage_obj is not None:
                obj["usage"] = usage_obj
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        async def gen():
            sent_r = sent_c = ""

            def deltas(text, final):
                nonlocal sent_r, sent_c
                r, c = split_reasoning(text, final)
                if tools and not final:
                    # don't stream raw <tool_call> markup; it is parsed at the end
                    if TOOL_OPEN in c:
                        c = c[: c.index(TOOL_OPEN)]
                    c = c[: len(c) - _partial_suffix_len(c, [TOOL_OPEN])]
                out = []
                if r.startswith(sent_r) and len(r) > len(sent_r):
                    out.append(chunk({"reasoning_content": r[len(sent_r):]}))
                    sent_r = r
                if c.startswith(sent_c) and len(c) > len(sent_c):
                    out.append(chunk({"content": c[len(sent_c):]}))
                    sent_c = c
                return out

            try:
                yield chunk({"role": "assistant", "content": ""})
                while True:
                    ev = await job.events.get()
                    if ev[0] == "text":
                        for s in deltas(ev[1], final=False):
                            yield s
                    elif ev[0] == "error":
                        err = {"error": {"message": ev[1], "type": "server_error", "param": None, "code": None}}
                        yield f"data: {json.dumps(err)}\n\n"
                        break
                    elif ev[0] == "done":
                        _, text, finish_reason, stats = ev
                        if tools:
                            _, c = split_reasoning(text, final=True)
                            c, calls = extract_tool_calls(c)
                            text_for_content = text if not calls else None
                        else:
                            calls = []
                            text_for_content = text
                        if text_for_content is not None:
                            for s in deltas(text_for_content, final=True):
                                yield s
                        else:
                            r, _ = split_reasoning(text, final=True)
                            if r.startswith(sent_r) and len(r) > len(sent_r):
                                yield chunk({"reasoning_content": r[len(sent_r):]})
                            if c.startswith(sent_c) and len(c) > len(sent_c):
                                yield chunk({"content": c[len(sent_c):]})
                            yield chunk(
                                {
                                    "tool_calls": [
                                        {"index": i, **tc} for i, tc in enumerate(calls)
                                    ]
                                }
                            )
                            finish_reason = "tool_calls"
                        yield chunk({}, finish_reason)
                        if include_usage:
                            yield chunk(None, usage_obj=usage(stats), choices=False)
                        break
                yield "data: [DONE]\n\n"
            except (asyncio.CancelledError, GeneratorExit):
                job.cancelled.set()
                raise
            finally:
                job.cancelled.set()  # no-op if already finished

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
