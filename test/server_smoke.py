#!/usr/bin/env python3
"""Smoke test for a running slc-server, using the official OpenAI client.

    uv run python test/server_smoke.py --base-url http://localhost:8000/v1
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    model = client.models.list().data[0].id
    print(f"model: {model}")

    # 1. non-stream
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Name three primary colors."}],
        max_tokens=48,
    )
    print("[non-stream]", repr(r.choices[0].message.content), r.choices[0].finish_reason, r.usage)

    # 2. stream: measure TTFT and decode speed
    t0 = time.perf_counter()
    ttft, n = None, 0
    text = ""
    for ch in client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Explain what a KV cache is in about 100 words."}],
        max_tokens=160,
        stream=True,
        stream_options={"include_usage": True},
    ):
        if ch.usage:
            n = ch.usage.completion_tokens
        if ch.choices and ch.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            text += ch.choices[0].delta.content
    total = time.perf_counter() - t0
    print(f"[stream] ttft={ttft:.2f}s total={total:.2f}s tokens={n} "
          f"decode={(n - 1) / (total - ttft):.1f} tok/s")
    print("         ", repr(text[:120]))

    # 3. stop sequence + extra unknown fields
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 10 separated by commas."}],
        max_tokens=48,
        temperature=0,
        stop=["5"],
        extra_body={"user_id": "u-123", "some_unknown_field": {"x": 1}},
    )
    print("[stop]", repr(r.choices[0].message.content), r.choices[0].finish_reason)

    # 4. thinking mode -> reasoning_content
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 17 + 25?"}],
        max_tokens=400,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    msg = r.choices[0].message
    print("[thinking] reasoning:", repr((getattr(msg, "reasoning_content", None) or "")[:80]),
          "| content:", repr(msg.content[:80]), r.choices[0].finish_reason)

    # 5. tools (should not crash; a 0.6B model may or may not call the tool)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }],
        max_tokens=128,
    )
    print("[tools]", r.choices[0].finish_reason, r.choices[0].message.tool_calls, repr(r.choices[0].message.content))

    # 6. concurrency: requests queue on the single engine worker
    def one(i):
        t = time.perf_counter()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Give me one fun fact about the number {i + 1}."}],
            max_tokens=64,
        )
        return i, time.perf_counter() - t, r.usage.completion_tokens, r.choices[0].message.content

    t = time.perf_counter()
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        results = list(ex.map(one, range(args.concurrency)))
    for i, dt, n, c in results:
        print(f"[concurrent {i}] {dt:.2f}s {n} tok: {c[:70]!r}")
    print(f"[concurrent] wall={time.perf_counter() - t:.2f}s for {args.concurrency} requests")

    health = json.load(urllib.request.urlopen(args.base_url.rsplit("/v1", 1)[0] + "/health"))
    print("[health]", health)


if __name__ == "__main__":
    main()
