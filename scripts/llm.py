#!/usr/bin/env python3
"""Command-line client for the Jetson Llama API (stdlib only).

  llm.py wait   [--timeout S]   block until GET /health returns 200
  llm.py ask    PROMPT          one-shot question via POST /chat
  llm.py chat                   interactive streaming chat with history (/v1)
  llm.py smoke                  end-to-end check: health, auth, chat, streaming, speed

Environment: API_URL (default http://127.0.0.1:8000), API_KEY (optional).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
# Without this models tend to claim they are a cloud service.
SYSTEM_PROMPT = (
    "You are {model}, running locally on an NVIDIA Jetson Orin Nano "
    "via llama.cpp. No cloud service is involved."
)


def request(method, path, body=None, key=API_KEY, timeout=600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API_URL + path, data=data, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def call(method, path, body=None, **kw):
    """Return (status, parsed JSON or text); HTTP errors are returned, not raised."""
    try:
        with request(method, path, body, **kw) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode(errors="replace")


def stream_chat(messages, max_tokens=512, on_token=None):
    """Stream a /v1/chat/completions reply; return (text, timings or None)."""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True}
    text, timings = [], None
    with request("POST", "/v1/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            timings = event.get("timings", timings)
            for choice in event.get("choices", []):
                token = (choice.get("delta") or {}).get("content") or ""
                text.append(token)
                if on_token:
                    on_token(token)
    return "".join(text), timings


def cmd_wait(args):
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        try:
            status, _ = call("GET", "/health", timeout=5)
            if status == 200:
                print("ready")
                return 0
        except (urllib.error.URLError, OSError):
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    print(f"\nnot ready after {args.timeout}s", file=sys.stderr)
    return 1


def cmd_ask(args):
    status, data = call("POST", "/chat", {"prompt": args.prompt, "max_tokens": args.max_tokens})
    if status != 200:
        print(f"HTTP {status}: {data}", file=sys.stderr)
        return 1
    print(data["response"])
    return 0


def served_model():
    """Name of the model llama-server is serving (its --alias), or None."""
    status, data = call("GET", "/v1/models", timeout=10)
    if status == 200 and data.get("data"):
        return data["data"][0]["id"]
    return None


def cmd_chat(args):
    model = served_model() or "a local model"
    system = args.system or SYSTEM_PROMPT.format(model=model)
    messages = [{"role": "system", "content": system}]
    print(f"Chatting with {model}. Ctrl+D or /exit to quit, /reset to clear history.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt in ("/exit", "/quit"):
            return 0
        if prompt == "/reset":
            messages = messages[:1]
            print("(history cleared)")
            continue
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        print("llm> ", end="", flush=True)
        try:
            reply, timings = stream_chat(
                messages, args.max_tokens, on_token=lambda t: print(t, end="", flush=True)
            )
        except urllib.error.HTTPError as e:
            # Most likely the context window (CONTEXT_SIZE) is full.
            print(f"\n[HTTP {e.code}: {e.read().decode(errors='replace')}] try /reset")
            messages.pop()
            continue
        except KeyboardInterrupt:
            print("\n(interrupted)")
            messages.pop()
            continue
        print()
        if timings:
            print(f"[{timings.get('predicted_per_second', 0):.1f} tok/s]")
        messages.append({"role": "assistant", "content": reply})


def cmd_smoke(args):
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    print(f"Smoke test against {API_URL}")

    status, data = call("GET", "/health", timeout=10)
    check("health", status == 200, f"HTTP {status} {data}")
    if status != 200:
        print("API not ready; aborting.")
        return 1

    if API_KEY:
        status, _ = call("POST", "/chat", {"prompt": "hi"}, key="wrong-key")
        check("rejects bad API key", status == 401, f"HTTP {status}")

    status, _ = call("POST", "/chat", {"prompt": ""})
    check("rejects invalid input", status == 422, f"HTTP {status}")

    status, data = call("GET", "/v1/models")
    model = data["data"][0]["id"] if status == 200 and data.get("data") else None
    check("lists model", model is not None, model or f"HTTP {status}")

    t0 = time.time()
    status, data = call("POST", "/chat", {
        "prompt": "What is 17 + 25? Answer with just the number.",
        "max_tokens": 16,
        "temperature": 0,
    })
    answer = data.get("response", "") if status == 200 else str(data)
    check("chat answers correctly", "42" in answer,
          f"{answer.strip()!r} in {time.time() - t0:.1f}s")

    t0 = time.time()
    text, timings = stream_chat(
        [{"role": "user", "content": "Write a Python function that reverses a string."}],
        max_tokens=128,
    )
    check("streaming", len(text) > 20, f"{len(text)} chars in {time.time() - t0:.1f}s")

    if timings:
        pp = timings.get("prompt_per_second", 0)
        tg = timings.get("predicted_per_second", 0)
        # The README reports ~11.7 tok/s on the Orin Nano; well below that means
        # the model is probably not fully offloaded to the GPU.
        check("generation speed", tg >= args.min_tps,
              f"{tg:.1f} tok/s generation, {pp:.1f} tok/s prompt (min {args.min_tps})")

    print("OK" if failures == 0 else f"{failures} check(s) failed")
    return 0 if failures == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("wait")
    w.add_argument("--timeout", type=int, default=300)
    w.set_defaults(func=cmd_wait)

    a = sub.add_parser("ask")
    a.add_argument("prompt")
    a.add_argument("--max-tokens", type=int, default=512)
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("chat")
    c.add_argument("--system", help="system prompt (default: says which local model it is)")
    c.add_argument("--max-tokens", type=int, default=512)
    c.set_defaults(func=cmd_chat)

    s = sub.add_parser("smoke")
    s.add_argument("--min-tps", type=float, default=5.0)
    s.set_defaults(func=cmd_smoke)

    args = p.parse_args()
    try:
        return args.func(args)
    except urllib.error.URLError as e:
        print(f"cannot reach {API_URL}: {e.reason} (is the stack up? try: make up)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
