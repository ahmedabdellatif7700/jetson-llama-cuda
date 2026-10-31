"""Thin REST API front-end for the llama-server running in the sibling container.

- GET  /health  public liveness/readiness check (no auth).
- POST /chat    simple prompt in, text out.
- /v1/*         transparent proxy to llama-server's OpenAI-compatible API,
                including SSE streaming.

Optional API-key auth: set API_KEY in the environment and pass it as the
X-API-Key header on /chat and /v1/* requests.
"""

import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://llama:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip() or "local-model"
# Tells the model what it is; without this models tend to claim they are a cloud service.
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "").strip() or (
    f"You are {MODEL_NAME}, running locally on an NVIDIA Jetson Orin Nano "
    "via llama.cpp. No cloud service is involved."
)

# Hop-by-hop headers must not be forwarded by a proxy (RFC 9110), plus headers
# that describe the original framing/encoding or carry our own credentials.
REQUEST_SKIP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
    "x-api-key",
}
# The response body is re-encoded by httpx/starlette, so drop encoding headers.
RESPONSE_SKIP = REQUEST_SKIP | {"content-encoding"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=LLAMA_BASE_URL,
        timeout=httpx.Timeout(600.0, connect=10.0),
    )
    yield
    await app.state.client.aclose()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def filter_headers(headers, skip: set) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in skip}


def upstream_unavailable(exc: httpx.RequestError) -> HTTPException:
    return HTTPException(status_code=502, detail=f"llama-server unreachable: {exc}")


app = FastAPI(title="Jetson Llama API", version="1.1.0", lifespan=lifespan)
protected = APIRouter(dependencies=[Depends(require_api_key)])


@app.get("/health")
async def health(request: Request):
    """200 when llama-server has a model loaded, 503 otherwise."""
    try:
        r = await request.app.state.client.get("/health", timeout=5.0)
        ready = r.status_code == 200
    except httpx.RequestError:
        ready = False
    return JSONResponse(
        {"status": "ok" if ready else "unavailable", "llama_ready": ready},
        status_code=200 if ready else 503,
    )


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str = SYSTEM_PROMPT
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    model: str
    response: str


@protected.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.prompt},
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": False,
    }
    try:
        r = await request.app.state.client.post("/v1/chat/completions", json=payload)
    except httpx.RequestError as exc:
        raise upstream_unavailable(exc)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    return ChatResponse(
        model=data.get("model", MODEL_NAME),
        response=data["choices"][0]["message"]["content"],
    )


@protected.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_v1(path: str, request: Request):
    client: httpx.AsyncClient = request.app.state.client
    query = request.url.query
    upstream = client.build_request(
        request.method,
        httpx.URL(path=f"/v1/{path}", query=query.encode() if query else None),
        headers=filter_headers(request.headers, REQUEST_SKIP),
        content=await request.body(),
    )
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.RequestError as exc:
        raise upstream_unavailable(exc)

    headers = filter_headers(resp.headers, RESPONSE_SKIP)
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        async def stream():
            # finally also runs when the client disconnects mid-stream (cancellation),
            # so the upstream connection is always released.
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(stream(), status_code=resp.status_code, headers=headers)

    try:
        content = await resp.aread()
    finally:
        await resp.aclose()
    return Response(content=content, status_code=resp.status_code, headers=headers)


app.include_router(protected)
