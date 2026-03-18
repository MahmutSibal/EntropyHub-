from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path
from collections import Counter

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, Counter as PromCounter, Gauge, Histogram, generate_latest

from core.chaos.nihde import NIHDE, RUST_CORE_AVAILABLE
from core.pqc.kyber768 import Kyber768

 
DEFAULT_DEMO_API_KEY = "teknofest-local-dev-key"


def _split_csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_api_keys() -> set[str]:
    keys = set(_split_csv_env("ENTROPYHUB_API_KEYS", ""))
    if keys:
        return keys
    return {DEFAULT_DEMO_API_KEY}


def _load_public_path_patterns() -> list[str]:
    default_public = "/,/health,/api/health,/metrics,/favicon.ico,/docs*,/openapi.json,/redoc*"
    return _split_csv_env("ENTROPYHUB_PUBLIC_PATHS", default_public)


def _is_public_path(path: str, method: str, public_patterns: list[str]) -> bool:
    if method == "OPTIONS":
        return True
    for pattern in public_patterns:
        if pattern.endswith("*"):
            if path.startswith(pattern[:-1]):
                return True
        elif path == pattern:
            return True
    return False


def _load_cors_origins() -> list[str]:
    default = "http://localhost:3000,http://127.0.0.1:3000"
    origins = _split_csv_env("ENTROPYHUB_CORS_ORIGINS", default)
    return origins if origins else ["http://localhost:3000"]


def _mask_api_key(api_key: str | None) -> str:
    if not api_key:
        return "missing"
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


class FixedWindowRateLimiter:
    """Simple in-memory fixed-window limiter for demo environments."""

    def __init__(self, limit_per_minute: int = 120):
        self.limit_per_minute = max(1, int(limit_per_minute))
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, int]] = {}

    def allow(self, identity: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            window_start, count = self._store.get(identity, (now, 0))

            if now - window_start >= 60:
                window_start = now
                count = 0

            if count >= self.limit_per_minute:
                retry_after = max(1, int(60 - (now - window_start)))
                return False, retry_after

            self._store[identity] = (window_start, count + 1)
            return True, 0

    def reset(self):
        with self._lock:
            self._store.clear()


def _build_audit_logger() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("entropyhub.audit")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("logs/api_audit.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _shannon_entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    total = len(values)
    counts = Counter(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


class RandomBytesRequest(BaseModel):
    count: int = Field(default=32, ge=1, le=4096)


class RandomIntegersRequest(BaseModel):
    count: int = Field(default=16, ge=1, le=4096)
    min_val: int = Field(default=0)
    max_val: int = Field(default=255)

    @field_validator("max_val")
    @classmethod
    def validate_range(cls, value: int, info):
        min_val = info.data.get("min_val", 0)
        if value < min_val:
            raise ValueError("max_val must be greater than or equal to min_val")
        return value


class ReseedRequest(BaseModel):
    source: str = Field(default="api")


class KyberEncapsRequest(BaseModel):
    public_key_hex: str = Field(min_length=Kyber768.PUBLIC_KEY_SIZE * 2, max_length=Kyber768.PUBLIC_KEY_SIZE * 2)


class KyberDecapsRequest(BaseModel):
    secret_key_hex: str = Field(min_length=Kyber768.SECRET_KEY_SIZE * 2, max_length=Kyber768.SECRET_KEY_SIZE * 2)
    ciphertext_hex: str = Field(min_length=Kyber768.CIPHERTEXT_SIZE * 2, max_length=Kyber768.CIPHERTEXT_SIZE * 2)


class EntropyHubService:
    def __init__(self):
        self._lock = threading.Lock()
        self._engine = NIHDE(use_live_qrng=True, reseed_interval=256)
        self._total_bytes = 0
        self._generation_count = 0

    def random_bytes(self, count: int) -> list[int]:
        with self._lock:
            values = [self._engine.decide() for _ in range(count)]
            self._total_bytes += count
            self._generation_count += 1
            return values

    def random_integers(self, count: int, min_val: int, max_val: int) -> list[int]:
        if min_val == max_val:
            return [min_val] * count
        span = max_val - min_val + 1
        values = self.random_bytes(count)
        return [min_val + (value % span) for value in values]

    def reseed(self, source: str) -> dict:
        with self._lock:
            self._engine.reseed_manual(source=source)
            return self._engine.profile()

    def profile(self) -> dict:
        with self._lock:
            return self._engine.profile()

    def stats(self) -> dict:
        with self._lock:
            profile = self._engine.profile()
            return {
                "total_bytes": self._total_bytes,
                "generation_count": self._generation_count,
                "reseed_count": profile.get("reseed_count", 0),
                "postprocessing": profile.get("postprocessing"),
            }


def _audit_error_summary(limit: int = 10) -> list[dict]:
    log_path = Path("logs") / "api_audit.log"
    if not log_path.exists():
        return []

    with log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    summary: list[dict] = []
    for raw_line in reversed(lines):
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if entry.get("event") in {"auth_failed", "rate_limited"} or int(entry.get("status", 200)) >= 500:
            summary.append(
                {
                    "ts": entry.get("ts"),
                    "event": entry.get("event", "request_error"),
                    "path": entry.get("path"),
                    "status": entry.get("status"),
                    "request_id": entry.get("request_id"),
                }
            )
            if len(summary) >= limit:
                break
    return summary


service = EntropyHubService()
audit_logger = _build_audit_logger()
configured_api_keys = _load_api_keys()
public_path_patterns = _load_public_path_patterns()
cors_origins = _load_cors_origins()
rate_limiter = FixedWindowRateLimiter(limit_per_minute=int(os.getenv("ENTROPYHUB_RATE_LIMIT_PER_MIN", "120")))

REQUEST_COUNT = PromCounter(
    "entropyhub_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "entropyhub_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
AUTH_FAILURES = PromCounter(
    "entropyhub_auth_failures_total",
    "Total failed authentication attempts",
)
RATE_LIMIT_REJECTIONS = PromCounter(
    "entropyhub_rate_limit_rejections_total",
    "Total requests rejected by rate limiting",
)
ACTIVE_REQUESTS = Gauge(
    "entropyhub_active_requests",
    "Active in-flight HTTP requests",
)
GENERATED_BYTES = PromCounter(
    "entropyhub_generated_bytes_total",
    "Total number of random bytes generated by API endpoints",
)

app = FastAPI(
    title="EntropyHub API",
    version="2.2",
    summary="Teknofest-ready chaos RNG and ML-KEM service",
    description=(
        "EntropyHub exposes a Rust-accelerated Rossler chaos engine with "
        "Von Neumann post-processing, live entropy reseeding and ML-KEM-768 helpers."
    ),
)
app.state.rate_limiter = rate_limiter

# React Frontend entegrasyonu için CORS ayarları
# Bu ayar, tarayıcının localhost:3000'den localhost:8000'e istek atmasına izin verir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_and_observability_middleware(request: Request, call_next):
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    path = request.url.path
    method = request.method
    client_ip = request.client.host if request.client else "unknown"

    ACTIVE_REQUESTS.inc()

    api_key = request.headers.get("x-api-key")
    is_public = _is_public_path(path, method, public_path_patterns)

    if not is_public:
        if not api_key or api_key not in configured_api_keys:
            AUTH_FAILURES.inc()
            REQUEST_COUNT.labels(method=method, path=path, status="401").inc()
            audit_logger.info(
                json.dumps(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "event": "auth_failed",
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "client_ip": client_ip,
                        "api_key": _mask_api_key(api_key),
                    },
                    ensure_ascii=True,
                )
            )
            ACTIVE_REQUESTS.dec()
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: valid x-api-key is required."},
                headers={"X-Request-ID": request_id, "WWW-Authenticate": "ApiKey"},
            )

    identity = api_key if api_key else client_ip
    allowed, retry_after = app.state.rate_limiter.allow(identity)
    if not allowed:
        RATE_LIMIT_REJECTIONS.inc()
        REQUEST_COUNT.labels(method=method, path=path, status="429").inc()
        audit_logger.info(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": "rate_limited",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "client_ip": client_ip,
                    "api_key": _mask_api_key(api_key),
                    "retry_after_s": retry_after,
                },
                ensure_ascii=True,
            )
        )
        ACTIVE_REQUESTS.dec()
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded."},
            headers={"Retry-After": str(retry_after), "X-Request-ID": request_id},
        )

    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(method=method, path=path, status=str(status_code)).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
        audit_logger.info(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event": "request",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status": status_code,
                    "latency_ms": round(elapsed * 1000, 3),
                    "client_ip": client_ip,
                    "api_key": _mask_api_key(api_key),
                },
                ensure_ascii=True,
            )
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id
        ACTIVE_REQUESTS.dec()


@app.get("/")
def root():
    return {
        "service": "EntropyHub API",
        "version": app.version,
        "docs": "/docs",
        "postprocessing": "von_neumann",
        "public_paths": public_path_patterns,
        "security": {
            "auth": "x-api-key",
            "rate_limit_per_minute": app.state.rate_limiter.limit_per_minute,
        },
    }


@app.get("/health")
def health():
    profile = service.profile()
    stats = service.stats()
    return {
        "status": "healthy",
        "version": app.version,
        "chaos_system": "Rossler",
        "rust_core_available": bool(RUST_CORE_AVAILABLE),
        "postprocessing": profile["postprocessing"],
        "live_entropy_reseed": profile["use_live_qrng"],
        "reseed_count": profile["reseed_count"],
        "generation_count": stats["generation_count"],
        "total_bytes": stats["total_bytes"],
        "server_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pqc": "ML-KEM-768",
    }


@app.get("/api/health")
def health_legacy():
    return health()


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/chaos/reseed",
    responses={
        401: {"description": "Unauthorized - x-api-key missing or invalid"},
        429: {"description": "Rate limit exceeded", "headers": {"Retry-After": {"schema": {"type": "string"}}}},
    },
)
def chaos_reseed(request: ReseedRequest):
    return service.reseed(request.source)


@app.post("/api/reseed")
def chaos_reseed_legacy(request: ReseedRequest):
    return chaos_reseed(request)


@app.post(
    "/random/bytes",
    responses={
        200: {
            "description": "Random bytes generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "bytes": [147, 203, 89, 11],
                        "count": 4,
                        "entropy_estimate": 2.0,
                        "postprocessing": "von_neumann",
                    }
                }
            },
        },
        401: {"description": "Unauthorized - x-api-key missing or invalid"},
        429: {"description": "Rate limit exceeded", "headers": {"Retry-After": {"schema": {"type": "string"}}}},
        422: {"description": "Validation error"},
    },
)
def random_bytes(
    request: RandomBytesRequest = Body(
        ...,
        examples={
            "default": {
                "summary": "Generate 32 bytes",
                "value": {"count": 32},
            },
            "large": {
                "summary": "Generate 1024 bytes",
                "value": {"count": 1024},
            },
        },
    )
):
    values = service.random_bytes(request.count)
    GENERATED_BYTES.inc(request.count)
    return {
        "bytes": values,
        "count": request.count,
        "entropy_estimate": round(_shannon_entropy(values), 6),
        "postprocessing": "von_neumann",
    }


@app.get("/api/generate")
def random_bytes_legacy(bytes: int = 32):
    values = service.random_bytes(bytes)
    GENERATED_BYTES.inc(bytes)
    return {
        "values": values,
        "bytes": bytes,
        "count": bytes,
        "entropy_estimate": round(_shannon_entropy(values), 6),
        "postprocessing": "von_neumann",
    }


@app.post(
    "/random/integers",
    responses={
        200: {
            "description": "Random integers generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "values": [4, 8, 1, 9],
                        "count": 4,
                        "min_val": 0,
                        "max_val": 10,
                    }
                }
            },
        },
        401: {"description": "Unauthorized - x-api-key missing or invalid"},
        429: {"description": "Rate limit exceeded", "headers": {"Retry-After": {"schema": {"type": "string"}}}},
        422: {"description": "Validation error"},
    },
)
def random_integers(
    request: RandomIntegersRequest = Body(
        ...,
        examples={
            "default": {
                "summary": "Generate range-bound integers",
                "value": {"count": 16, "min_val": 0, "max_val": 255},
            }
        },
    )
):
    GENERATED_BYTES.inc(request.count)
    return {
        "values": service.random_integers(request.count, request.min_val, request.max_val),
        "count": request.count,
        "min_val": request.min_val,
        "max_val": request.max_val,
    }


@app.get("/api/stats")
def stats_legacy():
    stats = service.stats()
    return {
        **stats,
        "last_request_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recent_errors": _audit_error_summary(limit=5),
    }


@app.post("/crypto/kyber768/keypair")
def kyber_keypair():
    public_key, secret_key = Kyber768.keygen()
    return {
        "public_key": public_key.hex(),
        "secret_key": secret_key.hex(),
        "public_key_size": len(public_key),
        "secret_key_size": len(secret_key),
        "algorithm": "ML-KEM-768",
    }


@app.post("/crypto/kyber768/encapsulate")
def kyber_encapsulate(request: KyberEncapsRequest):
    try:
        shared_secret, ciphertext = Kyber768.encaps(bytes.fromhex(request.public_key_hex))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ciphertext": ciphertext.hex(),
        "ciphertext_size": len(ciphertext),
        "shared_secret": shared_secret.hex(),
        "shared_secret_size": len(shared_secret),
        "algorithm": "ML-KEM-768",
    }


@app.post("/crypto/kyber768/decapsulate")
def kyber_decapsulate(request: KyberDecapsRequest):
    try:
        shared_secret = Kyber768.decaps(
            bytes.fromhex(request.secret_key_hex),
            bytes.fromhex(request.ciphertext_hex),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "shared_secret": shared_secret.hex(),
        "shared_secret_size": len(shared_secret),
        "algorithm": "ML-KEM-768",
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Proje kök dizinini Python yoluna ekle (core modülü için)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    uvicorn.run(app, host="0.0.0.0", port=8000)