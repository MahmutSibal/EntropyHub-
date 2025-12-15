# api/main.py
"""
Aether PRNG - FastAPI Web Service
High-performance quantum-seeded random number generation API
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.chaos.nihde import NIHDE
from core.pqc.kyber768 import Kyber768

# Global variables
engine = None
start_time = None
stats = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global engine, start_time, stats
    print("🚀 Initializing Aether PRNG Engine...")
    engine = NIHDE(use_live_qrng=False)
    start_time = time.time()
    stats = {
        'total_bytes': 0,
        'total_latency': 0.0,
        'requests': 0
    }
    print("✓ Aether PRNG Engine ready!")
    yield
    # Shutdown
    print("👋 Shutting down Aether PRNG Engine...")

# Initialize FastAPI
app = FastAPI(
    title="Aether PRNG API",
    description="High-performance quantum-seeded chaotic random number generator",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RandomBytesRequest(BaseModel):
    count: int = Field(default=32, ge=1, le=1000000, description="Number of random bytes to generate")

class RandomBytesResponse(BaseModel):
    bytes: List[int] = Field(description="List of random bytes (0-255)")
    count: int = Field(description="Number of bytes generated")
    timestamp: float = Field(description="Generation timestamp")
    latency_us: float = Field(description="Generation latency in microseconds")

class RandomIntegersRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=100000, description="Number of random integers")
    min_value: int = Field(default=0, description="Minimum value (inclusive)")
    max_value: int = Field(default=100, description="Maximum value (inclusive)")

class RandomIntegersResponse(BaseModel):
    integers: List[int] = Field(description="List of random integers")
    count: int = Field(description="Number of integers generated")
    range: str = Field(description="Range of values")

class RandomFloatsRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=100000, description="Number of random floats")
    min_value: float = Field(default=0.0, description="Minimum value")
    max_value: float = Field(default=1.0, description="Maximum value")

class RandomFloatsResponse(BaseModel):
    floats: List[float] = Field(description="List of random floats")
    count: int = Field(description="Number of floats generated")
    range: str = Field(description="Range of values")

class HealthResponse(BaseModel):
    status: str
    engine: str
    version: str
    uptime_seconds: float

class StatsResponse(BaseModel):
    total_bytes_generated: int
    average_latency_us: float
    requests_served: int
    uptime_seconds: float


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """API root endpoint"""
    return {
        "name": "Aether PRNG API",
        "version": "2.1.0",
        "description": "High-performance quantum-seeded chaotic random number generator",
        "endpoints": {
            "health": "/health",
            "random_bytes": "/random/bytes",
            "random_integers": "/random/integers",
            "random_floats": "/random/floats",
            "random_binary": "/random/binary",
            "crypto_keypair": "/crypto/kyber768/keypair",
            "stats": "/stats"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy",
        engine="Aether NIHDE v2.1 (Rust Core)",
        version="2.1.0",
        uptime_seconds=uptime
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API statistics"""
    uptime = time.time() - start_time
    avg_latency = stats['total_latency'] / stats['requests'] if stats['requests'] > 0 else 0
    
    return StatsResponse(
        total_bytes_generated=stats['total_bytes'],
        average_latency_us=avg_latency,
        requests_served=stats['requests'],
        uptime_seconds=uptime
    )

@app.post("/random/bytes", response_model=RandomBytesResponse)
async def generate_random_bytes(request: RandomBytesRequest):
    """Generate random bytes"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    start = time.perf_counter()
    random_bytes = [engine.decide() for _ in range(request.count)]
    end = time.perf_counter()
    
    latency_us = (end - start) * 1_000_000
    
    # Update stats
    stats['total_bytes'] += request.count
    stats['total_latency'] += latency_us
    stats['requests'] += 1
    
    return RandomBytesResponse(
        bytes=random_bytes,
        count=len(random_bytes),
        timestamp=time.time(),
        latency_us=latency_us
    )

@app.get("/random/bytes/{count}", response_model=RandomBytesResponse)
async def generate_random_bytes_get(count: int = Path(ge=1, le=1000000, description="Number of random bytes to generate")):
    """Generate random bytes (GET method)"""
    return await generate_random_bytes(RandomBytesRequest(count=count))

@app.post("/random/integers", response_model=RandomIntegersResponse)
async def generate_random_integers(request: RandomIntegersRequest):
    """Generate random integers in specified range"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if request.min_value >= request.max_value:
        raise HTTPException(status_code=400, detail="min_value must be less than max_value")
    
    range_size = request.max_value - request.min_value + 1
    integers = []
    
    for _ in range(request.count):
        # Generate enough bytes for the range
        num_bytes = (range_size.bit_length() + 7) // 8
        random_int = 0
        for _ in range(num_bytes):
            random_int = (random_int << 8) | engine.decide()
        
        # Map to range
        value = request.min_value + (random_int % range_size)
        integers.append(value)
    
    stats['requests'] += 1
    
    return RandomIntegersResponse(
        integers=integers,
        count=len(integers),
        range=f"[{request.min_value}, {request.max_value}]"
    )

@app.post("/random/floats", response_model=RandomFloatsResponse)
async def generate_random_floats(request: RandomFloatsRequest):
    """Generate random floats in specified range"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if request.min_value >= request.max_value:
        raise HTTPException(status_code=400, detail="min_value must be less than max_value")
    
    floats = []
    for _ in range(request.count):
        # Generate 8 bytes for double precision
        random_bytes = [engine.decide() for _ in range(8)]
        random_int = sum(b << (8*i) for i, b in enumerate(random_bytes))
        
        # Normalize to [0, 1)
        normalized = random_int / (2**64)
        
        # Scale to range
        value = request.min_value + normalized * (request.max_value - request.min_value)
        floats.append(value)
    
    stats['requests'] += 1
    
    return RandomFloatsResponse(
        floats=floats,
        count=len(floats),
        range=f"[{request.min_value}, {request.max_value})"
    )

@app.get("/random/binary/{num_bits}")
async def generate_random_binary(num_bits: int = Path(ge=1, le=1000000, description="Number of bits to generate")):
    """Generate random binary string"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    num_bytes = (num_bits + 7) // 8
    random_bytes = [engine.decide() for _ in range(num_bytes)]
    
    # Convert to binary string
    binary = ''.join(format(byte, '08b') for byte in random_bytes)[:num_bits]
    
    stats['requests'] += 1
    
    return {
        "binary": binary,
        "length": len(binary),
        "num_ones": binary.count('1'),
        "num_zeros": binary.count('0'),
        "balance": abs(binary.count('1') - binary.count('0')) / len(binary)
    }

@app.get("/random/hex/{num_bytes}")
async def generate_random_hex(num_bytes: int = Path(ge=1, le=100000, description="Number of bytes for hex string")):
    """Generate random hexadecimal string"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    random_bytes = bytes([engine.decide() for _ in range(num_bytes)])
    hex_string = random_bytes.hex()
    
    stats['requests'] += 1
    
    return {
        "hex": hex_string,
        "length": len(hex_string),
        "bytes": num_bytes
    }

@app.post("/crypto/kyber768/keypair")
async def generate_kyber_keypair():
    """Generate Kyber-768 quantum-resistant keypair"""
    try:
        pk, sk = Kyber768.keygen()
        
        return {
            "algorithm": "Kyber-768 (ML-KEM-768)",
            "public_key": pk.hex(),
            "secret_key": sk.hex(),
            "public_key_size": len(pk),
            "secret_key_size": len(sk),
            "security_level": "NIST Level 3 (192-bit quantum security)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key generation failed: {str(e)}")

@app.post("/crypto/kyber768/encapsulate")
async def kyber_encapsulate(public_key_hex: str):
    """Encapsulate shared secret with Kyber-768"""
    try:
        pk = bytes.fromhex(public_key_hex)
        shared_secret, ciphertext = Kyber768.encaps(pk)
        
        return {
            "shared_secret": shared_secret.hex(),
            "ciphertext": ciphertext.hex(),
            "ciphertext_size": len(ciphertext),
            "shared_secret_size": len(shared_secret)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Encapsulation failed: {str(e)}")

@app.get("/benchmark/latency")
async def benchmark_latency(iterations: int = Query(default=10000, ge=100, le=1000000)):
    """Benchmark PRNG latency"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    latencies = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        engine.decide()
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000)  # Convert to microseconds
    
    return {
        "iterations": iterations,
        "mean_latency_us": np.mean(latencies),
        "median_latency_us": np.median(latencies),
        "min_latency_us": np.min(latencies),
        "max_latency_us": np.max(latencies),
        "std_latency_us": np.std(latencies),
        "throughput_MB_per_sec": (iterations / sum(latencies)) * 1_000_000
    }


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   AETHER PRNG API SERVER                      ║
    ║                                                               ║
    ║  High-Performance Quantum-Seeded Chaotic RNG                  ║
    ║  Rust-Optimized Core • Kyber-768 PQC • NIST Compliant       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
