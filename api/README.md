# API Usage Guide

## Quick Start

### Install Dependencies
```bash
pip install fastapi uvicorn pydantic
```

### Run the Server
```bash
cd api
python main.py
```

Or with uvicorn directly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health & Status

#### GET /health
Check API health status
```bash
curl http://localhost:8000/health
```

#### GET /stats
Get API statistics
```bash
curl http://localhost:8000/stats
```

### Random Generation

#### POST /random/bytes
Generate random bytes
```bash
curl -X POST http://localhost:8000/random/bytes \
  -H "Content-Type: application/json" \
  -d '{"count": 32}'
```

#### GET /random/bytes/{count}
Generate random bytes (GET method)
```bash
curl http://localhost:8000/random/bytes/32
```

#### POST /random/integers
Generate random integers in range
```bash
curl -X POST http://localhost:8000/random/integers \
  -H "Content-Type: application/json" \
  -d '{"count": 10, "min_value": 0, "max_value": 100}'
```

#### POST /random/floats
Generate random floats in range
```bash
curl -X POST http://localhost:8000/random/floats \
  -H "Content-Type: application/json" \
  -d '{"count": 10, "min_value": 0.0, "max_value": 1.0}'
```

#### GET /random/binary/{num_bits}
Generate random binary string
```bash
curl http://localhost:8000/random/binary/128
```

#### GET /random/hex/{num_bytes}
Generate random hexadecimal string
```bash
curl http://localhost:8000/random/hex/32
```

### Cryptography

#### POST /crypto/kyber768/keypair
Generate Kyber-768 quantum-resistant keypair
```bash
curl -X POST http://localhost:8000/crypto/kyber768/keypair
```

#### POST /crypto/kyber768/encapsulate
Encapsulate shared secret
```bash
curl -X POST http://localhost:8000/crypto/kyber768/encapsulate \
  -H "Content-Type: application/json" \
  -d '{"public_key_hex": "YOUR_PUBLIC_KEY_HEX"}'
```

### Benchmarking

#### GET /benchmark/latency
Benchmark PRNG latency
```bash
curl "http://localhost:8000/benchmark/latency?iterations=10000"
```

## Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Generate random bytes
response = requests.post(f"{BASE_URL}/random/bytes", json={"count": 32})
random_bytes = response.json()["bytes"]
print(f"Random bytes: {random_bytes}")

# Generate random integers
response = requests.post(
    f"{BASE_URL}/random/integers",
    json={"count": 10, "min_value": 1, "max_value": 100}
)
random_ints = response.json()["integers"]
print(f"Random integers: {random_ints}")

# Generate Kyber-768 keypair
response = requests.post(f"{BASE_URL}/crypto/kyber768/keypair")
keypair = response.json()
print(f"Public key size: {keypair['public_key_size']} bytes")

# Benchmark latency
response = requests.get(f"{BASE_URL}/benchmark/latency?iterations=10000")
benchmark = response.json()
print(f"Mean latency: {benchmark['mean_latency_us']:.3f} µs")
```

## JavaScript Client Example

```javascript
// Generate random bytes
fetch('http://localhost:8000/random/bytes/32')
  .then(response => response.json())
  .then(data => console.log('Random bytes:', data.bytes));

// Generate random integers
fetch('http://localhost:8000/random/integers', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({count: 10, min_value: 0, max_value: 100})
})
  .then(response => response.json())
  .then(data => console.log('Random integers:', data.integers));
```

## Response Examples

### Random Bytes
```json
{
  "bytes": [123, 45, 67, 89, 12, 34, 56, 78],
  "count": 8,
  "timestamp": 1702567890.123,
  "latency_us": 45.678
}
```

### Random Integers
```json
{
  "integers": [42, 17, 89, 3, 56, 91, 23, 67, 8, 34],
  "count": 10,
  "range": "[0, 100]"
}
```

### Kyber-768 Keypair
```json
{
  "algorithm": "Kyber-768 (ML-KEM-768)",
  "public_key": "a1b2c3...",
  "secret_key": "d4e5f6...",
  "public_key_size": 1184,
  "secret_key_size": 2400,
  "security_level": "NIST Level 3 (192-bit quantum security)"
}
```

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t aether-api .
docker run -p 8000:8000 aether-api
```

## Security Considerations

- Use HTTPS in production
- Implement rate limiting
- Add authentication for sensitive endpoints
- Monitor resource usage
- Implement request validation
- Set up CORS properly for your domain

## Performance Tips

- Use connection pooling
- Implement caching for non-random endpoints
- Scale horizontally with load balancer
- Monitor latency metrics
- Use async clients for batch requests
