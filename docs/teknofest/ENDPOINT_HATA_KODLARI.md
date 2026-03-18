# Endpoint ve Hata Kodlari (Kisa)

## Temel Endpoint Listesi
- `GET /health` (public)
- `GET /metrics` (public, Prometheus)
- `POST /random/bytes` (x-api-key gerekli)
- `POST /random/integers` (x-api-key gerekli)
- `GET /api/stats` (x-api-key gerekli, frontend uyumlulugu icin mevcut)
- `POST /chaos/reseed` (x-api-key gerekli, frontend uyumlulugu icin mevcut)

## Public Path Kontrolu
`ENTROPYHUB_PUBLIC_PATHS` env ile wildcard destekli yonetilir.
Varsayilan: `/,/health,/api/health,/metrics,/docs*,/openapi.json,/redoc*`

## Kimlik Dogrulama
- Header: `x-api-key`
- Env: `ENTROPYHUB_API_KEYS` (virgulle ayrilmis)
- Basarisiz auth: `401 Unauthorized`

## Rate Limit
- Uygulama: `FixedWindowRateLimiter`
- Env: `ENTROPYHUB_RATE_LIMIT_PER_MIN`
- Asim durumunda:
  - `429 Too Many Requests`
  - `Retry-After` header doner

## Standart Hata Kodlari
- `400`: Gecersiz icerik (ozellikle PQC endpoint parse hatalari)
- `401`: API key hatasi
- `422`: Request body dogrulama hatasi
- `429`: Rate limit asimi (Retry-After)
- `500+`: Sunucu ici hata

## OpenAPI / Swagger
- `GET /docs`
- `GET /openapi.json`
- `GET /redoc`

`/random/bytes` ve `/random/integers` endpointlerinde ornek request/response tanimlari eklidir.
