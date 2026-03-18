# Entegrasyon Checklist ve Eksikler

## Frontend -> Backend Uyumu
- [x] `POST /random/bytes` frontendden cagriliyor
- [x] `GET /health` polling ile cagriliyor
- [x] `GET /api/stats` endpointi backendde mevcut
- [x] `POST /chaos/reseed` endpointi backendde mevcut

## Guvenlik ve Dayaniklilik
- [x] `x-api-key` dogrulamasi aktif
- [x] `ENTROPYHUB_API_KEYS` env ile anahtar yonetimi
- [x] Rate limit asiminda `429` + `Retry-After`
- [x] `logs/api_audit.log` JSON satir formati

## CORS
- [x] `CORSMiddleware` aktif
- [x] Origin listesi `ENTROPYHUB_CORS_ORIGINS` env ile yonetiliyor

## API Durum Ekranlari
- [x] Loading gorunumu
- [x] Error gorunumu
- [x] Timeout gorunumu
- [x] Retry davranisi

## Canli Gozlem (2026-03-18)
- Last API Request: `2026-03-18T11:37:26.783Z`
- `/health` Status: `healthy`
- Recent Error Summary:
	- `[auth_failed] /favicon.ico`
	- `[rate_limited] /random/bytes`
	- `[auth_failed] /random/bytes`

## Gozlem Yorumu
- `/health = healthy` sonucu backendin ayakta oldugunu dogrular.
- `/favicon.ico` icin `auth_failed` kaydi teknik olarak beklenebilir (tarayici otomatik istegi). Demo akisinda kritik degil.
- `/random/bytes` icin `rate_limited` kaydi, FixedWindowRateLimiter'in calistigini dogrular.
- `/random/bytes` icin `auth_failed` kaydi, API key kontrolunun calistigini dogrular.
