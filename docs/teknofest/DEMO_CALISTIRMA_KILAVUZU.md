# Demo Calistirma Kilavuzu (1 Sayfa)

## 1) Ortam Hazirligi
- Python bagimliliklari:
  - `pip install -r requirements.txt`
- Frontend bagimliliklari:
  - `cd frontend && npm install`

## 2) Konfigurasyon
- Backend icin proje kokunde `.env` dosyasi olustur:
  - `.env.example` icerigini baz alin
- Frontend icin `frontend/.env` dosyasi olustur:
  - `frontend/.env.example` icerigini baz alin

## 3) Baslatma
### Manuel
- API:
  - `uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`
- Frontend:
  - `cd frontend`
  - `npm start`

### Tek Komut Taslak
- `powershell -ExecutionPolicy Bypass -File scripts/dev-demo.ps1`

## 4) Demo Kontrol Noktalari
- UI acilis: `http://localhost:3000`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/metrics`
- Entropi uretimi:
  - Byte Count gir
  - Generate Data tikla
  - Output panelinde byte akisini dogrula

## 5) Son Kontrol
- `/health` donuyor mu
- `/random/bytes` UI uzerinden basarili mi
- `/metrics` metrik uretiyor mu
- `logs/api_audit.log` icinde JSON satirlar olusuyor mu

## 6) Raporlama Dosyalari
- Benchmark sonuclari: `benchmarks/benchmark_results*.json`
- Dogrulama raporlari: `docs/verification/*.json`
- Ekran goruntuleri: `docs/screenshots/*.png`
