# Teknofest Kisa Teknik Ozet

## Mimari
- NIHDE tabanli kaotik entropi motoru (`core/chaos/nihde.py`)
- Rust cekirdek hizlandirma (`core/chaos/entropyhub_core_rs`)
- FastAPI servis katmani (`api/main.py`)
- React tabanli operator arayuzu (`frontend/src`)

## Akis
1. Kaotik sistemden entropy byte uretimi
2. Von Neumann post-processing
3. FastAPI endpointleri ile dagitim (`/random/bytes`, `/random/integers`)
4. Frontendde durum yonetimi + gorunsel cikti

## Test / Validasyon Ozeti
- Temel endpoint smoke:
  - `/health`: PASS
  - `/random/bytes`: PASS
  - `/metrics`: PASS
- Guvenlik:
  - API key yoksa `401`: PASS
  - Rate limit asiminda `429 + Retry-After`: PASS (pytest)
- Log izlenebilirligi:
  - `logs/api_audit.log` JSON satir kaydi: PASS

## Benchmark + Dogrulama Calistirma Sirasi
1. `python benchmarks/real_entropyhub_benchmark.py`
2. `python benchmarks/comprehensive_benchmark.py`
3. `python formal/entropyhub_bounded_formal.py`
4. `python docs/verification/generate_teknofest_report.py`

## Cikti Dosya Isimlendirme Onerisi
- `docs/verification/teknofest_validation_YYYYMMDD.json`
- `benchmarks/benchmark_results_real_YYYYMMDD.json`
- `benchmarks/benchmark_results_YYYYMMDD.json`
