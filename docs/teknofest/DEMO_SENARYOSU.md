# Uctan Uca Demo Senaryosu

## On Kosul
- Python bagimliliklari kurulu (`pip install -r requirements.txt`)
- Frontend bagimliliklari kurulu (`cd frontend && npm install`)
- `.env.example` dosyalarindan ortama uygun `.env` olusturuldu

## Adimlar
1. API baslat:
   - `uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`
2. Frontend baslat:
   - `cd frontend`
   - `npm start`
3. Tarayici:
   - `http://localhost:3000`
4. Entropi uret:
   - `Byte Count` alanina deger gir (or. 64)
   - `Generate Data` tikla
   - Byte ciktilarini panelde dogrula
5. API sagligini dogrula:
   - UI `System Health` panelinde `/health` durumu
   - Son istek zamani ve hata ozetini kontrol et
6. Swagger dogrulama:
   - `http://127.0.0.1:8000/docs`
   - `/random/bytes` icin ornek istek gonder
7. Metrics dogrulama:
   - `http://127.0.0.1:8000/metrics`
8. Audit log dogrulama:
   - `logs/api_audit.log` dosyasinda JSON satirlar olusmali

## Tek Komut Onerisi
- `scripts/dev-demo.ps1` scripti ile API ve Frontend birlikte acilir.
