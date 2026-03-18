# EntropyHub Frontend

EntropyHub React arayuzu, FastAPI backend ile entropi uretimi ve sistem sagligi goruntuleme amaciyla gelistirildi.

## Hizli Baslangic

```bash
npm install
cp .env.example .env
npm start
```

Varsayilan adres: `http://localhost:3000`

## Ortam Degiskenleri

`.env.example` dosyasini temel alin:

- `REACT_APP_API_URL`: Backend adresi
- `REACT_APP_API_KEY`: `x-api-key` degeri
- `REACT_APP_API_TIMEOUT_MS`: istek timeout suresi (ms)
- `REACT_APP_API_RETRY_COUNT`: timeout/5xx icin retry adedi
- `REACT_APP_API_RETRY_DELAY_MS`: retry bekleme suresi (ms)

## Ozellikler

- Sidebar + main layout (1024px alti tek kolon responsive)
- Entropi uretimi icin kullanici byte sayisi ayari
- API durumlari: loading / error / timeout / retry
- Sistem Sagligi paneli:
	- `/health` sonucu
	- son istek zamani
	- hata log ozeti (`/api/stats.recent_errors`)

## Scriptler

- `npm start`: gelistirme
- `npm run build`: production build
- `npm test`: testler

## Ilgili Dokumanlar

- `../docs/teknofest/FRONTEND_API_KULLANIM.md`
- `../docs/teknofest/DEMO_SENARYOSU.md`
- `../docs/teknofest/DEMO_CALISTIRMA_KILAVUZU.md`
