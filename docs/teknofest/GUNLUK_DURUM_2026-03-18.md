# Gunluk Kisa Durum (2026-03-18)

## Frontend
- API durum yonetimi eklendi: loading / error / timeout / retry
- Entropi uretimi icin byte sayisi kullanici ayari eklendi
- Sistem sagligi paneli eklendi: `/health`, son istek zamani, hata ozeti
- 1024px alti responsive akis korunup iyilestirildi (sidebar -> ust blok)

## API
- Public path kontrolu env tabanli hale getirildi (`ENTROPYHUB_PUBLIC_PATHS`)
- API key mekanizmasi env ile netlestirildi (`ENTROPYHUB_API_KEYS`)
- CORS origin listesi env tabanli hale getirildi (`ENTROPYHUB_CORS_ORIGINS`)
- `/random/bytes` ve `/random/integers` icin OpenAPI ornekleri eklendi
- `/api/stats` endpointi son istek zamani + audit hata ozeti dondurur hale getirildi

## Entegrasyon
- Frontend cagrilari backend endpointleri ile uyumlu:
  - `GET /api/stats`: mevcut
  - `POST /chaos/reseed`: mevcut
- Checklist dosyasi olusturuldu

## Demo
- Demo script taslagi eklendi: `scripts/dev-demo.ps1`
- 1 sayfa demo kilavuzu eklendi

## Acik Is
- Ekran goruntulerinin alinip `docs/screenshots/` altina konulmasi
- Blokzincir demo kontrat adresinin gercek ortama gore guncellenmesi
