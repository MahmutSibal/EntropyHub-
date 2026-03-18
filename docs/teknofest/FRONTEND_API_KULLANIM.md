# Frontend Kullanim Ozeti (Teknofest)

## Amac
Bu dokuman, EntropyHub frontend ekraninda API baglanti durumlari ve entropi uretim akisinin hizli kullanimini ozetler.

## Ekran Akislari
- Ana layout: `sidebar + main`
- 1024px alti: sidebar ustte, icerik altta (tek kolon)
- API durumlari:
  - `loading`: entropi panelinde mavi bilgi karti
  - `error`: kirmizi hata karti + `Retry` butonu
  - `timeout`: timeout ozel mesaji + `Retry`

## Entropi Uretimi
1. Sidebar icindeki `Byte Count` alanina `1-4096` arasi deger girin.
2. `Generate Data` butonuna basin.
3. Sonuclar `Output (Byte Stream)` panelinde listelenir.
4. `System Health` panelinde son istek zamani ve hata ozeti gorulur.

## API Baglanti Ayarlari
`frontend/.env` veya `frontend/.env.example`:
- `REACT_APP_API_URL`
- `REACT_APP_API_KEY`
- `REACT_APP_API_TIMEOUT_MS`
- `REACT_APP_API_RETRY_COUNT`
- `REACT_APP_API_RETRY_DELAY_MS`

## Ekran Goruntusu Listesi
Asagidaki dosyalar `docs/screenshots/` altina alinmalidir:
- `01-home-sidebar-main.png`
- `02-loading-state.png`
- `03-timeout-or-error-state.png`
- `04-success-byte-output.png`
- `05-system-health-panel.png`
