# EntropyHub - Teknofest On Degerlendirme Raporu Taslak Metni

## Icindekiler
1. Proje Ozeti
2. Katma Deger ve Yenilikcilik
3. Teknoloji Kullanimi
4. Uygulanabilirlik
5. Yaygin Etki
6. Surdurulebilirlik
7. Proje Takvimi
8. Takim Yapisi
9. Kaynakca

---

## 1. Proje Ozeti
EntropyHub, kaotik dinamik tabanli rastgelelik uretimini post-quantum kriptografi ile birlestiren bir guvenlik cekirdegi calismasidir. Proje kapsaminda Rossler cekicisi tabanli entropy motoru, Von Neumann post-processing katmani, Rust hizlandirmali hesaplama cekirdegi ve ML-KEM-768 tabanli kriptografik yapi birlikte ele alinmistir [1], [2], [7].

Yarismadaki amac, iki kritik ihtiyaca ayni anda yanit vermektir:
- Yuksek kaliteli, olculebilir rastgelelik uretimi
- Gelecekteki kuantum tehditlerine dayanikli anahtar anlasma altyapisi

Secilen yontem, kaotik surekli sistemin hassas baslangic kosulu ozelligini sayisal ciktiya donusturmek, bias giderimi icin Von Neumann uygulamak ve ML-KEM-768 ile post-quantum guvenli paylasimli sir olusturmaktir [1], [3], [4].

Projenin olculen sonuclarina gore:
- Entropy: 7.9990756 bits/byte
- NIST Frequency p: 0.8125
- NIST Runs p: 0.2252
- Basic randomness pass: True
- KEM basari orani: %100 (1000 trial)
Bu degerler proje olcum raporlariyla desteklenmistir [2], [5], [6].

---

## 2. Katma Deger ve Yenilikcilik
### 2.1 Cozulen problem
Mevcut bircok uygulamada rastgelelik kaynaklari ya tek katmanli pseudo-random mekanizmalara dayalidir ya da post-quantum gecis planlari pratik urun seviyesine inmemistir. EntropyHub bu boslugu, kaotik entropy + post-processing + post-quantum KEM zinciriyle kapatir [1], [4].

### 2.2 Mevcut cozumlerden ayrisma
- Klasik CSPRNG yaklasimlarina ek olarak kaotik surekli sistem kullanir.
- Ham kaotik ciktiyi dogrudan vermek yerine bias azaltimi icin Von Neumann uygular.
- Rust cekirdek ile hesaplama performansini olculebilir sekilde artirir.
- ML-KEM-768 entegrasyonunu kriptografik akisla birlikte sunar.
- Bounded formal yaklasimla davranissal tutarlilik guclendirilir [1], [5], [6], [8].

### 2.3 Yenilikcilik ve ozgunluk
- Arastirma + urunlesme yaklasimi: sadece algoritma degil, olculmus metriklerle teknik olgunluga tasinmis bir tasarim.
- Bounded formal odak: kriptografik akis davranisinin net tanimlanmasi [6].
- Yerli gelistirme kabiliyeti: cekirdek ve ilgili teknik artefaktlar tek proje havuzunda yonetilir.

### 2.4 Akademik dayanak
Kaos teorisi, Von Neumann extraction, NIST SP 800-22 ve FIPS 203 ML-KEM standartlari uzerine kuruludur [3], [4], [9], [10], [11].

---

## 3. Teknoloji Kullanimi
### 3.1 Teknik mimari
Sistem katmanlari:
1. Entropy uretim katmani: Rossler kaotik modeli
2. Post-processing katmani: Von Neumann extractor
3. Cekirdek hizlandirma: Rust + Python entegrasyonu
4. Kriptografik katman: ML-KEM-768 kapsulleme/acma akislari
5. Formal katman: bounded formal kontrol yaklasimi [1], [5], [6], [7].

### 3.2 Teknik analiz yaklasimi
- RNG kalite olcumu: entropy, monobit/frequency, runs, chi-square, autocorrelation
- Performans olcumu: latency mean/median/p95, throughput
- KEM olcumu: keygen/encaps/decaps sureleri, success rate
- Formal kapsama: bounded formal sozlesme kontrolleri [2], [5], [6], [7].

### 3.3 Olculen teknik bulgular
Gercek benchmark ciktisina gore [2]:
- RNG throughput: 0.7144 Mbps
- RNG latency mean: 11.198 us
- Entropy: 7.9991 bits/byte
- Lag-1 autocorr: 0.00119
- KEM keygen mean: 330.586 us
- KEM encaps mean: 340.240 us
- KEM decaps mean: 118.247 us
- KEM success rate: 1.0

Bagimsiz dogrulama [5]:
- Entropy: 7.9991
- Monobit p-value: 0.7303
- Chi-square p-value: 0.6079
- KEM success rate: 1.0
- Durum: PASS

Bounded formal rapor [6]:
- Durum: PASS
- Bounded KEM domain: 16
- Input contract check: true
- RNG output range check: true

### 3.4 Teknolojik olgunluk
Temel teknolojik ciktilar:
- Istatistik kalite raporu (entropy, p-value, autocorrelation)
- Performans raporu (mean/median/p95 latency, throughput)
- KEM basari raporu (trials, success rate, sure dagilimi)
- Formal/bounded kontrol raporu (domain, input contract, output range) [2], [5], [6], [7].

---

## 4. Uygulanabilirlik
### 4.1 Hayata gecirme plani
Proje, moduler bir yapi ile prototipten urune gecis icin uygun durumdadir:
- Cekirdek modul bagimsiz bir guvenlik bileseni olarak konumlandirilabilir
- Parametre setleri teknik gereksinimlere gore belirlenebilir
- Farkli donanim profillerinde olculebilir performans profili cikarilabilir [2], [7], [8].

### 4.2 Ticari urune donusum potansiyeli
Olasilikli urunlestirme eksenleri:
- Cekirdek RNG modulu lisanslanabilir guvenlik bileseni
- PQ KEM destekli anahtar degisimi kutuphanesi
- Regule alanlarda (finans, savunma, kritik altyapi) guvenli rastgelelik ve anahtar degisimi
- Laboratuvar/akademik ortamlarda teknik referans paketi

### 4.3 Risk ve azaltim
- Risk: Kaotik model parametrelerinin yanlis secimi
  Azaltim: Parametre setlerinin resmi teknik kosullarla kilitlenmesi
- Risk: Donanim farkliligina bagli performans sapmasi
  Azaltim: Standart olcum seti + karsilastirmali p95/throughput raporu
- Risk: KEM uygulama entegrasyon hatasi
  Azaltim: Kriptografik vektor bazli encaps/decaps tutarlilik seti
- Risk: Standart degisimleri
  Azaltim: NIST/FIPS guncellemelerinin surekli takibi

---

## 5. Yaygin Etki
### 5.1 Toplumsal ve ekonomik etki
- Dijital guvenlik farkindaligi ve yerlilik kapasitesine katki
- Kritik sistemlerde guvenli rastgelelik kullaniminin artmasi
- Ulusal siber guvenlik urun ekosistemine teknik bir bilesen kazandirilmasi

### 5.2 Endustriyel etki
- Moduler cekirdek yapisi ile farkli platformlara hizli gecis
- Post-quantum gecis sureclerinde kullanilabilir referans uygulama
- Guvenlik urunlerinde entropy kalitesinin olculmesine yonelik pratik metrik seti

### 5.3 Olculen etki gostergeleri
- Kalite KPI: entropy, nist p-value, autocorrelation
- Performans KPI: mean/p95 latency, throughput
- Teknik KPI: sureklilik, tekrar uretilebilirlik, formal kontrol durumu

---

## 6. Surdurulebilirlik
### 6.1 Finansal surdurulebilirlik
- Moduler lisanslama: cekirdek + teknik dogrulama paketi + destek
- Hizmet bazli gelir modeli: istek/hacim tabanli fiyatlandirma
- Kurumsal destek modeline uygun teknik raporlama odakli gelistirme

### 6.2 Cevresel surdurulebilirlik
- Hesaplama maliyeti optimizasyonu icin Rust hizlandirma
- Gereksiz tekrar kosulari azaltan standardize isletim senaryolari
- Olceklenebilir altyapi ile kaynaklarin talebe gore kullanimi

### 6.3 Sosyal/organizasyonel surdurulebilirlik
- Teknik dokumantasyon odakli bilgi surekliligi
- Kod tabani icinde ayri dogrulama katmanlari (independent + formal)
- Olcumlenebilir kalite gostergeleri ile kolay bakim

### 6.4 Surdurulebilirlik riskleri ve yonetimi
- Kritik bagimlilik riski: alternatif paket/fallback stratejisi
- Ekip degisim riski: is paketi bazli dokumantasyon
- Tehdit evrimi riski: duzenli guvenlik incelemeleri ve standart takibi

---

## 7. Proje Takvimi
Asagidaki plan, on degerlendirme asamasi sonrasi 24 haftalik gelisim yol haritasi olarak onerilir.

| Is Paketi | Alt Faaliyet | Sure (Hafta) | Donem | Cikti |
|---|---|---:|---|---|
| WP1 Mimari Sertlestirme | Parametre seti dondurma, kod refactor | 4 | 1-4 | Stabil cekirdek surum |
| WP2 Teknik Derinlestirme | NIST kapsami genisletme, regresyon yonetimi | 4 | 5-8 | Teknik raporlar |
| WP3 Performans Iyilestirme | Rust optimizasyon turu, p95 azaltimi | 4 | 9-12 | Performans raporu |
| WP4 Guvenlik Sertlestirme | Kriptografik akis kontrolu, vektor analizi, hardening | 4 | 13-16 | Guvenlik kontrol listesi |
| WP5 Saha Hazirlik Paketi | Parametre kalibrasyonu, pilot hazirlik, dry-run | 4 | 17-20 | Uctan uca teknik paket |
| WP6 Urunlesme Hazirligi | Dokumantasyon, pilot kurulum, geri bildirim | 4 | 21-24 | Pilot cikis dosyasi |

---

## 8. Takim Yapisi
### Takim uye ve gorev dagilimi
| Uye | Rol | Sorumluluk Alani |
|---|---|---|
| AbdulKadir Guler | Proje Tasarim Lideri | Proje kapsamlandirma, sistem tasarimi, teknik yol haritasi |
| Sukran Akilidiz | Mentor | Teknik yonlendirme, karar kalite kontrolu, surec mentorlugu |
| Sukru Bas | Cekirdek Yapi ve Optimizasyon Muhendisi | Cekirdek yapi performans iyilestirme, optimizasyon ve profil calismalari |
| Ahmet Oyan | Teknik Raporlama Muhendisi | Teknik bulgularin raporlanmasi ve dokuman kalite kontrolu |
| Mahmut Sibal | Kriptografik Yapi Mimari Uzmani | KEM entegrasyonu, kriptografik akis dogrulamasi, guvenlik mimarisi |

### Ekip organizasyonu
- Teknik karar ve tasarim koordinasyonu: Proje Tasarim Lideri + Mentor
- Cekirdek performans ve sistem verimliligi: Cekirdek Yapi ve Optimizasyon Muhendisi
- Teknik bulgularin raporlanmasi: Teknik Raporlama Muhendisi
- Kriptografik guvenlik ve protokol katmani: Kriptografik Yapi Mimari Uzmani

### Not
- Teknofest final rapor tesliminde, kural geregi bu bolum kisi isimleri cikarilarak sadece rol bazli yapida sunulabilir.

### RACI ozet onerisi
- Sorumlu (R): Modul sahibinin kendisi
- Hesap veren (A): Proje Yoneticisi
- Danisilan (C): Domain uzmanlari
- Bilgilendirilen (I): Tum ekip ve paydaslar

---

## 9. Kaynakca
Metin ici atiflarda koseli parantez kullanilmistir. Ornek: [1], [4], [5-6].

### Dijital / Web ve Proje Ici Kaynaklar
[1] EntropyHub, Ana Proje Dokumani (README), erisim: 2026-03-19, erisim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/README.md

[2] EntropyHub, Gercek Benchmark Sonuclari (benchmark_results_real.json), 2026-03-04, erisim: 2026-03-19, erisim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/benchmarks/benchmark_results_real.json

[5] EntropyHub, Bagimsiz Dogrulama Raporu (independent_validation_report.json), 2026-03-04, erisim: 2026-03-19, erisim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/verification/independent_validation_report.json

[6] EntropyHub, Bounded Formal Dogrulama Raporu (formal_bounded_report.json), 2026-03-04, erisim: 2026-03-19, erisim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/verification/formal_bounded_report.json

[7] EntropyHub, Validation Pack README (docs/verification/README.md), erisim: 2026-03-19, erisim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/verification/README.md

[8] EntropyHub, Benchmark Raporu (docs/benchmarks/comprehensive_report.md), erisim: 2026-03-19, erisim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/benchmarks/comprehensive_report.md

### Akademik ve Standart Kaynaklar
[3] Rukhin, A. L., Soto, J., Nechvatal, J., et al., A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications, NIST SP 800-22 Rev.1a.

[4] National Institute of Standards and Technology, Module-Lattice-Based Key-Encapsulation Mechanism Standard, FIPS 203, 2024.

[9] Rossler, O. E., An equation for continuous chaos, Physics Letters A, 57(5), 397-398, 1976.

[10] Lorenz, E. N., Deterministic nonperiodic flow, Journal of the Atmospheric Sciences, 20(2), 130-141, 1963.

[11] von Neumann, J., Various techniques used in connection with random digits, NBS Applied Mathematics Series, 12, 36-38, 1951.
