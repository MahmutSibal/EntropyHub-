# EntropyHub - Teknofest Ön Değerlendirme Raporu Taslak Metni

## İçindekiler
1. Proje Özeti
2. Katma Değer ve Yenilikçilik
3. Teknoloji Kullanımı
4. Uygulanabilirlik
5. Yaygın Etki
6. Sürdürülebilirlik
7. Proje Takvimi
8. Takım Yapısı
9. Kaynakça

---

## 1. Proje Özeti
EntropyHub, kaotik dinamik tabanlı rastgelelik üretimini post-quantum kriptografi ile birleştiren bir güvenlik çekirdeği çalışmasıdır. Proje kapsamında Rössler çekicisi tabanlı entropy motoru, Von Neumann post-processing katmanı, Rust hızlandırmalı hesaplama çekirdeği ve ML-KEM-768 tabanlı kriptografik yapı birlikte ele alınmıştır [1], [2], [7].

Yarışmadaki amaç, iki kritik ihtiyaca aynı anda yanıt vermektir:
- Yüksek kaliteli, ölçülebilir rastgelelik üretimi
- Gelecekteki kuantum tehditlerine dayanıklı anahtar anlaşma altyapısı

Seçilen yöntem, kaotik sürekli sistemin hassas başlangıç koşulu özelliğini sayısal çıktıya dönüştürmek, bias giderimi için Von Neumann uygulamak ve ML-KEM-768 ile post-quantum güvenli paylaşımlı sır oluşturmaktır [1], [3], [4].

Projenin ölçülen sonuçlarına göre:
- Entropy: 7.9990756 bits/byte
- NIST Frequency p: 0.8125
- NIST Runs p: 0.2252
- Basic randomness pass: True
- KEM başarı oranı: %100 (1000 trial)
Bu değerler proje ölçüm raporlarıyla desteklenmiştir [2], [5], [6].

---

## 2. Katma Değer ve Yenilikçilik
### 2.1 Çözülen problem
Mevcut birçok uygulamada rastgelelik kaynakları ya tek katmanlı pseudo-random mekanizmalara dayalıdır ya da post-quantum geçiş planları pratik ürün seviyesine inmemiştir. EntropyHub bu boşluğu, kaotik entropy + post-processing + post-quantum KEM zinciriyle kapatır [1], [4].

### 2.2 Mevcut çözümlerden ayrışma
- Klasik CSPRNG yaklaşımlarına ek olarak kaotik sürekli sistem kullanır.
- Ham kaotik çıktıyı doğrudan vermek yerine bias azaltımı için Von Neumann uygular.
- Rust çekirdek ile hesaplama performansını ölçülebilir şekilde artırır.
- ML-KEM-768 entegrasyonunu kriptografik akışla birlikte sunar.
- Bounded formal yaklaşım ile davranışsal tutarlılık güçlendirilir [1], [5], [6], [8].

### 2.3 Yenilikçilik ve özgünlük
- Araştırma + ürünleşme yaklaşımı: sadece algoritma değil, ölçülmüş metriklerle teknik olgunluğa taşınmış bir tasarım.
- Bounded formal odak: kriptografik akış davranışının net tanımlanması [6].
- Yerli geliştirme kabiliyeti: çekirdek ve ilgili teknik artefaktlar tek proje havuzunda yönetilir.

### 2.4 Akademik dayanak
Kaos teorisi, Von Neumann extraction, NIST SP 800-22 ve FIPS 203 ML-KEM standartları üzerine kuruludur [3], [4], [9], [10], [11].

---

## 3. Teknoloji Kullanımı
### 3.1 Teknik mimari
Sistem katmanları:
1. Entropy üretim katmanı: Rössler kaotik modeli
2. Post-processing katmanı: Von Neumann extractor
3. Çekirdek hızlandırma: Rust + Python entegrasyonu
4. Kriptografik katman: ML-KEM-768 kapsülleme/açma akışları
5. Formal katman: bounded formal kontrol yaklaşımı [1], [5], [6], [7]

### 3.2 Teknik analiz yaklaşımı
- RNG kalite ölçümü: entropy, monobit/frequency, runs, chi-square, autocorrelation
- Performans ölçümü: latency mean/median/p95, throughput
- KEM ölçümü: keygen/encaps/decaps süreleri, success rate
- Formal kapsama: bounded formal sözleşme kontrolleri [2], [5], [6], [7]

### 3.3 Ölçülen teknik bulgular
Gerçek benchmark çıktısına göre [2]:
- RNG throughput: 0.7144 Mbps
- RNG latency mean: 11.198 us
- Entropy: 7.9991 bits/byte
- Lag-1 autocorr: 0.00119
- KEM keygen mean: 330.586 us
- KEM encaps mean: 340.240 us
- KEM decaps mean: 118.247 us
- KEM success rate: 1.0

Bağımsız doğrulama [5]:
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
Temel teknolojik çıktılar:
- İstatistik kalite raporu (entropy, p-value, autocorrelation)
- Performans raporu (mean/median/p95 latency, throughput)
- KEM başarı raporu (trials, success rate, süre dağılımı)
- Formal/bounded kontrol raporu (domain, input contract, output range) [2], [5], [6], [7]

---

## 4. Uygulanabilirlik
### 4.1 Hayata geçirme planı
Proje, modüler bir yapı ile prototipten ürüne geçiş için uygun durumdadır:
- Çekirdek modül bağımsız bir güvenlik bileşeni olarak konumlandırılabilir
- Parametre setleri teknik gereksinimlere göre belirlenebilir
- Farklı donanım profillerinde ölçülebilir performans profili çıkarılabilir [2], [7], [8]

### 4.2 Ticari ürüne dönüşüm potansiyeli
Olası ürünleştirme eksenleri:
- Çekirdek RNG modülü lisanslanabilir güvenlik bileşeni
- PQ KEM destekli anahtar değişimi kütüphanesi
- Regüle alanlarda (finans, savunma, kritik altyapı) güvenli rastgelelik ve anahtar değişimi
- Laboratuvar/akademik ortamlarda teknik referans paketi

### 4.3 Risk ve azaltım
- Risk: Kaotik model parametrelerinin yanlış seçimi
  Azaltım: Parametre setlerinin resmi teknik koşullarla kilitlenmesi
- Risk: Donanım farklılığına bağlı performans sapması
  Azaltım: Standart ölçüm seti + karşılaştırmalı p95/throughput raporu
- Risk: KEM uygulama entegrasyon hatası
  Azaltım: Kriptografik vektör bazlı encaps/decaps tutarlılık seti
- Risk: Standart değişimleri
  Azaltım: NIST/FIPS güncellemelerinin sürekli takibi

---

## 5. Yaygın Etki
### 5.1 Toplumsal ve ekonomik etki
- Dijital güvenlik farkındalığı ve yerlilik kapasitesine katkı
- Kritik sistemlerde güvenli rastgelelik kullanımının artması
- Ulusal siber güvenlik ürün ekosistemine teknik bir bileşen kazandırılması

### 5.2 Endüstriyel etki
- Modüler çekirdek yapısı ile farklı platformlara hızlı geçiş
- Post-quantum geçiş süreçlerinde kullanılabilir referans uygulama
- Güvenlik ürünlerinde entropy kalitesinin ölçülmesine yönelik pratik metrik seti

### 5.3 Ölçülen etki göstergeleri
- Kalite KPI: entropy, nist p-value, autocorrelation
- Performans KPI: mean/p95 latency, throughput
- Teknik KPI: süreklilik, tekrar üretilebilirlik, formal kontrol durumu

---

## 6. Sürdürülebilirlik
### 6.1 Finansal sürdürülebilirlik
- Modüler lisanslama: çekirdek + teknik doğrulama paketi + destek
- Hizmet bazlı gelir modeli: istek/hacim tabanlı fiyatlandırma
- Kurumsal destek modeline uygun teknik raporlama odaklı geliştirme

### 6.2 Çevresel sürdürülebilirlik
- Hesaplama maliyeti optimizasyonu için Rust hızlandırma
- Gereksiz tekrar koşuları azaltan standardize işletim senaryoları
- Ölçeklenebilir altyapı ile kaynakların talebe göre kullanımı

### 6.3 Sosyal/organizasyonel sürdürülebilirlik
- Teknik dokümantasyon odaklı bilgi sürekliliği
- Kod tabanı içinde ayrı doğrulama katmanları (independent + formal)
- Ölçümlenebilir kalite göstergeleri ile kolay bakım

### 6.4 Sürdürülebilirlik riskleri ve yönetimi
- Kritik bağımlılık riski: alternatif paket/fallback stratejisi
- Ekip değişim riski: iş paketi bazlı dokümantasyon
- Tehdit evrimi riski: düzenli güvenlik incelemeleri ve standart takibi

---

## 7. Proje Takvimi
Aşağıdaki plan, ön değerlendirme aşaması sonrası 24 haftalık gelişim yol haritası olarak önerilir.

| İş Paketi | Alt Faaliyet | Süre (Hafta) | Dönem | Çıktı |
|---|---|---:|---|---|
| WP1 Mimari Sertleştirme | Parametre seti dondurma, kod refactor | 4 | 1-4 | Stabil çekirdek sürüm |
| WP2 Teknik Derinleştirme | NIST kapsamı genişletme, regresyon yönetimi | 4 | 5-8 | Teknik raporlar |
| WP3 Performans İyileştirme | Rust optimizasyon turu, p95 azaltımı | 4 | 9-12 | Performans raporu |
| WP4 Güvenlik Sertleştirme | Kriptografik akış kontrolü, vektör analizi, hardening | 4 | 13-16 | Güvenlik kontrol listesi |
| WP5 Saha Hazırlık Paketi | Parametre kalibrasyonu, pilot hazırlık, dry-run | 4 | 17-20 | Uçtan uca teknik paket |
| WP6 Ürünleşme Hazırlığı | Dokümantasyon, pilot kurulum, geri bildirim | 4 | 21-24 | Pilot çıkış dosyası |

---

## 8. Takım Yapısı
### Takım üye ve görev dağılımı
| Üye | Rol | Sorumluluk Alanı |
|---|---|---|
| AbdülKadir Güler | Proje Tasarım Lideri | Proje kapsamlandırma, sistem tasarımı, teknik yol haritası |
| Şükran Akılıdız | Mentor | Teknik yönlendirme, karar kalite kontrolü, süreç mentörlüğü |
| Şükrü Baş | Çekirdek Yapı ve Optimizasyon Mühendisi | Çekirdek yapı performans iyileştirme, optimizasyon ve profil çalışmaları |
| Ahmet Oyan | Teknik Raporlama Mühendisi | Teknik bulguların raporlanması ve doküman kalite kontrolü |
| Mahmut Sibal | Kriptografik Yapı Mimari Uzmanı | KEM entegrasyonu, kriptografik akış doğrulaması, güvenlik mimarisi |

### Ekip organizasyonu
- Teknik karar ve tasarım koordinasyonu: Proje Tasarım Lideri + Mentor
- Çekirdek performans ve sistem verimliliği: Çekirdek Yapı ve Optimizasyon Mühendisi
- Teknik bulguların raporlanması: Teknik Raporlama Mühendisi
- Kriptografik güvenlik ve protokol katmanı: Kriptografik Yapı Mimari Uzmanı

### Not
- Teknofest final rapor tesliminde, kural gereği bu bölüm kişi isimleri çıkarılarak sadece rol bazlı yapıda sunulabilir.

### RACI özet önerisi
- Sorumlu (R): Modül sahibinin kendisi
- Hesap veren (A): Proje Yöneticisi
- Danışılan (C): Domain uzmanları
- Bilgilendirilen (I): Tüm ekip ve paydaşlar

---

## 9. Kaynakça
Metin içi atıflarda köşeli parantez kullanılmıştır. Örnek: [1], [4], [5-6].

### Dijital / Web ve Proje İçi Kaynaklar
[1] EntropyHub, Ana Proje Dokümanı (README), erişim: 2026-03-19, erişim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/README.md

[2] EntropyHub, Gerçek Benchmark Sonuçları (benchmark_results_real.json), 2026-03-04, erişim: 2026-03-19, erişim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/benchmarks/benchmark_results_real.json

[5] EntropyHub, Bağımsız Doğrulama Raporu (independent_validation_report.json), 2026-03-04, erişim: 2026-03-19, erişim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/verification/independent_validation_report.json

[6] EntropyHub, Bounded Formal Doğrulama Raporu (formal_bounded_report.json), 2026-03-04, erişim: 2026-03-19, erişim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/verification/formal_bounded_report.json

[7] EntropyHub, Validation Pack README (docs/verification/README.md), erişim: 2026-03-19, erişim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/verification/README.md

[8] EntropyHub, Benchmark Raporu (docs/benchmarks/comprehensive_report.md), erişim: 2026-03-19, erişim adresi: https://github.com/MahmutSibal/EntropyHub-/blob/main/docs/benchmarks/comprehensive_report.md

### Akademik ve Standart Kaynaklar
[3] Rukhin, A. L., Soto, J., Nechvatal, J., et al., A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications, NIST SP 800-22 Rev.1a.

[4] National Institute of Standards and Technology, Module-Lattice-Based Key-Encapsulation Mechanism Standard, FIPS 203, 2024.

[9] Rossler, O. E., An equation for continuous chaos, Physics Letters A, 57(5), 397-398, 1976.

[10] Lorenz, E. N., Deterministic nonperiodic flow, Journal of the Atmospheric Sciences, 20(2), 130-141, 1963.

[11] von Neumann, J., Various techniques used in connection with random digits, NBS Applied Mathematics Series, 12, 36-38, 1951.

