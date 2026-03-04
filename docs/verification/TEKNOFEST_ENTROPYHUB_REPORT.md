# EntropyHub Teknofest Teknik Raporu

- Rapor zamanı: 2026-03-04 20:21:27
- Benchmark kaynağı: `benchmarks/benchmark_results_real.json`
- Bağımsız doğrulama kaynağı: `docs/verification/independent_validation_report.json`
- Formal doğrulama kaynağı: `docs/verification/formal_bounded_report.json`

## 1) Yürütme Özeti

- Rust çekirdek aktif: **True**
- RNG gecikme (ortalama): **7.371 µs**
- RNG throughput: **1.085 Mbps**
- Entropy: **7.999076 bits/byte**
- KEM başarı oranı: **100.00%**

## 2) Ortam ve Metodoloji

- Python: `3.13.12 (tags/v3.13.12:1cbe481, Feb  3 2026, 18:22:25) [MSC v.1944 64 bit (AMD64)]`
- Çalışma dizini: `C:\Users\-\Desktop\YildizGithub\Aether-CC`
- RNG ölçümü: 10k latency iterasyonu + 200k byte kalite ölçümü
- KEM ölçümü: 1000 trial (keygen/encaps/decaps)
- İstatistik: mean/median/p95/min/max + NIST temel testleri

## 3) RNG Gerçek Ölçüm Sonuçları

| Metrik | Değer |
|---|---:|
| Latency mean (µs) | 7.371 |
| Latency median (µs) | 6.900 |
| Latency p95 (µs) | 8.300 |
| Throughput (Mbps) | 1.085345 |
| Entropy (bits/byte) | 7.999076 |
| NIST Frequency p | 0.812524 |
| NIST Runs p | 0.225249 |
| Chi-square p | 0.456384 |
| Lag-1 autocorr | 0.001193 |
| Basic randomness pass | True |

## 4) KEM Gerçek Ölçüm Sonuçları

| Metrik | Değer |
|---|---:|
| Trials | 1000 |
| Success Rate | 100.00% |
| Keygen mean (µs) | 47.502 |
| Encaps mean (µs) | 24.791 |
| Decaps mean (µs) | 22.209 |
| Keygen p95 (µs) | 70.300 |
| Encaps p95 (µs) | 38.575 |
| Decaps p95 (µs) | 35.290 |

## 5) Bağımsız Doğrulama

- Durum: **PASS**
- Çekirdek: `EntropyHubCore` (rust=True)
- Entropy: 7.999106
- Monobit p: 0.730329
- Chi-square p: 0.607859
- Lag-1 autocorr: 0.001106
- KEM success: 100.00%

## 6) Formal/Bağlı Doğrulama

- Durum: **PASS**
- Bounded KEM domain size: 16
- Input contract check: True
- RNG output range check: True

## 7) Yeniden Üretilebilirlik Komutları

```bash
python benchmarks/real_entropyhub_benchmark.py
python tests/independent_validation.py
python formal/entropyhub_bounded_formal.py
python docs/verification/generate_teknofest_report.py
```

## 8) Kaynakça

1. NIST SP 800-22 Rev.1a — Statistical Test Suite for Random and Pseudorandom Number Generators.
2. NIST FIPS 203 — Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM).
3. Rössler, O. E. (1976). An equation for continuous chaos.
4. EntropyHub kaynak kodu ve ölçüm artefaktları:
   - `benchmarks/real_entropyhub_benchmark.py`
   - `benchmarks/benchmark_results_real.json`
   - `tests/independent_validation.py`
   - `formal/entropyhub_bounded_formal.py`
   - `formal/EntropyHubKEM.tla`
