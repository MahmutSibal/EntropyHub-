# benchmarks/comprehensive_benchmark.py
"""
Comprehensive Benchmark Suite for Aether PRNG
Compares performance, entropy quality, and statistical properties
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import struct

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.chaos.nihde import NIHDE


class ComprehensiveBenchmark:
    """Complete benchmark suite for Aether PRNG"""
    
    def __init__(self):
        self.results = defaultdict(dict)
        self.engine = NIHDE()
        
    def benchmark_latency(self, iterations=100000):
        """Measure per-byte generation latency"""
        print(f"\n{'='*80}")
        print(f"LATENCY BENCHMARK - {iterations:,} iterations")
        print(f"{'='*80}")
        
        latencies = []
        
        # Warmup
        for _ in range(1000):
            self.engine.decide()
        
        # Actual benchmark
        for i in range(iterations):
            start = time.perf_counter()
            self.engine.decide()
            end = time.perf_counter()
            latencies.append((end - start) * 1_000_000)  # microseconds
            
            if (i + 1) % 10000 == 0:
                print(f"Progress: {i+1:,}/{iterations:,} iterations", end='\r')
        
        latencies = np.array(latencies)
        
        results = {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'throughput_MB_s': (iterations / np.sum(latencies)) * 1_000_000
        }
        
        print(f"\n\nResults:")
        print(f"  Mean latency:      {results['mean']:.3f} µs")
        print(f"  Median latency:    {results['median']:.3f} µs")
        print(f"  Std deviation:     {results['std']:.3f} µs")
        print(f"  Min latency:       {results['min']:.3f} µs")
        print(f"  Max latency:       {results['max']:.3f} µs")
        print(f"  95th percentile:   {results['p95']:.3f} µs")
        print(f"  99th percentile:   {results['p99']:.3f} µs")
        print(f"  Throughput:        {results['throughput_MB_s']:.2f} MB/s")
        
        self.results['latency'] = results
        self._plot_latency_distribution(latencies)
        
        return results
    
    def benchmark_entropy(self, num_bytes=1000000):
        """Measure entropy quality"""
        print(f"\n{'='*80}")
        print(f"ENTROPY BENCHMARK - {num_bytes:,} bytes")
        print(f"{'='*80}")
        
        # Generate data
        data = bytes([self.engine.decide() for _ in range(num_bytes)])
        
        # Shannon entropy
        byte_counts = np.bincount(list(data), minlength=256)
        probabilities = byte_counts / len(data)
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Min-entropy (worst-case)
        max_prob = np.max(probabilities)
        min_entropy = -np.log2(max_prob) if max_prob > 0 else 8.0
        
        # Byte frequency analysis
        expected_count = num_bytes / 256
        chi_squared = np.sum((byte_counts - expected_count)**2 / expected_count)
        
        # Bit balance
        bitstring = ''.join(format(byte, '08b') for byte in data)
        ones = bitstring.count('1')
        zeros = len(bitstring) - ones
        bit_balance = abs(ones - zeros) / len(bitstring)
        
        results = {
            'shannon_entropy': shannon_entropy,
            'min_entropy': min_entropy,
            'chi_squared': chi_squared,
            'bit_balance': bit_balance,
            'ones_ratio': ones / len(bitstring),
            'zeros_ratio': zeros / len(bitstring)
        }
        
        print(f"\nResults:")
        print(f"  Shannon Entropy:   {results['shannon_entropy']:.4f} bits/byte (ideal: 8.0)")
        print(f"  Min-Entropy:       {results['min_entropy']:.4f} bits/byte")
        print(f"  Chi-squared:       {results['chi_squared']:.2f}")
        print(f"  Bit balance:       {results['bit_balance']:.6f} (ideal: 0.0)")
        print(f"  Ones ratio:        {results['ones_ratio']:.6f} (ideal: 0.5)")
        print(f"  Zeros ratio:       {results['zeros_ratio']:.6f} (ideal: 0.5)")
        
        self.results['entropy'] = results
        self._plot_byte_distribution(byte_counts)
        
        return results
    
    def benchmark_correlation(self, num_samples=10000, lag_max=100):
        """Measure autocorrelation"""
        print(f"\n{'='*80}")
        print(f"CORRELATION BENCHMARK - {num_samples:,} samples, lag up to {lag_max}")
        print(f"{'='*80}")
        
        # Generate data
        data = np.array([self.engine.decide() for _ in range(num_samples)])
        
        # Compute autocorrelation
        mean = np.mean(data)
        variance = np.var(data)
        
        autocorr = []
        for lag in range(lag_max + 1):
            if lag == 0:
                autocorr.append(1.0)
            else:
                shifted = data[lag:]
                original = data[:-lag]
                corr = np.mean((original - mean) * (shifted - mean)) / variance
                autocorr.append(corr)
        
        autocorr = np.array(autocorr)
        
        # Find max absolute correlation (excluding lag=0)
        max_corr = np.max(np.abs(autocorr[1:]))
        max_corr_lag = np.argmax(np.abs(autocorr[1:])) + 1
        
        results = {
            'max_correlation': max_corr,
            'max_correlation_lag': max_corr_lag,
            'mean_abs_correlation': np.mean(np.abs(autocorr[1:])),
            'autocorrelation': autocorr.tolist()
        }
        
        print(f"\nResults:")
        print(f"  Max correlation:      {results['max_correlation']:.6f} (at lag {max_corr_lag})")
        print(f"  Mean abs correlation: {results['mean_abs_correlation']:.6f}")
        print(f"  Status: {'✓ GOOD' if max_corr < 0.1 else '✗ NEEDS IMPROVEMENT'}")
        
        self.results['correlation'] = results
        self._plot_autocorrelation(autocorr)
        
        return results
    
    def benchmark_pattern_detection(self, num_bytes=100000):
        """Detect repeating patterns"""
        print(f"\n{'='*80}")
        print(f"PATTERN DETECTION - {num_bytes:,} bytes")
        print(f"{'='*80}")
        
        data = bytes([self.engine.decide() for _ in range(num_bytes)])
        
        # Check for repeating sequences of different lengths
        pattern_counts = {}
        
        for pattern_length in [2, 4, 8, 16]:
            patterns = defaultdict(int)
            for i in range(len(data) - pattern_length + 1):
                pattern = data[i:i+pattern_length]
                patterns[pattern] += 1
            
            # Find most common pattern
            max_count = max(patterns.values())
            expected_count = len(data) / (256 ** pattern_length)
            
            pattern_counts[pattern_length] = {
                'max_count': max_count,
                'expected_count': expected_count,
                'ratio': max_count / expected_count if expected_count > 0 else 0,
                'unique_patterns': len(patterns)
            }
            
            print(f"\n  Pattern length {pattern_length} bytes:")
            print(f"    Unique patterns:    {pattern_counts[pattern_length]['unique_patterns']:,}")
            print(f"    Max repetitions:    {max_count}")
            print(f"    Expected avg:       {expected_count:.2f}")
            print(f"    Ratio:              {pattern_counts[pattern_length]['ratio']:.2f}x")
        
        self.results['patterns'] = pattern_counts
        return pattern_counts
    
    def benchmark_comparison(self, iterations=50000):
        """Compare with os.urandom and numpy.random"""
        print(f"\n{'='*80}")
        print(f"COMPARISON BENCHMARK - {iterations:,} iterations")
        print(f"{'='*80}")
        
        # Aether PRNG
        start = time.perf_counter()
        for _ in range(iterations):
            self.engine.decide()
        aether_time = time.perf_counter() - start
        
        # os.urandom
        start = time.perf_counter()
        for _ in range(iterations):
            os.urandom(1)
        urandom_time = time.perf_counter() - start
        
        # numpy random
        start = time.perf_counter()
        for _ in range(iterations):
            np.random.randint(0, 256)
        numpy_time = time.perf_counter() - start
        
        results = {
            'aether': {
                'total_time': aether_time,
                'per_byte_us': (aether_time / iterations) * 1_000_000,
                'speedup_vs_urandom': urandom_time / aether_time,
                'speedup_vs_numpy': numpy_time / aether_time
            },
            'urandom': {
                'total_time': urandom_time,
                'per_byte_us': (urandom_time / iterations) * 1_000_000
            },
            'numpy': {
                'total_time': numpy_time,
                'per_byte_us': (numpy_time / iterations) * 1_000_000
            }
        }
        
        print(f"\nResults:")
        print(f"  Aether PRNG:")
        print(f"    Total time:        {results['aether']['total_time']:.4f} s")
        print(f"    Per-byte latency:  {results['aether']['per_byte_us']:.3f} µs")
        print(f"  ")
        print(f"  os.urandom:")
        print(f"    Total time:        {results['urandom']['total_time']:.4f} s")
        print(f"    Per-byte latency:  {results['urandom']['per_byte_us']:.3f} µs")
        print(f"    Speedup:           {results['aether']['speedup_vs_urandom']:.2f}x")
        print(f"  ")
        print(f"  numpy.random:")
        print(f"    Total time:        {results['numpy']['total_time']:.4f} s")
        print(f"    Per-byte latency:  {results['numpy']['per_byte_us']:.3f} µs")
        print(f"    Speedup:           {results['aether']['speedup_vs_numpy']:.2f}x")
        
        self.results['comparison'] = results
        self._plot_comparison(results)
        
        return results
    
    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print(f"\n{'#'*80}")
        print(f"#  AETHER PRNG - COMPREHENSIVE BENCHMARK SUITE")
        print(f"#  Version: 2.1.0")
        print(f"#  Engine: Rust-optimized Rössler chaotic core")
        print(f"{'#'*80}")
        
        # Run all benchmarks
        self.benchmark_latency(iterations=100000)
        self.benchmark_entropy(num_bytes=1000000)
        self.benchmark_correlation(num_samples=10000, lag_max=100)
        self.benchmark_pattern_detection(num_bytes=100000)
        self.benchmark_comparison(iterations=50000)
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"\n{'#'*80}")
        print(f"#  BENCHMARK COMPLETE")
        print(f"{'#'*80}\n")
        
        return self.results
    
    def _plot_latency_distribution(self, latencies):
        """Plot latency distribution histogram"""
        plt.figure(figsize=(12, 6))
        plt.hist(latencies, bins=100, color='#00ffff', alpha=0.7, edgecolor='black')
        plt.xlabel('Latency (µs)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Aether PRNG - Latency Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('docs/figures/latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_byte_distribution(self, byte_counts):
        """Plot byte frequency distribution"""
        plt.figure(figsize=(14, 6))
        plt.bar(range(256), byte_counts, color='#00ff88', alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.axhline(y=np.mean(byte_counts), color='red', linestyle='--', label='Expected (uniform)')
        plt.xlabel('Byte Value (0-255)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Aether PRNG - Byte Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('docs/figures/byte_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_autocorrelation(self, autocorr):
        """Plot autocorrelation function"""
        plt.figure(figsize=(12, 6))
        plt.plot(autocorr, color='#00ffff', linewidth=2, marker='o', markersize=3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='±0.1 threshold')
        plt.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Lag', fontsize=12)
        plt.ylabel('Autocorrelation', fontsize=12)
        plt.title('Aether PRNG - Autocorrelation Function', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('docs/figures/autocorrelation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comparison(self, results):
        """Plot performance comparison"""
        generators = ['Aether', 'os.urandom', 'numpy.random']
        latencies = [
            results['aether']['per_byte_us'],
            results['urandom']['per_byte_us'],
            results['numpy']['per_byte_us']
        ]
        colors = ['#00ffff', '#ff6b6b', '#4ecdc4']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(generators, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        plt.ylabel('Latency per byte (µs)', fontsize=12)
        plt.title('Performance Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, latency in zip(bars, latencies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{latency:.2f} µs',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('docs/figures/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self):
        """Generate markdown summary report"""
        report = f"""# Aether PRNG - Benchmark Report

## Summary

**Engine:** Rust-optimized Rössler chaotic core  
**Version:** 2.1.0  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## Performance Metrics

### Latency
- **Mean:** {self.results['latency']['mean']:.3f} µs
- **Median:** {self.results['latency']['median']:.3f} µs
- **95th percentile:** {self.results['latency']['p95']:.3f} µs
- **99th percentile:** {self.results['latency']['p99']:.3f} µs
- **Throughput:** {self.results['latency']['throughput_MB_s']:.2f} MB/s

### Entropy Quality
- **Shannon Entropy:** {self.results['entropy']['shannon_entropy']:.4f} bits/byte (ideal: 8.0)
- **Min-Entropy:** {self.results['entropy']['min_entropy']:.4f} bits/byte
- **Chi-squared:** {self.results['entropy']['chi_squared']:.2f}
- **Bit Balance:** {self.results['entropy']['bit_balance']:.6f}

### Correlation
- **Max Autocorrelation:** {self.results['correlation']['max_correlation']:.6f}
- **Mean Abs Correlation:** {self.results['correlation']['mean_abs_correlation']:.6f}

### Comparison
- **vs os.urandom:** {self.results['comparison']['aether']['speedup_vs_urandom']:.2f}x faster
- **vs numpy.random:** {self.results['comparison']['aether']['speedup_vs_numpy']:.2f}x faster

---

## Visualizations

![Latency Distribution](figures/latency_distribution.png)
![Byte Distribution](figures/byte_distribution.png)
![Autocorrelation](figures/autocorrelation.png)
![Performance Comparison](figures/performance_comparison.png)

---

**Generated by Aether Comprehensive Benchmark Suite**
"""
        
        os.makedirs('docs/benchmarks', exist_ok=True)
        with open('docs/benchmarks/comprehensive_report.md', 'w') as f:
            f.write(report)
        
        print(f"\n📊 Summary report saved to: docs/benchmarks/comprehensive_report.md")


if __name__ == "__main__":
    benchmark = ComprehensiveBenchmark()
    results = benchmark.run_all_benchmarks()
