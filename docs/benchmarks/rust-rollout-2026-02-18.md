# Indexing Benchmark Report

- Fixture: `120` files (`.py=60`, `.md=36`, `.rst=24`)
- Parse workers: `1`
- Batch size: `32`

| Profile | Parser mode | Scan files/s | Parse files/s | Embed chunks/s | Total wall s |
|---|---|---:|---:|---:|---:|
| baseline_full | sequential | 40,079.1 | 46,403.0 | 108,860.7 | 0.01 |
| optimized_full | sequential_fallback | 48,442.4 | 47,858.0 | 2,151,951.9 | 0.01 |
| rust_enabled_full | sequential_fallback | 32,713.2 | 992.2 | 2,024,601.6 | 0.12 |
| optimized_rerun_hash_skip | n/a | 117,025.6 | 0.0 | 0.0 | 0.00 |

## Speedups

- Parse speedup: `1.03x`
- Embed speedup: `19.77x`
- Total speedup: `1.37x`

## Python vs Rust-Enabled Comparisons

| Workflow | Python throughput/s | Rust-enabled throughput/s | Speedup |
|---|---:|---:|---:|
| index commits prep | 516,229.0 | 110,562.9 | 0.21x |
| query baseline | 367.0 | 362.5 | 0.99x |
