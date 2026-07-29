# canns-lib CANN FFI Benchmark Report

**Platform**: x86_64 (Python 3.12.12)
**JAX**: 0.9.0
**FFI enabled**: True
**Configs**: 39 (model × n × T)

All times in **milliseconds** (median of 50 iters, 20 warmup).
Correctness = `max abs diff` vs canns upstream reference (numpy iteration).
Speedup = `ms_pure_jax / ms_ffi`.

## CANN1D

| model | n | T | diff_pj | ms_pj | diff_ffi | ms_ffi | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| CANN1D | 64 | 100 | 3.73e-08 | 1.65 | n/a | n/a | n/a |
| CANN1D | 64 | 500 | 1.04e-07 | 7.53 | n/a | n/a | n/a |
| CANN1D | 64 | 1000 | 1.19e-07 | 14.89 | n/a | n/a | n/a |
| CANN1D | 128 | 100 | 2.98e-08 | 1.68 | n/a | n/a | n/a |
| CANN1D | 128 | 500 | 5.96e-08 | 7.68 | n/a | n/a | n/a |
| CANN1D | 128 | 1000 | 2.98e-08 | 15.01 | n/a | n/a | n/a |
| CANN1D | 256 | 100 | 4.47e-08 | 1.64 | n/a | n/a | n/a |
| CANN1D | 256 | 500 | 8.94e-08 | 7.63 | n/a | n/a | n/a |
| CANN1D | 256 | 1000 | 5.96e-08 | 15.05 | n/a | n/a | n/a |


## CANN2D

| model | n | T | diff_pj | ms_pj | diff_ffi | ms_ffi | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| CANN2D | 64 | 100 | 7.45e-09 | 1.18 | n/a | n/a | n/a |
| CANN2D | 64 | 500 | 3.73e-09 | 5.54 | n/a | n/a | n/a |
| CANN2D | 64 | 1000 | 3.73e-09 | 9.75 | n/a | n/a | n/a |
| CANN2D | 256 | 100 | 7.45e-09 | 1.28 | n/a | n/a | n/a |
| CANN2D | 256 | 500 | 7.45e-09 | 5.36 | n/a | n/a | n/a |
| CANN2D | 256 | 1000 | 2.61e-08 | 10.04 | n/a | n/a | n/a |
| CANN2D | 1024 | 100 | 1.12e-08 | 1.88 | n/a | n/a | n/a |
| CANN2D | 1024 | 500 | 1.49e-08 | 6.92 | n/a | n/a | n/a |
| CANN2D | 1024 | 1000 | 1.12e-08 | 13.59 | n/a | n/a | n/a |


## GridCell

| model | n | T | diff_pj | ms_pj | diff_ffi | ms_ffi | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| GridCell | 64 | 100 | 1.16e-10 | 1.69 | n/a | n/a | n/a |
| GridCell | 64 | 500 | 2.33e-10 | 7.78 | n/a | n/a | n/a |
| GridCell | 64 | 1000 | 1.86e-09 | 15.14 | n/a | n/a | n/a |
| GridCell | 128 | 100 | 1.16e-10 | 1.68 | n/a | n/a | n/a |
| GridCell | 128 | 500 | 3.73e-09 | 7.71 | n/a | n/a | n/a |
| GridCell | 128 | 1000 | 4.66e-10 | 15.15 | n/a | n/a | n/a |
| GridCell | 256 | 100 | 3.73e-09 | 1.70 | n/a | n/a | n/a |
| GridCell | 256 | 500 | 1.16e-10 | 7.73 | n/a | n/a | n/a |
| GridCell | 256 | 1000 | 2.33e-10 | 15.28 | n/a | n/a | n/a |


## CANN-ND

| model | n | T | diff_pj | ms_pj | diff_ffi | ms_ffi | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| CANN-ND | 8 | 100 | 1.46e-11 | 1.26 | n/a | n/a | n/a |
| CANN-ND | 8 | 500 | 0.00e+00 | 4.96 | n/a | n/a | n/a |
| CANN-ND | 8 | 1000 | 5.82e-11 | 9.73 | n/a | n/a | n/a |
| CANN-ND | 16 | 100 | 1.82e-12 | 1.10 | n/a | n/a | n/a |
| CANN-ND | 16 | 500 | 4.66e-10 | 4.54 | n/a | n/a | n/a |
| CANN-ND | 16 | 1000 | 0.00e+00 | 8.68 | n/a | n/a | n/a |
| CANN-ND | 64 | 100 | 9.31e-10 | 1.22 | n/a | n/a | n/a |
| CANN-ND | 64 | 500 | 1.86e-09 | 4.87 | n/a | n/a | n/a |
| CANN-ND | 64 | 1000 | 9.31e-10 | 9.77 | n/a | n/a | n/a |
| CANN-ND | 256 | 100 | 4.66e-09 | 1.26 | n/a | n/a | n/a |
| CANN-ND | 256 | 500 | 3.73e-09 | 5.33 | n/a | n/a | n/a |
| CANN-ND | 256 | 1000 | 3.73e-09 | 10.15 | n/a | n/a | n/a |


## Cross-model at equivalent n (T=1000)

Compare speedup across models at comparable neuron counts.

| model | n | ms_pure_jax | ms_ffi | speedup |
|---|---:|---:|---:|---:|
| CANN1D | 64 | 14.89 | n/a | n/a |
| CANN1D | 128 | 15.01 | n/a | n/a |
| CANN1D | 256 | 15.05 | n/a | n/a |
| CANN2D | 64 | 9.75 | n/a | n/a |
| CANN2D | 256 | 10.04 | n/a | n/a |
| CANN2D | 1024 | 13.59 | n/a | n/a |
| GridCell | 64 | 15.14 | n/a | n/a |
| GridCell | 128 | 15.15 | n/a | n/a |
| GridCell | 256 | 15.28 | n/a | n/a |
| CANN-ND | 8 | 9.73 | n/a | n/a |
| CANN-ND | 16 | 8.68 | n/a | n/a |
| CANN-ND | 64 | 9.77 | n/a | n/a |
| CANN-ND | 256 | 10.15 | n/a | n/a |


## Per-step cost (single call, T=1)

Per-step latency (median across T values for the same model+n).
This is the true FFI speedup since scan wrapper overhead is removed.

| model | n | ms_pure_jax (step) | ms_ffi (step) | speedup |
|---|---:|---:|---:|---:|

