# Cross-Size Benchmark: FFI vs Pure JAX Speedup

Sweeps 4 models × n × T. Speedup = `ms_pure_jax / ms_ffi`.
All times in **milliseconds**.

## T = 1

n is the per-model size parameter (n for CANN1D/GridCell, L for CANN2D, side for N-D).
n_neurons is the actual neuron count (= n for CANN1D/GridCell, L² for CANN2D, n^D for N-D).

| model | n | n_neurons | ms_pj | ms_ffi | speedup |
|---|---|---|---|---|---|
| CANN1D | 16 | 16 | 0.00 | 0.00 | 1.03× |
| CANN1D | 32 | 32 | 0.00 | 0.00 | 1.11× |
| CANN1D | 64 | 64 | 0.00 | 0.00 | 1.17× |
| CANN1D | 128 | 128 | 0.01 | 0.00 | 1.54× |
| CANN1D | 256 | 256 | 0.02 | 0.01 | 2.52× |
| CANN1D | 512 | 512 | 0.02 | 0.01 | 1.54× |
| CANN2D | 4 | 16 | 0.00 | 0.00 | 1.01× |
| CANN2D | 6 | 36 | 0.00 | 0.00 | 1.26× |
| CANN2D | 8 | 64 | 0.00 | 0.00 | 1.18× |
| CANN2D | 12 | 144 | 0.01 | 0.00 | 1.48× |
| CANN2D | 16 | 256 | 0.01 | 0.01 | 2.01× |
| CANN2D | 20 | 400 | 0.02 | 0.01 | 1.66× |
| CANN2D | 24 | 576 | 0.02 | 0.02 | 1.38× |
| GridCell | 16 | 16 | 0.00 | 0.00 | 1.01× |
| GridCell | 32 | 32 | 0.00 | 0.00 | 1.22× |
| GridCell | 64 | 64 | 0.00 | 0.00 | 1.07× |
| GridCell | 128 | 128 | 0.01 | 0.00 | 1.48× |
| GridCell | 256 | 256 | 0.01 | 0.01 | 2.30× |
| GridCell | 512 | 512 | 0.03 | 0.01 | 1.89× |
| CANN-ND | 4 | 256 | 0.01 | 0.01 | 2.36× |
| CANN-ND | 6 | 216 | 0.01 | 0.01 | 2.68× |
| CANN-ND | 8 | 512 | 0.02 | 0.01 | 1.74× |
| CANN-ND | 12 | 144 | 0.01 | 0.00 | 1.51× |
| CANN-ND | 16 | 16 | 0.00 | 0.00 | 1.19× |

## T = 100

n is the per-model size parameter (n for CANN1D/GridCell, L for CANN2D, side for N-D).
n_neurons is the actual neuron count (= n for CANN1D/GridCell, L² for CANN2D, n^D for N-D).

| model | n | n_neurons | ms_pj | ms_ffi | speedup |
|---|---|---|---|---|---|
| CANN1D | 16 | 16 | 0.01 | 0.02 | 0.63× |
| CANN1D | 32 | 32 | 0.02 | 0.02 | 1.05× |
| CANN1D | 64 | 64 | 0.05 | 0.03 | 1.45× |
| CANN1D | 128 | 128 | 0.09 | 0.08 | 1.19× |
| CANN1D | 256 | 256 | 0.38 | 0.33 | 1.13× |
| CANN1D | 512 | 512 | 1.21 | 1.18 | 1.03× |
| CANN2D | 4 | 16 | 0.03 | 0.02 | 1.55× |
| CANN2D | 6 | 36 | 0.04 | 0.02 | 1.70× |
| CANN2D | 8 | 64 | 0.05 | 0.04 | 1.49× |
| CANN2D | 12 | 144 | 0.10 | 0.09 | 1.11× |
| CANN2D | 16 | 256 | 0.38 | 0.33 | 1.15× |
| CANN2D | 20 | 400 | 0.82 | 0.76 | 1.08× |
| CANN2D | 24 | 576 | 1.55 | 1.52 | 1.02× |
| GridCell | 16 | 16 | 0.01 | 0.02 | 0.59× |
| GridCell | 32 | 32 | 0.02 | 0.02 | 1.12× |
| GridCell | 64 | 64 | 0.05 | 0.04 | 1.37× |
| GridCell | 128 | 128 | 0.09 | 0.08 | 1.13× |
| GridCell | 256 | 256 | 0.37 | 0.37 | 1.00× |
| GridCell | 512 | 512 | 1.22 | 1.21 | 1.01× |
| CANN-ND | 4 | 256 | 0.38 | 0.33 | 1.15× |
| CANN-ND | 6 | 216 | 0.27 | 0.25 | 1.08× |
| CANN-ND | 8 | 512 | 1.23 | 1.17 | 1.05× |
| CANN-ND | 12 | 144 | 0.11 | 0.09 | 1.14× |
| CANN-ND | 16 | 16 | 0.03 | 0.02 | 1.49× |

## T = 1000

n is the per-model size parameter (n for CANN1D/GridCell, L for CANN2D, side for N-D).
n_neurons is the actual neuron count (= n for CANN1D/GridCell, L² for CANN2D, n^D for N-D).

| model | n | n_neurons | ms_pj | ms_ffi | speedup |
|---|---|---|---|---|---|
| CANN1D | 16 | 16 | 0.08 | 0.14 | 0.56× |
| CANN1D | 32 | 32 | 0.14 | 0.18 | 0.75× |
| CANN1D | 64 | 64 | 0.38 | 0.30 | 1.28× |
| CANN1D | 128 | 128 | 0.79 | 0.69 | 1.15× |
| CANN1D | 256 | 256 | 3.53 | 3.24 | 1.09× |
| CANN1D | 512 | 512 | 12.21 | 11.67 | 1.05× |
| CANN2D | 4 | 16 | 0.22 | 0.15 | 1.47× |
| CANN2D | 6 | 36 | 0.31 | 0.21 | 1.46× |
| CANN2D | 8 | 64 | 0.41 | 0.32 | 1.31× |
| CANN2D | 12 | 144 | 0.90 | 0.90 | 1.00× |
| CANN2D | 16 | 256 | 3.60 | 3.21 | 1.12× |
| CANN2D | 20 | 400 | 8.03 | 7.45 | 1.08× |
| CANN2D | 24 | 576 | 15.56 | 14.99 | 1.04× |
| GridCell | 16 | 16 | 0.08 | 0.17 | 0.49× |
| GridCell | 32 | 32 | 0.13 | 0.19 | 0.71× |
| GridCell | 64 | 64 | 0.39 | 0.33 | 1.18× |
| GridCell | 128 | 128 | 0.78 | 0.75 | 1.04× |
| GridCell | 256 | 256 | 3.53 | 3.57 | 0.99× |
| GridCell | 512 | 512 | 12.17 | 12.27 | 0.99× |
| CANN-ND | 4 | 256 | 3.62 | 3.17 | 1.14× |
| CANN-ND | 6 | 216 | 2.49 | 2.42 | 1.03× |
| CANN-ND | 8 | 512 | 12.29 | 11.84 | 1.04× |
| CANN-ND | 12 | 144 | 0.91 | 0.89 | 1.02× |
| CANN-ND | 16 | 16 | 0.23 | 0.15 | 1.54× |
