# Cross-Size Benchmark: FFI vs Pure JAX Speedup

Sweeps 4 models × n × T. Speedup = `ms_pure_jax / ms_ffi`.
All times in **milliseconds**.

## T = 1

n is the per-model size parameter (n for CANN1D/GridCell, L for CANN2D, side for N-D).
n_neurons is the actual neuron count (= n for CANN1D/GridCell, L² for CANN2D, n^D for N-D).

| model | n | n_neurons | ms_pj | ms_ffi | speedup |
|---|---|---|---|---|---|
| CANN1D | 16 | 16 | 0.00 | 0.00 | 1.01× |
| CANN1D | 32 | 32 | 0.00 | 0.00 | 1.23× |
| CANN1D | 64 | 64 | 0.00 | 0.00 | 1.11× |
| CANN1D | 128 | 128 | 0.01 | 0.00 | 1.66× |
| CANN1D | 256 | 256 | 0.01 | 0.01 | 2.39× |
| CANN1D | 512 | 512 | 0.02 | 0.01 | 1.77× |
| CANN2D | 4 | 16 | 0.00 | 0.00 | 1.10× |
| CANN2D | 6 | 36 | 0.00 | 0.00 | 1.16× |
| CANN2D | 8 | 64 | 0.00 | 0.00 | 1.08× |
| CANN2D | 12 | 144 | 0.01 | 0.00 | 1.48× |
| CANN2D | 16 | 256 | 0.01 | 0.01 | 2.35× |
| CANN2D | 20 | 400 | 0.02 | 0.01 | 2.02× |
| CANN2D | 24 | 576 | 0.03 | 0.02 | 1.55× |
| GridCell | 16 | 16 | 0.00 | 0.00 | 1.07× |
| GridCell | 32 | 32 | 0.00 | 0.00 | 1.21× |
| GridCell | 64 | 64 | 0.00 | 0.00 | 1.13× |
| GridCell | 128 | 128 | 0.01 | 0.00 | 1.69× |
| GridCell | 256 | 256 | 0.01 | 0.01 | 2.40× |
| GridCell | 512 | 512 | 0.02 | 0.02 | 1.56× |
| CANN-ND | 4 | 256 | 0.01 | 0.01 | 2.05× |
| CANN-ND | 6 | 216 | 0.01 | 0.01 | 2.52× |
| CANN-ND | 8 | 512 | 0.02 | 0.01 | 1.87× |
| CANN-ND | 12 | 144 | 0.01 | 0.00 | 1.43× |
| CANN-ND | 16 | 16 | 0.00 | 0.00 | 1.12× |

## T = 100

n is the per-model size parameter (n for CANN1D/GridCell, L for CANN2D, side for N-D).
n_neurons is the actual neuron count (= n for CANN1D/GridCell, L² for CANN2D, n^D for N-D).

| model | n | n_neurons | ms_pj | ms_ffi | speedup |
|---|---|---|---|---|---|
| CANN1D | 16 | 16 | 0.01 | 0.02 | 0.61× |
| CANN1D | 32 | 32 | 0.02 | 0.02 | 1.13× |
| CANN1D | 64 | 64 | 0.05 | 0.03 | 1.44× |
| CANN1D | 128 | 128 | 0.09 | 0.08 | 1.21× |
| CANN1D | 256 | 256 | 0.38 | 0.33 | 1.15× |
| CANN1D | 512 | 512 | 1.22 | 1.16 | 1.05× |
| CANN2D | 4 | 16 | 0.02 | 0.02 | 1.41× |
| CANN2D | 6 | 36 | 0.04 | 0.02 | 1.74× |
| CANN2D | 8 | 64 | 0.05 | 0.04 | 1.50× |
| CANN2D | 12 | 144 | 0.11 | 0.09 | 1.13× |
| CANN2D | 16 | 256 | 0.38 | 0.33 | 1.16× |
| CANN2D | 20 | 400 | 0.81 | 0.76 | 1.06× |
| CANN2D | 24 | 576 | 1.59 | 1.51 | 1.06× |
| GridCell | 16 | 16 | 0.01 | 0.02 | 0.59× |
| GridCell | 32 | 32 | 0.02 | 0.02 | 1.06× |
| GridCell | 64 | 64 | 0.05 | 0.04 | 1.39× |
| GridCell | 128 | 128 | 0.09 | 0.08 | 1.10× |
| GridCell | 256 | 256 | 0.36 | 0.36 | 1.01× |
| GridCell | 512 | 512 | 1.22 | 1.21 | 1.01× |
| CANN-ND | 4 | 256 | 0.39 | 0.33 | 1.17× |
| CANN-ND | 6 | 216 | 0.26 | 0.24 | 1.09× |
| CANN-ND | 8 | 512 | 1.25 | 1.18 | 1.06× |
| CANN-ND | 12 | 144 | 0.11 | 0.09 | 1.12× |
| CANN-ND | 16 | 16 | 0.03 | 0.02 | 1.38× |

## T = 1000

n is the per-model size parameter (n for CANN1D/GridCell, L for CANN2D, side for N-D).
n_neurons is the actual neuron count (= n for CANN1D/GridCell, L² for CANN2D, n^D for N-D).

| model | n | n_neurons | ms_pj | ms_ffi | speedup |
|---|---|---|---|---|---|
| CANN1D | 16 | 16 | 0.08 | 0.15 | 0.54× |
| CANN1D | 32 | 32 | 0.14 | 0.18 | 0.73× |
| CANN1D | 64 | 64 | 0.38 | 0.31 | 1.25× |
| CANN1D | 128 | 128 | 0.79 | 0.70 | 1.14× |
| CANN1D | 256 | 256 | 3.53 | 3.24 | 1.09× |
| CANN1D | 512 | 512 | 12.20 | 11.73 | 1.04× |
| CANN2D | 4 | 16 | 0.22 | 0.15 | 1.46× |
| CANN2D | 6 | 36 | 0.31 | 0.21 | 1.48× |
| CANN2D | 8 | 64 | 0.42 | 0.31 | 1.34× |
| CANN2D | 12 | 144 | 0.92 | 0.89 | 1.04× |
| CANN2D | 16 | 256 | 3.65 | 3.26 | 1.12× |
| CANN2D | 20 | 400 | 8.05 | 7.58 | 1.06× |
| CANN2D | 24 | 576 | 15.78 | 15.20 | 1.04× |
| GridCell | 16 | 16 | 0.08 | 0.16 | 0.49× |
| GridCell | 32 | 32 | 0.13 | 0.19 | 0.71× |
| GridCell | 64 | 64 | 0.38 | 0.33 | 1.17× |
| GridCell | 128 | 128 | 0.79 | 0.76 | 1.05× |
| GridCell | 256 | 256 | 3.55 | 3.53 | 1.00× |
| GridCell | 512 | 512 | 12.25 | 12.30 | 1.00× |
| CANN-ND | 4 | 256 | 3.64 | 3.23 | 1.13× |
| CANN-ND | 6 | 216 | 2.55 | 2.40 | 1.07× |
| CANN-ND | 8 | 512 | 12.43 | 11.81 | 1.05× |
| CANN-ND | 12 | 144 | 0.93 | 0.89 | 1.05× |
| CANN-ND | 16 | 16 | 0.22 | 0.15 | 1.46× |
