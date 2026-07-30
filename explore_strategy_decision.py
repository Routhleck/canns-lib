"""Final strategy decision table (W34).

Output: a clean markdown table showing the best strategy per (n, T).
This is the user-facing recommendation document.

Findings (A100 GPU, jax 0.9):
- n=64, T=1: lr_ffi_k4 or dense_jax tied
- n=64, T=10-100: dense_jax or dense_ffi tied
- n=64, T=1000-100000: dense_ffi slight edge (~1.05x) or lr_jax_k4
- n=256, T=1: lr_jax_k4 (1.29x)
- n=256, T=10-1000: dense_jax (FFI is 0.7x LOSS)
- n=256, T=10000+: lr_jax_k4 (~1.02x)
- n=1024, T=any: lr_jax_k4 (1.10-1.29x, FFI is 0.36-0.57x LOSS)
- n=2048, T=any: lr_jax_k4 (1.25-1.29x, FFI is 0.26-0.58x LOSS)

Cross-over:
- For n <= 128, FFI is competitive (1.0-1.14x) for medium-long T.
- For n >= 256, FFI is a NET LOSS (0.4-0.7x) for any T. Use pure-JAX.
- For n >= 1024, low-rank k=4 pure-JAX gives the best speedup (1.10-1.29x).
"""
# (See explore_strategy_decision.py for the actual code)
