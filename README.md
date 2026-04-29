# garch-lnmm

Log-normal moment-matching simulation for European call option pricing under the GARCH diffusion model

## Problem

The continuous-time GARCH diffusion model is useful for stochastic volatility option pricing, but standard Euler discretization can generate negative variance values. In practice, truncation is often applied, which may introduce discretization bias under coarse time steps.

This package implements and compares a positivity-preserving Log-Normal Moment-Matching (LN-MM) scheme against Euler-conditional Monte Carlo and a Taylor-2 closed-form analytical approximation.

## Installation

```bash
pip install garch-lnmm
```
## Quick Start
```python
from garch_lnmm import GarchDiffusionMC

model = GarchDiffusionMC(
    S0=100,
    V0=0.09,
    r=0.05,
    kappa=0.75,
    theta=0.04,
    sigma=0.85,
    rho=0,
    T=5,
)

price = model.simulate_moment_matching_cond_mc(
    N_paths=1000000,
    N_steps=24,
    K=130,
    seed=42,
)

print(price)
```

## API Reference

### `GarchDiffusionMC(S0, V0, r, kappa, theta, sigma, rho, T)`

Main simulator class for the continuous-time GARCH diffusion model.

| Parameter | Type | Description |
|---|---|---|
| `S0` | float | Initial stock price |
| `V0` | float | Initial variance |
| `r` | float | Risk-free interest rate |
| `kappa` | float | Mean-reversion speed of variance |
| `theta` | float | Long-run variance level |
| `sigma` | float | Volatility of variance |
| `rho` | float | Correlation parameter |
| `T` | float | Maturity |

---

### `simulate_euler_cond_mc(N_paths, N_steps, K, seed=None, return_stats=False)`

Prices a European call using truncated Euler conditional Monte Carlo.

| Parameter | Type | Description |
|---|---|---|
| `N_paths` | int | Number of Monte Carlo paths |
| `N_steps` | int | Number of time steps |
| `K` | float | Strike price |
| `seed` | int or None | Random seed |
| `return_stats` | bool | If True, return price and error statistics |

Returns:

| Field | Type | Description |
|---|---|---|
| `price` | float | Estimated option price |
| `std_dev` | float | Sample standard deviation |
| `std_error` | float | Monte Carlo standard error |

---

### `simulate_moment_matching_cond_mc(N_paths, N_steps, K, seed=None, return_stats=False)`

Prices a European call using the Log-Normal Moment-Matching scheme.

| Parameter | Type | Description |
|---|---|---|
| `N_paths` | int | Number of Monte Carlo paths |
| `N_steps` | int | Number of time steps |
| `K` | float | Strike price |
| `seed` | int or None | Random seed |
| `return_stats` | bool | If True, return price and error statistics |

Returns:

| Field | Type | Description |
|---|---|---|
| `price` | float | Estimated option price |
| `std_dev` | float | Sample standard deviation |
| `std_error` | float | Monte Carlo standard error |

---

### `garch_taylor2_call(K)`

Computes the Taylor-2 analytical approximation.

| Parameter | Type | Description |
|---|---|---|
| `K` | float | Strike price |

Returns:

| Field | Type | Description |
|---|---|---|
| price | float | Taylor-2 approximated option price |

## License
MIT

## Demo Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JM357/MATH5030Group11/blob/main/notebooks/demo.ipynb)
