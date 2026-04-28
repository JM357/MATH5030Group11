import numpy as np
from garch_lnmm import GarchDiffusionMC


def test_taylor2_price_is_finite():
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
    price = model.garch_taylor2_call(K=130)
    assert np.isfinite(price)


def test_lnmm_price_is_positive():
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
        N_paths=1000,
        N_steps=6,
        K=130,
        seed=42,
    )
    assert np.isfinite(price)
    assert price > 0
