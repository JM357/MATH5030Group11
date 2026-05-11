import numpy as np
from garch_lnmm import GarchDiffusionMC

def make_model():
    return GarchDiffusionMC(
        S0=100,
        V0=0.09,
        r=0.05,
        kappa=0.75,
        theta=0.04,
        sigma=0.85,
        rho=0,
        T=5,
    )

def test_lnmm_price_is_positive():
    model = make_model()
    price = model.simulate_moment_matching_cond_mc(
        N_paths=1000,
        N_steps=6,
        K=130,
        seed=42,
    )
    assert np.isfinite(price)
    assert price > 0

def test_slnmm_price_is_positive_and_finite():
    model = make_model()
    price = model.simulate_shifted_lognormal_cond_mc(
        N_paths=1000,
        N_steps=6,
        K=130,
        seed=42,
    )

    assert np.isfinite(price)
    assert price > 0
