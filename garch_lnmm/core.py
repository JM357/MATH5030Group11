import numpy as np
from scipy.stats import norm


class GarchDiffusionMC:
    """
    Monte Carlo simulator for the continuous-time GARCH diffusion model.

    The current implementation focuses on the rho = 0 case, where conditional
    Monte Carlo pricing can be used for European call options.
    """

    def __init__(self, S0, V0, r, kappa, theta, sigma, rho, T):
        self.S0 = S0
        self.V0 = V0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.T = T

    def _price_summary(self, conditional_prices):
        price = float(np.mean(conditional_prices))
        std_dev = float(np.std(conditional_prices, ddof=1))
        std_error = float(std_dev / np.sqrt(conditional_prices.size))

        return {
            "price": price,
            "std_dev": std_dev,
            "std_error": std_error,
        }

    def _fill_antithetic_normals(self, rng, Z):
        """
        Fill Z with antithetic standard-normal pairs.
        """
        half_paths = Z.size // 2
        Z_half = rng.standard_normal(half_paths)

        Z[:half_paths] = Z_half
        Z[half_paths:2 * half_paths] = -Z_half

        if Z.size % 2:
            Z[-1] = rng.standard_normal()

        return Z

    def bs_call_from_variance(self, avg_var, K):
        """
        Black-Scholes call price using average variance over [0, T].

        Parameters
        ----------
        avg_var : float or array-like
            Average integrated variance.
        K : float
            Strike price.

        Returns
        -------
        float or np.ndarray
            Black-Scholes call price conditional on avg_var.
        """
        avg_var = np.maximum(avg_var, 1e-14)
        vol = np.sqrt(avg_var)
        T = self.T

        d1 = (
            np.log(self.S0 / K)
            + (self.r + 0.5 * avg_var) * T
        ) / (vol * np.sqrt(T))

        d2 = d1 - vol * np.sqrt(T)

        return (
            self.S0 * norm.cdf(d1)
            - K * np.exp(-self.r * T) * norm.cdf(d2)
        )

    def simulate_euler_cond_mc(
        self,
        N_paths,
        N_steps,
        K,
        seed=None,
        return_stats=False,
    ):
        """
        Price a European call using truncated Euler conditional Monte Carlo.

        This method assumes rho = 0.
        """
        assert self.rho == 0, "Conditional Monte Carlo requires rho = 0."

        rng = np.random.default_rng(seed)

        dt = self.T / N_steps
        sqrt_dt = np.sqrt(dt)

        V = np.full(N_paths, self.V0, dtype=np.float64)
        int_var = np.zeros(N_paths, dtype=np.float64)
        Z = np.empty(N_paths, dtype=np.float64)

        for _ in range(N_steps):
            V_old = V.copy()
            self._fill_antithetic_normals(rng, Z)

            V_next = (
                V_old
                + self.kappa * (self.theta - V_old) * dt
                + self.sigma * V_old * sqrt_dt * Z
            )

            V = np.maximum(V_next, 1e-8)
            int_var += 0.5 * (V_old + V) * dt

        avg_var = int_var / self.T
        conditional_prices = self.bs_call_from_variance(avg_var, K)
        summary = self._price_summary(conditional_prices)

        if return_stats:
            return summary

        return summary["price"]

    def simulate_moment_matching_cond_mc(
        self,
        N_paths,
        N_steps,
        K,
        seed=None,
        return_stats=False,
    ):
        """
        Price a European call using log-normal moment-matching conditional Monte Carlo.

        This method assumes rho = 0.
        """
        assert self.rho == 0, "Conditional Monte Carlo requires rho = 0."

        rng = np.random.default_rng(seed)

        dt = self.T / N_steps

        V = np.full(N_paths, self.V0, dtype=np.float64)
        int_var = np.zeros(N_paths, dtype=np.float64)
        Z = np.empty(N_paths, dtype=np.float64)

        c2 = self.kappa
        c3 = self.sigma
        theta = self.theta

        exp_c2 = np.exp(-c2 * dt)
        exp_2c2 = np.exp(-2.0 * c2 * dt)
        exp_c3_2_minus_2c2 = np.exp((c3**2 - 2.0 * c2) * dt)

        for _ in range(N_steps):
            V_old = V.copy()
            self._fill_antithetic_normals(rng, Z)

            M = theta + (V_old - theta) * exp_c2

            Var_exact = (
                theta**2 / (2.0 * c2 / c3**2 - 1.0)
                + exp_c2
                * (2.0 * theta * (V_old - theta))
                / (c2 / c3**2 - 1.0)
                - exp_2c2 * (V_old - theta) ** 2
                + exp_c3_2_minus_2c2
                * (
                    V_old**2
                    - (2.0 * V_old * theta) / (1.0 - c3**2 / c2)
                    + theta**2
                    / (
                        (1.0 - c3**2 / (2.0 * c2))
                        * (1.0 - c3**2 / c2)
                    )
                )
            )

            M = np.maximum(M, 1e-14)
            Var_exact = np.maximum(Var_exact, 0.0)

            ln_var = np.log1p(Var_exact / M**2)
            ln_vol = np.sqrt(ln_var)
            ln_mean = np.log(M) - 0.5 * ln_var

            V = np.exp(ln_mean + ln_vol * Z)
            int_var += 0.5 * (V_old + V) * dt

        avg_var = int_var / self.T
        conditional_prices = self.bs_call_from_variance(avg_var, K)
        summary = self._price_summary(conditional_prices)

        if return_stats:
            return summary

        return summary["price"]

    def bs_second_derivative_avg_variance(self, Vbar, K):
        """
        Second derivative of the Black-Scholes call price with respect to average variance.
        """
        Vbar = np.maximum(Vbar, 1e-16)
        T = self.T

        m = np.log(self.S0 / K) + self.r * T
        d1 = (m + 0.5 * Vbar * T) / np.sqrt(Vbar * T)

        dC_dV = (
            self.S0
            * np.sqrt(T)
            * np.exp(-0.5 * d1**2)
            / np.sqrt(8.0 * np.pi * Vbar)
        )

        bracket = (
            0.5 * m**2 / (Vbar * T) ** 2
            - 1.0 / (2.0 * Vbar * T)
            - 1.0 / 8.0
        )

        return dC_dV * bracket * T

    def garch_M1_M2c_integrated_variance(self):
        """
        First and centered second moment of average integrated variance
        for the GARCH diffusion model.
        """
        c1 = self.kappa * self.theta
        c2 = self.kappa
        c3 = self.sigma
        V0 = self.V0
        T = self.T

        assert 2 * c2 > c3**2, "Need 2 * kappa > sigma^2 for finite second moment."

        exp1 = np.exp(-c2 * T)
        exp2 = np.exp(-2.0 * c2 * T)
        exp3 = np.exp((c3**2 - 2.0 * c2) * T)

        M1 = c1 / c2 + (V0 - c1 / c2) * (1.0 - exp1) / (c2 * T)

        term1 = -exp2 * (c2 * V0 - c1) ** 2 / (T**2 * c2**4)

        term2 = (
            2.0
            * exp3
            * (
                2.0 * c1**2
                + 2.0 * c1 * (c3**2 - 2.0 * c2) * V0
                + (2.0 * c2**2 - 3.0 * c2 * c3**2 + c3**4) * V0**2
            )
            / (T**2 * (c2 - c3**2) ** 2 * (2.0 * c2 - c3**2) ** 2)
        )

        term3 = (
            -c3**2
            * (
                c1**2 * (4.0 * c2 * (3.0 - T * c2) + (2.0 * T * c2 - 5.0) * c3**2)
                + 2.0 * c1 * c2 * (-2.0 * c2 + c3**2) * V0
                + c2**2 * (-2.0 * c2 + c3**2) * V0**2
            )
            / (T**2 * c2**4 * (-2.0 * c2 + c3**2) ** 2)
        )

        term4 = (
            2.0
            * exp1
            * c3**2
            * (
                2.0 * c1**2 * (T * c2**2 - (1.0 + T * c2) * c3**2)
                + 2.0 * c1 * c2**2 * (1.0 - T * c2 + T * c3**2) * V0
                + c2**2 * (c3**2 - c2) * V0**2
            )
            / (T**2 * c2**4 * (c2 - c3**2) ** 2)
        )

        M2c = term1 + term2 + term3 + term4

        return M1, M2c

    def garch_taylor2_call(self, K):
        """
        Taylor-2 analytical approximation for the GARCH diffusion call price.
        """
        M1, M2c = self.garch_M1_M2c_integrated_variance()

        bs_base = self.bs_call_from_variance(M1, K)
        second_deriv = self.bs_second_derivative_avg_variance(M1, K)

        price = bs_base + 0.5 * M2c * second_deriv

        return float(price)
