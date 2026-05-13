import numpy as np
from scipy.stats import norm
from scipy.linalg import expm


class GarchDiffusionMC:
    """
    Monte Carlo simulator for the continuous-time GARCH diffusion model.
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

    # --- Basic Utilities ---
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
        vol = np.sqrt(np.maximum(avg_var, 1e-14))
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
    
    # --- In-House Euler conditional Monte Carlo method ---
    def simulate_euler_cond_mc(self, N_paths, N_steps, K, seed=None, return_stats=False):
        rng = np.random.default_rng(seed)

        dt = self.T / N_steps
        sqrt_dt = np.sqrt(dt)

        V = np.full(N_paths, self.V0)
        int_var = np.zeros(N_paths)
        Z = np.empty(N_paths, dtype=np.float64)

        for _ in range(N_steps):
            V_old = V.copy()
            self._fill_antithetic_normals(rng, Z)

            V_next = V_old + self.kappa * (self.theta - V_old) * dt + self.sigma * V_old * sqrt_dt * Z
            V = np.maximum(V_next, 1e-12)

            int_var += 0.5 * (V_old + V) * dt

        avg_var = int_var / self.T
        summary = self._price_summary(self.bs_call_from_variance(avg_var, K))

        if return_stats:
            return summary

        return summary["price"]

    
    # --- Helper functions for the shifted-lognormal moment-matching method ---
    def _transition_moment_matrix(self, dt):
        """
        Matrix exponential for the first three raw transition moments of V.

        For dV = kappa(theta - V)dt + sigma V dW,

        M_n'(h) = n c1 M_{n-1}(h)
                  + [-n c2 + 0.5 n(n-1)c3^2] M_n(h),

        where c1 = kappa * theta, c2 = kappa, c3 = sigma.

        The state vector is [1, E[V], E[V^2], E[V^3]].
        """
        c1 = self.kappa * self.theta
        c2 = self.kappa
        c3 = self.sigma

        a1 = -c2
        a2 = -2.0 * c2 + c3**2
        a3 = -3.0 * c2 + 3.0 * c3**2

        A = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [c1, a1, 0.0, 0.0],
                [0.0, 2.0 * c1, a2, 0.0],
                [0.0, 0.0, 3.0 * c1, a3],
            ],
            dtype=np.float64,
        )

        return expm(A * dt)

    def _transition_raw_moments_first3(self, V, moment_matrix):
        """
        Return raw moments m1, m2, m3 of V_{t+dt} conditional on V_t = V.
        """
        V2 = V * V
        V3 = V2 * V

        m1 = (
            moment_matrix[1, 0]
            + moment_matrix[1, 1] * V
            + moment_matrix[1, 2] * V2
            + moment_matrix[1, 3] * V3
        )

        m2 = (
            moment_matrix[2, 0]
            + moment_matrix[2, 1] * V
            + moment_matrix[2, 2] * V2
            + moment_matrix[2, 3] * V3
        )

        m3 = (
            moment_matrix[3, 0]
            + moment_matrix[3, 1] * V
            + moment_matrix[3, 2] * V2
            + moment_matrix[3, 3] * V3
        )

        return m1, m2, m3

    def _sln_params_from_raw_moments(self, m1, m2, m3):
        """
        Convert the first three raw moments into shifted-lognormal parameters.

        The shifted-lognormal update has the form

            Y = mu * [(1 - lam) + lam * exp(sigma_sln * Z - 0.5 * sigma_sln^2)],

        where Z is standard normal. The parameters are chosen to match the
        conditional mean, variance, and skewness implied by m1, m2, and m3.
        """
        mu = np.maximum(m1, 1e-14)

        var = m2 - m1**2
        var = np.maximum(var, 0.0)

        std = np.sqrt(var)
        cv = std / mu   # Coefficient of variation

        mu3_central = m3 - 3.0 * m1 * m2 + 2.0 * m1**3

        skew = np.zeros_like(mu)
        valid_var = var > 1e-20

        skew[valid_var] = (
            mu3_central[valid_var]
            / (var[valid_var] ** 1.5)
        )

        # The SLN formula requires positive skewness.
        skew_pos = np.maximum(skew, 0.0)

        w = np.zeros_like(mu)
        valid_skew = skew_pos > 1e-12

        w[valid_skew] = (
            4.0
            * np.sinh(
                (1.0 / 6.0)
                * np.arccosh(1.0 + 0.5 * skew_pos[valid_skew] ** 2)
            ) ** 2
        )

        sigma_sln = np.zeros_like(mu)
        lam = np.ones_like(mu)

        valid_w = w > 1e-20
        sigma_sln[valid_w] = np.sqrt(np.log1p(w[valid_w]))
        lam[valid_w] = cv[valid_w] / np.sqrt(w[valid_w])

        return mu, sigma_sln, lam, var, skew

    # --- LNMM method using first and second moments ---
    def simulate_moment_matching_cond_mc(self, N_paths, N_steps, K, seed=None, return_stats=False):
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
                + exp_c2 * (2.0 * theta * (V_old - theta)) / (c2 / c3**2 - 1.0)
                - exp_2c2 * (V_old - theta)**2
                + exp_c3_2_minus_2c2 * (
                    V_old**2
                    - (2.0 * V_old * theta) / (1.0 - c3**2 / c2)
                    + theta**2 / (
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

    # --- shifted-lognormal moment-matching method using first three moments ---
    def simulate_shifted_lognormal_cond_mc(
        self,
        N_paths,
        N_steps,
        K,
        seed=None,
        return_stats=False,
    ):
        """
        Price a European call using skewness-matched shifted-lognormal
        moment-matching conditional Monte Carlo.

        This version follows the direct shifted-lognormal implementation:
        after matching the first three transition moments, it always samples
        from the shifted-lognormal update and does not fall back to ordinary
        two-moment LNMM.
        """
        assert self.rho == 0, "Conditional Monte Carlo requires rho = 0."

        rng = np.random.default_rng(seed)

        dt = self.T / N_steps

        V = np.full(N_paths, self.V0, dtype=np.float64)
        int_var = np.zeros(N_paths, dtype=np.float64)
        Z = np.empty(N_paths, dtype=np.float64)

        moment_matrix = self._transition_moment_matrix(dt)

        for _ in range(N_steps):
            V_old = V.copy()
            self._fill_antithetic_normals(rng, Z)

            m1, m2, m3 = self._transition_raw_moments_first3(
                V_old,
                moment_matrix,
            )

            mu, sigma_sln, lam, var, skew = self._sln_params_from_raw_moments(
                m1,
                m2,
                m3,
            )

            V_next = mu * (
                (1.0 - lam)
                + lam * np.exp(sigma_sln * Z - 0.5 * sigma_sln**2)
            )

            V = np.maximum(V_next, 1e-12)
            int_var += 0.5 * (V_old + V) * dt

        avg_var = int_var / self.T
        conditional_prices = self.bs_call_from_variance(avg_var, K)
        summary = self._price_summary(conditional_prices)

        if return_stats:
            return summary

        return summary["price"]

    
