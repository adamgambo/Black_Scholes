"""
Black-Scholes Model Engine
Contains all pricing and Greeks calculation functions.
All core functions are numpy-vectorised — they accept scalars or arrays.
"""

import numpy as np
from scipy.stats import norm
from datetime import datetime


class BlackScholesModel:
    """Black-Scholes-Merton model for European option pricing with dividend yield support."""

    # ------------------------------------------------------------------ #
    #  Core helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_time_to_expiry(expiration_date: datetime) -> float:
        """Return time to expiry in years (minimum 1/365 to avoid divide-by-zero)."""
        days = (expiration_date - datetime.now()).days
        return max(days / 365.0, 1 / 365)

    @staticmethod
    def calculate_d1_d2(S, K, r, sigma, T, q=0.0):
        """
        Vectorised d1/d2 calculation. S and sigma may be scalars or numpy arrays.

        Parameters:
            S     : Spot price (scalar or array)
            K     : Strike price (scalar)
            r     : Risk-free rate (scalar)
            sigma : Volatility (scalar or array)
            T     : Time to expiry in years (scalar)
            q     : Continuous dividend yield (scalar, default 0)
        """
        S = np.maximum(S, 1e-8)
        sigma = np.maximum(sigma, 1e-8)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    # ------------------------------------------------------------------ #
    #  Pricing                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_option_price(S, K, r, sigma, T, option_type='call', q=0.0):
        """
        Black-Scholes-Merton option price. Accepts scalars or numpy arrays.

        Parameters:
            S           : Spot price
            K           : Strike price
            r           : Risk-free rate
            sigma       : Volatility
            T           : Time to expiry in years
            option_type : 'call' or 'put'
            q           : Continuous dividend yield (default 0)

        Returns:
            Option price (same shape as S / sigma inputs)
        """
        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, r, sigma, T, q)
        disc = np.exp(-r * T)
        S_disc = S * np.exp(-q * T)

        if option_type.lower() == 'call':
            price = S_disc * norm.cdf(d1) - K * disc * norm.cdf(d2)
        else:
            price = K * disc * norm.cdf(-d2) - S_disc * norm.cdf(-d1)

        return np.maximum(price, 0.0)

    # ------------------------------------------------------------------ #
    #  Greeks                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_greeks(S, K, r, sigma, T, option_type='call', q=0.0):
        """
        Calculate all Greeks for a single option (scalar inputs).

        Returns:
            dict with keys: delta, gamma, vega, theta, rho
        """
        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, r, sigma, T, q)

        sqrt_T = np.sqrt(T)
        exp_neg_rT = np.exp(-r * T)
        exp_neg_qT = np.exp(-q * T)
        n_prime_d1 = norm.pdf(d1)

        gamma = n_prime_d1 * exp_neg_qT / (S * sigma * sqrt_T)
        vega = S * exp_neg_qT * n_prime_d1 * sqrt_T / 100   # per 1% vol move

        if option_type.lower() == 'call':
            delta = exp_neg_qT * norm.cdf(d1)
            theta = (
                -S * exp_neg_qT * n_prime_d1 * sigma / (2 * sqrt_T)
                - r * K * exp_neg_rT * norm.cdf(d2)
                + q * S * exp_neg_qT * norm.cdf(d1)
            ) / 365
            rho = K * T * exp_neg_rT * norm.cdf(d2) / 100
        else:
            delta = exp_neg_qT * (norm.cdf(d1) - 1)
            theta = (
                -S * exp_neg_qT * n_prime_d1 * sigma / (2 * sqrt_T)
                + r * K * exp_neg_rT * norm.cdf(-d2)
                - q * S * exp_neg_qT * norm.cdf(-d1)
            ) / 365
            rho = -K * T * exp_neg_rT * norm.cdf(-d2) / 100

        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega':  float(vega),
            'theta': float(theta),
            'rho':   float(rho),
        }

    @staticmethod
    def calculate_greeks_grid(S_grid, vol_grid, K, r, T, option_type='call', q=0.0):
        """
        Vectorised Greeks over a 2-D meshgrid of spot prices × volatilities.
        Used for the 3-D surface plot — replaces the nested Python loop.

        Parameters:
            S_grid   : 2-D numpy array of spot prices (from np.meshgrid)
            vol_grid : 2-D numpy array of volatilities (from np.meshgrid)
            K, r, T, option_type, q : scalar BSM parameters

        Returns:
            dict mapping Greek name → 2-D numpy array (same shape as S_grid)
        """
        d1, d2 = BlackScholesModel.calculate_d1_d2(S_grid, K, r, vol_grid, T, q)

        sqrt_T = np.sqrt(T)
        exp_neg_rT = np.exp(-r * T)
        exp_neg_qT = np.exp(-q * T)
        n_prime_d1 = norm.pdf(d1)

        gamma = n_prime_d1 * exp_neg_qT / (S_grid * vol_grid * sqrt_T)
        vega  = S_grid * exp_neg_qT * n_prime_d1 * sqrt_T / 100

        if option_type.lower() == 'call':
            delta = exp_neg_qT * norm.cdf(d1)
            theta = (
                -S_grid * exp_neg_qT * n_prime_d1 * vol_grid / (2 * sqrt_T)
                - r * K * exp_neg_rT * norm.cdf(d2)
                + q * S_grid * exp_neg_qT * norm.cdf(d1)
            ) / 365
            rho = K * T * exp_neg_rT * norm.cdf(d2) / 100
        else:
            delta = exp_neg_qT * (norm.cdf(d1) - 1)
            theta = (
                -S_grid * exp_neg_qT * n_prime_d1 * vol_grid / (2 * sqrt_T)
                + r * K * exp_neg_rT * norm.cdf(-d2)
                - q * S_grid * exp_neg_qT * norm.cdf(-d1)
            ) / 365
            rho = -K * T * exp_neg_rT * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega':  vega,
            'theta': theta,
            'rho':   rho,
        }

    # ------------------------------------------------------------------ #
    #  Historical volatility                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_historical_volatility(price_data, window=30):
        """
        Annualised historical volatility using log returns (more accurate than
        simple percentage returns, especially over longer windows).

        Parameters:
            price_data : pd.Series of closing prices
            window     : Rolling window in trading days

        Returns:
            float: Most recent annualised HV estimate
        """
        log_returns = np.log(price_data / price_data.shift(1)).dropna()
        hv_series = log_returns.rolling(window=window).std() * np.sqrt(252)
        return float(hv_series.iloc[-1]) if len(hv_series) > 0 else 0.20

    # ------------------------------------------------------------------ #
    #  Sensitivity data (fully vectorised — no Python loop)               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_sensitivity_data(S, K, r, sigma, T, option_type='call',
                                  param='spot', range_pct=0.2, points=50, q=0.0):
        """
        Generate option price and Greeks across a parameter range.
        Fully vectorised — no Python-level loop.

        Parameters:
            param : 'spot' | 'strike' | 'volatility' | 'time'

        Returns:
            x_values (array), prices (array), greeks_dict (dict of arrays)
        """
        if param == 'spot':
            x_values = np.linspace(S * (1 - range_pct), S * (1 + range_pct), points)
            S_arr, K_arr, sigma_arr, T_arr = x_values, K, sigma, T
        elif param == 'strike':
            x_values = np.linspace(K * (1 - range_pct), K * (1 + range_pct), points)
            S_arr, K_arr, sigma_arr, T_arr = S, x_values, sigma, T
        elif param == 'volatility':
            x_values = np.linspace(max(sigma * 0.5, 0.01), sigma * 1.5, points)
            S_arr, K_arr, sigma_arr, T_arr = S, K, x_values, T
        elif param == 'time':
            x_values = np.linspace(1 / 365, T, points)
            S_arr, K_arr, sigma_arr, T_arr = S, K, sigma, x_values
        else:
            raise ValueError(f"Unknown param: {param}")

        prices = BlackScholesModel.calculate_option_price(
            S_arr, K_arr, r, sigma_arr, T_arr, option_type, q
        )

        d1, d2 = BlackScholesModel.calculate_d1_d2(S_arr, K_arr, r, sigma_arr, T_arr, q)
        sqrt_T = np.sqrt(T_arr)
        exp_neg_rT = np.exp(-r * T_arr)
        exp_neg_qT = np.exp(-q * T_arr)
        n_prime_d1 = norm.pdf(d1)

        gamma = n_prime_d1 * exp_neg_qT / (np.maximum(S_arr, 1e-8) * np.maximum(sigma_arr, 1e-8) * sqrt_T)
        vega  = np.maximum(S_arr, 1e-8) * exp_neg_qT * n_prime_d1 * sqrt_T / 100

        if option_type.lower() == 'call':
            delta = exp_neg_qT * norm.cdf(d1)
            theta = (
                -np.maximum(S_arr, 1e-8) * exp_neg_qT * n_prime_d1 * np.maximum(sigma_arr, 1e-8) / (2 * sqrt_T)
                - r * K_arr * exp_neg_rT * norm.cdf(d2)
                + q * np.maximum(S_arr, 1e-8) * exp_neg_qT * norm.cdf(d1)
            ) / 365
            rho = K_arr * T_arr * exp_neg_rT * norm.cdf(d2) / 100
        else:
            delta = exp_neg_qT * (norm.cdf(d1) - 1)
            theta = (
                -np.maximum(S_arr, 1e-8) * exp_neg_qT * n_prime_d1 * np.maximum(sigma_arr, 1e-8) / (2 * sqrt_T)
                + r * K_arr * exp_neg_rT * norm.cdf(-d2)
                - q * np.maximum(S_arr, 1e-8) * exp_neg_qT * norm.cdf(-d1)
            ) / 365
            rho = -K_arr * T_arr * exp_neg_rT * norm.cdf(-d2) / 100

        greeks_dict = {
            'delta': delta,
            'gamma': gamma,
            'vega':  vega,
            'theta': theta,
            'rho':   rho,
        }

        return x_values, prices, greeks_dict
