"""
Black-Scholes Model Engine
Contains all pricing and Greeks calculation functions
"""

import numpy as np
from scipy.stats import norm
from datetime import datetime


class BlackScholesModel:
    """
    Black-Scholes Model for European Option Pricing
    """
    
    @staticmethod
    def calculate_time_to_expiry(expiration_date):
        """
        Calculate time to expiry as a fraction of a year
        
        Args:
            expiration_date: datetime object of expiration date
            
        Returns:
            T: Time to expiry in years
        """
        current_date = datetime.now()
        days_to_expiry = (expiration_date - current_date).days
        T = max(days_to_expiry / 365.0, 0.0001)  # Avoid division by zero
        return T
    
    @staticmethod
    def calculate_d1_d2(S, K, r, sigma, T):
        """
        Calculate d1 and d2 parameters for Black-Scholes formula
        
        Args:
            S: Current stock price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiry
            
        Returns:
            d1, d2: Black-Scholes parameters
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def calculate_option_price(S, K, r, sigma, T, option_type='call'):
        """
        Calculate option price using Black-Scholes formula
        
        Args:
            S: Current stock price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiry
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, r, sigma, T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        return price
    
    @staticmethod
    def calculate_greeks(S, K, r, sigma, T, option_type='call'):
        """
        Calculate all Greeks for an option
        
        Args:
            S: Current stock price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiry
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary containing all Greeks
        """
        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, r, sigma, T)
        
        # Common calculations
        sqrt_T = np.sqrt(T)
        exp_neg_rT = np.exp(-r * T)
        n_prime_d1 = norm.pdf(d1)  # N'(d1)
        
        greeks = {}
        
        if option_type.lower() == 'call':
            # Call Greeks
            greeks['delta'] = norm.cdf(d1)
            greeks['gamma'] = n_prime_d1 / (S * sigma * sqrt_T)
            greeks['vega'] = S * n_prime_d1 * sqrt_T / 100  # Divided by 100 for 1% change
            greeks['theta'] = (
                -S * n_prime_d1 * sigma / (2 * sqrt_T) 
                - r * K * exp_neg_rT * norm.cdf(d2)
            ) / 365  # Divided by 365 for daily theta
            greeks['rho'] = K * T * exp_neg_rT * norm.cdf(d2) / 100  # Divided by 100 for 1% change
        else:  # put
            # Put Greeks
            greeks['delta'] = norm.cdf(d1) - 1
            greeks['gamma'] = n_prime_d1 / (S * sigma * sqrt_T)
            greeks['vega'] = S * n_prime_d1 * sqrt_T / 100  # Divided by 100 for 1% change
            greeks['theta'] = (
                -S * n_prime_d1 * sigma / (2 * sqrt_T) 
                + r * K * exp_neg_rT * norm.cdf(-d2)
            ) / 365  # Divided by 365 for daily theta
            greeks['rho'] = -K * T * exp_neg_rT * norm.cdf(-d2) / 100  # Divided by 100 for 1% change
            
        return greeks
    
    @staticmethod
    def calculate_historical_volatility(price_data, window=30):
        """
        Calculate historical volatility from price data
        
        Args:
            price_data: Pandas Series of historical prices
            window: Number of days for calculation
            
        Returns:
            Annualized historical volatility
        """
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        # Calculate volatility
        daily_vol = returns.rolling(window=window).std()
        annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days per year
        
        return annualized_vol.iloc[-1] if len(annualized_vol) > 0 else 0.2  # Default to 20%
    
    @staticmethod
    def generate_sensitivity_data(S, K, r, sigma, T, option_type='call', 
                                  param='spot', range_pct=0.2, points=50):
        """
        Generate data for sensitivity analysis
        
        Args:
            S, K, r, sigma, T: Black-Scholes parameters
            option_type: 'call' or 'put'
            param: Parameter to vary ('spot', 'strike', 'volatility', 'time')
            range_pct: Percentage range to vary the parameter
            points: Number of data points
            
        Returns:
            x_values, prices, greeks_dict
        """
        base_params = {'S': S, 'K': K, 'r': r, 'sigma': sigma, 'T': T}
        
        if param == 'spot':
            x_values = np.linspace(S * (1 - range_pct), S * (1 + range_pct), points)
            vary_param = 'S'
        elif param == 'strike':
            x_values = np.linspace(K * (1 - range_pct), K * (1 + range_pct), points)
            vary_param = 'K'
        elif param == 'volatility':
            x_values = np.linspace(sigma * 0.5, sigma * 1.5, points)
            vary_param = 'sigma'
        elif param == 'time':
            x_values = np.linspace(0.01, T, points)
            vary_param = 'T'
        
        prices = []
        greeks_dict = {greek: [] for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']}
        
        for x in x_values:
            params = base_params.copy()
            params[vary_param] = x
            
            price = BlackScholesModel.calculate_option_price(
                params['S'], params['K'], params['r'], 
                params['sigma'], params['T'], option_type
            )
            greeks = BlackScholesModel.calculate_greeks(
                params['S'], params['K'], params['r'], 
                params['sigma'], params['T'], option_type
            )
            
            prices.append(price)
            for greek, value in greeks.items():
                greeks_dict[greek].append(value)
        
        return x_values, prices, greeks_dict