# Black-Scholes Option Pricing Application

A comprehensive Streamlit application for real-time option pricing using the Black-Scholes model with live market data integration.

## Features

### Core Functionality
- **Real-time Stock Data**: Fetches live prices using Yahoo Finance
- **Option Pricing**: Calculates Call and Put option prices using Black-Scholes formula
- **Greeks Calculation**: Complete Greeks (Delta, Gamma, Vega, Theta, Rho)
- **Historical Volatility**: Calculates volatility from historical price data
- **Auto-refresh**: Optional automatic data refresh every 10 seconds

### Visualizations
1. **Price History Chart**: 3-month historical stock prices with volume
2. **Sensitivity Analysis**: Interactive charts showing how option price and Greeks change with parameters
3. **3D Greeks Surface**: 3D visualization of Greeks across spot price and volatility ranges
4. **Payoff Diagram**: Option payoff at expiration with breakeven analysis

### User Inputs
- Stock ticker symbol
- Option type (Call/Put)
- Strike price
- Expiration date
- Risk-free rate
- Implied volatility
- Number of contracts
- Contract multiplier

## Installation

1. Clone or download the project files:
```
black_scholes_app/
├── app.py                 # Main Streamlit application
├── bs_model.py           # Black-Scholes model engine
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The app will open in your browser at `http://localhost:8501`

3. Enter your parameters in the sidebar:
   - Stock ticker (e.g., AAPL, TSLA, MSFT)
   - Option type and parameters
   - Adjust volatility and risk-free rate as needed

4. View results:
   - Current option price and total value
   - All Greeks with explanations
   - Interactive charts and analysis

## Key Components

### bs_model.py
The Black-Scholes engine includes:
- `calculate_option_price()`: Core pricing formula
- `calculate_greeks()`: All five Greeks calculations
- `calculate_historical_volatility()`: Historical volatility from price data
- `generate_sensitivity_data()`: Data for sensitivity analysis

### app.py
The Streamlit interface provides:
- Clean, intuitive UI with sidebar inputs
- Real-time data fetching and display
- Interactive Plotly charts
- Auto-refresh capability
- Error handling for invalid inputs

## Mathematical Formulas

### Black-Scholes Formula
- **d₁** = [ln(S/K) + (r + σ²/2)T] / (σ√T)
- **d₂** = d₁ - σ√T
- **Call Price**: C = S·N(d₁) - K·e^(-rT)·N(d₂)
- **Put Price**: P = K·e^(-rT)·N(-d₂) - S·N(-d₁)

### Greeks
- **Delta (Δ)**: Rate of change in option price with respect to stock price
- **Gamma (Γ)**: Rate of change in delta with respect to stock price
- **Vega (ν)**: Sensitivity to volatility changes
- **Theta (Θ)**: Time decay of option value
- **Rho (ρ)**: Sensitivity to interest rate changes

## Notes

- The model assumes European-style options (exercise only at expiration)
- No dividend payments are considered
- Assumes constant volatility and risk-free rate
- Market prices may differ due to bid-ask spreads, liquidity, and other factors

## Troubleshooting

- **Invalid ticker**: Ensure you're using valid stock symbols
- **No data**: Check internet connection for live data fetching
- **Calculation errors**: Verify all inputs are positive and expiration is in the future

## Future Enhancements

Potential additions:
- Implied volatility solver
- American option pricing
- Dividend adjustment
- Option chains display
- Portfolio analysis
- Risk metrics (VaR, etc.)

## License

This is an educational tool for understanding option pricing. Use at your own risk for actual trading decisions.
