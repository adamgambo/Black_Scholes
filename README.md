# Black-Scholes Option Pricer

A Streamlit web app for real-time European option pricing using the Black-Scholes-Merton model with live Yahoo Finance data.

## Features

- **Live market data** — spot price, option chain, bid/ask, volume, open interest
- **BSM pricing** — call and put prices with continuous dividend yield support
- **Full Greeks** — Delta, Gamma, Vega, Theta, Rho
- **Historical volatility** — log-return based, configurable rolling window
- **Four analysis tabs**:
  - Price History — 3-month OHLCV chart with volume bars
  - Sensitivity Analysis — option price and Greeks vs any BSM parameter (fully vectorised)
  - Greeks Surface — 3D surface plot across spot × volatility grid (vectorised, no Python loop)
  - Payoff Diagram — expiration P&L with breakeven, strike, and spot markers
- **Caching** — market data cached (30–300 s TTL) so widget changes don't trigger refetches
- **Auto-refresh** — optional 30-second refresh cycle

## Installation

```bash
git clone https://github.com/adamgambo/Black_Scholes.git
cd Black_Scholes

python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

1. Enter a stock ticker in the sidebar (e.g. `AAPL`, `SPY`, `TSLA`)
2. Select expiration date and strike price — market IV is auto-populated
3. Adjust risk-free rate and contract parameters as needed
4. Explore the four analysis tabs

## Project Structure

```
Black_Scholes/
├── app.py          # Streamlit UI — data fetching, layout, charts
├── bs_model.py     # BlackScholesModel class — pricing, Greeks, sensitivity
├── requirements.txt
└── README.md
```

## Model Notes

- Implements Black-Scholes-Merton (continuous dividend yield `q`)
- Historical volatility uses log returns annualised over 252 trading days
- All sensitivity and surface calculations are numpy-vectorised (no Python loops)
- Assumes European-style options; American early exercise is not modelled

## Disclaimer

Educational tool only. Not financial advice. Options trading involves significant risk.
