"""
Black-Scholes Option Pricing Application
A Streamlit app for option pricing with live market data.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from bs_model import BlackScholesModel

# ------------------------------------------------------------------ #
#  Page config                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Black-Scholes Option Pricer",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------------------------ #
#  Colour palette — single source of truth                            #
# ------------------------------------------------------------------ #

COLOURS = {
    'primary':    '#1E88E5',
    'secondary':  '#43A047',
    'danger':     '#E53935',
    'warning':    '#FB8C00',
    'neutral':    '#757575',
    'volume':     '#90CAF9',
    'call':       '#43A047',
    'put':        '#E53935',
    'strike':     '#E53935',
    'spot':       '#FB8C00',
    'breakeven':  '#7B1FA2',
}

# ------------------------------------------------------------------ #
#  Cached data fetchers                                                #
# ------------------------------------------------------------------ #

@st.cache_data(ttl=60)
def fetch_price_history(ticker: str, period: str = '3mo') -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period)


@st.cache_data(ttl=30)
def fetch_spot(ticker: str) -> float:
    data = yf.Ticker(ticker).history(period='1d', interval='1m')
    if not data.empty:
        return float(data['Close'].iloc[-1])
    daily = yf.Ticker(ticker).history(period='5d')
    if not daily.empty:
        return float(daily['Close'].iloc[-1])
    info = yf.Ticker(ticker).info
    for key in ('currentPrice', 'regularMarketPrice', 'previousClose'):
        if info.get(key):
            return float(info[key])
    raise ValueError(f"Cannot determine price for {ticker}")


@st.cache_data(ttl=120)
def fetch_option_chain(ticker: str, expiry: str):
    """Return (calls_df, puts_df) as independent copies."""
    chain = yf.Ticker(ticker).option_chain(expiry)
    return chain.calls.copy(), chain.puts.copy()


@st.cache_data(ttl=300)
def fetch_expiry_dates(ticker: str):
    return yf.Ticker(ticker).options

# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.header("⚙️ Option Parameters")

    ticker = st.text_input("Stock Ticker", value="AAPL",
                           help="Enter stock symbol (e.g., AAPL, TSLA)").upper().strip()

    # ---- expiry / strike selection --------------------------------- #
    option_type = st.radio("Option Type", ["Call", "Put"])

    try:
        expiry_dates = fetch_expiry_dates(ticker)
    except Exception:
        expiry_dates = ()

    if expiry_dates:
        expiration_date_str = st.selectbox("Expiration Date", expiry_dates)
        expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d').date()

        try:
            spot_price_sidebar = fetch_spot(ticker)
        except Exception:
            spot_price_sidebar = 100.0

        calls_df, puts_df = fetch_option_chain(ticker, expiration_date_str)
        options_df = calls_df if option_type == "Call" else puts_df

        available_strikes = sorted(options_df['strike'].unique())
        atm_idx = min(range(len(available_strikes)),
                      key=lambda i: abs(available_strikes[i] - spot_price_sidebar))

        strike_price = st.selectbox("Strike Price ($)", available_strikes, index=atm_idx)

        selected_option = options_df[options_df['strike'] == strike_price].iloc[0]
        last_price   = max(0.0, float(selected_option.get('lastPrice', 0) or 0))
        bid          = max(0.0, float(selected_option.get('bid', 0) or 0))
        ask          = max(0.0, float(selected_option.get('ask', 0) or 0))
        volume       = int(selected_option.get('volume', 0) or 0)
        open_int     = int(selected_option.get('openInterest', 0) or 0)
        market_iv    = float(selected_option.get('impliedVolatility', 0) or 0)

        with st.expander("📊 Market Data for Selected Option"):
            c1, c2 = st.columns(2)
            c1.metric("Last Price", f"${last_price:.2f}")
            c1.metric("Bid",        f"${bid:.2f}")
            c1.metric("Ask",        f"${ask:.2f}")
            c2.metric("Volume",        f"{volume:,}")
            c2.metric("Open Interest", f"{open_int:,}")
            c2.metric("Implied Vol",   f"{market_iv * 100:.1f}%")

        use_market_iv = st.checkbox("Use Market Implied Volatility", value=True)
        if use_market_iv and market_iv > 0:
            volatility = market_iv
            st.info(f"Using market IV: {volatility * 100:.1f}%")
        else:
            volatility = st.number_input("Volatility (%)", 0.1, 200.0, 20.0, 1.0) / 100

    else:
        st.warning(f"No options data available for {ticker}.")
        expiration_date_str = None
        expiration_date = (datetime.now() + timedelta(days=30)).date()
        strike_price = st.number_input("Strike Price ($)", min_value=0.01, value=150.0, step=1.0)
        volatility = st.number_input("Volatility (%)", 0.1, 200.0, 20.0, 1.0) / 100

    risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 5.0, 0.1) / 100

    st.divider()
    st.subheader("📊 Optional Parameters")
    num_contracts       = st.number_input("Number of Contracts",  min_value=1,  value=1,   step=1)
    contract_multiplier = st.number_input("Contract Multiplier",  min_value=1,  value=100, step=1)
    hist_vol_window     = st.number_input("Historical Vol. Window (days)", 10, 252, 30,
                                          help="Rolling window for historical volatility")
    auto_refresh        = st.checkbox("Auto-refresh (every 30 s)")
    if auto_refresh:
        st.caption("⚡ Auto-refresh enabled")

# ------------------------------------------------------------------ #
#  Main content                                                        #
# ------------------------------------------------------------------ #

st.title("📈 Black-Scholes Option Pricing Model")
st.markdown("### Real-time option pricing with Greeks calculation")

try:
    spot_price = fetch_spot(ticker)
    hist_data  = fetch_price_history(ticker)

    T = BlackScholesModel.calculate_time_to_expiry(
        datetime.combine(expiration_date, datetime.min.time())
    )

    option_price = BlackScholesModel.calculate_option_price(
        spot_price, strike_price, risk_free_rate, volatility, T,
        option_type.lower()
    )
    total_value = float(option_price) * num_contracts * contract_multiplier
    hist_vol    = BlackScholesModel.calculate_historical_volatility(
        hist_data['Close'], window=hist_vol_window
    )
    greeks = BlackScholesModel.calculate_greeks(
        spot_price, strike_price, risk_free_rate, volatility, T,
        option_type.lower()
    )

    # ---- Key metrics row ------------------------------------------ #
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spot Price",         f"${spot_price:.2f}")
    m2.metric("Days to Expiry",     f"{max(int(T * 365), 0)}")
    m3.metric(f"{option_type} Price", f"${float(option_price):.4f}")
    m4.metric("Total Value",        f"${total_value:,.2f}")
    m5.metric("Historical Vol.",    f"{hist_vol * 100:.1f}%")

    st.divider()

    # ---- Greeks --------------------------------------------------- #
    st.subheader("🎯 Option Greeks")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
    g1.caption("Price Δ per $1 stock move")
    g2.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
    g2.caption("Delta Δ per $1 stock move")
    g3.metric("Vega (ν)",  f"{greeks['vega']:.4f}")
    g3.caption("Price Δ per 1% vol move")
    g4.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
    g4.caption("Daily time decay")
    g5.metric("Rho (ρ)",   f"{greeks['rho']:.4f}")
    g5.caption("Price Δ per 1% rate move")

    st.divider()

    # ---------------------------------------------------------------- #
    #  Tabs                                                             #
    # ---------------------------------------------------------------- #
    st.subheader("📊 Analysis & Visualisation")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price History", "Sensitivity Analysis", "Greeks Surface", "Payoff Diagram"]
    )

    # ---- Tab 1: Price History ------------------------------------- #
    with tab1:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=hist_data.index, y=hist_data['Close'],
            mode='lines', name='Close Price',
            line=dict(color=COLOURS['primary'], width=2)
        ))
        fig_price.add_hline(
            y=strike_price, line_dash="dash", line_color=COLOURS['strike'],
            annotation_text=f"Strike: ${strike_price}"
        )
        fig_price.update_layout(
            title=f"{ticker} Price History (Last 3 Months)",
            xaxis_title="Date", yaxis_title="Price ($)", height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)

        fig_vol = go.Figure(go.Bar(
            x=hist_data.index, y=hist_data['Volume'],
            name='Volume', marker_color=COLOURS['volume']
        ))
        fig_vol.update_layout(
            title="Trading Volume", xaxis_title="Date",
            yaxis_title="Volume", height=200
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ---- Tab 2: Sensitivity --------------------------------------- #
    with tab2:
        sensitivity_param = st.selectbox(
            "Parameter to analyse",
            ["Spot Price", "Strike Price", "Volatility", "Time to Expiry"]
        )
        param_map = {
            "Spot Price":    "spot",
            "Strike Price":  "strike",
            "Volatility":    "volatility",
            "Time to Expiry":"time",
        }

        x_values, prices, greeks_sens = BlackScholesModel.generate_sensitivity_data(
            spot_price, strike_price, risk_free_rate, volatility, T,
            option_type.lower(), param=param_map[sensitivity_param]
        )

        current_x_map = {
            "spot": spot_price, "strike": strike_price,
            "volatility": volatility, "time": T
        }
        current_x = current_x_map[param_map[sensitivity_param]]

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=x_values, y=prices, mode='lines',
            name=f'{option_type} Price',
            line=dict(color=COLOURS['secondary'], width=3)
        ))
        fig_sens.add_trace(go.Scatter(
            x=[current_x], y=[float(option_price)],
            mode='markers', name='Current Value',
            marker=dict(size=10, color=COLOURS['danger'])
        ))
        fig_sens.update_layout(
            title=f"Option Price vs {sensitivity_param}",
            xaxis_title=sensitivity_param, yaxis_title="Option Price ($)", height=400
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        st.subheader("Greeks Sensitivity")
        greek_choice = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"])
        fig_greek = go.Figure()
        fig_greek.add_trace(go.Scatter(
            x=x_values, y=greeks_sens[greek_choice.lower()],
            mode='lines', name=greek_choice,
            line=dict(color=COLOURS['primary'], width=3)
        ))
        fig_greek.update_layout(
            title=f"{greek_choice} vs {sensitivity_param}",
            xaxis_title=sensitivity_param, yaxis_title=greek_choice, height=350
        )
        st.plotly_chart(fig_greek, use_container_width=True)

    # ---- Tab 3: 3D Greeks Surface --------------------------------- #
    with tab3:
        st.subheader("Greeks 3D Surface")
        greek_surface = st.selectbox(
            "Greek for 3D plot", ["Delta", "Gamma", "Vega", "Theta"]
        )

        spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 40)
        vol_range  = np.linspace(max(volatility * 0.4, 0.01), volatility * 2.0, 40)
        S_grid, vol_grid = np.meshgrid(spot_range, vol_range)

        # Single vectorised call — replaces 1,600-iteration nested loop
        grid_greeks = BlackScholesModel.calculate_greeks_grid(
            S_grid, vol_grid, strike_price, risk_free_rate, T, option_type.lower()
        )
        Z = grid_greeks[greek_surface.lower()]

        fig_3d = go.Figure(data=[go.Surface(
            x=S_grid, y=vol_grid, z=Z,
            colorscale='Viridis',
            colorbar=dict(title=greek_surface)
        )])
        fig_3d.update_layout(
            title=f"{greek_surface} Surface — {option_type}",
            scene=dict(
                xaxis_title="Spot Price ($)",
                yaxis_title="Volatility",
                zaxis_title=greek_surface,
            ),
            height=550
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # ---- Tab 4: Payoff -------------------------------------------- #
    with tab4:
        st.subheader("Option Payoff at Expiration")

        spot_range_payoff = np.linspace(spot_price * 0.5, spot_price * 1.5, 200)
        if option_type.lower() == 'call':
            payoff = np.maximum(spot_range_payoff - strike_price, 0)
            breakeven = strike_price + float(option_price)
            max_profit_label = "Unlimited"
        else:
            payoff = np.maximum(strike_price - spot_range_payoff, 0)
            breakeven = strike_price - float(option_price)
            max_profit_label = f"${strike_price - float(option_price):.2f}"

        profit_loss = payoff - float(option_price)

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(
            x=spot_range_payoff, y=payoff,
            mode='lines', name='Payoff',
            line=dict(color=COLOURS['primary'], width=3)
        ))
        fig_payoff.add_trace(go.Scatter(
            x=spot_range_payoff, y=profit_loss,
            mode='lines', name='Profit / Loss',
            line=dict(color=COLOURS['secondary'], width=3, dash='dash')
        ))
        fig_payoff.add_hline(y=0, line_dash="dot", line_color=COLOURS['neutral'])
        fig_payoff.add_vline(
            x=strike_price, line_dash="dash", line_color=COLOURS['strike'],
            annotation_text=f"Strike: ${strike_price}"
        )
        fig_payoff.add_vline(
            x=spot_price, line_dash="dash", line_color=COLOURS['spot'],
            annotation_text=f"Spot: ${spot_price:.2f}"
        )
        fig_payoff.add_vline(
            x=breakeven, line_dash="dot", line_color=COLOURS['breakeven'],
            annotation_text=f"Breakeven: ${breakeven:.2f}"
        )
        fig_payoff.update_layout(
            title=f"{option_type} Option Payoff Diagram",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Payoff / P&L ($)",
            height=450, hovermode='x unified'
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

        p1, p2, p3 = st.columns(3)
        p1.metric("Breakeven Price", f"${breakeven:.2f}")
        p2.metric("Max Loss",        f"${float(option_price):.2f} per contract")
        p3.metric("Max Profit",      max_profit_label)

    # ---- Auto-refresh -------------------------------------------- #
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check your ticker symbol and ensure you have an internet connection.")

# ------------------------------------------------------------------ #
#  Footer                                                              #
# ------------------------------------------------------------------ #
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "<small>Black-Scholes assumes European options, constant volatility and interest rates, "
    "and no dividends. Market prices may differ.</small></div>",
    unsafe_allow_html=True
)
