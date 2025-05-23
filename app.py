"""
Black-Scholes Option Pricing Application
A comprehensive Streamlit app for option pricing with live market data
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from bs_model import BlackScholesModel

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Black-Scholes Option Pricing Model")
st.markdown("### Real-time option pricing with Greeks calculation")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Option Parameters")
    
    # Stock ticker input
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, TSLA)")
    
    # Initialize session state for options data
    if 'options_data' not in st.session_state:
        st.session_state.options_data = None
    if 'selected_expiry' not in st.session_state:
        st.session_state.selected_expiry = None
    
    # Fetch available options when ticker changes
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expiry_dates = stock.options
        
        if len(expiry_dates) > 0:
            # Option type
            option_type = st.radio("Option Type", ["Call", "Put"])
            
            # Expiration date selector
            expiration_date_str = st.selectbox(
                "Select Expiration Date",
                expiry_dates,
                help="Available expiration dates for this option"
            )
            
            # Convert string to date
            expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d').date()
            
            # Get option chain for selected expiry
            opt_chain = stock.option_chain(expiration_date_str)
            
            if option_type == "Call":
                options_df = opt_chain.calls
            else:
                options_df = opt_chain.puts
            
            # Get available strikes
            available_strikes = sorted(options_df['strike'].unique())
            
            # Find default strike (closest to current price)
            try:
                current_price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 100))
            except:
                current_price = 100
            
            closest_strike_idx = min(range(len(available_strikes)), 
                                   key=lambda i: abs(available_strikes[i] - current_price))
            
            # Strike price selector
            strike_price = st.selectbox(
                "Select Strike Price ($)",
                available_strikes,
                index=closest_strike_idx,
                help="Available strike prices for selected expiration"
            )
            
            # Display market data for selected option
            selected_option = options_df[options_df['strike'] == strike_price].iloc[0]
            
            with st.expander("📊 Market Data for Selected Option"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Last Price", f"${selected_option.get('lastPrice', 0):.2f}")
                    st.metric("Bid", f"${selected_option.get('bid', 0):.2f}")
                    st.metric("Ask", f"${selected_option.get('ask', 0):.2f}")
                with col2:
                    st.metric("Volume", f"{selected_option.get('volume', 0):,}")
                    st.metric("Open Interest", f"{selected_option.get('openInterest', 0):,}")
                    st.metric("Implied Volatility", f"{selected_option.get('impliedVolatility', 0)*100:.1f}%")
            
            # Option to use market implied volatility
            use_market_iv = st.checkbox("Use Market Implied Volatility", value=True)
            
            if use_market_iv and selected_option.get('impliedVolatility', 0) > 0:
                volatility = selected_option['impliedVolatility']
                st.info(f"Using market IV: {volatility*100:.1f}%")
            else:
                # Manual volatility input
                volatility = st.number_input(
                    "Volatility (%)", 
                    min_value=0.1, 
                    max_value=200.0, 
                    value=20.0, 
                    step=1.0,
                    help="Annual implied volatility (e.g., 20.0 for 20%)"
                ) / 100
        else:
            st.error(f"No options data available for {ticker}")
            # Fallback to manual inputs
            option_type = st.radio("Option Type", ["Call", "Put"])
            strike_price = st.number_input("Strike Price ($)", min_value=0.01, value=150.0, step=1.0)
            expiration_date = st.date_input(
                "Expiration Date",
                min_value=datetime.now().date() + timedelta(days=1),
                value=datetime.now().date() + timedelta(days=30)
            )
            volatility = st.number_input(
                "Volatility (%)", 
                min_value=0.1, 
                max_value=200.0, 
                value=20.0, 
                step=1.0,
                help="Annual implied volatility (e.g., 20.0 for 20%)"
            ) / 100
            
    except Exception as e:
        st.warning(f"Could not fetch options data: {str(e)}")
        # Fallback to manual inputs
        option_type = st.radio("Option Type", ["Call", "Put"])
        strike_price = st.number_input("Strike Price ($)", min_value=0.01, value=150.0, step=1.0)
        expiration_date = st.date_input(
            "Expiration Date",
            min_value=datetime.now().date() + timedelta(days=1),
            value=datetime.now().date() + timedelta(days=30)
        )
        volatility = st.number_input(
            "Volatility (%)", 
            min_value=0.1, 
            max_value=200.0, 
            value=20.0, 
            step=1.0,
            help="Annual implied volatility (e.g., 20.0 for 20%)"
        ) / 100
    
    # Risk-free rate
    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=5.0, 
        step=0.1,
        help="Annual risk-free interest rate (e.g., 5.0 for 5%)"
    ) / 100
    
    st.divider()
    
    # Optional inputs
    st.subheader("📊 Optional Parameters")
    
    # Number of contracts
    num_contracts = st.number_input("Number of Contracts", min_value=1, value=1, step=1)
    
    # Contract multiplier
    contract_multiplier = st.number_input("Contract Multiplier", min_value=1, value=100, step=1)
    
    # Historical volatility window
    hist_vol_window = st.number_input(
        "Historical Vol. Window (days)", 
        min_value=10, 
        max_value=252, 
        value=30,
        help="Number of days for historical volatility calculation"
    )
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (every 10 seconds)")
    
    if auto_refresh:
        st.caption("⚡ Auto-refresh is enabled")

# Main content area
try:
    # Fetch stock data
    stock = yf.Ticker(ticker)
    
    # Get current price
    current_data = stock.history(period='1d', interval='1m')
    if len(current_data) > 0:
        spot_price = current_data['Close'].iloc[-1]
    else:
        # Fallback to daily data
        current_data = stock.history(period='5d')
        spot_price = current_data['Close'].iloc[-1]
    
    # Get historical data for volatility calculation
    hist_data = stock.history(period='3mo')
    
    # Calculate time to expiry
    T = BlackScholesModel.calculate_time_to_expiry(
        datetime.combine(expiration_date, datetime.min.time())
    )
    
    # Create columns for main display
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.metric("Current Stock Price", f"${spot_price:.2f}")
        st.metric("Days to Expiry", f"{int(T * 365)}")
    
    with col2:
        # Calculate option price
        option_price = BlackScholesModel.calculate_option_price(
            spot_price, strike_price, risk_free_rate, volatility, T, option_type.lower()
        )
        
        # Calculate total value
        total_value = option_price * num_contracts * contract_multiplier
        
        st.metric(f"{option_type} Option Price", f"${option_price:.4f}")
        st.metric("Total Value", f"${total_value:,.2f}")
    
    with col3:
        # Calculate historical volatility
        hist_vol = BlackScholesModel.calculate_historical_volatility(
            hist_data['Close'], window=hist_vol_window
        )
        st.metric("Historical Vol.", f"{hist_vol*100:.1f}%")
    
    st.divider()
    
    # Greeks section
    st.subheader("🎯 Option Greeks")
    
    # Calculate Greeks
    greeks = BlackScholesModel.calculate_greeks(
        spot_price, strike_price, risk_free_rate, volatility, T, option_type.lower()
    )
    
    # Display Greeks
    greeks_col1, greeks_col2, greeks_col3, greeks_col4, greeks_col5 = st.columns(5)
    
    with greeks_col1:
        st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
        st.caption("Price change per $1 stock move")
    
    with greeks_col2:
        st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
        st.caption("Delta change per $1 stock move")
    
    with greeks_col3:
        st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
        st.caption("Price change per 1% vol move")
    
    with greeks_col4:
        st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
        st.caption("Daily time decay")
    
    with greeks_col5:
        st.metric("Rho (ρ)", f"{greeks['rho']:.4f}")
        st.caption("Price change per 1% rate move")
    
    # Visualization section
    st.divider()
    st.subheader("📊 Analysis & Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Price History", "Sensitivity Analysis", "Greeks Surface", "Payoff Diagram"])
    
    with tab1:
        # Historical price chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add strike price line
        fig_price.add_hline(
            y=strike_price, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Strike: ${strike_price}"
        )
        
        fig_price.update_layout(
            title=f"{ticker} Price History (Last 3 Months)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Volume subplot
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        fig_volume.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=200
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        # Sensitivity analysis
        sensitivity_param = st.selectbox(
            "Select parameter to analyze",
            ["Spot Price", "Strike Price", "Volatility", "Time to Expiry"]
        )
        
        param_map = {
            "Spot Price": "spot",
            "Strike Price": "strike",
            "Volatility": "volatility",
            "Time to Expiry": "time"
        }
        
        # Generate sensitivity data
        x_values, prices, greeks_sens = BlackScholesModel.generate_sensitivity_data(
            spot_price, strike_price, risk_free_rate, volatility, T,
            option_type.lower(), param=param_map[sensitivity_param]
        )
        
        # Price sensitivity chart
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=x_values,
            y=prices,
            mode='lines',
            name=f'{option_type} Price',
            line=dict(color='green', width=3)
        ))
        
        # Add current value marker
        current_x = {
            "spot": spot_price,
            "strike": strike_price,
            "volatility": volatility,
            "time": T
        }[param_map[sensitivity_param]]
        
        fig_sens.add_trace(go.Scatter(
            x=[current_x],
            y=[option_price],
            mode='markers',
            name='Current Value',
            marker=dict(size=10, color='red')
        ))
        
        fig_sens.update_layout(
            title=f"Option Price Sensitivity to {sensitivity_param}",
            xaxis_title=sensitivity_param,
            yaxis_title="Option Price ($)",
            height=400
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        
        # Greeks sensitivity
        st.subheader("Greeks Sensitivity")
        
        greek_choice = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"])
        
        fig_greek = go.Figure()
        fig_greek.add_trace(go.Scatter(
            x=x_values,
            y=greeks_sens[greek_choice.lower()],
            mode='lines',
            name=greek_choice,
            line=dict(width=3)
        ))
        
        fig_greek.update_layout(
            title=f"{greek_choice} vs {sensitivity_param}",
            xaxis_title=sensitivity_param,
            yaxis_title=greek_choice,
            height=350
        )
        st.plotly_chart(fig_greek, use_container_width=True)
    
    with tab3:
        # 3D Greeks surface
        st.subheader("Greeks 3D Surface")
        
        # Generate grid data
        spot_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 30)
        vol_range = np.linspace(volatility * 0.5, volatility * 1.5, 30)
        
        greek_surface = st.selectbox("Select Greek for 3D plot", ["Delta", "Gamma", "Vega"])
        
        # Calculate surface
        X, Y = np.meshgrid(spot_range, vol_range)
        Z = np.zeros_like(X)
        
        for i in range(len(spot_range)):
            for j in range(len(vol_range)):
                greeks_3d = BlackScholesModel.calculate_greeks(
                    X[j, i], strike_price, risk_free_rate, Y[j, i], T, option_type.lower()
                )
                Z[j, i] = greeks_3d[greek_surface.lower()]
        
        fig_3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        fig_3d.update_layout(
            title=f"{greek_surface} Surface",
            scene=dict(
                xaxis_title="Spot Price ($)",
                yaxis_title="Volatility",
                zaxis_title=greek_surface
            ),
            height=500
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab4:
        # Payoff diagram
        st.subheader("Option Payoff at Expiration")
        
        # Generate payoff data
        spot_range_payoff = np.linspace(spot_price * 0.5, spot_price * 1.5, 100)
        
        if option_type.lower() == 'call':
            payoff = np.maximum(spot_range_payoff - strike_price, 0)
        else:
            payoff = np.maximum(strike_price - spot_range_payoff, 0)
        
        profit_loss = payoff - option_price
        
        fig_payoff = go.Figure()
        
        # Payoff line
        fig_payoff.add_trace(go.Scatter(
            x=spot_range_payoff,
            y=payoff,
            mode='lines',
            name='Payoff',
            line=dict(color='blue', width=3)
        ))
        
        # Profit/Loss line
        fig_payoff.add_trace(go.Scatter(
            x=spot_range_payoff,
            y=profit_loss,
            mode='lines',
            name='Profit/Loss',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        # Add breakeven line
        fig_payoff.add_hline(y=0, line_dash="dot", line_color="gray")
        
        # Add strike price line
        fig_payoff.add_vline(
            x=strike_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Strike: ${strike_price}"
        )
        
        # Add current price line
        fig_payoff.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Current: ${spot_price:.2f}"
        )
        
        fig_payoff.update_layout(
            title=f"{option_type} Option Payoff Diagram",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Payoff / Profit-Loss ($)",
            height=400,
            hovermode='x'
        )
        st.plotly_chart(fig_payoff, use_container_width=True)
        
        # Breakeven calculation
        if option_type.lower() == 'call':
            breakeven = strike_price + option_price
        else:
            breakeven = strike_price - option_price
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Breakeven Price", f"${breakeven:.2f}")
        with col2:
            st.metric("Max Loss", f"${option_price:.2f}")
        with col3:
            if option_type.lower() == 'call':
                st.metric("Max Profit", "Unlimited")
            else:
                st.metric("Max Profit", f"${strike_price - option_price:.2f}")
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check your inputs and ensure the ticker symbol is valid.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    Black-Scholes Model assumes European options, no dividends, constant volatility and interest rates.<br>
    Real market prices may differ due to market conditions and other factors.
    </small>
</div>
""", unsafe_allow_html=True)