import streamlit as st
import pandas as pd
import numpy as np
from utils import FinancialDataProcessor, ChartGenerator, ForecastingEngine, get_stock_info, format_currency
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Author information
st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Vikas Ramaswamy")
st.sidebar.markdown("**Version:** 1.0")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Set up the analysis tools
if 'processor' not in st.session_state:
    st.session_state.processor = FinancialDataProcessor()
if 'chart_generator' not in st.session_state:
    st.session_state.chart_generator = ChartGenerator()
if 'forecasting_engine' not in st.session_state:
    st.session_state.forecasting_engine = ForecastingEngine()

def main():
    st.markdown('<h1 class="main-header">Financial Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Professional header info
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #1f77b4;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>Author:</strong> Vikas Ramaswamy | <strong>Version:</strong> 1.0 | <strong>Technology:</strong> Python, yfinance, Prophet, Plotly
            </div>
            <div style="color: #6c757d; font-size: 0.9rem;">
                Professional Financial Analytics & Forecasting Platform
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Single Stock Analysis", "Multi-Stock Comparison", "Portfolio Analysis", "Forecasting"]
    )
    
    if analysis_type == "Single Stock Analysis":
        single_stock_analysis()
    elif analysis_type == "Multi-Stock Comparison":
        multi_stock_comparison()
    elif analysis_type == "Portfolio Analysis":
        portfolio_analysis()
    elif analysis_type == "Forecasting":
        forecasting_analysis()

def single_stock_analysis():
    st.subheader("Single Stock Analysis")
    
    # Choose which stock to analyze
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, GOOGL, TSLA").upper()
    
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.button("Analyze Stock", type="primary"):
        try:
            with st.spinner(f"Getting data for {ticker}..."):
                # Get the stock data and add indicators
                data = st.session_state.processor.fetch_data(ticker, period)
                data = st.session_state.processor.calculate_moving_averages(data)
                data = st.session_state.processor.calculate_technical_indicators(data)
                
                # Get basic company info
                stock_info = get_stock_info(ticker)
                
                # Show the results
                st.success(f"Analysis complete for {stock_info['name']} ({ticker})")
                
                # Show key numbers
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.1f}%)")
                
                with col2:
                    st.metric("Market Cap", format_currency(stock_info['market_cap']))
                
                with col3:
                    st.metric("P/E Ratio", f"{stock_info['pe_ratio']:.2f}" if isinstance(stock_info['pe_ratio'], (int, float)) else "N/A")
                
                with col4:
                    volatility = data['Daily_Return'].std() * np.sqrt(252) * 100
                    st.metric("Annual Volatility", f"{volatility:.1f}%")
                
                # Show the price chart
                st.subheader("Price Chart with Moving Averages")
                price_chart = st.session_state.chart_generator.create_price_chart(data, ticker)
                st.plotly_chart(price_chart, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Trading Volume")
                    volume_chart = st.session_state.chart_generator.create_volume_chart(data, ticker)
                    st.plotly_chart(volume_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Returns Distribution")
                    returns_data = data['Daily_Return'].dropna()
                    fig = go.Figure(data=[go.Histogram(x=returns_data, nbinsx=50)])
                    fig.update_layout(title="Daily Returns Distribution", template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show technical analysis
                st.subheader("Technical Indicators")
                tech_chart = st.session_state.chart_generator.create_technical_indicators_chart(data, ticker)
                st.plotly_chart(tech_chart, use_container_width=True)
                
                # Show data summary
                st.subheader("Summary Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Price Statistics**")
                    price_stats = data['Close'].describe()
                    st.dataframe(price_stats)
                
                with col2:
                    st.write("**Returns Statistics**")
                    returns_stats = data['Daily_Return'].describe()
                    st.dataframe(returns_stats)
                
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")

def multi_stock_comparison():
    st.subheader("Multi-Stock Comparison")
    
    # Choose multiple stocks to compare
    tickers_input = st.text_input(
        "Enter Stock Tickers (comma-separated)", 
        value="AAPL,GOOGL,MSFT,TSLA",
        help="e.g., AAPL,GOOGL,MSFT,TSLA"
    )
    
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    
    if st.button("Compare Stocks", type="primary"):
        try:
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
            
            with st.spinner("Getting data for all stocks..."):
                comparison_data = {}
                
                for ticker in tickers:
                    data = st.session_state.processor.fetch_data(ticker, period)
                    comparison_data[ticker] = data['Close']
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Make all stocks start at the same point for fair comparison
                normalized_df = comparison_df / comparison_df.iloc[0]
                
                # Show how the stocks performed relative to each other
                st.subheader("Price Comparison (Normalized)")
                fig = go.Figure()
                
                for ticker in tickers:
                    fig.add_trace(go.Scatter(
                        x=normalized_df.index,
                        y=normalized_df[ticker],
                        mode='lines',
                        name=ticker
                    ))
                
                fig.update_layout(
                    title="Stock Price Comparison (Normalized to 1.0)",
                    yaxis_title="Normalized Price",
                    xaxis_title="Date",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # See how closely the stocks move together
                returns_df = comparison_df.pct_change().dropna()
                correlation_matrix = returns_df.corr()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Correlation Heatmap")
                    corr_chart = st.session_state.chart_generator.create_correlation_heatmap(correlation_matrix)
                    st.plotly_chart(corr_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Performance Metrics")
                    
                    metrics_data = []
                    for ticker in tickers:
                        returns = returns_df[ticker]
                        total_return = (comparison_df[ticker].iloc[-1] / comparison_df[ticker].iloc[0] - 1) * 100
                        volatility = returns.std() * np.sqrt(252) * 100
                        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
                        
                        metrics_data.append({
                            'Ticker': ticker,
                            'Total Return (%)': f"{total_return:.2f}",
                            'Volatility (%)': f"{volatility:.2f}",
                            'Sharpe Ratio': f"{sharpe:.2f}"
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in comparison: {str(e)}")

def portfolio_analysis():
    st.subheader("Portfolio Analysis")
    
    # Set up your portfolio
    st.write("**Configure Your Portfolio**")
    
    num_stocks = st.slider("Number of Stocks", min_value=2, max_value=10, value=4)
    
    tickers = []
    weights = []
    
    cols = st.columns(2)
    
    for i in range(num_stocks):
        with cols[i % 2]:
            ticker = st.text_input(f"Stock {i+1}", value=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"][i] if i < 8 else "")
            weight = st.number_input(f"Weight {i+1} (%)", min_value=0.0, max_value=100.0, value=100.0/num_stocks, step=1.0)
            
            if ticker:
                tickers.append(ticker.upper())
                weights.append(weight/100)
    
    # Make sure weights add up to 100%
    if sum(weights) != 0:
        weights = [w/sum(weights) for w in weights]
    
    period = st.selectbox("Analysis Period", ["6mo", "1y", "2y", "5y"], index=1)
    
    if st.button("Analyze Portfolio", type="primary") and tickers:
        try:
            with st.spinner("Crunching portfolio numbers..."):
                portfolio_metrics = st.session_state.processor.calculate_portfolio_metrics(tickers, weights, period)
                
                # Show key portfolio stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{portfolio_metrics['total_return']*100:.2f}%")
                
                with col2:
                    st.metric("Annual Return", f"{portfolio_metrics['annual_return']*100:.2f}%")
                
                with col3:
                    st.metric("Volatility", f"{portfolio_metrics['volatility']*100:.2f}%")
                
                with col4:
                    st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
                
                # Show how your portfolio performed over time
                st.subheader("Portfolio Performance")
                portfolio_chart = st.session_state.chart_generator.create_portfolio_chart(portfolio_metrics['cumulative_returns'])
                st.plotly_chart(portfolio_chart, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Portfolio Composition")
                    composition_data = pd.DataFrame({
                        'Ticker': tickers,
                        'Weight (%)': [w*100 for w in weights]
                    })
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=composition_data['Ticker'],
                        values=composition_data['Weight (%)'],
                        hole=0.3
                    )])
                    
                    fig.update_layout(title="Portfolio Allocation", template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Correlation Matrix")
                    corr_chart = st.session_state.chart_generator.create_correlation_heatmap(portfolio_metrics['correlation_matrix'])
                    st.plotly_chart(corr_chart, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in portfolio analysis: {str(e)}")

def forecasting_analysis():
    st.subheader("Stock Price Forecasting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    
    with col2:
        forecast_days = st.slider("Days to Predict", min_value=7, max_value=90, value=30)
    
    with col3:
        model_type = st.selectbox("AI Model to Use", ["Prophet", "ARIMA"])
    
    if st.button("Generate Forecast", type="primary"):
        try:
            with st.spinner(f"Predicting {ticker} prices for the next {forecast_days} days..."):
                # Get historical data to train the model
                data = st.session_state.processor.fetch_data(ticker, "2y")
                
                if model_type == "Prophet":
                    forecast = st.session_state.forecasting_engine.prophet_forecast(data, forecast_days)
                    
                    # Create forecast chart
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    forecast_data = forecast.tail(forecast_days)
                    fig.add_trace(go.Scatter(
                        x=forecast_data['ds'],
                        y=forecast_data['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Confidence intervals
                    fig.add_trace(go.Scatter(
                        x=forecast_data['ds'],
                        y=forecast_data['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_data['ds'],
                        y=forecast_data['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Confidence Interval',
                        fillcolor='rgba(255,0,0,0.2)'
                    ))
                    
                else:  # ARIMA
                    forecast = st.session_state.forecasting_engine.arima_forecast(data, forecast_days)
                    
                    # Create forecast chart
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast['Date'],
                        y=forecast['Forecast'],
                        mode='lines+markers',
                        name='ARIMA Forecast',
                        line=dict(color='red', dash='dash'),
                        marker=dict(size=4)
                    ))
                
                fig.update_layout(
                    title=f'{ticker} Price Forecast ({model_type} Model)',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                current_price = data['Close'].iloc[-1]
                
                if model_type == "Prophet":
                    forecast_price = forecast_data['yhat'].iloc[-1]
                    price_change = ((forecast_price - current_price) / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        st.metric("Forecast Price", f"${forecast_price:.2f}")
                    
                    with col3:
                        st.metric("Expected Change", f"{price_change:+.2f}%")
                
                else:  # ARIMA
                    forecast_price = forecast['Forecast'].iloc[-1]
                    price_change = ((forecast_price - current_price) / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        st.metric("Forecast Price", f"${forecast_price:.2f}")
                    
                    with col3:
                        st.metric("Expected Change", f"{price_change:+.2f}%")
                
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

if __name__ == "__main__":
    main()
    
    # Footer disclaimer
    st.markdown("---")
    with st.expander("Important Disclaimer - Click to Read"):
        st.markdown("""
        **Educational Use Only:** This tool is for educational purposes only and is provided as an open source project.
        
        **Not Financial Advice:** This should NOT be considered as financial advice. Do not use for actual investment decisions.
        
        **Open Source:** Free to use, modify, and distribute. Created by Vikas Ramaswamy.
        
        **No Guarantees:** Past performance does not guarantee future results. Always consult with qualified financial professionals before making investment decisions.
        """)
    
    # Professional Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
        <div style="margin-bottom: 1rem;">
            <strong>Financial Analytics Dashboard</strong>
        </div>
        <div style="color: #6c757d; margin-bottom: 1rem;">
            Professional financial analysis platform with real-time data, technical indicators, and ML forecasting
        </div>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <div><strong>Features:</strong> Stock Analysis | Portfolio Management | Price Forecasting</div>
            <div><strong>Technology:</strong> Python | yfinance | Prophet | Plotly</div>
        </div>
        <div style="color: #6c757d; font-size: 0.9rem;">
            © 2024 Vikas Ramaswamy | Professional Analytics Portfolio | Educational Purpose Only
        </div>
    </div>
    """, unsafe_allow_html=True)