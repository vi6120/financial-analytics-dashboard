# Financial Analytics Dashboard

A comprehensive Python-based financial analytics dashboard that demonstrates end-to-end data analytics skills including data collection, cleaning, exploratory data analysis (EDA), visualization, and forecasting.

## Project Overview

This dashboard provides professional-grade financial analysis tools for individual stocks, multi-stock comparisons, portfolio analysis, and price forecasting. Built with modern Python libraries and deployed using Streamlit for an interactive web interface.

## Features

### Basic Version
- **Stock Selection**: Analyze any publicly traded stock by ticker symbol
- **Historical Data**: Pull 1 month to 5 years of daily price data
- **Price Visualization**: Interactive candlestick charts with volume
- **Moving Averages**: 20-day, 50-day, and 200-day moving averages
- **Volume Analysis**: Trading volume trends and patterns

### Intermediate Version
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Multi-Stock Comparison**: Compare performance of multiple stocks
- **Correlation Analysis**: Correlation heatmap between selected stocks
- **Volatility Analysis**: Daily and annualized volatility calculations
- **Returns Distribution**: Statistical analysis of daily returns

### Advanced Version
- **Price Forecasting**: Prophet and ARIMA models for price prediction
- **Portfolio Analysis**: Multi-asset portfolio performance evaluation
- **Risk Metrics**: Sharpe ratio, volatility, and drawdown analysis
- **Interactive Dashboard**: Professional web interface with real-time updates

## Technology Stack

- **Data Collection**: yfinance (Yahoo Finance API)
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Technical Analysis**: ta (Technical Analysis library)
- **Forecasting**: prophet (Facebook Prophet), statsmodels (ARIMA)
- **Web Framework**: streamlit
- **Machine Learning**: scikit-learn

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd financial-analytics-dashboard
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the dashboard**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

3. **Select analysis type**:
   - **Single Stock Analysis**: Comprehensive analysis of individual stocks
   - **Multi-Stock Comparison**: Compare multiple stocks side-by-side
   - **Portfolio Analysis**: Analyze portfolio performance and allocation
   - **Forecasting**: Generate price predictions using ML models

## Dashboard Sections

### Single Stock Analysis
- Real-time stock information and key metrics
- Interactive price charts with technical indicators
- Volume analysis and returns distribution
- Comprehensive summary statistics

### Multi-Stock Comparison
- Normalized price comparison charts
- Correlation analysis between stocks
- Performance metrics comparison table
- Risk-return visualization

### Portfolio Analysis
- Custom portfolio configuration with weights
- Portfolio performance tracking
- Asset allocation visualization
- Correlation matrix and risk analysis

### Forecasting
- Prophet model for trend and seasonality analysis
- ARIMA model for time series forecasting
- Confidence intervals and forecast accuracy
- Visual forecast charts with historical context

## Key Metrics and Indicators

### Price Metrics
- Current price and daily change
- Moving averages (20, 50, 200-day)
- Support and resistance levels

### Technical Indicators
- **RSI**: Relative Strength Index for momentum analysis
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility and mean reversion indicator

### Risk Metrics
- **Volatility**: Annualized price volatility
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline

### Portfolio Metrics
- **Total Return**: Cumulative portfolio performance
- **Annual Return**: Annualized return calculation
- **Asset Allocation**: Portfolio composition and weights
- **Correlation Matrix**: Inter-asset correlation analysis

## Data Sources

- **Yahoo Finance**: Historical price data, volume, and basic fundamentals
- **Real-time Updates**: Live market data during trading hours
- **Global Markets**: Support for international stock exchanges

## File Structure

```
financial-analytics-dashboard/
├── data/              # Sample datasets (optional)
├── notebooks/         # Jupyter notebooks for EDA
├── app.py            # Main Streamlit application
├── utils.py          # Helper functions and classes
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

## Professional Applications

This dashboard demonstrates key skills valuable in financial technology:

1. **Data Engineering**: Automated data collection and processing pipelines
2. **Quantitative Analysis**: Statistical analysis and risk modeling
3. **Visualization**: Interactive charts and professional dashboards
4. **Machine Learning**: Time series forecasting and predictive modeling
5. **Web Development**: Full-stack application deployment

## Future Enhancements

- **Real-time Streaming**: WebSocket integration for live data
- **Advanced Models**: Deep learning models for price prediction
- **Options Analysis**: Options pricing and Greeks calculation
- **Backtesting**: Strategy backtesting framework
- **API Integration**: RESTful API for programmatic access

## Deployment

The dashboard can be deployed on various platforms:

- **Streamlit Cloud**: Free hosting for Streamlit applications
- **Heroku**: Cloud platform with easy deployment
- **AWS/GCP**: Enterprise-grade cloud deployment
- **Docker**: Containerized deployment for any environment

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.