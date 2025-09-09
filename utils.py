import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class FinancialDataProcessor:
    """Core class for financial data processing and analysis"""
    
    def __init__(self):
        self.data = None
        self.ticker = None
    
    def fetch_data(self, ticker, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            self.data = data
            self.ticker = ticker
            return data
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def calculate_moving_averages(self, data):
        """Calculate moving averages"""
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        return data
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        data['BB_Middle'] = bollinger.bollinger_mavg()
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Daily Returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        return data
    
    def calculate_portfolio_metrics(self, tickers, weights, period="1y"):
        """Calculate portfolio performance metrics"""
        portfolio_data = {}
        
        for ticker in tickers:
            stock_data = yf.Ticker(ticker).history(period=period)
            portfolio_data[ticker] = stock_data['Close']
        
        portfolio_df = pd.DataFrame(portfolio_data)
        returns = portfolio_df.pct_change().dropna()
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        return {
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'correlation_matrix': returns.corr()
        }

class ChartGenerator:
    """Generate interactive charts using Plotly"""
    
    @staticmethod
    def create_price_chart(data, ticker):
        """Create interactive price chart with moving averages"""
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker
        ))
        
        # Moving averages
        if 'MA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], 
                                   name='MA 20', line=dict(color='orange')))
        if 'MA_50' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], 
                                   name='MA 50', line=dict(color='blue')))
        if 'MA_200' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_200'], 
                                   name='MA 200', line=dict(color='red')))
        
        fig.update_layout(
            title=f'{ticker} Stock Price with Moving Averages',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_volume_chart(data, ticker):
        """Create volume chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'{ticker} Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_technical_indicators_chart(data, ticker):
        """Create technical indicators chart"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
            vertical_spacing=0.1
        )
        
        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal'), row=2, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram'), row=2, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='Upper Band'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='Lower Band'), row=3, col=1)
        
        fig.update_layout(
            title=f'{ticker} Technical Indicators',
            template='plotly_white',
            height=800
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(correlation_matrix):
        """Create correlation heatmap"""
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Stock Correlation Matrix"
        )
        
        fig.update_layout(template='plotly_white')
        return fig
    
    @staticmethod
    def create_portfolio_chart(cumulative_returns):
        """Create portfolio performance chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Cumulative Returns',
            yaxis_title='Cumulative Return',
            xaxis_title='Date',
            template='plotly_white'
        )
        
        return fig

class ForecastingEngine:
    """Stock price forecasting using Prophet and ARIMA"""
    
    @staticmethod
    def prophet_forecast(data, days=30):
        """Generate forecast using Facebook Prophet"""
        try:
            # Prepare data for Prophet
            df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # Remove timezone from ds column
            df['ds'] = df['ds'].dt.tz_localize(None)
            
            # Create and fit model
            model = Prophet(daily_seasonality=True)
            model.fit(df)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        except Exception as e:
            raise Exception(f"Prophet forecasting error: {str(e)}")
    
    @staticmethod
    def arima_forecast(data, days=30):
        """Generate forecast using ARIMA"""
        try:
            # Fit ARIMA model
            model = ARIMA(data['Close'], order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=days)
            
            # Create forecast dataframe
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast
            })
            
            return forecast_df
        
        except Exception as e:
            raise Exception(f"ARIMA forecasting error: {str(e)}")

def get_stock_info(ticker):
    """Get basic stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
    except:
        return {'name': ticker, 'sector': 'N/A', 'industry': 'N/A', 'market_cap': 'N/A', 'pe_ratio': 'N/A'}

def format_currency(value):
    """Format currency values"""
    if isinstance(value, (int, float)):
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:,.2f}"
    return "N/A"