import os
import time
import datetime
import smtplib
import requests
import numpy as np
import pandas as pd
import scipy.stats as stats
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pycoingecko import CoinGeckoAPI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables for local testing
load_dotenv()

# --- Configuration ---
COIN_ID = "the-sandbox"
VS_CURRENCY = "usd"
DAYS_HISTORY = 30  # Fetch 30 days of hourly data for statistical significance
# GARCH model parameters
# Using 'GARCH' model with p=1, q=1 by default in arch library
SIMULATION_PATHS = 10000
FORECAST_HORIZON_HOURS = 24

# Secrets
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", EMAIL_USER) # Default to self

class MarketDataFetcher:
    def __init__(self):
        self.api_key = os.getenv("COINGECKO_API_KEY")
        if self.api_key:
            self.cg = CoinGeckoAPI(demo_api_key=self.api_key)
        else:
            self.cg = CoinGeckoAPI()
        
    def fetch_price_history(self, days=DAYS_HISTORY):
        """Fetches hourly historical data with exponential backoff."""
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            try:
                # Fetch hourly data (CoinGecko provides hourly for ranges > 1 day and <= 90 days)
                data = self.cg.get_coin_market_chart_by_id(
                    id=COIN_ID,
                    vs_currency=VS_CURRENCY,
                    days=days
                )
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                wait_time = 2 ** retries
                print(f"API Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
        
        raise Exception("Failed to fetch market data after max retries.")

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.api_key = CRYPTOPANIC_API_KEY

    def fetch_news(self):
        """Fetches news from CryptoPanic."""
        if not self.api_key:
            print("WARNING: CryptoPanic API Key not found. Skipping sentiment analysis.")
            return []
            
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.api_key}&currencies=SAND&kind=news"
        try:
            response = requests.get(url)
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            print(f"Sentiment Fetch Error: {e}")
            return []

    def analyze_sentiment(self, news_items):
        """Calculates average sentiment and returns a DataFrame of sentiment over time."""
        if not news_items:
            return 0, pd.DataFrame()

        sentiments = []
        for item in news_items:
            title = item['title']
            published_at = item['published_at']
            score = self.analyzer.polarity_scores(title)['compound']
            sentiments.append({'timestamp': published_at, 'score': score})
            
        df = pd.DataFrame(sentiments)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Sort by time
        df.sort_values('timestamp', inplace=True)
        
        avg_sentiment = df['score'].mean() if not df.empty else 0
        return avg_sentiment, df

class QuantModel:
    def __init__(self, price_df):
        self.price_df = price_df
        # Calculate Log Returns: ln(P_t / P_{t-1})
        self.price_df['returns'] = np.log(self.price_df['price'] / self.price_df['price'].shift(1))
        self.price_df.dropna(inplace=True)
        self.last_price = self.price_df['price'].iloc[-1]
        
    def fit_garch(self):
        """
        Fits a GARCH(1,1) model to the returns.
        Volitility Clustering: GARCH captures the phenomenon where periods of high volatility 
        tend to cluster together (large changes follow large changes).
        
        Formula: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        """
        # Scale returns by 100 for better numerical stability in optimization
        returns_scaled = self.price_df['returns'] * 100
        
        # GARCH(1,1) is the default; p=1 (lag of symmetric innovation), q=1 (lag of conditional variance)
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        
        # Forecast variance for next step
        forecast = res.forecast(horizon=1)
        next_vol_scaled = np.sqrt(forecast.variance.iloc[-1, 0])
        
        # Rescale volatility back to original scale
        self.current_volatility = next_vol_scaled / 100
        
        return res, self.current_volatility

    def simulate_gbm(self, days_forecast=1):
        """
        Simulates future price paths using Geometric Brownian Motion.
        Equation: dS_t = mu * S_t * dt + sigma * S_t * dW_t
        
        W_t is a Wiener Process (Brownian Motion), characterized by independent, 
        stationary increments with normal distribution N(0, t).
        """
        dt = 1 / 24  # Time step is 1 hour (assuming hourly data frequency for 24 hours)
        num_steps = int(days_forecast * 24)
        
        # Drift (mu) assumed to be average historical return (hourly)
        mu = self.price_df['returns'].mean()
        sigma = self.current_volatility # Use GARCH forecasted volatility
        
        # Simulation: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        # We simulate stepwise increments
        
        paths = np.zeros((num_steps + 1, SIMULATION_PATHS))
        paths[0] = self.last_price
        
        for t in range(1, num_steps + 1):
            rand = np.random.standard_normal(SIMULATION_PATHS)
            # Discrete approximation: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            # Where Z ~ N(0,1)
            paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)
            
        return paths

    def calc_correlation(self, sentiment_df):
        """Calculates Pearson correlation between Sentiment and Price Velocity."""
        if sentiment_df.empty:
            return 0, 0
            
        # Resample sentiment to hourly to match price data
        sentiment_hourly = sentiment_df.set_index('timestamp').resample('H').mean().interpolate()
        
        # Calculate Price Velocity (Simple Returns or % Change)
        # Using returns calculated in __init__
        aligned_data = pd.concat([self.price_df['returns'], sentiment_hourly['score']], axis=1, join='inner')
        aligned_data.dropna(inplace=True)
        
        if len(aligned_data) < 2:
            return 0, 0
            
        correlation, p_value = stats.pearsonr(aligned_data['returns'], aligned_data['score'])
        return correlation, p_value

    def hypothesis_test(self):
        """
        Tests the Null Hypothesis: Price follows a Random Walk.
        We can use the Augmented Dickey-Fuller test or Variance Ratio test.
        Here we'll use a simple check on Autocorrelation of returns.
        If returns are white noise, price is a random walk.
        """
        # Ljung-Box test on returns
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(self.price_df['returns'], lags=[10], return_df=True)
        p_value = lb_test['lb_pvalue'].iloc[0]
        
        is_random_walk = p_value > 0.05 # Fail to reject null (Null = no autocorrelation)
        return is_random_walk, p_value

class ReportGenerator:
    def __init__(self, last_price):
        self.last_price = last_price

    def generate_plots(self, simulation_paths, volatility, upper_conf, lower_conf):
        # 1. Monte Carlo Paths
        plt.figure(figsize=(10, 6))
        plt.plot(simulation_paths[:, :100], color='blue', alpha=0.1) # Plot first 100 paths
        plt.axhline(self.last_price, color='black', linestyle='--', label='Start Price')
        plt.title(f"Monte Carlo Simulation: 100 paths (out of {SIMULATION_PATHS})")
        plt.xlabel("Hours into Future")
        plt.ylabel("Price (USD)")
        plt.savefig("monte_carlo_paths.png")
        plt.close()

        # 2. Volatility Cone (Conceptual - showing CI expansion)
        steps = np.arange(len(simulation_paths))
        plt.figure(figsize=(10, 6))
        
        # Calculate simple cone based on drift + diffusion
        # Upper/Lower Expected Bounds
        mu = 0 # Assuming neutral drift for cone visual
        
        plt.fill_between(steps, 
                         [self.last_price for _ in steps], 
                         [self.last_price * (1 + volatility * np.sqrt(t/24)) for t in steps],
                         color='orange', alpha=0.3, label='1 Sigma Volatility')
        plt.fill_between(steps, 
                         [self.last_price for _ in steps], 
                         [self.last_price * (1 - volatility * np.sqrt(t/24)) for t in steps],
                         color='orange', alpha=0.3)
                         
        plt.title("Volatility Cone (1 Sigma)")
        plt.xlabel("Hours into Future")
        plt.ylabel("Price")
        plt.savefig("volatility_cone.png")
        plt.close()

    def create_markdown(self, last_price, forecasted_price, garch_res, sentiment_score, corr, is_random_walk, p_value, ci_95):
        upper_price = ci_95[1]
        lower_price = ci_95[0]
        
        report = f"""# MARKET INTELLIGENCE REPORT: Sandbox (SAND)
**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Current Price:** ${last_price:.4f}
- **Expected Price (24h):** ${forecasted_price:.4f} (Mean of Monte Carlo)
- **95% Confidence Interval:** [${lower_price:.4f}, ${upper_price:.4f}]
- **Market Sentiment Score:** {sentiment_score:.4f} (Scale: -1 to 1)
- **Sentiment-Price Correlation:** {corr:.4f}

## 1. Mathematical Framework & Derivation

### A. GARCH(1,1) Volatility Modeling
We utilized a GARCH(1,1) model to estimate the current conditional volatility.
Equation: $\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$

**Model Results:**
- **Annualized Volatility:** {(garch_res.conditional_volatility[-1] * np.sqrt(24*365)):.2f}% (approx)
- **AIC:** {garch_res.aic:.2f}

Only GARCH captures volatility clustering, allowing us to react to periods of high risk dynamically compared to static standard deviation.

### B. Geometric Brownian Motion (Stochastic Prediction)
We simulated {SIMULATION_PATHS} future price paths using:
$dS_t = \\mu S_t dt + \\sigma S_t dW_t$

Where $dW_t$ is a Wiener process (Independent Gaussian increments).

## 2. Hypothesis Testing Results
**Null Hypothesis ($H_0$):** SAND price returns follow a Random Walk (No Autocorrelation).
**Test Used:** Ljung-Box Test on Log Returns.

- **P-Value:** {p_value:.6f}
- **Result:** {"Reject Null (Predictable Patterns Exist)" if not is_random_walk else "Fail to Reject Null (Price is Random Walk)"}

## 3. Visuals

### Monte Carlo Simulation Paths
![Monte Carlo](monte_carlo_paths.png)

### Volatility Cone
![Volatility Cone](volatility_cone.png)

***
*Generated by Automated Quant Engine*
"""
        with open("MARKET_REPORT.md", "w") as f:
            f.write(report)
        
        return report

class Notifier:
    def send_email(self, report_body):
        if not EMAIL_USER or not EMAIL_PASSWORD:
            print("Skipping email: Credentials not set.")
            return

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = f"SAND Market Report - {datetime.datetime.now().strftime('%Y-%m-%d')}"

        msg.attach(MIMEText(report_body, 'markdown'))

        # Attach images
        for file in ["monte_carlo_paths.png", "volatility_cone.png"]:
            if os.path.exists(file):
                with open(file, "rb") as f:
                    mime = MIMEBase('image', 'png', filename=file)
                    mime.add_header('Content-Disposition', 'attachment', filename=file)
                    mime.set_payload(f.read())
                    encoders.encode_base64(mime)
                    msg.attach(mime)

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.send_message(msg)
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")

def main():
    print("Starting Quant Engine...")
    
    # 1. Data Ingestion
    fetcher = MarketDataFetcher()
    try:
        price_df = fetcher.fetch_price_history()
        print(f"Fetched {len(price_df)} data points.")
    except Exception as e:
        print(f"Critical Error Fetching Data: {e}")
        return

    # 2. Sentiment Analysis
    analyzer = SentimentAnalyzer()
    news = analyzer.fetch_news()
    avg_sentiment, sentiment_df = analyzer.analyze_sentiment(news)
    print(f"Sentiment Analysis Complete. Score: {avg_sentiment}")

    # 3. Modeling
    model = QuantModel(price_df)
    garch_res, current_vol = model.fit_garch()
    print(f"GARCH Model Fitted. Current Volatility: {current_vol}")
    
    simulations = model.simulate_gbm()
    final_prices = simulations[-1]
    
    # Calculate Statistics
    expected_price = np.mean(final_prices)
    ci_95 = np.percentile(final_prices, [2.5, 97.5])
    
    corr, p_corr = model.calc_correlation(sentiment_df)
    is_random_walk, p_val_rw = model.hypothesis_test()

    # 4. Reporting
    report_gen = ReportGenerator(model.last_price)
    report_gen.generate_plots(simulations, current_vol, ci_95[1], ci_95[0])
    report_md = report_gen.create_markdown(
        model.last_price, expected_price, garch_res, avg_sentiment, corr, is_random_walk, p_val_rw, ci_95
    )
    print("Report Generated.")

    # 5. Notification
    notifier = Notifier()
    notifier.send_email(report_md)

if __name__ == "__main__":
    main()
