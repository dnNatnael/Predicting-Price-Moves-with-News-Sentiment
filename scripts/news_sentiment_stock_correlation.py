"""
Compute correlation between aggregated news sentiment and daily stock returns.

Steps:
- Clean and align news + price dates
- Score sentiment per headline with TextBlob
- Average sentiment per date with configurable windows
- Compute daily returns from price data
- Merge and compute Pearson/Spearman correlation with p-values
- Handle timezone issues for news publication times
- Test different sentiment aggregation windows (T-1 vs same-day)

Robustness Features:
- Timezone handling for news timestamps
- Multiple sentiment aggregation strategies
- Statistical significance testing (p-values)
- Confidence intervals for correlations
- Configurable date ranges via CLI
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from textblob import TextBlob


@dataclass
class CorrelationResult:
    """Container for correlation analysis results"""
    pearson_corr: float
    spearman_corr: float
    pearson_pvalue: float
    spearman_pvalue: float
    confidence_interval_95: Tuple[float, float]
    n_observations: int
    aggregation_method: str


def load_news(news_path: Path, ticker: str, 
              start_date: Optional[str] = None,
              end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load and filter news data for a specific ticker
    
    Args:
        news_path: Path to news CSV file
        ticker: Stock ticker symbol
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        Filtered DataFrame with news data
    """
    news = pd.read_csv(
        news_path,
        usecols=["headline", "date", "date_only", "stock", "hour"],
        parse_dates=["date", "date_only"],
    )
    news = news[news["stock"] == ticker].copy()
    news["headline"] = news["headline"].fillna("")
    
    # Apply date filters
    if start_date:
        start = pd.to_datetime(start_date)
        news = news[news["date_only"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        news = news[news["date_only"] <= end]
    
    return news


def handle_timezone(news: pd.DataFrame, 
                   news_timezone: str = "America/New_York",
                   market_timezone: str = "America/New_York") -> pd.DataFrame:
    """
    Handle timezone conversion for news timestamps
    
    News is assumed to be published in the specified timezone (default: ET)
    Market closes at 4 PM ET, so news after 4 PM may affect next day prices
    
    Args:
        news: DataFrame with 'date' column
        news_timezone: Timezone of news timestamps
        market_timezone: Timezone of market (for market hours)
        
    Returns:
        DataFrame with timezone-adjusted dates
    """
    news = news.copy()
    
    # Check if date column is timezone-naive
    if news["date"].dt.tz is None:
        # Localize to news timezone
        news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(news_timezone)
    
    # Convert to market timezone
    news["date_market_tz"] = news["date"].dt.tz_convert(market_timezone)
    
    # Extract hour in market timezone
    news["hour_market"] = news["date_market_tz"].dt.hour
    
    # News published after 4 PM ET (market close) affects next trading day
    # This is a key insight: post-market news should be attributed to next day
    news["effective_date"] = np.where(
        news["hour_market"] >= 16,  # After 4 PM
        news["date_only"] + pd.Timedelta(days=1),
        news["date_only"]
    )
    
    return news


def score_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """
    Score sentiment of each headline using TextBlob
    
    Args:
        news: DataFrame with 'headline' column
        
    Returns:
        DataFrame with added 'sentiment' column
    """
    news = news.copy()
    news["sentiment"] = news["headline"].apply(
        lambda text: TextBlob(str(text)).sentiment.polarity
    )
    return news


def aggregate_daily_sentiment(news: pd.DataFrame, 
                              method: str = "mean",
                              date_column: str = "effective_date") -> pd.DataFrame:
    """
    Aggregate sentiment scores by date using various methods
    
    Args:
        news: DataFrame with sentiment scores
        method: Aggregation method ('mean', 'median', 'weighted')
        date_column: Column to use for grouping ('effective_date', 'date_only')
        
    Returns:
        DataFrame with aggregated daily sentiment
    """
    if method == "mean":
        sentiment = (
            news.groupby(date_column)["sentiment"]
            .mean()
            .rename("avg_daily_sentiment")
            .reset_index()
        )
    elif method == "median":
        sentiment = (
            news.groupby(date_column)["sentiment"]
            .median()
            .rename("avg_daily_sentiment")
            .reset_index()
        )
    elif method == "weighted":
        # Weight by headline length (longer headlines = more weight)
        news_copy = news.copy()
        news_copy["weight"] = news_copy["headline"].str.len()
        weighted_sum = news_copy.groupby(date_column).apply(
            lambda x: (x["sentiment"] * x["weight"]).sum() / x["weight"].sum()
        )
        sentiment = weighted_sum.rename("avg_daily_sentiment").reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    sentiment.columns = ["date_only", "avg_daily_sentiment"]
    return sentiment


def create_lagged_sentiment(sentiment: pd.DataFrame, 
                            lag: int = 1) -> pd.DataFrame:
    """
    Create lagged sentiment features
    
    Args:
        sentiment: DataFrame with daily sentiment
        lag: Number of days to lag (1 = previous day)
        
    Returns:
        DataFrame with lagged sentiment
    """
    sentiment = sentiment.copy()
    sentiment = sentiment.sort_values("date_only")
    sentiment[f"sentiment_lag_{lag}"] = sentiment["avg_daily_sentiment"].shift(lag)
    return sentiment


def load_prices(price_path: Path, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess price data
    
    Args:
        price_path: Path to price CSV file
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        DataFrame with daily returns
    """
    prices = pd.read_csv(price_path, parse_dates=["Date"])
    prices = prices.rename(columns={"Date": "date_only"})
    prices["daily_return"] = prices["Close"].pct_change()
    
    # Apply date filters
    if start_date:
        start = pd.to_datetime(start_date)
        prices = prices[prices["date_only"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        prices = prices[prices["date_only"] <= end]
    
    return prices[["date_only", "daily_return"]].dropna()


def compute_correlation_with_stats(sentiment: pd.DataFrame, 
                                   returns: pd.DataFrame,
                                   use_lagged: bool = False,
                                   lag: int = 1) -> CorrelationResult:
    """
    Compute correlation with statistical significance
    
    Args:
        sentiment: DataFrame with sentiment scores
        returns: DataFrame with daily returns
        use_lagged: Whether to use lagged sentiment
        lag: Lag in days for sentiment
        
    Returns:
        CorrelationResult with statistics
    """
    merged = pd.merge(sentiment, returns, on="date_only", how="inner").dropna()
    
    if merged.empty:
        raise ValueError("No overlapping dates between sentiment and returns.")
    
    # Determine which sentiment column to use
    if use_lagged:
        sentiment_col = f"sentiment_lag_{lag}"
        if sentiment_col not in merged.columns:
            # Create lag if not exists
            merged = create_lagged_sentiment(merged, lag)
        sentiment_values = merged[sentiment_col].dropna()
        returns_values = merged.loc[sentiment_values.index, "daily_return"]
    else:
        sentiment_values = merged["avg_daily_sentiment"]
        returns_values = merged["daily_return"]
    
    # Remove any remaining NaN
    valid_idx = ~(sentiment_values.isna() | returns_values.isna())
    sentiment_values = sentiment_values[valid_idx]
    returns_values = returns_values[valid_idx]
    
    if len(sentiment_values) < 10:
        raise ValueError("Insufficient observations for correlation analysis (minimum 10 required)")
    
    # Pearson correlation with p-value
    pearson_corr, pearson_pvalue = stats.pearsonr(sentiment_values, returns_values)
    
    # Spearman correlation (rank-based, more robust to outliers)
    spearman_corr, spearman_pvalue = stats.spearmanr(sentiment_values, returns_values)
    
    # Confidence interval for Pearson correlation (Fisher z-transformation)
    n = len(sentiment_values)
    z = np.arctanh(pearson_corr)  # Fisher z-transformation
    se = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se
    z_upper = z + 1.96 * se
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    method = f"lag_{lag}" if use_lagged else "same_day"
    
    return CorrelationResult(
        pearson_corr=pearson_corr,
        spearman_corr=spearman_corr,
        pearson_pvalue=pearson_pvalue,
        spearman_pvalue=spearman_pvalue,
        confidence_interval_95=(ci_lower, ci_upper),
        n_observations=n,
        aggregation_method=method
    )


def run(news_path: Path, price_path: Path, ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_timezone: bool = True,
        aggregation_method: str = "mean",
        use_lagged: bool = False,
        lag: int = 1,
        verbose: bool = True) -> CorrelationResult:
    """
    Run the complete correlation analysis
    
    Args:
        news_path: Path to news CSV
        price_path: Path to price CSV
        ticker: Stock ticker
        start_date: Start date filter
        end_date: End date filter
        use_timezone: Whether to handle timezone for news
        aggregation_method: How to aggregate sentiment ('mean', 'median', 'weighted')
        use_lagged: Whether to use lagged sentiment
        lag: Days to lag sentiment
        verbose: Print results
        
    Returns:
        CorrelationResult with statistics
    """
    # Load data
    news = load_news(news_path, ticker, start_date, end_date)
    returns = load_prices(price_path, start_date, end_date)
    
    if verbose:
        print(f"Ticker: {ticker}")
        print(f"News articles: {len(news)}")
        print(f"Trading days: {len(returns)}")
    
    # Handle timezone
    if use_timezone:
        news = handle_timezone(news)
        date_col = "effective_date"
    else:
        date_col = "date_only"
    
    # Score sentiment
    news = score_sentiment(news)
    
    # Aggregate sentiment
    sentiment = aggregate_daily_sentiment(news, method=aggregation_method, 
                                         date_column=date_col)
    
    if verbose:
        print(f"Days with sentiment: {len(sentiment)}")
    
    # Compute correlation with statistics
    result = compute_correlation_with_stats(sentiment, returns, 
                                           use_lagged=use_lagged, lag=lag)
    
    if verbose:
        print(f"\nCorrelation Analysis Results:")
        print(f"  Aggregation: {result.aggregation_method}")
        print(f"  Observations: {result.n_observations}")
        print(f"  Pearson correlation: {result.pearson_corr:.4f}")
        print(f"  Pearson p-value: {result.pearson_pvalue:.4f}")
        print(f"  Spearman correlation: {result.spearman_corr:.4f}")
        print(f"  Spearman p-value: {result.spearman_pvalue:.4f}")
        print(f"  95% CI: [{result.confidence_interval_95[0]:.4f}, {result.confidence_interval_95[1]:.4f}]")
        
        # Interpret significance
        if result.pearson_pvalue < 0.05:
            print("  Significance: STATISTICALLY SIGNIFICANT at α=0.05")
        else:
            print("  Significance: NOT significant at α=0.05")
    
    return result


def run_comprehensive_analysis(news_path: Path, price_path: Path, ticker: str,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> dict:
    """
    Run multiple correlation analyses with different configurations
    
    Tests:
    - Same-day vs lagged sentiment
    - Different aggregation methods
    
    Args:
        news_path: Path to news CSV
        price_path: Path to price CSV
        ticker: Stock ticker
        start_date: Start date filter
        end_date: End date filter
        
    Returns:
        Dictionary with results from all configurations
    """
    results = {}
    
    # Test same-day vs lagged
    for use_lagged in [False, True]:
        for lag in [1, 2, 3]:
            key = f"lag_{lag}" if use_lagged else "same_day"
            try:
                result = run(news_path, price_path, ticker, 
                           start_date, end_date,
                           use_timezone=True,
                           aggregation_method="mean",
                           use_lagged=use_lagged,
                           lag=lag,
                           verbose=False)
                results[key] = result
            except Exception as e:
                results[key] = {"error": str(e)}
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate news sentiment with stock price movements."
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol to filter news and pick price file (default: AAPL).",
    )
    parser.add_argument(
        "--news-path",
        type=Path,
        default=Path("output/data/articles_with_topics.csv"),
        help="Path to the news CSV file.",
    )
    parser.add_argument(
        "--price-path",
        type=Path,
        default=Path("data") / "AAPL.csv",
        help="Path to the price CSV file (default assumes ticker.csv inside data/).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for analysis (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for analysis (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--no-timezone",
        action="store_true",
        help="Disable timezone handling for news timestamps.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "median", "weighted"],
        default="mean",
        help="Sentiment aggregation method (default: mean).",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Number of days to lag sentiment (default: 1).",
    )
    parser.add_argument(
        "--use-lagged",
        action="store_true",
        help="Use lagged sentiment instead of same-day.",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive analysis with multiple configurations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Resolve price path
    resolved_price_path = args.price_path
    if resolved_price_path.is_dir():
        resolved_price_path = resolved_price_path / f"{args.ticker.upper()}.csv"
    
    if args.comprehensive:
        print("Running comprehensive correlation analysis...")
        results = run_comprehensive_analysis(
            args.news_path, resolved_price_path, args.ticker,
            args.start_date, args.end_date
        )
        print("\nSummary of all configurations:")
        print("-" * 60)
        for key, result in results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"{key}: Error - {result['error']}")
            else:
                sig = "SIGNIFICANT" if result.pearson_pvalue < 0.05 else "not sig"
                print(f"{key}: r={result.pearson_corr:.4f}, p={result.pearson_pvalue:.4f} ({sig})")
    else:
        run(
            args.news_path, resolved_price_path, args.ticker,
            args.start_date, args.end_date,
            use_timezone=not args.no_timezone,
            aggregation_method=args.aggregation,
            use_lagged=args.use_lagged,
            lag=args.lag,
            verbose=True
        )
