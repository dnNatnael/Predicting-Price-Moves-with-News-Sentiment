"""
Compute correlation between aggregated news sentiment and daily stock returns.

Steps:
- Clean and align news + price dates
- Score sentiment per headline with TextBlob
- Average sentiment per date
- Compute daily returns from price data
- Merge and compute Pearson correlation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from textblob import TextBlob


def load_news(news_path: Path, ticker: str) -> pd.DataFrame:
    news = pd.read_csv(
        news_path,
        usecols=["headline", "date_only", "stock"],
        parse_dates=["date_only"],
    )
    news = news[news["stock"] == ticker].copy()
    news["headline"] = news["headline"].fillna("")
    return news


def score_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    news["sentiment"] = news["headline"].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )
    return news


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    return (
        news.groupby("date_only")["sentiment"]
        .mean()
        .rename("avg_daily_sentiment")
        .reset_index()
    )


def load_prices(price_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(price_path, parse_dates=["Date"])
    prices = prices.rename(columns={"Date": "date_only"})
    prices["daily_return"] = prices["Close"].pct_change()
    return prices[["date_only", "daily_return"]].dropna()


def compute_correlation(sentiment: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(sentiment, returns, on="date_only", how="inner").dropna()
    if merged.empty:
        raise ValueError("No overlapping dates between sentiment and returns.")
    merged["pearson_corr"] = merged["avg_daily_sentiment"].corr(merged["daily_return"])
    return merged


def run(news_path: Path, price_path: Path, ticker: str) -> None:
    news = load_news(news_path, ticker)
    news = score_sentiment(news)
    sentiment = aggregate_daily_sentiment(news)
    returns = load_prices(price_path)
    merged = compute_correlation(sentiment, returns)

    corr_value = merged["pearson_corr"].iloc[0]
    print(f"Ticker: {ticker}")
    print(f"Rows after merge: {len(merged)}")
    print(f"Pearson correlation: {corr_value:.4f}")
    print("\nSample merged rows:")
    print(merged.head())


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_price_path = args.price_path
    if resolved_price_path.is_dir():
        resolved_price_path = resolved_price_path / f"{args.ticker.upper()}.csv"
    run(args.news_path, resolved_price_path, args.ticker.upper())

