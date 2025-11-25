import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynance as pn
import talib


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "AAPL.csv"
OUTPUT_DIR = BASE_DIR / "output" / "technical_analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"

plt.style.use("seaborn-v0_8-darkgrid")


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().title() for col in df.columns]
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("Date").set_index("Date")
    df = df.replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["SMA_20"] = talib.SMA(close, timeperiod=20)
    df["SMA_50"] = talib.SMA(close, timeperiod=50)
    df["SMA_100"] = talib.SMA(close, timeperiod=100)

    df["EMA_20"] = talib.EMA(close, timeperiod=20)
    df["EMA_50"] = talib.EMA(close, timeperiod=50)

    df["RSI_14"] = talib.RSI(close, timeperiod=14)

    slowk, slowd = talib.STOCH(
        high,
        low,
        close,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    df["STOCH_%K"] = slowk
    df["STOCH_%D"] = slowd

    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)

    upper, middle, lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BB_Upper"] = upper
    df["BB_Middle"] = middle
    df["BB_Lower"] = lower
    return df


def compute_pynance_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    close_frame = df[["Close"]].copy()

    daily_ret = pn.tech.ret(df, selection="Close", outputcol="Daily_Return")
    df["Daily_Return"] = daily_ret["Daily_Return"]
    df["Daily_Return"] = df["Daily_Return"].fillna(0.0)

    log_ret = pn.tech.ln_growth(df, selection="Close", outputcol="Log_Return")
    df["Log_Return"] = log_ret["Log_Return"]
    df["Log_Return"] = df["Log_Return"].fillna(0.0)

    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    rolling_vol = pn.tech.volatility(
        close_frame, selection="Close", window=30, outputcol="Rolling_Volatility_30"
    )
    df["Rolling_Volatility_30"] = rolling_vol["Rolling_Volatility_30"]

    returns_frame = df[["Daily_Return"]].copy()
    returns_frame.columns = ["Daily_Return"]
    rolling_return_vol = pn.tech.volatility(
        returns_frame, selection="Daily_Return", window=30, outputcol="Return_Vol_30"
    )
    df["Return_Vol_30"] = rolling_return_vol["Return_Vol_30"]

    latest_vol = df["Return_Vol_30"].dropna().iloc[-1]
    annualized_vol = latest_vol * math.sqrt(252)
    metrics["Annualized_Volatility"] = annualized_vol

    risk_free_daily = 0.02 / 252
    excess_returns = df["Daily_Return"] - risk_free_daily
    sharpe = (
        excess_returns.mean()
        / excess_returns.std(ddof=0)
        * math.sqrt(252)
        if excess_returns.std(ddof=0) != 0
        else np.nan
    )
    metrics["Sharpe_Ratio"] = sharpe

    metrics["Return_Summary"] = {
        "avg_daily_return": df["Daily_Return"].mean(),
        "avg_log_return": df["Log_Return"].mean(),
        "last_cumulative_return": df["Cumulative_Return"].iloc[-1],
    }

    metrics["Correlation_Matrix"] = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].corr()

    metrics["Autocorrelation_Returns_lag1"] = df["Daily_Return"].autocorr(lag=1)
    metrics["Autocorrelation_Returns_lag5"] = df["Daily_Return"].autocorr(lag=5)
    return metrics


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_plot(fig, name: str):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def create_plots(df: pd.DataFrame):
    last_year = df.index >= (df.index.max() - pd.DateOffset(years=1))
    plot_df = df.loc[last_year]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Close"], label="Close", color="#1f77b4")
    ax.set_title("Closing Price")
    ax.set_ylabel("Price ($)")
    ax.legend()
    _save_plot(fig, "closing_price")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Close"], label="Close", color="#1f77b4")
    ax.plot(plot_df.index, plot_df["SMA_20"], label="SMA 20", color="#ff7f0e")
    ax.plot(plot_df.index, plot_df["SMA_50"], label="SMA 50", color="#2ca02c")
    ax.plot(plot_df.index, plot_df["EMA_20"], label="EMA 20", color="#d62728")
    ax.plot(plot_df.index, plot_df["EMA_50"], label="EMA 50", color="#9467bd")
    ax.set_title("Price with SMA and EMA Overlays")
    ax.set_ylabel("Price ($)")
    ax.legend()
    _save_plot(fig, "price_with_sma_ema")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(plot_df.index, plot_df["RSI_14"], label="RSI", color="#9467bd")
    ax.axhline(70, color="red", linestyle="--", linewidth=1)
    ax.axhline(30, color="green", linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_title("RSI (14)")
    ax.set_ylabel("RSI")
    ax.legend()
    _save_plot(fig, "rsi")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["MACD"], label="MACD", color="#1f77b4")
    ax.plot(plot_df.index, plot_df["MACD_Signal"], label="Signal", color="#ff7f0e")
    ax.bar(plot_df.index, plot_df["MACD_Hist"], label="Histogram", color="#2ca02c")
    ax.set_title("MACD")
    ax.set_ylabel("Value")
    ax.legend()
    _save_plot(fig, "macd")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(plot_df.index, plot_df["Volume"] / 1e6, color="#17becf")
    ax.set_title("Volume (Millions)")
    ax.set_ylabel("Volume (M)")
    _save_plot(fig, "volume")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Close"], label="Close", color="#1f77b4")
    ax.fill_between(
        plot_df.index,
        plot_df["BB_Lower"],
        plot_df["BB_Upper"],
        color="#c6dbef",
        alpha=0.4,
        label="Bollinger Bands",
    )
    ax.plot(plot_df.index, plot_df["BB_Middle"], label="BB Middle", color="#2ca02c")
    ax.set_title("Bollinger Bands")
    ax.set_ylabel("Price ($)")
    ax.legend()
    _save_plot(fig, "bollinger_bands")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(plot_df["Daily_Return"].dropna(), bins=30, color="#ff7f0e", alpha=0.7)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    _save_plot(fig, "daily_returns_hist")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Cumulative_Return"], color="#2ca02c")
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    _save_plot(fig, "cumulative_returns")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Rolling_Volatility_30"], color="#d62728")
    ax.set_title("30-Day Rolling Volatility (Price)")
    ax.set_ylabel("Volatility")
    _save_plot(fig, "rolling_volatility")


def main():
    ensure_output_dirs()
    df = load_and_clean(DATA_PATH)
    df = add_ta_indicators(df)
    metrics = compute_pynance_metrics(df)
    create_plots(df)

    snapshot_path = OUTPUT_DIR / "dataframe_snapshot.csv"
    df.tail(10).to_csv(snapshot_path)
    corr_path = OUTPUT_DIR / "correlation_matrix.csv"
    metrics["Correlation_Matrix"].to_csv(corr_path)

    summary = {
        "annualized_volatility": metrics["Annualized_Volatility"],
        "sharpe_ratio": metrics["Sharpe_Ratio"],
        "last_cumulative_return": metrics["Return_Summary"]["last_cumulative_return"],
        "autocorr_lag1": metrics["Autocorrelation_Returns_lag1"],
        "autocorr_lag5": metrics["Autocorrelation_Returns_lag5"],
        "snapshot_path": snapshot_path.as_posix(),
        "plots_dir": PLOTS_DIR.as_posix(),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)
    print("Analysis complete. Summary saved to", summary_path)


if __name__ == "__main__":
    main()

