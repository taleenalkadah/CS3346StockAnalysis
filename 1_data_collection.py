import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

import yfinance as yf
import pandas as pd
import requests

# -------------------- define constants --------------------
DEFAULT_TICKER = "AAPL"
DEFAULT_STOCK_START_DATE = "2020-01-01"

# how many days of news to fetch relative to "now"
NEWS_LOOKBACK_DAYS = 8

DATE_FMT = "%Y-%m-%d"

# NewsAPI configuration
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_API_KEY = "176edb56f96746ba8f621df345ec8bee"

# optional richer queries per ticker (not used when querying with just ticker)
TICKER_QUERIES = {
    "AAPL": '',
    # "MSFT": 'Microsoft OR MSFT OR "Microsoft Corp" OR Windows OR Azure',
}


# -------------------- core functions: stock data --------------------
# fetch historical stock data for a ticker from Yahoo Finance using yfinance
# takes: ticker, a start date in YYYY-MM-DD format
# returns: pd.DataFrame containing historical OHLCV data
def fetch_stock_data(ticker: str, start_date: str = DEFAULT_STOCK_START_DATE) -> pd.DataFrame:
    """
    Fetch historical stock data for a ticker from Yahoo Finance using yfinance.
    """
    end_date = datetime.now().strftime(DATE_FMT)
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")

    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"]).dt.strftime(DATE_FMT)

    return data


# -------------------- core functions: news via NewsAPI --------------------
# fetch news headlines for a ticker using the NewsAPI /v2/everything endpoint
# requests only one page with pageSize <= 100 (developer account limit)
# does not send `from=`, and instead filters by [start_date, end_date] locally
# takes: ticker, start date, end date in YYYY-MM-DD format
# returns: dictionary mapping date string to list of headline dictionaries
def fetch_news_headlines_newsapi(
    ticker: str,
    start_date: str,
    end_date: str,
    page_size: int = 100,
    max_pages: int = 5,  # kept just for signature compatibility; ignored
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch news headlines for a ticker using the NewsAPI /v2/everything endpoint.

    - Requests only ONE page with pageSize <= 100 (limit of dev accounts).
    - Does NOT send `from=`; instead, we filter by [start_date, end_date] locally.
    """
    if not NEWS_API_KEY:
        print("WARNING: NEWS_API_KEY is not set. Skipping NewsAPI.")
        return {}

    print(f"Fetching news from NewsAPI for {ticker}")

    all_headlines: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # hard cap of 100 results
    page_size = min(page_size, 100)

    params = {
        "q": ticker,
        "to": end_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "page": 1,  # single page only
        "apiKey": NEWS_API_KEY,
    }

    total_articles = 0
    start_dt = datetime.strptime(start_date, DATE_FMT)
    end_dt = datetime.strptime(end_date, DATE_FMT)

    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=15)
    except Exception as e:
        print(f"Error connecting to NewsAPI: {e}")
        return {}

    if resp.status_code != 200:
        print(f"NewsAPI returned status {resp.status_code}: {resp.text[:200]}")
        return {}

    payload = resp.json()
    status = payload.get("status")
    if status != "ok":
        print(f"NewsAPI error status: {status}, message: {payload.get('message')}")
        return {}

    articles = payload.get("articles", [])
    if not articles:
        print("NewsAPI returned 0 articles.")
        return {}

    for art in articles:
        published_at = art.get("publishedAt")
        if not published_at:
            continue

        try:
            pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except Exception:
            continue

        # filter locally to [start_date, end_date]
        if not (start_dt <= pub_dt.replace(tzinfo=None) <= end_dt):
            continue

        date_str = pub_dt.strftime(DATE_FMT)

        all_headlines[date_str].append(
            {
                "heading": art.get("title", "") or "",
                "summary": art.get("description", "") or "",
                "link": art.get("url", "") or "",
                "source": (art.get("source") or {}).get("name", "") or "",
            }
        )
        total_articles += 1

    print(f"NewsAPI fetched {total_articles} articles total (after date filtering).")
    return dict(all_headlines)


# -------------------- news headline scraping (NewsAPI only) --------------------
# fetch news headlines for a ticker across a date range using only NewsAPI
# ensures every date in the range exists as a key (mapping to a list, possibly empty)
# takes: ticker, start date, end date in YYYY-MM-DD format
# returns: dictionary mapping date to list of headline dictionaries
def scrape_news_headlines(
    ticker: str,
    start_date: str,
    end_date: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch news headlines for a ticker across a date range using **only** NewsAPI.
    Ensures every date in the range exists as a key (possibly empty list).
    """
    headlines = fetch_news_headlines_newsapi(ticker, start_date, end_date) or {}

    total_headlines = sum(len(v) for v in headlines.values())
    print(f"Total headlines collected from NewsAPI: {total_headlines}")

    # ensure every date exists as a key
    start = datetime.strptime(start_date, DATE_FMT)
    end = datetime.strptime(end_date, DATE_FMT)
    current = start

    while current <= end:
        date_str = current.strftime(DATE_FMT)
        headlines.setdefault(date_str, [])
        current += timedelta(days=1)

    return headlines


# -------------------- save stock and news data --------------------
# save stock and news data to disk under ./data/stocks and ./data/news
# takes: stock data in pd.DataFrame format, headlines dictionary, ticker
def save_data(
    stock_data: pd.DataFrame,
    headlines: Dict[str, List[Dict[str, Any]]],
    ticker: str
) -> None:
    """
    Save stock and news data to disk under ./data/stocks and ./data/news.
    """
    os.makedirs("data/stocks", exist_ok=True)
    os.makedirs("data/news", exist_ok=True)

    stock_path = f"data/stocks/{ticker}_data.csv"
    stock_data.to_csv(stock_path, index=False)
    print(f"Stock data saved to {stock_path}")

    headlines_path = f"data/news/{ticker}_headlines.json"
    with open(headlines_path, "w", encoding="utf-8") as f:
        json.dump(headlines, f, ensure_ascii=False, indent=2)
    print(f"Headlines saved to {headlines_path}")

    total_headlines = sum(len(v) for v in headlines.values())
    days_with_news = sum(1 for v in headlines.values() if v)
    print("\nSummary:")
    print(f"  Total headlines collected: {total_headlines}")
    print(f"  Days with at least one headline: {days_with_news}")
    print(f"  Stock data rows: {len(stock_data)}")


# -------------------- main / CLI helpers --------------------
# CLI argument parsing
# usage: script.py [TICKER] [STOCK_START_DATE]
def parse_args(argv: List[str]) -> tuple[str, str]:
    """
    Usage: script.py [TICKER] [STOCK_START_DATE]
    """
    ticker = DEFAULT_TICKER
    stock_start_date = DEFAULT_STOCK_START_DATE

    if len(argv) > 1:
        ticker = argv[1].upper()
    if len(argv) > 2:
        stock_start_date = argv[2]

    return ticker, stock_start_date


# main entry point
def main(argv: List[str]) -> None:
    ticker, stock_start_date = parse_args(argv)

    end_date = datetime.now().strftime(DATE_FMT)
    news_start_date = (datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime(DATE_FMT)

    print("\n" + "=" * 60)
    print(f"Data Collection for {ticker}")
    print("=" * 60 + "\n")

    # stock data
    print(f"Stock data: Fetching from {stock_start_date} to {end_date}")
    stock_data = fetch_stock_data(ticker, stock_start_date)
    print(f"Stock data shape: {stock_data.shape}")

    # news data
    print(f"\nNews data: Fetching from {news_start_date} to {end_date} (past {NEWS_LOOKBACK_DAYS} days)")
    headlines = scrape_news_headlines(ticker, news_start_date, end_date)

    # save data
    save_data(stock_data, headlines, ticker)

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main(sys.argv)
