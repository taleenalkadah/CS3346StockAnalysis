import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import feedparser
import json
import requests
from bs4 import BeautifulSoup
import time
import os

# swap start date if need
def fetch_stock_data(ticker, start_date="2015-01-01"):
    # fetch stock data from yfinance
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    return data

def scrape_news_headlines_yahoo(ticker, start_date, end_date):
    # fetch news data from yfinance rss
    all_headlines = {}
    yahoo_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    
    try:
        print(f"Fetching news from Yahoo Finance RSS for {ticker}...")
        feed = feedparser.parse(yahoo_rss)
        
        for entry in feed.entries:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
                date_str = pub_date.strftime("%Y-%m-%d")
                
                if start_date <= date_str <= end_date:
                    if date_str not in all_headlines:
                        all_headlines[date_str] = []
                    all_headlines[date_str].append({
                        "heading": entry.title,
                        "summary": entry.get('summary', ''),
                        "link": entry.link
                    })
            except Exception as e:
                continue
    except Exception as e:
        print(f"Error fetching news from Yahoo RSS: {e}")
    
    return all_headlines

def scrape_news_headlines_alternative(ticker, start_date, end_date):
    # if yfinance not working => fetch from yfinane news page
    all_headlines = {}
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('h3', class_='Mb(5px)')
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            # limit to 20 articles
            for article in articles[:20]:
                heading = article.get_text(strip=True)
                if heading:
                    if current_date not in all_headlines:
                        all_headlines[current_date] = []
                    all_headlines[current_date].append({
                        "heading": heading,
                        "summary": "",
                        "link": ""
                    })
    except Exception as e:
        print(f"Error in alternative news scraping: {e}")
    
    return all_headlines

def scrape_news_headlines(ticker, start_date, end_date):
    # scrape news headlines
    all_headlines = scrape_news_headlines_yahoo(ticker, start_date, end_date)
    
    # if no headlines found, try alternative method
    if not all_headlines:
        print("RSS feed returned no results, trying alternative method...")
        all_headlines = scrape_news_headlines_alternative(ticker, start_date, end_date)
    
    # fill in missing dates with empty lists
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        if date_str not in all_headlines:
            all_headlines[date_str] = []
        current += timedelta(days=1)
    
    return all_headlines

def save_data(stock_data, headlines, ticker):
    # ensure directories exist
    os.makedirs('data/stocks', exist_ok=True)
    os.makedirs('data/news', exist_ok=True)
    
    # save stock data
    stock_path = f'data/stocks/{ticker}_data.csv'
    stock_data.to_csv(stock_path, index=False)
    print(f"Stock data saved to {stock_path}")
    
    # save headlines
    headlines_path = f'data/news/{ticker}_headlines.json'
    with open(headlines_path, 'w', encoding='utf-8') as f:
        json.dump(headlines, f, ensure_ascii=False, indent=2)
    print(f"Headlines saved to {headlines_path}")
    
    # print summary
    total_headlines = sum(len(v) for v in headlines.values())
    days_with_news = sum(1 for v in headlines.values() if len(v) > 0)
    print(f"\nSummary:")
    print(f"  Total headlines collected: {total_headlines}")
    print(f"  Days with news: {days_with_news}")
    print(f"  Stock data rows: {len(stock_data)}")

if __name__ == "__main__":
    import sys
    
    # default to apple
    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()
    
    STOCK_START_DATE = "2015-01-01"
    if len(sys.argv) > 2:
        STOCK_START_DATE = sys.argv[2]
    
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    # fetch news from past 30 days only
    NEWS_START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Data Collection for {TICKER}")
    print(f"{'='*60}\n")
    
    print(f"Stock data: Fetching from {STOCK_START_DATE} to {END_DATE}")
    stock_data = fetch_stock_data(TICKER, STOCK_START_DATE)
    print(f"Stock data shape: {stock_data.shape}")
    
    print(f"\nNews data: Fetching from {NEWS_START_DATE} to {END_DATE} (past 30 days)")
    headlines = scrape_news_headlines(TICKER, NEWS_START_DATE, END_DATE)
    
    save_data(stock_data, headlines, TICKER)
    
    print(f"\n{'='*60}")
    print("Data collection complete!")
    print(f"{'='*60}\n")

