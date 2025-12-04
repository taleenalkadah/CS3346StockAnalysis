"""
Data Preparation Script
Merges stock data with sentiment scores
"""

import pandas as pd
import json
import numpy as np
import os


def merge_stock_and_sentiment(ticker):
    """Merge stock data with sentiment scores"""
    # Load stock data
    stock_path = f'data/stocks/{ticker}_data.csv'
    if not os.path.exists(stock_path):
        print(f"Error: {stock_path} not found. Please run data collection first.")
        return None
    
    print(f"Loading stock data from {stock_path}...")
    stock_df = pd.read_csv(stock_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # Load sentiment scores
    sentiment_path = f'data/news/{ticker}_daily_scores.json'
    if not os.path.exists(sentiment_path):
        print(f"Error: {sentiment_path} not found. Please run sentiment analysis first.")
        return None
    
    print(f"Loading sentiment scores from {sentiment_path}...")
    with open(sentiment_path, 'r', encoding='utf-8') as f:
        sentiment_scores = json.load(f)
    
    # Convert JSON to DataFrame
    sentiment_df = pd.DataFrame([
        [pd.to_datetime(date)] + scores
        for date, scores in sentiment_scores.items()
    ], columns=['Date', 'sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4', 'sentiment_5'])
    
    # Merge on date
    print("Merging stock data with sentiment scores...")
    merged_df = pd.merge(stock_df, sentiment_df, on='Date', how='left')
    
    # Fill missing sentiment values with neutral defaults
    sentiment_cols = ['sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4', 'sentiment_5']
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0.2)
    
    # Drop unnecessary columns if exist
    if 'Adj Close' in merged_df.columns:
        merged_df = merged_df.drop(columns=['Adj Close'])
    
    # Sort by date
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    # Remove invalid rows
    merged_df = merged_df.dropna(subset=['Date'])
    merged_df = merged_df[pd.to_numeric(merged_df['Close'], errors='coerce').notna()]

    # Save merged data
    output_path = f'data/stocks/{ticker}_merged.csv'
    os.makedirs('data/stocks', exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Merged data saved to {output_path}")
    print(f"Shape: {merged_df.shape}")
    print(f"\nColumns: {list(merged_df.columns)}")
    print(f"\nFirst few rows:")
    print(merged_df.head())
    print(f"\nLast few rows:")
    print(merged_df.tail())
    print(f"{'='*60}\n")
    
    return merged_df

if __name__ == "__main__":
    import sys
    
    # Default ticker
    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()
    
    print(f"\n{'='*60}")
    print(f"Data Preparation for {TICKER}")
    print(f"{'='*60}\n")
    
    merge_stock_and_sentiment(TICKER)
