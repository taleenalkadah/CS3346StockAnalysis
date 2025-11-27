import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
from datetime import datetime
import os

def load_sentiment_model():
    # pretrained sentiment analyzer
    print("Loading sentiment analysis model...")
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Model loaded successfully!")
    return tokenizer, model

def calculate_daily_sentiment(headlines, tokenizer, model):
    if not headlines or len(headlines) == 0:
        # 0.2 for neutral if no news
        return [0.2, 0.2, 0.2, 0.2, 0.2]
    
    texts = [h['heading'] for h in headlines if 'heading' in h and h['heading']]
    
    if not texts:
        return [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # process in batches
    batch_size = 32
    all_scores = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_attention_mask=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            scores = logits.softmax(dim=1)
            all_scores.append(scores)
    
    if len(all_scores) > 1:
        combined_scores = torch.cat(all_scores, dim=0)
    else:
        combined_scores = all_scores[0]
    
    average_score = combined_scores.mean(dim=0).tolist()
    
    return average_score

def analyze_and_save_sentiment(ticker):
    input_file = f'data/news/{ticker}_headlines.json'
    output_file = f'data/news/{ticker}_daily_scores.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run data collection first.")
        return
    
    print(f"Loading headlines from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Loading sentiment model...")
    tokenizer, model = load_sentiment_model()
    
    result = {}
    total_days = len(data)
    processed = 0
    
    print(f"\nAnalyzing sentiment for {total_days} days...")
    print("-" * 60)
    
    for date, headlines in sorted(data.items()):
        average_score = calculate_daily_sentiment(headlines, tokenizer, model)
        result[date] = average_score
        processed += 1
        
        if processed % 50 == 0 or processed == total_days:
            print(f"Processed {processed}/{total_days} days...")
    
    os.makedirs('data/news', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Sentiment analysis complete!")
    print(f"Results saved to {output_file}")
    print(f"Total days analyzed: {len(result)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()
    
    print(f"\n{'='*60}")
    print(f"Sentiment Analysis for {TICKER}")
    print(f"{'='*60}\n")
    
    analyze_and_save_sentiment(TICKER)

