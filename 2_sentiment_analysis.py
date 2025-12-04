"""
Sentiment Analysis Script
Performs sentiment analysis on news headlines using a BERT model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

# -------------------- core functions --------------------
# load pre-trained sentiment analysis model (BERT)
# returns: tokenizer and model in eval mode
def load_sentiment_model():
    """Load pre-trained sentiment analysis model."""
    print("Loading sentiment analysis model (nlptown/bert-base-multilingual-uncased-sentiment)...")
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model

# compute average sentiment distribution for a single day's headlines
# input: list of headline dictionaries, tokenizer, model
# output: list of 5 floats representing probabilities for 1..5 stars
def calculate_daily_sentiment(headlines, tokenizer, model):
    """
    Calculate average sentiment for a day's headlines.
    Returns a list of 5 floats (probabilities for 1..5 stars).
    """
    # no headlines for that day -> return neutral distribution
    if not headlines or len(headlines) == 0:
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    texts = [h.get("heading", "") for h in headlines if h.get("heading")]
    # no valid headline texts -> return neutral distribution
    if not texts:
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    batch_size = 16
    all_scores = []

    # process headlines in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            scores = logits.softmax(dim=1)  # shape: (batch, 5)
            all_scores.append(scores)

    # safety check: if something went wrong, fall back to neutral
    if not all_scores:
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    combined_scores = torch.cat(all_scores, dim=0)
    average_score = combined_scores.mean(dim=0).tolist()

    return average_score

# run sentiment analysis for all days and save daily average scores
# reads: data/news/{ticker}_headlines.json
# writes: data/news/{ticker}_daily_scores.json
def analyze_and_save_sentiment(ticker):
    """Analyze sentiment for all headlines and save daily average scores."""
    input_file = f"data/news/{ticker}_headlines.json"
    output_file = f"data/news/{ticker}_daily_scores.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run data collection first.")
        return

    print(f"Loading headlines from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loading sentiment model...")
    tokenizer, model = load_sentiment_model()

    result = {}
    total_days = len(data)
    processed = 0
    total_headlines_used = 0

    print(f"\nAnalyzing sentiment for {total_days} days...")
    print("-" * 60)

    # iterate over dates in sorted order for reproducible output
    for date in sorted(data.keys()):
        headlines = data[date]
        num_heads = len(headlines)
        total_headlines_used += num_heads

        avg_score = calculate_daily_sentiment(headlines, tokenizer, model)

        # safety: ensure we always store a 5-element list
        if not isinstance(avg_score, list) or len(avg_score) != 5:
            avg_score = [0.2, 0.2, 0.2, 0.2, 0.2]

        result[date] = avg_score
        processed += 1

        # progress logging every 25 days or at the end
        if processed % 25 == 0 or processed == total_days:
            print(f"Processed {processed}/{total_days} days (headlines this day: {num_heads})")

    os.makedirs("data/news", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Sentiment analysis complete!")
    print(f"Results saved to {output_file}")
    print(f"Total days analyzed: {len(result)}")
    print(f"Total headlines used: {total_headlines_used}")
    print(f"{'='*60}\n")


# -------------------- main --------------------
if __name__ == "__main__":
    import sys

    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()

    print(f"\n{'='*60}")
    print(f"Sentiment Analysis for {TICKER}")
    print(f"{'='*60}\n")

    analyze_and_save_sentiment(TICKER)
