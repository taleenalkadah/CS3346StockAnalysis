# CS3346StockAnalysis

Created by Nada, Hartej, Nish, and Taleen.

# Quick Start Guide

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Run (All Steps)

Run the complete pipeline for a stock ticker:

```bash
python run_all.py AAPL
```

This will:

1. Collect stock data and news
2. Analyze sentiment
3. Prepare merged dataset
4. Train the model
5. Make predictions

You can find stock tickers on Yahoo Finance. Here are a couple for reference:

Apple = AAPL
Tesla = TSLA
Google = GOOGL
Microsoft = MSFT

## Step-by-Step Run

### 1. Collect Data

```bash
python 1_data_collection.py AAPL
```

### 2. Analyze Sentiment

```bash
python 2_sentiment_analysis.py AAPL
```

_Note: First run will download the BERT model (~600MB)_

### 3. Prepare Data

```bash
python 3_data_preparation.py AAPL
```

### 4. Train Model

```bash
python 4_model_training.py AAPL
```

### 5. Get Predictions

```bash
python 5_prediction.py AAPL
```

## Customization

### Change Training Parameters

```bash
python 4_model_training.py AAPL 100 50 32
```

### Use Different Start Date

```bash
python 1_data_collection.py AAPL 2020-01-01
```

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed
- **Model not found**: Run step 4 (training) first
- **No news data**: System will use neutral sentiment (0.2)
- **Memory errors**: Reduce batch_size or look_back in training

## Expected Runtime

- Data collection: 1-5 minutes
- Sentiment analysis: 5-15 minutes (first time includes model download)
- Data preparation: < 1 minute
- Model training: 5-10 minutes
- Prediction: < 1 minute
