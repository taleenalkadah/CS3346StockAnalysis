"""
Script to train LSTM model on merged stock and sentiment data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import pickle

# Function to create sequences for LSTM training
def create_dataset(dataset, target, look_back=100):
    # sequences for training
    dataX, dataY = [], []
    for i in range(look_back, len(dataset)):
        dataX.append(dataset[i-look_back:i, :])
        dataY.append(target[i])
    return np.array(dataX), np.array(dataY)

# Function to build LSTM model architecture
def build_lstm_model(input_shape, neurons=100, dropout=0.3):
    # build LSTM model architecture
    model = Sequential()
    
    model.add(LSTM(units=neurons, return_sequences=True, activation='tanh', input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=neurons, return_sequences=True, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(LSTM(units=neurons, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='linear', activity_regularizer=regularizers.l1(0.00001)))
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    return model

# Train the LSTM model
def train_model(ticker, look_back=100, epochs=50, batch_size=32, neurons=100, dropout=0.3):
    # Load merged data
    merged_path = f'data/stocks/{ticker}_merged.csv'
    if not os.path.exists(merged_path):
        print(f"Error: {merged_path} not found. Please run data preparation first.")
        return None
    
    # Load merged data
    print(f"Loading merged data from {merged_path}...")
    df = pd.read_csv(merged_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # print data shape and date range
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Prepare features (exclude Date and Close)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
    print(f"\nFeature columns: {feature_cols}")
    features = df[feature_cols].values
    target = df['Close'].values.reshape(-1, 1)
    
    # scale features and target
    print("\nScaling features and target...")
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target)
    
    # create sequences
    print(f"\nCreating sequences with look_back={look_back}...")
    X, y = create_dataset(features_scaled, target_scaled, look_back)
    print(f"Sequences shape: X={X.shape}, y={y.shape}")
    
    # split train/test (80/20)
    train_size = int(len(X) * 0.8)
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = y[:train_size], y[train_size:]
    
    print(f"\nTrain set: {trainX.shape[0]} samples")
    print(f"Test set: {testX.shape[0]} samples")
    
    # build model
    print(f"\nBuilding LSTM model...")
    model = build_lstm_model((look_back, features.shape[1]), neurons=neurons, dropout=dropout)
    model.summary()
    
    # train model
    print(f"\nTraining model for {epochs} epochs...")
    print("-" * 60)
    history = model.fit(
        trainX, trainY, 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=1, 
        validation_split=0.2
    )
    
    # evaluate
    print("\nEvaluating model...")
    train_predict = model.predict(trainX, verbose=0)
    test_predict = model.predict(testX, verbose=0)
    
    # inverse transform
    train_predict_inv = scaler_target.inverse_transform(train_predict)
    trainY_inv = scaler_target.inverse_transform(trainY.reshape(-1, 1))
    test_predict_inv = scaler_target.inverse_transform(test_predict)
    testY_inv = scaler_target.inverse_transform(testY.reshape(-1, 1))
    
    # Calculating metrics
    train_rmse = np.sqrt(mean_squared_error(trainY_inv, train_predict_inv))
    test_rmse = np.sqrt(mean_squared_error(testY_inv, test_predict_inv))
    train_mae = mean_absolute_error(trainY_inv, train_predict_inv)
    test_mae = mean_absolute_error(testY_inv, test_predict_inv)
    
    train_accuracy = 100 - (train_rmse / np.mean(trainY_inv) * 100)
    test_accuracy = 100 - (test_rmse / np.mean(testY_inv) * 100)
    
    print(f"\n{'='*60}")
    print("Model Performance Metrics:")
    print(f"{'='*60}")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Testing RMSE: ${test_rmse:.2f}")
    print(f"Training MAE: ${train_mae:.2f}")
    print(f"Testing MAE: ${test_mae:.2f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # save model and scalers
    os.makedirs('models', exist_ok=True)
    
    model_path = f'models/{ticker}_lstm_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    scaler_features_path = f'models/{ticker}_scaler_features.pkl'
    with open(scaler_features_path, 'wb') as f:
        pickle.dump(scaler_features, f)
    print(f"Feature scaler saved to {scaler_features_path}")
    
    scaler_target_path = f'models/{ticker}_scaler_target.pkl'
    with open(scaler_target_path, 'wb') as f:
        pickle.dump(scaler_target, f)
    print(f"Target scaler saved to {scaler_target_path}")
    
    feature_cols_path = f'models/{ticker}_feature_cols.pkl'
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Feature columns saved to {feature_cols_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # plot training set: actual vs predicted (last 200 samples)
    plt.subplot(1, 2, 2)
    plt.plot(trainY_inv[-200:], label='Actual Train', alpha=0.7)
    plt.plot(train_predict_inv[-200:], label='Predicted Train', alpha=0.7)
    plt.title('Training Set: Actual vs Predicted (Last 200 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plot_path = f'models/{ticker}_training_plot.png'
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()
    
    # Plot test predictions
    plt.figure(figsize=(15, 6))
    plt.plot(testY_inv, label='Actual Test', linewidth=2)
    plt.plot(test_predict_inv, label='Predicted Test', linewidth=2, alpha=0.8)
    plt.title(f'Test Set: Actual vs Predicted Prices for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    test_plot_path = f'models/{ticker}_test_predictions.png'
    plt.savefig(test_plot_path)
    print(f"Test predictions plot saved to {test_plot_path}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("Model training complete!")
    print(f"{'='*60}\n")
    
    return model, scaler_features, scaler_target, feature_cols

# main function
if __name__ == "__main__":
    import sys
    
    # default ticker
    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()
    
    # hyperparameters
    LOOK_BACK = 100
    EPOCHS = 50
    BATCH_SIZE = 32
    NEURONS = 100
    DROPOUT = 0.3
    
    if len(sys.argv) > 2:
        LOOK_BACK = int(sys.argv[2])
    if len(sys.argv) > 3:
        EPOCHS = int(sys.argv[3])
    if len(sys.argv) > 4:
        BATCH_SIZE = int(sys.argv[4])
    
    # print hyperparameters
    print(f"\n{'='*60}")
    print(f"Model Training for {TICKER}")
    print(f"{'='*60}")
    print(f"Look back: {LOOK_BACK}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Neurons: {NEURONS}")
    print(f"Dropout: {DROPOUT}")
    print(f"{'='*60}\n")
    
    train_model(TICKER, LOOK_BACK, EPOCHS, BATCH_SIZE, NEURONS, DROPOUT)

