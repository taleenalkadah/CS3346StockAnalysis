"""
Convenience script to run all steps in sequence
Usage: python run_all.py AAPL
"""

import sys
import subprocess
import os

def run_script(script_name, ticker, *args):
    """Run a Python script and check for errors"""
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}\n")
    
    cmd = [sys.executable, script_name, ticker] + list(args)
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\nError: {script_name} failed with exit code {result.returncode}")
        return False
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <TICKER> [start_date]")
        print("Example: python run_all.py AAPL")
        print("Example: python run_all.py TSLA 2020-01-01")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2015-01-01"
    
    print(f"\n{'='*70}")
    print(f"Unified Stock Predictor - Full Pipeline")
    print(f"Ticker: {ticker}")
    print(f"Start Date: {start_date}")
    print(f"{'='*70}\n")
    
    steps = [
        ("1_data_collection.py", [start_date]),
        ("2_sentiment_analysis.py", []),
        ("3_data_preparation.py", []),
        # ("4_model_training.py", []),
        # ("5_prediction.py", [])
    ]
    
    for script, args in steps:
        if not run_script(script, ticker, *args):
            print(f"\nPipeline stopped at {script}")
            print("Please fix the error and rerun from this step.")
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print("Pipeline completed successfully!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
