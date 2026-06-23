import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))
FILE_NAME = os.path.join(DATA_DIR, 'weather_dataset.csv')

PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'results_lstm')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES =['temp', 'rh', 'dew', 'pressure', 'cloud', 'wx', 'wy', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
TARGETS = ['temp', 'rh', 'dew', 'pressure', 'cloud', 'wx', 'wy']

LOOKBACK = 120
HORIZON = 24

BEST_PARAMS = {
    'lstm_units': 128, 
    'dropout': 0.1, 
    'batch_size': 64, 
    'initial_lr': 0.001
}

SEEDS = [42, 123, 2026]