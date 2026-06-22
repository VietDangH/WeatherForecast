import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from . import config

def set_deterministic_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_and_train_model(X_train, y_train, X_val, y_val, seed):
    set_deterministic_seed(seed)
    params = config.BEST_PARAMS
    
    model = Sequential([
        LSTM(params['lstm_units'], input_shape=(config.LOOKBACK, len(config.FEATURES)), return_sequences=False),
        Dropout(params['dropout']),
        Dense(params['lstm_units'], activation='relu'),
        Dense(config.HORIZON * len(config.TARGETS)),
        Reshape((config.HORIZON, len(config.TARGETS)))
    ])

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=params['initial_lr']))
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=params['batch_size'],
        validation_data=(X_val, y_val), 
        callbacks=[reduce_lr, early_stop], 
        verbose=2, shuffle=False
    )
    
    return model, history.history

def predict_and_inverse(model, X_test, scaler_y, N_win):
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_real = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, len(config.TARGETS))
    ).reshape(N_win, config.HORIZON, len(config.TARGETS))
    return y_pred_real