import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Dropout

# === Random Forest ===
def fit_predict_block_rf(X_train, y_train, X_test, n_estimators=350, max_depth=None, seed=42, n_jobs=-1):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced")
    rf.fit(X_train_s, y_train)
    prob_up = rf.predict_proba(X_test_s)[:, 1]
    return prob_up, rf, scaler

# === XGBoost ===
def fit_predict_block_xgb(X_train, y_train, X_test, seed=42):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train_s, y_train)
    prob_up = model.predict_proba(X_test_s)[:, 1]
    return prob_up, model, scaler

# === CNN ===
def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def fit_predict_block_cnn(X_train, y_train, X_test, epochs=10, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn(input_shape)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
    prob_up = model.predict(X_test).flatten()
    return prob_up, model, history

# === LSTM ===
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def fit_predict_block_lstm(X_train, y_train, X_test, epochs=10, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm(input_shape)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
    prob_up = model.predict(X_test).flatten()
    return prob_up, model, history

# === CNN-LSTM Hybrid ===
def build_cnnlstm(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def fit_predict_block_cnnlstm(X_train, y_train, X_test, epochs=10, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnnlstm(input_shape)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
    prob_up = model.predict(X_test).flatten()
    return prob_up, model, history
