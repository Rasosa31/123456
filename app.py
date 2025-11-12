# app.py
from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Cargar modelo y features (entrenado con nombres genéricos)
model = joblib.load('best_model.pkl')
features = joblib.load('data/features.pkl')

def download_and_align(pa_ticker, ca_ticker, period="10y"):
    print(f"Descargando {pa_ticker} y {ca_ticker}...")
    end = datetime.today()
    start = end - timedelta(days=365*10)

    # Descargar PA
    pa_df = yf.download(pa_ticker, start=start, end=end, progress=False)
    ca_df = yf.download(ca_ticker, start=start, end=end, progress=False)

    if pa_df.empty or ca_df.empty:
        raise ValueError("No se pudieron descargar los datos")

    # Renombrar columnas
    pa_df = pa_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    ca_df = ca_df[['Open', 'High', 'Low', 'Close', 'Volume']]

    pa_df.columns = [f'PA_{col}' for col in pa_df.columns]
    ca_df.columns = [f'CA_{col}' for col in ca_df.columns]

    # Alinear por índice
    df = pa_df.copy()
    for col in ca_df.columns:
        df[col] = ca_df[col].reindex(df.index).ffill()

    df = df.dropna()
    return df

def build_features(df):
    df['CA_Change'] = df['CA_Close'].pct_change()
    df['PA_CA_Ratio'] = df['PA_Close'] / (df['CA_Close'] + 1e-8)
    df['CA_Volatility'] = df['CA_Close'].rolling(14).std()
    df['SMA_100'] = df['PA_Close'].rolling(100).mean()

    # Añade más si tu modelo las usa
    df['PA_Return'] = df['PA_Close'].pct_change()
    df['PA_Volume_Change'] = df['PA_Volume'].pct_change()

    df = df.dropna()
    return df

@app.route('/predict/<pa_ticker>/<ca_ticker>')
def predict(pa_ticker, ca_ticker):
    try:
        # 1. Descargar y alinear
        df = download_and_align(pa_ticker, ca_ticker)
        
        # 2. Construir features
        df = build_features(df)
        
        if len(df) == 0:
            return jsonify({"error": "No hay datos suficientes"}), 400

        # 3. Última fila
        X = df[features].tail(1)
        
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        confidence = float(max(prob))
        
        meaning = "0 = SUBE mañana" if pred == 0 else "1 = BAJA mañana"

        return jsonify({
            "pa_ticker": pa_ticker,
            "ca_ticker": ca_ticker,
            "prediction": int(pred),
            "confidence": round(confidence, 3),
            "meaning": meaning,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "features_used": features
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)