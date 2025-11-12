FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# requirements.txt (alternativa)
flask
pandas
numpy>=2.0.0
scikit-learn>=1.7.0
joblib
yfinance

# Copiar modelo y features
COPY best_model.pkl .
COPY data/features.pkl data/features.pkl

EXPOSE 5000
CMD ["python", "app.py"]