# test_api.py
import requests
import sys

# Uso: python test_api.py [PA_TICKER] [CA_TICKER]
# Ejemplos:
#   python test_api.py NVDA ^GSPC
#   python test_api.py QLYS ^RUT
#   python test_api.py EC CL=F
#   python test_api.py        → usa EC + CL=F por defecto

pa_ticker = sys.argv[1] if len(sys.argv) > 1 else "EC"
ca_ticker = sys.argv[2] if len(sys.argv) > 2 else "CL=F"

url = f"http://localhost:5001/predict/{pa_ticker}/{ca_ticker}"

try:
    response = requests.get(url, timeout=15)
    result = response.json()
    print("\n" + "="*60)
    print("PREDICCIÓN UNIVERSAL")
    print("="*60)
    print(f"Activo Principal (PA): {result['pa_ticker']}")
    print(f"Activo Correlacionado (CA): {result['ca_ticker']}")
    print(f"Predicción: {result['prediction']} → {result['meaning']}")
    print(f"Confianza: {result['confidence']:.1%}")
    print(f"Fecha: {result['date']}")
    print("="*60)
except Exception as e:
    print(f"Error: {e}")
    print("Asegúrate de que el contenedor esté corriendo:")
    print("docker run -p 5001:5000 stock-predictor")