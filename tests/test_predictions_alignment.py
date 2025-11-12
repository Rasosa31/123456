import pandas as pd


def test_predictions_align_with_df_ml():
    """Verifica que data/predictions_df_ml.csv tenga el mismo número de filas que data/df_ml.csv
    y contenga las columnas mínimas esperadas.
    """
    df = pd.read_csv('data/df_ml.csv')
    preds = pd.read_csv('data/predictions_df_ml.csv')

    # Igual número de filas (sin contar header)
    assert len(df) == len(preds), f"Filas df_ml={len(df)} != preds={len(preds)}"

    # Columnas esperadas en el CSV de predicciones
    expected_cols = {'prediction', 'prediction_label'}
    missing = expected_cols - set(preds.columns)
    assert not missing, f"Faltan columnas en predictions_df_ml.csv: {missing}"

    # Si existen probabilidades, deben sumar aproximadamente 1 por fila
    if {'prob_Sube', 'prob_Baja'}.issubset(preds.columns):
        sums = preds['prob_Sube'].fillna(0) + preds['prob_Baja'].fillna(0)
        # permitir pequeña tolerancia
        assert (sums - 1.0).abs().max() < 1e-6, "Las probabilidades no suman 1 en al menos una fila"
