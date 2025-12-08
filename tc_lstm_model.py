"""
Módulo LSTM para pronosticar el nivel del TC SUNAT (sin simulaciones).
Se usa como complemento del módulo de riesgo (GARCH/Monte Carlo), no lo reemplaza.
"""

from datetime import date
from typing import Tuple

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def set_global_seed(seed: int = 42) -> None:
    """
    Fija la semilla de las fuentes de aleatoriedad para que el
    entrenamiento sea (lo más) reproducible posible.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _construir_ventanas(serie: np.ndarray, ventana: int):
    """
    Crea ventanas (X, y) para entrenamiento LSTM.
    serie: array 1D ya escalado (0-1).
    ventana: número de observaciones pasadas que se usan para predecir la siguiente.
    """
    X, y = [], []
    for i in range(len(serie) - ventana):
        X.append(serie[i:i + ventana])
        y.append(serie[i + ventana])
    X = np.array(X)
    y = np.array(y)
    # reshape para LSTM: (n_muestras, ventana, n_features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def entrenar_lstm_tc(
    serie_tc: pd.Series,
    fecha_corte: date,
    ventana: int = 60,
    epochs: int = 25,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[Sequential, MinMaxScaler, int]:
    """
    Entrena un LSTM univariado sobre el TC SUNAT hasta 'fecha_corte' (inclusive).

    - serie_tc: pd.Series con índice datetime y valores de TC (por ejemplo df_sunat_habiles["tc_sunat"])
    - fecha_corte: sólo se usan datos <= fecha_corte para entrenar.
    - ventana: tamaño de la ventana de entrada (nº de días pasados).
    """
    set_global_seed(seed)

    # 1) Filtrar sólo datos <= fecha_corte
    serie_train = serie_tc.loc[serie_tc.index.date <= fecha_corte].dropna()
    if len(serie_train) <= ventana + 10:
        raise ValueError(
            f"No hay suficientes datos para entrenar LSTM. "
            f"Se requieren al menos ventana+10={ventana+10} observaciones."
        )

    # 2) Escalar a [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(serie_train.values.reshape(-1, 1)).flatten()

    # 3) Construir ventanas
    X, y = _construir_ventanas(data_scaled, ventana)

    # 4) Definir modelo LSTM sencillo
    model = Sequential()
    model.add(LSTM(32, input_shape=(ventana, 1)))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False,
    )

    return model, scaler, ventana


def pronosticar_lstm_tc(
    model: Sequential,
    scaler: MinMaxScaler,
    serie_tc: pd.Series,
    fecha_corte: date,
    fechas_future: pd.DatetimeIndex,
    ventana: int,
) -> pd.Series:
    """
    Genera un camino determinista del TC (sin simulaciones) usando LSTM.

    - model, scaler, ventana: objetos devueltos por entrenar_lstm_tc
    - serie_tc: misma serie original de TC SUNAT (df_sunat_habiles["tc_sunat"])
    - fecha_corte: se toman los últimos 'ventana' datos <= fecha_corte para arrancar el pronóstico
    - fechas_future: índice de fechas hábiles futuras (el mismo que ya usas en GARCH/Monte Carlo)

    Devuelve:
      pd.Series con índice = fechas_future y valores = TC pronosticado por LSTM.
    """
    serie_train = serie_tc.loc[serie_tc.index.date <= fecha_corte].dropna()
    values = serie_train.values

    if len(values) < ventana:
        raise ValueError(
            f"No hay suficientes datos para pronosticar con LSTM. "
            f"Se requieren al menos {ventana} observaciones."
        )

    # Tomamos los últimos 'ventana' valores y los escalamos
    ultimos = values[-ventana:].reshape(-1, 1)
    ultimos_scaled = scaler.transform(ultimos).flatten().tolist()

    preds_scaled = []

    # Forecast iterativo paso a paso
    for _ in range(len(fechas_future)):
        x_input = np.array(ultimos_scaled[-ventana:]).reshape((1, ventana, 1))
        y_hat = model.predict(x_input, verbose=0)[0, 0]  # escalar (0-1)
        preds_scaled.append(y_hat)
        ultimos_scaled.append(y_hat)

    # Volver a nivel de TC
    preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
    preds_tc = scaler.inverse_transform(preds_scaled_arr).flatten()

    return pd.Series(data=preds_tc, index=fechas_future, name="tc_lstm")
