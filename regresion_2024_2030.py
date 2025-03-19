import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow import random

# Configuración inicial
backend.clear_session()
random.set_seed(42)

# Carga del archivo CSV
# Ajustamos la lectura para manejar problemas de codificación
with open('años_demandatotal.csv', 'r', encoding='latin1') as file:
    data = pd.read_csv(file)
data.columns = ['año', 'demanda total']  # Ajusta los nombres según el archivo
data = data.set_index('año')

# Escalamiento de los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data['demanda total'] = scaler.fit_transform(data[['demanda total']])

# División de datos en entrenamiento y prueba
train_data = data.loc[1992:2018]
test_data = data.loc[2019:2023]

# Creación de secuencias para la red neuronal
def create_sequences(data, input_length):
    X, y = [], []
    for i in range(len(data) - input_length):
        X.append(data[i:i + input_length])
        y.append(data[i + input_length])
    return np.array(X), np.array(y)

input_length = 5  # Usamos 5 años como ventana para capturar tendencias
X_train, y_train = create_sequences(train_data['demanda total'].values, input_length)
X_test, y_test = create_sequences(test_data['demanda total'].values, input_length)

# Si no hay suficientes datos de prueba, usa los últimos datos de entrenamiento para predicción
if X_test.size == 0:
    print("Advertencia: Insuficientes datos de prueba. Usando datos de entrenamiento para predicción.")
    X_test = X_train[-1:].copy()
    y_test = y_train[-1:].copy()

# Ajustar las dimensiones para LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Creación del modelo LSTM
model = Sequential([
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, activation='relu'),
    Dropout(0.2),
    Dense(units=1)  # Salida lineal para regresión
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Evaluación del modelo
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba: {loss}")

# Predicción de datos futuros
future_years = np.arange(2024, 2031)
predictions = []
current_input = X_test[-1]  # Usa la última secuencia disponible como entrada inicial

for _ in future_years:
    next_pred = model.predict(current_input.reshape((1, input_length, 1)))[0, 0]
    predictions.append(next_pred)
    next_pred_reshaped = np.array([[next_pred]])
    current_input = np.append(current_input[1:], next_pred_reshaped, axis=0)  

# Desescalado de predicciones
predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actual_values = scaler.inverse_transform(data['demanda total'].values.reshape(-1, 1))

# Gráfica de resultados
plt.figure(figsize=(12, 6))
plt.plot(data.index, actual_values, label='Datos Reales (1992-2023)', color='blue')
plt.plot(future_years, predicted_values, label='Proyección (2024-2030)', color='green', linestyle='--')
plt.scatter(future_years[-1], predicted_values[-1][0], color='black', label=f'Año: {future_years[-1]} - Demanda total: {predicted_values[-1][0]:.2f} GWh')
plt.xlabel('Años')
plt.ylabel('Demanda Total (GWh)')
plt.title('Proyección de Demanda Total de Energía Eléctrica en Argentina')
plt.legend()
plt.grid()
plt.show()

# 10. Análisis del rendimiento
#plt.figure(figsize=(12, 6))
#plt.plot(history.history['loss'], label='Pérdida en Entrenamiento', color='blue')
#plt.plot(history.history['val_loss'], label='Pérdida en Validación', color='orange')
#plt.xlabel('Épocas')
#plt.ylabel('Pérdida')
#plt.title('Curvas de Entrenamiento y Validación')
#plt.legend()
#plt.grid()
#plt.show()