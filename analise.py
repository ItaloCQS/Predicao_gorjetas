import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold

df = pd.read_csv('data\dados_gorjetas.csv')

print(df.head())
print()
print(df.info())
print()
print(df.isnull().sum())
df['Gorjeta'].describe()

colunas_categoricas = ['Genero', 'Fumante', 'Dia', 'Horario', 'Possui_Criancas']

# One-Hot Encoding
df_codificado = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True)

# Define X e y
X = df_codificado.drop('Gorjeta', axis=1)
y = df_codificado['Gorjeta']

numerical_cols = ['Conta_Total', 'Tamanho_Mesa', 'Avaliacao_Servico', 'Percentual_Esperado_Gorjeta']

def criar_modelo(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

X_array = X.copy()
y_array = y.copy()

for fold, (train_index, test_index) in enumerate(kf.split(X_array), start=1):

    print(f"\n🔹 Fold {fold}")

    # Divisão dos folds
    X_train_fold = X_array.iloc[train_index].copy()
    X_test_fold = X_array.iloc[test_index].copy()
    y_train_fold = y_array.iloc[train_index]
    y_test_fold = y_array.iloc[test_index]

    # Escalonamento apenas dentro do fold
    scaler = StandardScaler()
    X_train_fold[numerical_cols] = scaler.fit_transform(X_train_fold[numerical_cols])
    X_test_fold[numerical_cols] = scaler.transform(X_test_fold[numerical_cols])

    # Criar novo modelo para este fold
    model = criar_modelo(input_dim=X_train_fold.shape[1])

    # Treino
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32)

    # Previsões
    y_pred = model.predict(X_test_fold).flatten()

    # Métricas do fold
    mae = mean_absolute_error(y_test_fold, y_pred)
    mse = mean_squared_error(y_test_fold, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_fold, y_pred)

    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

print("MAE por fold:", mae_scores)
print("MSE por fold:", mse_scores)
print("RMSE por fold:", rmse_scores)
print("R² por fold:", r2_scores)

print(f"\nMAE - Erro Absoluto Médio: {np.mean(mae_scores):.4f}")
print(f"MSE - Erro Quadrático Médio: {np.mean(mse_scores):.4f}")
print(f"RMSE - Raiz do Erro Quadrático Médio: {np.mean(rmse_scores):.4f}")
print(f"R² - Coeficiente de Determinação: {np.mean(r2_scores):.4f}")

import matplotlib.pyplot as plt

folds = range(1, kf.n_splits + 1)

plt.figure(figsize=(12,6))
width = 0.2

plt.bar([f - width for f in folds], mae_scores, width=0.2, label="MAE")
plt.bar(folds, mse_scores, width=0.2, label="MSE")
plt.bar([f + width for f in folds], rmse_scores, width=0.2, label="RMSE")

plt.title("Comparação das Métricas por Fold")
plt.xlabel("Fold")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.bar(folds, r2_scores, color='purple')
plt.title("R² por Fold")
plt.xlabel("Fold")
plt.ylabel("R²")
plt.axhline(0, color='red', linestyle='--')
plt.grid(True)
plt.show()

