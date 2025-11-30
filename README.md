# Tip Prediction Model – Predição de Gorjetas com Rede Neural

Projeto de machine learning focado em prever valores de gorjetas com base em variáveis numéricas e categóricas. 
O modelo utiliza uma rede neural construída com TensorFlow/Keras e um pipeline completo de pré-processamento, validação cruzada e análise de métricas. 
Este projeto demonstra habilidades essenciais em análise de dados, engenharia de atributos e modelagem preditiva.

## 📊 Objetivo do Projeto
Desenvolver um modelo de regressão capaz de prever o valor da gorjeta com base em características como:

- Valor total da conta  
- Tamanho da mesa  
- Avaliação do serviço  
- Gênero  
- Dia da semana  
- Horário  
- Entre outras variáveis categóricas

O objetivo é construir um pipeline de machine learning completo e avaliar a performance do modelo usando métricas de erro e validação cruzada.

## 🛠 Tecnologias Utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **TensorFlow / Keras**
- **Scikit-Learn**
- **Matplotlib**
- **Seaborn**

## 🔄 Pipeline do Projeto

### 1. **Carregamento e inspeção dos dados**
- Leitura do dataset  
- Verificação de tipos e valores ausentes  
- Estatísticas descritivas  

### 2. **Pré-processamento**
- One-Hot Encoding para variáveis categóricas  
- Padronização de variáveis numéricas com **StandardScaler**  
- Divisão dos dados por meio do **K-Fold Cross Validation**

### 3. **Construção do Modelo**
- Rede neural com:
  - Camada densa (64 neurônios, ReLU)
  - Camada densa (32 neurônios, ReLU)
  - Camada densa (16 neurônios, ReLU)
  - Saída linear (regressão)
- Otimizador **Adam**
- Função de perda **MSE**

### 4. **Validação Cruzada (K-Fold)**
- Avaliação robusta do modelo
- Resultados por fold:
  - MAE
  - MSE
  - RMSE
  - R²

### 5. **Visualização dos Resultados**
- Gráficos das métricas por fold 
- Gráfico de R² 

Os resultados variam de acordo com os folds, mas demonstram que o modelo possui boa capacidade preditiva para um problema de regressão multivariada.
