"""
==============================================================================
MODELAGEM PREDITIVA ANUAL E PROJEÇÃO DE DEMANDA DE ÁGUA DE RECIFE (2030 / 2050)
==============================================================================
Autora     : Mariana (IC / UFRPE)
Orientação : Especialista em Ciência de Dados / Recursos Hídricos
Descrição  : Script de modelagem em escala ANUAL (32 observações, 1995-2026),
             idêntico à estrutura do script de Caruaru (projecao_consumo_agua_v2.py).
             Permite a comparação direta entre a modelagem ANUAL vs MENSAL.
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Configurar saída UTF-8 no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==============================================================================
# 1. CARREGAMENTO DO DATASET ANUAL
# ==============================================================================
print("=" * 70)
print(" 1. CARREGANDO DATASET ANUAL DE RECIFE (1995 - 2026) ")
print("=" * 70)

caminho_anual = 'Caruaru_Dataset_Atualizado.csv'
if not os.path.exists(caminho_anual):
    caminho_anual = os.path.join('database', 'Caruaru_Dataset_Atualizado.csv')

df_anual = pd.read_csv(caminho_anual)
df_anual = df_anual.sort_values('Ano').reset_index(drop=True)

# Tratar valores ausentes (NaNs em anos mais antigos)
df_anual = df_anual.bfill().ffill()

print(f"Dataset Anual Carregado: {df_anual.shape[0]} anos de dados ({df_anual['Ano'].min()} - {df_anual['Ano'].max()}).")

# Definir target e features
target = 'Vol_produzido'
colunas_remover = ['Ano', target, 'Volume de água micromedido', 'Volume de água macromedido']
features = [c for c in df_anual.columns if c not in colunas_remover]

print(f"Número de variáveis preditoras (features): {len(features)}")

# ==============================================================================
# 2. PCA NO DATASET ANUAL — MESMA METODOLOGIA CARUARU
# ==============================================================================
print("\n" + "=" * 70)
print(" 2. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA ANUAL) ")
print("=" * 70)

scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(df_anual[features])

pca = PCA()
pca_transformed = pca.fit_transform(X_pca_scaled)

var_explicada = pca.explained_variance_ratio_
var_acumulada = np.cumsum(var_explicada)

print("Variância Explicada por Componente (Anual):")
for i, (ve, va) in enumerate(zip(var_explicada, var_acumulada), 1):
    print(f"  PC{i}: {ve*100:6.2f}% (Acumulada: {va*100:6.2f}%)")

# Scree plot anual
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(var_explicada) + 1), var_explicada * 100, color='mediumseagreen', edgecolor='darkgreen', alpha=0.8)
plt.plot(range(1, len(var_explicada) + 1), var_acumulada * 100, color='crimson', marker='o', linewidth=2)
plt.axhline(y=90, color='gray', linestyle='--', label='90% Variância Explicada')
plt.title('Scree Plot — Modelo Anual Recife (PCA)')
plt.xlabel('Componente Principal')
plt.ylabel('Variância Explicada (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('PCA_Anual_ScreePlot.png', dpi=300)
plt.close()

# ==============================================================================
# 3. DIVISÃO TREINO E TESTE TEMPORAL (ANUAL)
# ==============================================================================
print("\n" + "=" * 70)
print(" 3. CORTE TEMPORAL TREINO E TESTE (ANUAL) ")
print("=" * 70)

# Treino: 1995-2021 (27 anos) | Teste: 2022-2026 (5 anos)
train_mask = df_anual['Ano'] <= 2021
test_mask = df_anual['Ano'] >= 2022

X_train_raw = df_anual.loc[train_mask, features]
y_train = df_anual.loc[train_mask, target]

X_test_raw = df_anual.loc[test_mask, features]
y_test = df_anual.loc[test_mask, target]

scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print(f"Treino Anual: {X_train.shape[0]} anos | Teste Anual: {X_test.shape[0]} anos")

# ==============================================================================
# 4. BENCHMARK DE MODELOS ANUAIS (MESMA SUÍTE DE CARUARU)
# ==============================================================================
print("\n" + "=" * 70)
print(" 4. TREINAMENTO E BENCHMARK DE MODELOS ANUAIS ")
print("=" * 70)

modelos = {
    'Linear-MV': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=10.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

resultados_anual = {}

for nome, mod in modelos.items():
    mod.fit(X_train, y_train_scaled)
    preds_scaled = mod.predict(X_test)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    
    resultados_anual[nome] = {
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'preds': preds
    }

df_res_anual = pd.DataFrame([
    {'Modelo': k, 'RMSE (m³/ano)': v['rmse'], 'MAPE (%)': v['mape'], 'R²': v['r2']}
    for k, v in resultados_anual.items()
]).sort_values('R²', ascending=False)

print("\nRESULTADOS DO BENCHMARK ANUAL (2022-2026):")
print(df_res_anual.to_string(index=False))

# ==============================================================================
# 5. PROJEÇÃO ANUAL PARA 2030 E 2050 COM IC 95%
# ==============================================================================
print("\n" + "=" * 70)
print(" 5. PROJEÇÃO ANUAL DE DEMANDA (2027 A 2050) ")
print("=" * 70)

anos_futuros = np.arange(2027, 2051)
df_futuro_anual = pd.DataFrame({'Ano': anos_futuros})

# Projeção temporal das exógenas
ano_hist = df_anual['Ano'].values.reshape(-1, 1)

for col in features:
    reg = LinearRegression()
    reg.fit(ano_hist, df_anual[col])
    df_futuro_anual[col] = reg.predict(df_futuro_anual['Ano'].values.reshape(-1, 1))

X_futuro_anual_scaled = scaler_X.transform(df_futuro_anual[features])

# Modelo Linear Re-treinado com todos os dados
lr_full_anual = LinearRegression()
lr_full_anual.fit(scaler_X.transform(df_anual[features]), scaler_y.transform(df_anual[target].values.reshape(-1, 1)).flatten())

y_pred_hist_scaled = lr_full_anual.predict(scaler_X.transform(df_anual[features]))
y_pred_hist = scaler_y.inverse_transform(y_pred_hist_scaled.reshape(-1, 1)).flatten()

rmse_in_sample = np.sqrt(mean_squared_error(df_anual[target], y_pred_hist))

pred_linear_scaled = lr_full_anual.predict(X_futuro_anual_scaled)
df_futuro_anual['Previsao_Linear_Anual'] = scaler_y.inverse_transform(pred_linear_scaled.reshape(-1, 1)).flatten()

# XGBoost Anual Full
xgb_full_anual = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_full_anual.fit(scaler_X.transform(df_anual[features]), scaler_y.transform(df_anual[target].values.reshape(-1, 1)).flatten())
pred_xgb_scaled = xgb_full_anual.predict(X_futuro_anual_scaled)
df_futuro_anual['Previsao_XGBoost_Anual'] = scaler_y.inverse_transform(pred_xgb_scaled.reshape(-1, 1)).flatten()

# IC 95%
df_futuro_anual['IC95_Inf_Anual'] = df_futuro_anual['Previsao_Linear_Anual'] - 1.96 * rmse_in_sample
df_futuro_anual['IC95_Sup_Anual'] = df_futuro_anual['Previsao_Linear_Anual'] + 1.96 * rmse_in_sample

p2030_anual = df_futuro_anual[df_futuro_anual['Ano'] == 2030].iloc[0]
p2050_anual = df_futuro_anual[df_futuro_anual['Ano'] == 2050].iloc[0]

print("\nPROJEÇÕES DO MODELO ANUAL:")
print(f"2030 (Linear Anual) : {p2030_anual['Previsao_Linear_Anual']:,.2f} m³/ano [IC95%: {p2030_anual['IC95_Inf_Anual']:,.2f} - {p2030_anual['IC95_Sup_Anual']:,.2f}]")
print(f"2030 (XGBoost Anual): {p2030_anual['Previsao_XGBoost_Anual']:,.2f} m³/ano")
print(f"2050 (Linear Anual) : {p2050_anual['Previsao_Linear_Anual']:,.2f} m³/ano [IC95%: {p2050_anual['IC95_Inf_Anual']:,.2f} - {p2050_anual['IC95_Sup_Anual']:,.2f}]")
print(f"2050 (XGBoost Anual): {p2050_anual['Previsao_XGBoost_Anual']:,.2f} m³/ano")

# Exportar resultados anuais
df_res_anual.to_csv('Resultados_Modelo_Anual.csv', index=False)
df_futuro_anual.to_csv('Projeções_Demanda_Anual.csv', index=False)

print("\nModelo Anual executado com sucesso.")
