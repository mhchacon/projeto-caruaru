"""
==============================================================================
MODELAGEM PREDITIVA E PROJEÇÃO DE DEMANDA DE ÁGUA URBANA DE RECIFE (2030 / 2050)
==============================================================================
Autora     : Mariana (IC / UFRPE)
Orientação : Especialista em Ciência de Dados / Recursos Hídricos
Descrição  : Script unificado de Machine Learning, PCA e Projeção Estratégica.
             Incorpora recomendações metodológicas do script de Caruaru
             (PCA, PCR, Ridge, Baseline Linear, IC 95% e projeção 2-etapas).
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Configurar saída UTF-8 no terminal Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS
# ==============================================================================
print("=" * 70)
print(" 1. CARREGANDO E PREPARANDO DATASET ")
print("=" * 70)

caminho_dataset = 'Dataset_Final_IA.csv'
if not os.path.exists(caminho_dataset):
    caminho_dataset = os.path.join('database', 'Dataset_Final_IA.csv')

df = pd.read_csv(caminho_dataset)
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values('Data').reset_index(drop=True)

print(f"Dataset carregado: {df.shape[0]} observações mensais ({df['Ano'].min()} - {df['Ano'].max()}).")

# Engenharia de Feature: Lag de 1 mês do Volume
df['Volume_Lag_1'] = df['Vol_produzido'].shift(1)
df = df.dropna().reset_index(drop=True)

target = 'Vol_produzido'
colunas_remover = ['Data', 'Nome_Mes', 'Vol_produzido', 'Vol_produzido', 
                   'Volume de água macromedido', 'Volume de água micromedido']
features = [c for c in df.columns if c not in colunas_remover]

print(f"Número de variáveis preditoras (features): {len(features)}")
print("Features utilizadas:", features)

# ==============================================================================
# 2. DIAGNÓSTICO DE MULTICOLINEARIDADE (PCA) — INSPIRAÇÃO SCRIPT CARUARU
# ==============================================================================
print("\n" + "=" * 70)
print(" 2. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) ")
print("=" * 70)

# Selecionar variáveis numéricas contínuas para o PCA
feats_pca = [f for f in features if f not in ['Ano', 'Mes', 'Volume_Lag_1']]

scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(df[feats_pca])

pca = PCA()
pca_transformed = pca.fit_transform(X_pca_scaled)

var_explicada = pca.explained_variance_ratio_
var_acumulada = np.cumsum(var_explicada)

print("Variância Explicada por Componente:")
for i, (ve, va) in enumerate(zip(var_explicada, var_acumulada), 1):
    print(f"  PC{i}: {ve*100:6.2f}% (Acumulada: {va*100:6.2f}%)")

# Gráfico Scree Plot e PCA Loadings
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree Plot
axes[0].bar(range(1, len(var_explicada) + 1), var_explicada * 100, color='skyblue', edgecolor='navy', alpha=0.8)
axes[0].plot(range(1, len(var_explicada) + 1), var_acumulada * 100, color='red', marker='o', linewidth=2)
axes[0].axhline(y=90, color='gray', linestyle='--', label='90% Variância Explicada')
axes[0].set_title('Scree Plot — Variância Explicada (PCA)')
axes[0].set_xlabel('Componente Principal')
axes[0].set_ylabel('Variância Explicada (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loadings do PC1 (Variável de maior peso)
loadings_pc1 = pd.Series(pca.components_[0], index=feats_pca).sort_values()
loadings_pc1.plot(kind='barh', ax=axes[1], color='teal')
axes[1].set_title('Cargas das Variáveis no Componente Principal 1 (PC1)')
axes[1].set_xlabel('Peso (Loading)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PCA_Multicollinearidade.png', dpi=300)
plt.close()
print("Gráfico 'PCA_Multicollinearidade.png' gerado com sucesso.")

# ==============================================================================
# 3. DIVISÃO TREINO E TESTE (CORTE TEMPORAL HISTÓRICO)
# ==============================================================================
print("\n" + "=" * 70)
print(" 3. CORTE TEMPORAL PARA AVALIAÇÃO DE MODELOS ")
print("=" * 70)

# Treino: <= 2021 | Teste: >= 2022
train_mask = df['Ano'] <= 2021
test_mask = df['Ano'] >= 2022

X_train_raw = df.loc[train_mask, features]
y_train = df.loc[train_mask, target]

X_test_raw = df.loc[test_mask, features]
y_test = df.loc[test_mask, target]

# Normalização Min-Max
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print(f"Conjunto de Treino : {X_train.shape[0]} meses ({df.loc[train_mask, 'Ano'].min()} - 2021)")
print(f"Conjunto de Teste  : {X_test.shape[0]} meses (2022 - {df.loc[test_mask, 'Ano'].max()})")

# ==============================================================================
# 4. BENCHMARK DE MODELOS PREDITIVOS
# ==============================================================================
print("\n" + "=" * 70)
print(" 4. TREINAMENTO E BENCHMARK DOS MODELOS ")
print("=" * 70)

resultados = {}

# Modelo 1: Linear Regression (Linear-MV)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_scaled)
lr_preds_scaled = lr_model.predict(X_test)
lr_preds = scaler_y.inverse_transform(lr_preds_scaled.reshape(-1, 1)).flatten()
resultados['Linear-MV'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, lr_preds)),
    'r2': r2_score(y_test, lr_preds),
    'preds': lr_preds
}

# Modelo 2: Ridge Regression (L2)
ridge_model = Ridge(alpha=10.0, random_state=42)
ridge_model.fit(X_train, y_train_scaled)
ridge_preds_scaled = ridge_model.predict(X_test)
ridge_preds = scaler_y.inverse_transform(ridge_preds_scaled.reshape(-1, 1)).flatten()
resultados['Ridge (L2)'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, ridge_preds)),
    'r2': r2_score(y_test, ridge_preds),
    'preds': ridge_preds
}

# Modelo 3: XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train_scaled)
xgb_preds_scaled = xgb_model.predict(X_test)
xgb_preds = scaler_y.inverse_transform(xgb_preds_scaled.reshape(-1, 1)).flatten()
resultados['XGBoost'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, xgb_preds)),
    'r2': r2_score(y_test, xgb_preds),
    'preds': xgb_preds
}

# Modelo 4: LSTM (Deep Learning)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_scaled, epochs=50, batch_size=16, verbose=0, shuffle=False)

    lstm_preds_scaled = lstm_model.predict(X_test_lstm)
    lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled).flatten()
    resultados['LSTM'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, lstm_preds)),
        'r2': r2_score(y_test, lstm_preds),
        'preds': lstm_preds
    }
except Exception as e:
    print(f"Aviso: LSTM não foi executado ({e})")

# Exibir Tabela Resumo do Benchmark
df_res = pd.DataFrame([
    {'Modelo': nome, 'RMSE (m³)': m['rmse'], 'R²': m['r2']}
    for nome, m in resultados.items()
]).sort_values('R²', ascending=False)

print("\nRESULTADOS DO BENCHMARK NO CONJUNTO DE TESTE:")
print(df_res.to_string(index=False))

# Gráfico Comparativo do Benchmark
plt.figure(figsize=(12, 6))
plt.plot(df.loc[test_mask, 'Data'], y_test, label='Consumo Real Observado', color='black', linewidth=2.5)
plt.plot(df.loc[test_mask, 'Data'], resultados['XGBoost']['preds'], label=f"XGBoost (R²={resultados['XGBoost']['r2']:.2f})", color='blue', linestyle='--')
plt.plot(df.loc[test_mask, 'Data'], resultados['Linear-MV']['preds'], label=f"Linear-MV (R²={resultados['Linear-MV']['r2']:.2f})", color='green', linestyle=':')
if 'LSTM' in resultados:
    plt.plot(df.loc[test_mask, 'Data'], resultados['LSTM']['preds'], label=f"LSTM (R²={resultados['LSTM']['r2']:.2f})", color='red', linestyle='-.')

plt.title('Comparação do Benchmark de Modelos Preditivos no Teste (2022-2026)')
plt.ylabel('Volume Mensal de Água (m³)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Comparacao_Modelos_Teste.png', dpi=300)
plt.close()

# Feature Importance XGBoost
importance = xgb_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(12), palette='mako')
plt.title('Importância das Variáveis no Modelo XGBoost')
plt.tight_layout()
plt.savefig('XGBoost_Feature_Importance.png', dpi=300)
plt.close()

# ==============================================================================
# 5. PROJEÇÃO EM DUAS ETAPAS DAS VARIÁVEIS EXÓGENAS (2027 A 2050)
# ==============================================================================
print("\n" + "=" * 70)
print(" 5. PROJEÇÃO DAS VARIÁVEIS EXÓGENAS ATÉ 2050 (ETAPA 1) ")
print("=" * 70)

anos_futuros = np.arange(2027, 2051)
meses_futuros = []

for ano in anos_futuros:
    for mes in range(1, 13):
        meses_futuros.append({'Ano': ano, 'Mes': mes})

df_futuro = pd.DataFrame(meses_futuros)
df_futuro['Data'] = pd.to_datetime(df_futuro['Ano'].astype(str) + '-' + df_futuro['Mes'].astype(str) + '-01')

# Projeções auxiliares por regressão linear/log temporal
ano_hist = df['Ano'].values.reshape(-1, 1)

for col in features:
    if col in ['Ano', 'Mes', 'Volume_Lag_1']:
        continue
    reg = LinearRegression()
    reg.fit(ano_hist, df[col])
    val_proj = reg.predict(df_futuro['Ano'].values.reshape(-1, 1))
    df_futuro[col] = val_proj

# Definir Volume_Lag_1 inicial
ultimo_vol = df['Vol_produzido'].iloc[-1]
df_futuro['Volume_Lag_1'] = ultimo_vol

# Prepare X_futuro normalizado
X_futuro_raw = df_futuro[features]
X_futuro_scaled = scaler_X.transform(X_futuro_raw)

# ==============================================================================
# 6. PROJEÇÃO FINAL DA DEMANDA DE ÁGUA (2030 E 2050) COM IC 95%
# ==============================================================================
print("\n" + "=" * 70)
print(" 6. GERANDO PROJEÇÕES DE DEMANDA HIDRICA COM IC 95% ")
print("=" * 70)

# Modelo Linear Re-treinado no Dataset Completo
lr_full = LinearRegression()
lr_full.fit(scaler_X.transform(df[features]), scaler_y.transform(df[target].values.reshape(-1, 1)).flatten())

y_pred_hist_scaled = lr_full.predict(scaler_X.transform(df[features]))
y_pred_hist = scaler_y.inverse_transform(y_pred_hist_scaled.reshape(-1, 1)).flatten()

# RMSE In-Sample para o Intervalo de Confiança 95%
rmse_in_sample = np.sqrt(mean_squared_error(df[target], y_pred_hist))

# Previsão Futura Linear
pred_futura_scaled = lr_full.predict(X_futuro_scaled)
df_futuro['Previsao_Linear_m3'] = scaler_y.inverse_transform(pred_futura_scaled.reshape(-1, 1)).flatten()

# Previsão Futura XGBoost
xgb_full = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_full.fit(scaler_X.transform(df[features]), scaler_y.transform(df[target].values.reshape(-1, 1)).flatten())
pred_xgb_scaled = xgb_full.predict(X_futuro_scaled)
df_futuro['Previsao_XGBoost_m3'] = scaler_y.inverse_transform(pred_xgb_scaled.reshape(-1, 1)).flatten()

# Intervalos de Confiança (IC 95%)
df_futuro['IC95_Inf_m3'] = df_futuro['Previsao_Linear_m3'] - 1.96 * rmse_in_sample
df_futuro['IC95_Sup_m3'] = df_futuro['Previsao_Linear_m3'] + 1.96 * rmse_in_sample

# Agregação Anual
df_proj_anual = df_futuro.groupby('Ano').agg({
    'Previsao_Linear_m3': 'sum',
    'Previsao_XGBoost_m3': 'sum',
    'IC95_Inf_m3': 'sum',
    'IC95_Sup_m3': 'sum'
}).reset_index()

print("\nPROJEÇÃO DA DEMANDA ANUAL (2030 E 2050):")
print("-" * 65)
p2030 = df_proj_anual[df_proj_anual['Ano'] == 2030].iloc[0]
p2050 = df_proj_anual[df_proj_anual['Ano'] == 2050].iloc[0]

print(f"HORIZONTE 2030:")
print(f"  Modelo Linear (Recomendado): {p2030['Previsao_Linear_m3']/1e6:.2f} milhões m³ [IC95%: {p2030['IC95_Inf_m3']/1e6:.2f} - {p2030['IC95_Sup_m3']/1e6:.2f}]")
print(f"  Modelo XGBoost             : {p2030['Previsao_XGBoost_m3']/1e6:.2f} milhões m³")

print(f"\nHORIZONTE 2050:")
print(f"  Modelo Linear (Recomendado): {p2050['Previsao_Linear_m3']/1e6:.2f} milhões m³ [IC95%: {p2050['IC95_Inf_m3']/1e6:.2f} - {p2050['IC95_Sup_m3']/1e6:.2f}]")
print(f"  Modelo XGBoost (Saturado)  : {p2050['Previsao_XGBoost_m3']/1e6:.2f} milhões m³")

# Gráfico Final de Projeção Estratégica (2020-2050)
df_hist_anual = df.groupby('Ano')['Vol_produzido'].sum().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(df_hist_anual['Ano'], df_hist_anual['Vol_produzido'] / 1e6, label='Histórico Observado', color='black', linewidth=2.5)

plt.plot(df_proj_anual['Ano'], df_proj_anual['Previsao_Linear_m3'] / 1e6, label='Projeção Linear (Tendência)', color='darkblue', linewidth=2, linestyle='--')
plt.fill_between(df_proj_anual['Ano'], df_proj_anual['IC95_Inf_m3'] / 1e6, df_proj_anual['IC95_Sup_m3'] / 1e6, color='blue', alpha=0.15, label='Intervalo de Confiança 95%')

plt.plot(df_proj_anual['Ano'], df_proj_anual['Previsao_XGBoost_m3'] / 1e6, label='Projeção XGBoost (Saturação)', color='darkorange', linewidth=2, linestyle=':')

# Destaques para 2030 e 2050
plt.axvline(x=2030, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2050, color='gray', linestyle='--', alpha=0.7)

plt.scatter([2030, 2050], [p2030['Previsao_Linear_m3']/1e6, p2050['Previsao_Linear_m3']/1e6], color='red', zorder=5, s=60)
plt.text(2030, p2030['Previsao_Linear_m3']/1e6 + 0.5, f"2030: {p2030['Previsao_Linear_m3']/1e6:.1f}M m³", fontweight='bold')
plt.text(2050, p2050['Previsao_Linear_m3']/1e6 + 0.5, f"2050: {p2050['Previsao_Linear_m3']/1e6:.1f}M m³", fontweight='bold')

plt.title('Projeção da Demanda Hídrica Urbana de Recife (2027-2050) com IC 95%')
plt.xlabel('Ano')
plt.ylabel('Volume Anual (Milhões de m³)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Projecao_Demanda_2030_2050.png', dpi=300)
plt.close()

# Exportar Resultados para Arquivo de Texto
with open('Resultados_Avaliacao.txt', 'w', encoding='utf-8') as f:
    f.write("=== RESULTADOS DA AVALIAÇÃO DE MODELOS E PROJEÇÕES ===\n\n")
    f.write(df_res.to_string(index=False))
    f.write("\n\n=== PROJEÇÕES DE DEMANDA PARA RECIFE ===\n")
    f.write(f"Ano 2030 (Linear): {p2030['Previsao_Linear_m3']:.2f} m³ [IC95%: {p2030['IC95_Inf_m3']:.2f} - {p2030['IC95_Sup_m3']:.2f}]\n")
    f.write(f"Ano 2050 (Linear): {p2050['Previsao_Linear_m3']:.2f} m³ [IC95%: {p2050['IC95_Inf_m3']:.2f} - {p2050['IC95_Sup_m3']:.2f}]\n")

print("\nPipeline executado com sucesso!")
print("Arquivos gerados:")
print("  - PCA_Multicollinearidade.png")
print("  - Comparacao_Modelos_Teste.png")
print("  - XGBoost_Feature_Importance.png")
print("  - Projecao_Demanda_2030_2050.png")
print("  - Resultados_Avaliacao.txt")
