"""
==============================================================================
MODELAGEM PREDITIVA ANUAL E PROJEÇÃO DE DEMANDA DE ÁGUA DE CARUARU (2030/2040/2050)
==============================================================================
Autora     : Mariana (IC / UFRPE)
Orientação : Especialista em Ciência de Dados / Recursos Hídricos
Descrição  : Script de modelagem em escala ANUAL para Caruaru - PE.
             Gera projeções de volume de água consumido e variáveis
             demográficas (pop. urbana, pop. rural, área urbanizada)
             para os horizontes de 2030, 2040 e 2050.

NOTA DE UNIDADE: O campo 'Vol_produzido' no dataset SNIS está em
             1.000 m³/ano (mil metros cúbicos por ano). Para converter
             para m³ absolutos, multiplique por 1.000.
             Referência real 2022: ~29.720 × 1.000 = 29.720.750 m³/ano
             (~29,7 milhões de m³/ano, compatível com os ~19 milhões de
             m³/ano reportados pelo aguaesaneamento.org.br para anos
             anteriores ao pico de 2022).
=============================================================================="""

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

# Diretório base do script (disponível globalmente)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
projeto_dir = os.path.dirname(script_dir)  # pasta pai = projeto caruaru

# ==============================================================================
# 1. CARREGAMENTO DO DATASET ANUAL
# ==============================================================================
print("=" * 70)
print(" 1. CARREGANDO DATASET ANUAL DE CARUARU (1995 - 2026) ")
print("=" * 70)

candidatos_dataset = [
    os.path.join(script_dir, 'Caruaru_Dataset_Atualizado.csv'),
    os.path.join(projeto_dir, 'database', 'Caruaru_Dataset_Atualizado.csv'),
    os.path.join(projeto_dir, 'Caruaru_Dataset_Atualizado.csv'),
    'Caruaru_Dataset_Atualizado.csv'
]
caminho_anual = next((p for p in candidatos_dataset if os.path.exists(p)), 'Caruaru_Dataset_Atualizado.csv')

df_anual = pd.read_csv(caminho_anual)
df_anual = df_anual.sort_values('Ano').reset_index(drop=True)

# Separar apenas os anos com dados REAIS de Vol_produzido
df_historico = df_anual.dropna(subset=['Vol_produzido']).copy()
df_historico = df_historico.bfill().ffill().reset_index(drop=True)

print(f"Dataset total: {df_anual.shape[0]} linhas ({df_anual['Ano'].min()} – {df_anual['Ano'].max()})")
print(f"Anos com Vol_produzido real: {len(df_historico)} ({df_historico['Ano'].min()} – {df_historico['Ano'].max()})")
print("\nResumo dos dados disponíveis:")
print(f"  {'Ano':<6} {'Vol_produzido':>16} {'Pop_Total':>12} {'Pop_Urbana':>12} {'Pop_Rural':>11}")
print("  " + "-" * 60)
for _, r in df_historico.tail(10).iterrows():
    pop_u = f"{r['Pop_Urbana']:,.0f}" if pd.notna(r.get('Pop_Urbana')) else 'N/D'
    pop_r = f"{r['Pop_Rural']:,.0f}" if pd.notna(r.get('Pop_Rural')) else 'N/D'
    print(f"  {int(r['Ano']):<6} {r['Vol_produzido']:>16,.2f} {r['Pop_Total']:>12,.0f} {pop_u:>12} {pop_r:>11}")

# Usar df_historico como base de modelagem
df_anual = df_historico

# Definir target e features (aqui, após filtrar apenas dados reais)
target = 'Vol_produzido'
colunas_remover = ['Ano', target, 'Vol_micromedido']
features = [c for c in df_anual.columns if c not in colunas_remover]

print(f"\nNúmero de variáveis preditoras (features): {len(features)}")
print(f"Features: {features}")

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
plt.title('Scree Plot — Modelo Anual Caruaru (PCA)')
plt.xlabel('Componente Principal')
plt.ylabel('Variância Explicada (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
caminho_scree = os.path.join(projeto_dir, 'PCA_Anual_ScreePlot.png')
plt.savefig(caminho_scree, dpi=300)
plt.close()

# ==============================================================================
# 3. DIVISÃO TREINO E TESTE TEMPORAL (ANUAL)
# ==============================================================================
print("\n" + "=" * 70)
print(" 3. CORTE TEMPORAL TREINO E TESTE (ANUAL) ")
print("=" * 70)

# Usa apenas anos com Vol_produzido real para treino e teste
# Treino: todos até 2018 | Teste: 2019-2022 (últimos anos com dados reais)
anos_reais = sorted(df_anual['Ano'].unique())
ano_corte  = anos_reais[int(len(anos_reais) * 0.75)]  # ~75% treino

print(f"\nAnos com dados reais: {anos_reais}")
print(f"Corte treino/teste  : até {ano_corte} (treino) | > {ano_corte} (teste)")

train_mask = df_anual['Ano'] <= ano_corte
test_mask  = df_anual['Ano'] >  ano_corte

X_train_raw = df_anual.loc[train_mask, features]
y_train     = df_anual.loc[train_mask, target]

X_test_raw = df_anual.loc[test_mask, features]
y_test     = df_anual.loc[test_mask, target]

scaler_X = MinMaxScaler()
X_train  = scaler_X.fit_transform(X_train_raw)
X_test   = scaler_X.transform(X_test_raw)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print(f"Treino: {X_train.shape[0]} anos ({df_anual.loc[train_mask,'Ano'].min()}–{df_anual.loc[train_mask,'Ano'].max()})")
print(f"Teste : {X_test.shape[0]} anos ({df_anual.loc[test_mask,'Ano'].min()}–{df_anual.loc[test_mask,'Ano'].max()})")

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
    {'Modelo': k, 'RMSE (×1.000 m³)': v['rmse'], 'MAPE (%)': v['mape'], 'R²': v['r2']}
    for k, v in resultados_anual.items()
]).sort_values('R²', ascending=False)

ano_teste_ini = df_anual.loc[test_mask, 'Ano'].min()
ano_teste_fim = df_anual.loc[test_mask, 'Ano'].max()
print(f"\nRESULTADOS DO BENCHMARK ANUAL — CARUARU ({ano_teste_ini}–{ano_teste_fim}):")
print("(Vol_produzido em 1.000 m³/ano — RMSE na mesma unidade)")
print(df_res_anual.to_string(index=False))

# --- Gráfico: Comparação de Modelos no Teste ---
try:
    anos_teste = df_anual.loc[test_mask, 'Ano'].values
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(anos_teste, y_test.values, 'o-', color='black', linewidth=2,
            label='Observado', zorder=5)
    cores_mod = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for (nome, res), cor in zip(resultados_anual.items(), cores_mod):
        ax.plot(anos_teste, res['preds'], 's--', color=cor,
                alpha=0.8, label=f"{nome} (R²={res['r2']:.3f})")
    ax.set_title(f'Comparação de Modelos — Teste {ano_teste_ini}–{ano_teste_fim} — Caruaru/PE',
                 fontweight='bold')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Vol. Produzido (×1.000 m³/ano)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    caminho_graf = os.path.join(projeto_dir, 'Comparacao_Modelos_Teste.png')
    plt.savefig(caminho_graf, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Gráfico salvo: {caminho_graf}")
except Exception as e:
    print(f"  [AVISO] Gráfico de benchmark: {e}")


# ==============================================================================
# 5. PROJEÇÃO ANUAL PARA 2030, 2040 E 2050 COM IC 95%
# ==============================================================================
print("\n" + "=" * 70)
print(" 5. PROJEÇÃO ANUAL DE DEMANDA — CARUARU (2027 A 2050) ")
print("=" * 70)
print("  NOTA: Vol_produzido em 1.000 m³/ano (unidade SNIS).")
print("        Para converter para m³ absolutos, multiplique por 1.000.")

anos_futuros = np.arange(2027, 2051)
df_futuro_anual = pd.DataFrame({'Ano': anos_futuros})

# Projeção temporal das variáveis exógenas (regressão linear ano a ano)
ano_hist = df_anual['Ano'].values.reshape(-1, 1)

for col in features:
    reg = LinearRegression()
    reg.fit(ano_hist, df_anual[col])
    df_futuro_anual[col] = reg.predict(df_futuro_anual['Ano'].values.reshape(-1, 1))

X_futuro_anual_scaled = scaler_X.transform(df_futuro_anual[features])

# Modelo Linear Re-treinado com todos os dados históricos
lr_full_anual = LinearRegression()
lr_full_anual.fit(
    scaler_X.transform(df_anual[features]),
    scaler_y.transform(df_anual[target].values.reshape(-1, 1)).flatten()
)

y_pred_hist_scaled = lr_full_anual.predict(scaler_X.transform(df_anual[features]))
y_pred_hist = scaler_y.inverse_transform(y_pred_hist_scaled.reshape(-1, 1)).flatten()

rmse_in_sample = np.sqrt(mean_squared_error(df_anual[target], y_pred_hist))

pred_linear_scaled = lr_full_anual.predict(X_futuro_anual_scaled)
df_futuro_anual['Previsao_Linear_Anual'] = scaler_y.inverse_transform(
    pred_linear_scaled.reshape(-1, 1)
).flatten()

# XGBoost — re-treinado com todos os dados históricos
xgb_full_anual = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_full_anual.fit(
    scaler_X.transform(df_anual[features]),
    scaler_y.transform(df_anual[target].values.reshape(-1, 1)).flatten()
)
pred_xgb_scaled = xgb_full_anual.predict(X_futuro_anual_scaled)
df_futuro_anual['Previsao_XGBoost_Anual'] = scaler_y.inverse_transform(
    pred_xgb_scaled.reshape(-1, 1)
).flatten()

# IC 95% baseado no RMSE in-sample do modelo linear
df_futuro_anual['IC95_Inf_Anual'] = df_futuro_anual['Previsao_Linear_Anual'] - 1.96 * rmse_in_sample
df_futuro_anual['IC95_Sup_Anual'] = df_futuro_anual['Previsao_Linear_Anual'] + 1.96 * rmse_in_sample

# Marcos temporais de interesse
anos_marco = [2030, 2040, 2050]
projecoes_marco = {}
for ano in anos_marco:
    row = df_futuro_anual[df_futuro_anual['Ano'] == ano].iloc[0]
    projecoes_marco[ano] = row

# ==============================================================================
# 5.1 - EXIBIÇÃO DAS PROJEÇÕES DE VOLUME DE ÁGUA
# ==============================================================================
print("\n" + "-" * 70)
print("  PROJEÇÕES DE VOLUME DE ÁGUA PRODUZIDO — CARUARU")
print("-" * 70)
print(f"  {'Ano':<6} {'Linear (×1.000 m³)':<22} {'Linear (m³ total)':<22} {'IC95% Inf (m³)':<20} {'IC95% Sup (m³)':<20} {'XGBoost (×1.000 m³)':<22}")
print("  " + "-" * 112)
for ano in anos_marco:
    row = projecoes_marco[ano]
    vol_1000 = row['Previsao_Linear_Anual']
    vol_m3   = vol_1000 * 1000
    ic_inf   = row['IC95_Inf_Anual'] * 1000
    ic_sup   = row['IC95_Sup_Anual'] * 1000
    vol_xgb  = row['Previsao_XGBoost_Anual']
    print(f"  {ano:<6} {vol_1000:<22,.2f} {vol_m3:<22,.0f} {ic_inf:<20,.0f} {ic_sup:<20,.0f} {vol_xgb:<22,.2f}")

# ==============================================================================
# 5.2 - EXIBIÇÃO DAS PROJEÇÕES DEMOGRÁFICAS
# ==============================================================================
# Colunas demográficas disponíveis no dataset
cols_demo = {
    'Pop_Urbana':      'População Urbana (hab.)',
    'Pop_Rural':       'População Rural (hab.)',
    'Area_urbanizada': 'Área Urbanizada (m²)',
}

print("\n" + "-" * 70)
print("  PROJEÇÕES DEMOGRÁFICAS — CARUARU")
print("-" * 70)
print(f"  {'Variável':<30} {'2030':>18} {'2040':>18} {'2050':>18}")
print("  " + "-" * 86)
for col, label in cols_demo.items():
    if col in df_futuro_anual.columns:
        vals = [projecoes_marco[a][col] for a in anos_marco]
        # Area_urbanizada em m², formatar diferente
        if 'Area' in col:
            print(f"  {label:<30} {vals[0]:>18,.0f} {vals[1]:>18,.0f} {vals[2]:>18,.0f}")
        else:
            print(f"  {label:<30} {vals[0]:>18,.0f} {vals[1]:>18,.0f} {vals[2]:>18,.0f}")
    else:
        print(f"  {label:<30} {'N/D':>18} {'N/D':>18} {'N/D':>18}")

# Pop Rural = Pop Total - Pop Urbana (caso não exista diretamente)
if 'Pop_Total' in df_futuro_anual.columns and 'Pop_Urbana' in df_futuro_anual.columns:
    print(f"\n  Nota: Pop_Rural projetada = Pop_Total - Pop_Urbana (modelo linear)")

# ==============================================================================
# 5.3 - VERIFICAÇÃO DE SANIDADE DO CONSUMO PER CAPITA
# ==============================================================================
print("\n" + "-" * 70)
print("  VERIFICAÇÃO DE SANIDADE — CONSUMO PER CAPITA")
print("-" * 70)
print(f"  {'Ano':<6} {'Vol (m³/ano)':<20} {'Pop Total (hab.)':<20} {'L/hab/dia':<15} {'Referência OMS: >50 L/hab/dia'}")
print("  " + "-" * 70)
for ano in anos_marco:
    row = projecoes_marco[ano]
    vol_m3 = row['Previsao_Linear_Anual'] * 1000  # converte ×1000 m³ → m³
    pop    = row.get('Pop_Total', None)
    if pop and pop > 0:
        litros_dia = (vol_m3 * 1000) / (pop * 365)  # m³→L, /hab, /dia
        status = 'OK' if litros_dia > 50 else 'ATENÇÃO: ABAIXO DO MÍNIMO'
        print(f"  {ano:<6} {vol_m3:<20,.0f} {pop:<20,.0f} {litros_dia:<15.1f} {status}")
    else:
        print(f"  {ano:<6} {vol_m3:<20,.0f} {'N/D':<20} {'N/D':<15}")

# ==============================================================================
# 5.4 - EXPORTAÇÃO DOS RESULTADOS + GRÁFICOS
# ==============================================================================

# --- Gráfico: Projeção de Demanda 2030-2050 ---
try:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Projeções de Demanda de Água — Caruaru/PE', fontsize=13, fontweight='bold')

    # Painel superior: Volume histórico + projeções
    ax1 = axes[0]
    anos_hist = df_anual['Ano'].values
    vols_hist = df_anual[target].values * 1000  # converte para m³
    vols_proj = df_futuro_anual['Previsao_Linear_Anual'].values * 1000
    ic_inf    = df_futuro_anual['IC95_Inf_Anual'].values * 1000
    ic_sup    = df_futuro_anual['IC95_Sup_Anual'].values * 1000
    anos_proj = df_futuro_anual['Ano'].values

    ax1.plot(anos_hist, vols_hist, 'o-', color='#2c3e50', linewidth=2, label='Histórico (real)')
    ax1.plot(anos_proj, vols_proj, 's--', color='#e74c3c', linewidth=2, label='Projeção Linear')
    ax1.fill_between(anos_proj, ic_inf, ic_sup, alpha=0.15, color='#e74c3c', label='IC 95%')
    for ano in [2030, 2040, 2050]:
        row = projecoes_marco[ano]
        ax1.axvline(x=ano, color='gray', linestyle=':', alpha=0.7)
        ax1.annotate(f'{ano}\n{row["Previsao_Linear_Anual"]*1000:,.0f} m³',
                     xy=(ano, row['Previsao_Linear_Anual']*1000),
                     xytext=(ano+0.3, row['Previsao_Linear_Anual']*1000 * 0.96),
                     fontsize=8, color='#e74c3c', fontweight='bold')
    ax1.set_ylabel('Volume Produzido (m³/ano)')
    ax1.set_xlabel('Ano')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # Painel inferior: Projeções demográficas
    ax2 = axes[1]
    pop_urb  = df_futuro_anual['Pop_Urbana'].values  if 'Pop_Urbana' in df_futuro_anual.columns else None
    pop_rur  = df_futuro_anual['Pop_Rural'].values   if 'Pop_Rural'  in df_futuro_anual.columns else None
    pop_urb_h = df_anual['Pop_Urbana'].values if 'Pop_Urbana' in df_anual.columns else None
    pop_rur_h = df_anual['Pop_Rural'].values  if 'Pop_Rural'  in df_anual.columns else None

    if pop_urb is not None:
        ax2.plot(anos_hist, pop_urb_h, 'o-', color='#2980b9', linewidth=2, label='Pop. Urbana (real)')
        ax2.plot(anos_proj, pop_urb,   's--', color='#2980b9', linewidth=1.5, alpha=0.7, label='Pop. Urbana (proj.)')
    if pop_rur is not None:
        ax2.plot(anos_hist, pop_rur_h, 'o-', color='#27ae60', linewidth=2, label='Pop. Rural (real)')
        ax2.plot(anos_proj, pop_rur,   's--', color='#27ae60', linewidth=1.5, alpha=0.7, label='Pop. Rural (proj.)')
    for ano in [2030, 2040, 2050]:
        ax2.axvline(x=ano, color='gray', linestyle=':', alpha=0.7)
    ax2.set_ylabel('População (hab.)')
    ax2.set_xlabel('Ano')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

    plt.tight_layout()
    caminho_proj_png = os.path.join(projeto_dir, 'Projecao_Demanda_2030_2050.png')
    plt.savefig(caminho_proj_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Gráfico de projeções salvo: {caminho_proj_png}")
except Exception as e:
    print(f"  [AVISO] Gráfico de projeções: {e}")

caminho_res_csv = os.path.join(projeto_dir, 'Resultados_Modelo_Anual.csv')
caminho_proj_csv = os.path.join(projeto_dir, 'Projeções_Demanda_Anual.csv')
df_res_anual.to_csv(caminho_res_csv, index=False)
df_futuro_anual.to_csv(caminho_proj_csv, index=False)

# Gera Resultados_Avaliacao.txt com informações completas e unidades corretas
caminho_resultado = os.path.join(projeto_dir, 'Resultados_Avaliacao.txt')

with open(caminho_resultado, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("  RESULTADOS DA AVALIAÇÃO DE MODELOS E PROJEÇÕES — CARUARU/PE\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"--- BENCHMARK DE MODELOS (período de teste: {ano_teste_ini}–{ano_teste_fim}) ---\n")
    f.write("  Vol_produzido em 1.000 m³/ano (unidade SNIS). RMSE na mesma unidade.\n\n")
    f.write(df_res_anual.to_string(index=False) + "\n\n")

    f.write("=" * 70 + "\n")
    f.write("  PROJEÇÕES DE DEMANDA DE ÁGUA — CARUARU\n")
    f.write("  UNIDADE: ×1.000 m³/ano (SNIS) | Para m³ total, multiplique por 1.000\n")
    f.write("=" * 70 + "\n\n")

    for ano in anos_marco:
        row = projecoes_marco[ano]
        vol_1000 = row['Previsao_Linear_Anual']
        vol_m3   = vol_1000 * 1000
        ic_inf   = row['IC95_Inf_Anual'] * 1000
        ic_sup   = row['IC95_Sup_Anual'] * 1000
        vol_xgb  = row['Previsao_XGBoost_Anual'] * 1000
        f.write(f"Ano {ano} — Modelo Linear:\n")
        f.write(f"  Volume projetado : {vol_1000:>10,.2f} × 1.000 m³/ano\n")
        f.write(f"                   = {vol_m3:>14,.0f} m³/ano\n")
        f.write(f"  IC 95%           : [{ic_inf:,.0f}  —  {ic_sup:,.0f}] m³/ano\n")
        f.write(f"  Modelo XGBoost   : {vol_xgb:>14,.0f} m³/ano\n\n")

    f.write("=" * 70 + "\n")
    f.write("  PROJEÇÕES DEMOGRÁFICAS — CARUARU\n")
    f.write("=" * 70 + "\n\n")

    header = f"  {'Variável':<30} {'2030':>15} {'2040':>15} {'2050':>15}\n"
    f.write(header)
    f.write("  " + "-" * 75 + "\n")

    for col, label in cols_demo.items():
        if col in df_futuro_anual.columns:
            vals = [projecoes_marco[a][col] for a in anos_marco]
            f.write(f"  {label:<30} {vals[0]:>15,.0f} {vals[1]:>15,.0f} {vals[2]:>15,.0f}\n")

    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write("  VERIFICAÇÃO DE SANIDADE — CONSUMO PER CAPITA\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"  {'Ano':<6} {'Vol (m³/ano)':<20} {'Pop Total':<18} {'L/hab/dia':<12} {'Status'}\n")
    f.write("  " + "-" * 70 + "\n")
    for ano in anos_marco:
        row = projecoes_marco[ano]
        vol_m3 = row['Previsao_Linear_Anual'] * 1000
        pop    = row.get('Pop_Total', None)
        if pop and pop > 0:
            litros_dia = (vol_m3 * 1000) / (pop * 365)
            status = 'OK (acima de 50 L/hab/dia)' if litros_dia > 50 else 'ATENÇÃO: abaixo do mínimo OMS'
            f.write(f"  {ano:<6} {vol_m3:<20,.0f} {pop:<18,.0f} {litros_dia:<12.1f} {status}\n")

    f.write("\n")
    f.write("  Nota sobre unidade:\n")
    f.write("  O SNIS reporta Vol_produzido em 1.000 m³/ano.\n")
    f.write("  Referência real 2022: 29.720 × 1.000 = 29.720.750 m³/ano.\n")
    f.write("  Fonte de validação: aguaesaneamento.org.br (~19.184.400 m³/ano\n")
    f.write("  referente a período anterior ao crescimento pós-2020).\n")

print(f"\n  Resultados exportados para: {caminho_resultado}")
print("\nModelo Anual — Caruaru — executado com sucesso.")
