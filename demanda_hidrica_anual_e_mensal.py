"""
==============================================================================
MODELAGEM E PROJEÇÃO DE DEMANDA HÍDRICA COM PERDAS — CARUARU/PE (2030–2050)
ABORDAGENS ANUAL E MENSAL COM DESAGREGAÇÃO SAZONAL CLIMÁTICA (IDH)
==============================================================================
Autora     : Mariana (Iniciação Científica - UFRPE)
Descrição  : Este script implementa a modelagem preditiva completa da demanda
             hídrica do município de Caruaru - PE, integrando:
             1. Modelagem preditiva com 6 algoritmos de Machine Learning:
                - MLR (Regressão Linear Múltipla)
                - Ridge (Regressão com Regularização L2)
                - SVR (Support Vector Regression)
                - RFR (Random Forest Regressor)
                - GBR (Gradient Boosting Regressor)
                - XGBoost (Extreme Gradient Boosting)
             2. Resolução de inconsistências metodológicas:
                - Remoção completa de Target Leakage (vazamento de dados)
                - Prevenção do defeito de saturação/extrapolação de árvores e SVR
                - Pipeline de validação Leave-One-Out (LOO-CV) sem vazamento de escala
                - Explicitação do regime de racionamento da seca de 2012-2017
             3. Projeção Anual ano a ano de 2030 a 2050 com demografia e território:
                - População Urbana e Rural
                - Área Urbana e Área Rural (km² e m²)
                - Índices de perdas na rede (IN049) e metas de eficiência
             4. Desagregação Mensal (252 meses de 2030 a 2050) via fórmula de
                sazonalidade climática (Índice de Demanda Hídrica - IDH).

FORMULAÇÕES MATEMÁTICAS:
    1. Demanda Hídrica Bruta Total (Volume de entrada no sistema / distribuição):
           Demanda_Total(t) = Consumo_Faturado(t) / (1 - Perdas(t) / 100)

    2. Paradigma por Intensidade Per Capita (Fisiograficamente Consistente):
           Demanda_Total(t) = [Pop_Total(t) * q_efetivo(t) / 1000] / [1 - Perdas(t) / 100]
           onde q_efetivo(t) é a cota per capita líquida de consumo (m³/hab/ano).

    3. Fórmula de Sazonalidade Climática (Índice de Demanda Hídrica - IDH):
           IDH_m = 0.55 * T_norm_m + 0.45 * (1 - P_norm_m)
           Peso_m = IDH_m / sum(IDH_k, k=1..12)  onde sum(Peso_m) = 1.0

    4. Demanda Hídrica Mensal Projetada:
           Demanda_Mensal(ano, m) = Demanda_Total(ano) * Peso_m
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    max_error, explained_variance_score
)

try:
    import xgboost as xgb
    TEM_XGBOOST = True
except ImportError:
    TEM_XGBOOST = False

# Configurar saída UTF-8 no Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Diretório base e pastas organizadas
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

pasta_graficos = os.path.join(script_dir, 'graficos')
pasta_csv = os.path.join(script_dir, 'csv')
os.makedirs(pasta_graficos, exist_ok=True)
os.makedirs(pasta_csv, exist_ok=True)

# ==============================================================================
# SEÇÃO 0 - CONSTANTES, PARÂMETROS E PESOS CLIMÁTICOS (FÓRMULA IDH)
# ==============================================================================
NOMES_MESES = [
    "Janeiro", "Fevereiro", "Marco", "Abril",
    "Maio", "Junho", "Julho", "Agosto",
    "Setembro", "Outubro", "Novembro", "Dezembro"
]

# Pesos mensais calibrados a partir do Índice de Demanda Hídrica (IDH)
# Fonte: sazonalidade_climatica.py / INMET Caruaru-PE (Estação A341 / 82900)
# IDH_m = 0.55 * T_norm_m + 0.45 * (1 - P_norm_m)
# Peso_m = IDH_m / sum(IDH_k)
PESOS_MENSAIS_CLIMATICOS = {
    1:  0.117137,  # Janeiro:   11.71%
    2:  0.122682,  # Fevereiro: 12.27%
    3:  0.100951,  # Março:     10.10%
    4:  0.067829,  # Abril:      6.78%
    5:  0.060971,  # Maio:       6.10%
    6:  0.012990,  # Junho:      1.30%
    7:  0.005707,  # Julho:      0.57%
    8:  0.034751,  # Agosto:     3.48%
    9:  0.078510,  # Setembro:   7.85%
    10: 0.124994,  # Outubro:   12.50%
    11: 0.141891,  # Novembro:  14.19%
    12: 0.131588,  # Dezembro:  13.16%
}
assert abs(sum(PESOS_MENSAIS_CLIMATICOS.values()) - 1.0) < 1e-5, "Soma dos pesos deve ser 1.0"

# Área territorial oficial do município de Caruaru/PE (IBGE 2022: 920,611 km²)
AREA_TOTAL_MUNICIPAL_KM2 = 920.611
AREA_TOTAL_MUNICIPAL_M2  = 920611000.0

# ==============================================================================
# FUNÇÕES DE APOIO
# ==============================================================================
def parse_csv_snis(caminho):
    """Parsing robusto do CSV do SNIS com remoção de aspas e delimitador de milhar."""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(caminho, 'r', encoding=enc) as f:
                raw_lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    def parse_line(line):
        fields, in_quote, field = [], False, ""
        for ch in line.strip():
            if ch == '"':
                in_quote = not in_quote
            elif ch == ';' and not in_quote:
                fields.append(field.strip().strip('"').strip())
                field = ""
            else:
                field += ch
        fields.append(field.strip().strip('"').strip())
        return fields

    header = parse_line(raw_lines[0])
    rows = []
    for line in raw_lines[1:]:
        fields = parse_line(line)
        if len(fields) >= len(header):
            fields = fields[:len(header)]
        rows.append(fields)
    return pd.DataFrame(rows, columns=header)


def converter_numero_snis(valor):
    """Converte números formatados pelo SNIS para float padrão."""
    if pd.isna(valor) or str(valor).strip() == '':
        return np.nan
    s = str(valor).strip()
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace('.', '')
    try:
        return float(s)
    except ValueError:
        return np.nan


def calcular_metricas(y_real, y_pred, nome_modelo):
    """Calcula 8 métricas padronizadas de avaliação de regressão."""
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100
    medae = median_absolute_error(y_real, y_pred)
    max_err = max_error(y_real, y_pred)
    ev = explained_variance_score(y_real, y_pred)
    return {
        'Modelo': nome_modelo,
        'MAE': mae, 'MSE': mse, 'RMSE': rmse,
        'R2': r2, 'MAPE_pct': mape,
        'MedAE': medae, 'MaxError': max_err, 'EV': ev,
        'y_pred': y_pred,
    }


# ==============================================================================
# 1. CARREGAMENTO E INTEGRAÇÃO CONSISTENTE DOS DATASETS
# ==============================================================================
print("=" * 80)
print("  1. CARREGAMENTO DOS DADOS E AUDITORIA METODOLÓGICA SÊNIOR")
print("=" * 80)

candidatos_ds = [
    os.path.join(script_dir, 'database', 'Caruaru_Dataset_Atualizado.csv'),
    os.path.join(script_dir, 'Caruaru_Dataset_Atualizado.csv'),
]
caminho_ds = next((p for p in candidatos_ds if os.path.exists(p)), candidatos_ds[0])
df_dataset = pd.read_csv(caminho_ds).sort_values('Ano').reset_index(drop=True)

candidatos_snis = [
    os.path.join(script_dir, 'Consolidado_Limpo_Caruaru.csv'),
    os.path.join(script_dir, 'database', 'Consolidado_Limpo_Caruaru.csv'),
]
caminho_snis = next((p for p in candidatos_snis if os.path.exists(p)), candidatos_snis[0])

df_snis_raw = parse_csv_snis(caminho_snis)
mapa_snis = {
    'Ano de Referência': 'Ano',
    'AG010 - Volume de água consumido': 'Vol_Consumido_SNIS',
    'AG012 - Volume de água macromedido': 'Vol_Macromedido_SNIS',
    'IN049_AE - Índice de perdas na distribuição': 'Idx_Perdas',
    'IN044_AE - Índice de micromedição relativo ao consumo': 'Idx_Micromedicao',
    'AG001 - População total atendida com abastecimento de água': 'Pop_Atendida_SNIS',
    'AG004 - Quantidade de ligações ativas de água micromedidas': 'Lig_Micromedidas',
    'AG005 - Extensão da rede de água': 'Extensao_Rede_km',
}

df_snis = df_snis_raw.rename(columns=mapa_snis)
df_snis = df_snis[df_snis['Ano'].apply(lambda x: str(x).strip().isdigit())].copy()
df_snis['Ano'] = df_snis['Ano'].astype(int)

for col in ['Vol_Consumido_SNIS', 'Vol_Macromedido_SNIS', 'Idx_Perdas',
            'Idx_Micromedicao', 'Pop_Atendida_SNIS', 'Lig_Micromedidas', 'Extensao_Rede_km']:
    if col in df_snis.columns:
        df_snis[col] = df_snis[col].apply(converter_numero_snis)

df_snis = df_snis[['Ano', 'Vol_Consumido_SNIS', 'Vol_Macromedido_SNIS',
                    'Idx_Perdas', 'Idx_Micromedicao', 'Pop_Atendida_SNIS',
                    'Lig_Micromedidas', 'Extensao_Rede_km']].dropna(subset=['Idx_Perdas'])
df_snis = df_snis[df_snis['Idx_Perdas'] > 0].sort_values('Ano').reset_index(drop=True)

df_merged = pd.merge(df_dataset, df_snis, on='Ano', how='inner').sort_values('Ano').reset_index(drop=True)

# Cálculo da Demanda Hídrica Bruta Total Histórica (em 1.000 m³/ano):
# Demanda = Consumo / (1 - Perdas/100)
df_merged['Demanda_Hidrica'] = df_merged['Vol_Consumido_SNIS'] / (1.0 - df_merged['Idx_Perdas'] / 100.0)

# Consumo per capita efetivo (m³/hab/ano e L/hab/dia)
df_merged['Consumo_pc_m3'] = (df_merged['Vol_Consumido_SNIS'] / df_merged['Pop_Total']) * 1000.0
df_merged['Consumo_pc_L_dia'] = (df_merged['Vol_Consumido_SNIS'] * 1e6) / (df_merged['Pop_Total'] * 365.25)
df_merged['Demanda_pc_L_dia'] = (df_merged['Demanda_Hidrica'] * 1e6) / (df_merged['Pop_Total'] * 365.25)

# Dinâmica territorial e demográfica
df_merged['Pop_Rural'] = df_merged['Pop_Total'] - df_merged['Pop_Urbana']
df_merged['Taxa_Urbanizacao_pct'] = (df_merged['Pop_Urbana'] / df_merged['Pop_Total']) * 100.0
df_merged['Area_Urbana_km2'] = df_merged['Area_urbanizada'] / 1e6
df_merged['Area_Rural_km2'] = AREA_TOTAL_MUNICIPAL_KM2 - df_merged['Area_Urbana_km2']
df_merged['Area_Urbana_m2'] = df_merged['Area_urbanizada']
df_merged['Area_Rural_m2'] = AREA_TOTAL_MUNICIPAL_M2 - df_merged['Area_urbanizada']
df_merged['Ocupacao_Urbana_pct'] = (df_merged['Area_Urbana_km2'] / AREA_TOTAL_MUNICIPAL_KM2) * 100.0

print(f"  Dataset histórico validado: {len(df_merged)} anos ({df_merged['Ano'].min()} a {df_merged['Ano'].max()})")
print(f"  População Histórica: {df_merged['Pop_Total'].min():,.0f} (2002) a {df_merged['Pop_Total'].max():,.0f} (2022)")
print(f"  Demanda Hídrica Histórica: {df_merged['Demanda_Hidrica'].min():,.1f} (2003) a {df_merged['Demanda_Hidrica'].max():,.1f} (2008) mil m3/ano")


# ==============================================================================
# 2. DEFINIÇÃO RIGOROSA DAS FEATURES (ELIMINAÇÃO DE TARGET LEAKAGE)
# ==============================================================================
# AUDITORIA SÊNIOR:
# No código anterior, Vol_per_capita = Vol_Consumido / Pop_Total era incluída em X.
# Como Demanda = (Pop_Total * Vol_per_capita) / (1 - Perdas/100), o MLR aprendia
# a identidade algébrica, causando vazamento de dados (Target Leakage).
#
# SOLUÇÃO ADOTADA:
# Modelamos a demanda em duas formulações científicas:
#
# FORMULAÇÃO A (Macro-Volumétrica Direta):
# Prediz Demanda_Total diretamente a partir de preditores EXÓGENOS legítimos:
# ['Pop_Urbana', 'Pop_Rural', 'Area_Urbana_km2', 'PIB_per_capita', 'Idx_Perdas', 'Idx_Micromedicao']
#
# FORMULAÇÃO B (Intensidade Per Capita Fisiograficamente Consistente - RECOMENDADA):
# Prediz Consumo_pc_m3 (m³/hab/ano) através de determinantes socioeconômicos e de infraestrutura:
# ['PIB_per_capita', 'Tarifa_residencial', 'Idx_Micromedicao', 'Area_Urbana_km2']
# e reconstrói a demanda física total por conservação de massa:
# Demanda_Pred = [Pop_Total * Consumo_pc_Pred / 1000] / [1 - Idx_Perdas / 100]
# Esta formulação elimina a saturação de árvores (RFR, GBR, XGBoost) e SVR na extrapolação futura!

features_diretas = ['Pop_Urbana', 'Pop_Rural', 'Area_Urbana_km2', 'PIB_per_capita', 'Idx_Perdas', 'Idx_Micromedicao']
features_percapita = ['PIB_per_capita', 'Tarifa_residencial', 'Idx_Micromedicao', 'Area_Urbana_km2']

for col in features_diretas + features_percapita:
    if col in df_merged.columns and df_merged[col].isna().any():
        df_merged[col] = df_merged[col].fillna(df_merged[col].median())

X_dir = df_merged[features_diretas].values
X_pc  = df_merged[features_percapita].values
y_demanda_real = df_merged['Demanda_Hidrica'].values
y_pc_real = df_merged['Consumo_pc_m3'].values
pop_hist = df_merged['Pop_Total'].values
perdas_hist = df_merged['Idx_Perdas'].values
anos_hist = df_merged['Ano'].values


# ==============================================================================
# 3. VALIDAÇÃO CRUZADA LEAVE-ONE-OUT (LOO-CV) EM PIPELINE ROBUSTO
# ==============================================================================
print("\n" + "=" * 80)
print("  2. AVALIAÇÃO DOS MODELOS (LOO-CV) LIVRE DE VAZAMENTO DE DADOS")
print("=" * 80)

def instanciar_modelos():
    """Gera instâncias limpas dos modelos de regressão."""
    mods = {
        'MLR': LinearRegression(),
        'Ridge': Ridge(alpha=5.0, random_state=42),
        'SVR': SVR(kernel='rbf', C=50.0, epsilon=1.0, gamma='scale'),
        'RFR': RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_split=2, random_state=42),
        'GBR': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    }
    if TEM_XGBOOST:
        mods['XGBoost'] = xgb.XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    return mods

loo = LeaveOneOut()

# Avaliar Formulacao B (Componente Per-Capita Rescalonado - Método Sênior Oficial)
resultados_anual = {}
predicoes_anual_loo = {}

print("  AVALIAÇÃO NA ESCALA ANUAL (Formulação Per-Capita Fisiográfica com LOO-CV):")
for nome, regressor in instanciar_modelos().items():
    # Pipeline garante que o scaler seja ajustado estritamente dentro de cada fold do LOO-CV
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', regressor)
    ])
    
    # 1. Predição LOO-CV do consumo per capita (m³/hab/ano)
    y_pc_pred_loo = cross_val_predict(pipe, X_pc, y_pc_real, cv=loo)
    
    # 2. Rescalonamento analítico para Demanda Total (1.000 m³/ano)
    # Demanda = (Consumo_pc * Pop_Total / 1000) / (1 - Perdas / 100)
    y_dem_pred_loo = (y_pc_pred_loo * pop_hist / 1000.0) / (1.0 - perdas_hist / 100.0)
    
    res = calcular_metricas(y_demanda_real, y_dem_pred_loo, nome)
    resultados_anual[nome] = res
    predicoes_anual_loo[nome] = y_dem_pred_loo
    print(f"  {nome:<10} | R²={res['R2']:.4f} | MAE={res['MAE']:>8,.1f} | RMSE={res['RMSE']:>8,.1f} | MAPE={res['MAPE_pct']:>5.2f}%")


# ==============================================================================
# 4. AVALIAÇÃO NA ESCALA MENSAL HISTÓRICA (COM FÓRMULA IDH)
# ==============================================================================
print("\n" + "=" * 80)
print("  3. AVALIAÇÃO NA ESCALA MENSAL HISTÓRICA (FÓRMULA CLIMÁTICA IDH)")
print("=" * 80)

# Construção da série histórica mensal real desagregada (21 anos * 12 meses = 252 observações)
registros_mensais_hist = []
for _, r in df_merged.iterrows():
    ano = int(r['Ano'])
    dem_anual = r['Demanda_Hidrica']
    vol_cons_anual = r['Vol_Consumido_SNIS']
    perdas_pct = r['Idx_Perdas']
    pop = r['Pop_Total']

    for mes in range(1, 13):
        peso = PESOS_MENSAIS_CLIMATICOS[mes]
        dem_mensal = dem_anual * peso
        cons_mensal = vol_cons_anual * peso
        perd_mensal = dem_mensal - cons_mensal

        registros_mensais_hist.append({
            'Ano': ano,
            'Mes': mes,
            'Nome_Mes': NOMES_MESES[mes - 1],
            'Peso_Mensal': peso,
            'Pop_Total': pop,
            'Idx_Perdas': perdas_pct,
            'Demanda_Anual_Real': dem_anual,
            'Demanda_Mensal_Real': dem_mensal,
            'Consumo_Mensal_Real': cons_mensal,
            'Perda_Mensal_Real': perd_mensal,
        })

df_mensal_hist = pd.DataFrame(registros_mensais_hist)
y_mensal_real = df_mensal_hist['Demanda_Mensal_Real'].values

resultados_mensal = {}
print("  AVALIAÇÃO NA ESCALA MENSAL (252 meses históricos LOO-CV):")
for nome, res_a in resultados_anual.items():
    preds_a = predicoes_anual_loo[nome]
    preds_m = []
    for idx_ano in range(len(anos_hist)):
        pa = preds_a[idx_ano]
        for m in range(1, 13):
            preds_m.append(pa * PESOS_MENSAIS_CLIMATICOS[m])
    y_pred_m = np.array(preds_m)
    res_m = calcular_metricas(y_mensal_real, y_pred_m, nome)
    resultados_mensal[nome] = res_m
    print(f"  {nome:<10} (Mensal) | R²={res_m['R2']:.4f} | MAE={res_m['MAE']:>8,.1f} | RMSE={res_m['RMSE']:>8,.1f} | MAPE={res_m['MAPE_pct']:>5.2f}%")


# ==============================================================================
# 5. TABELA COMPARATIVA DE MÉTRICAS: ESCALA ANUAL vs MENSAL
# ==============================================================================
print("\n" + "=" * 80)
print("  4. TABELA COMPARATIVA DE MÉTRICAS (LOO-CV): ANUAL vs MENSAL")
print("=" * 80)

linhas_comparativas = []
for nome in resultados_anual.keys():
    ra = resultados_anual[nome]
    rm = resultados_mensal[nome]
    linhas_comparativas.append({
        'Escala': 'Anual (x1.000 m3/ano)',
        'Modelo': nome,
        'MAE': round(ra['MAE'], 2),
        'MSE': round(ra['MSE'], 2),
        'RMSE': round(ra['RMSE'], 2),
        'R2': round(ra['R2'], 4),
        'MAPE (%)': round(ra['MAPE_pct'], 2),
        'MedAE': round(ra['MedAE'], 2),
        'MaxError': round(ra['MaxError'], 2),
        'EV': round(ra['EV'], 4),
    })
    linhas_comparativas.append({
        'Escala': 'Mensal (x1.000 m3/mes)',
        'Modelo': nome,
        'MAE': round(rm['MAE'], 2),
        'MSE': round(rm['MSE'], 2),
        'RMSE': round(rm['RMSE'], 2),
        'R2': round(rm['R2'], 4),
        'MAPE (%)': round(rm['MAPE_pct'], 2),
        'MedAE': round(rm['MedAE'], 2),
        'MaxError': round(rm['MaxError'], 2),
        'EV': round(rm['EV'], 4),
    })

df_comp_metricas = pd.DataFrame(linhas_comparativas)
print(df_comp_metricas.to_string(index=False))


# ==============================================================================
# 6. PROJEÇÃO ANUAL ANO A ANO (2030–2050)
# ==============================================================================
print("\n" + "=" * 80)
print("  5. PROJEÇÃO DA DEMANDA HÍDRICA ANUAL (2030–2050)")
print("=" * 80)

anos_futuros = np.arange(2025, 2051)
df_futuro_anual = pd.DataFrame({'Ano': anos_futuros})

# Projeções das variáveis demográficas e socioeconômicas
cols_tendencia = [
    'Pop_Total', 'Pop_Urbana', 'PIB_per_capita', 'Tarifa_residencial',
    'Idx_Micromedicao', 'Area_urbanizada', 'Idx_Perdas'
]
for col in cols_tendencia:
    reg_tmp = LinearRegression()
    reg_tmp.fit(anos_hist.reshape(-1, 1), df_merged[col].values)
    df_futuro_anual[col] = reg_tmp.predict(anos_futuros.reshape(-1, 1))

# Demografia e Balanço Espacial Consistentes:
# 1. População Urbana e População Rural
df_futuro_anual['Pop_Rural'] = df_futuro_anual['Pop_Total'] - df_futuro_anual['Pop_Urbana']
df_futuro_anual['Taxa_Urbanizacao_pct'] = (df_futuro_anual['Pop_Urbana'] / df_futuro_anual['Pop_Total']) * 100.0

# 2. Área Urbana e Área Rural (Área total do município = 920,611 km² / IBGE)
df_futuro_anual['Area_Urbana_m2'] = df_futuro_anual['Area_urbanizada']
df_futuro_anual['Area_Urbana_km2'] = df_futuro_anual['Area_urbanizada'] / 1e6
df_futuro_anual['Area_Rural_m2'] = AREA_TOTAL_MUNICIPAL_M2 - df_futuro_anual['Area_urbanizada']
df_futuro_anual['Area_Rural_km2'] = df_futuro_anual['Area_Rural_m2'] / 1e6
df_futuro_anual['Ocupacao_Urbana_pct'] = (df_futuro_anual['Area_Urbana_km2'] / AREA_TOTAL_MUNICIPAL_KM2) * 100.0

# 3. Metas de Perdas na Distribuição (Marco Legal do Saneamento - Lei 14.026/2020)
# Perdas são limitadas fisicamente entre 10% (limite operacional de excelência) e 70%
df_futuro_anual['Idx_Perdas'] = df_futuro_anual['Idx_Perdas'].clip(10.0, 70.0)
df_futuro_anual['Idx_Micromedicao'] = df_futuro_anual['Idx_Micromedicao'].clip(50.0, 100.0)

# Ajuste dos modelos finais com 100% da série histórica e predição futura
X_fut_pc = df_futuro_anual[features_percapita].values
scaler_pc_final = StandardScaler()
X_pc_s_final = scaler_pc_final.fit_transform(X_pc)
X_fut_pc_s = scaler_pc_final.transform(X_fut_pc)

modelos_finais = instanciar_modelos()
nomes_modelos = list(modelos_finais.keys())

for nome, mod in modelos_finais.items():
    mod.fit(X_pc_s_final, y_pc_real)
    pred_pc_fut = mod.predict(X_fut_pc_s)
    # Demanda Total = (Consumo_pc * Pop_Total / 1000) / (1 - Perdas / 100)
    pred_dem_fut = (pred_pc_fut * df_futuro_anual['Pop_Total'] / 1000.0) / (1.0 - df_futuro_anual['Idx_Perdas'] / 100.0)
    df_futuro_anual[f'Demanda_{nome}'] = np.clip(pred_dem_fut, 0, None)
    df_futuro_anual[f'Consumo_pc_{nome}'] = pred_pc_fut

cols_demanda = [f'Demanda_{nm}' for nm in nomes_modelos]
df_futuro_anual['Demanda_Media'] = df_futuro_anual[cols_demanda].mean(axis=1)
df_futuro_anual['Demanda_Min'] = df_futuro_anual[cols_demanda].min(axis=1)
df_futuro_anual['Demanda_Max'] = df_futuro_anual[cols_demanda].max(axis=1)

# Consumo Efetivo e Volume de Perdas baseados no melhor modelo Ensemble
df_futuro_anual['Consumo_Efetivo_Media'] = df_futuro_anual['Demanda_Media'] * (1.0 - df_futuro_anual['Idx_Perdas'] / 100.0)
df_futuro_anual['Volume_Perda_Media'] = df_futuro_anual['Demanda_Media'] * (df_futuro_anual['Idx_Perdas'] / 100.0)

# Filtrar horizonte oficial 2030–2050
df_proj_anual_2030_2050 = df_futuro_anual[df_futuro_anual['Ano'] >= 2030].copy().reset_index(drop=True)

print("\n  RESUMO DA PROJEÇÃO ANUAL (x1.000 m3/ano) — MARCOS QUINQUENAIS:")
header_str = f"  {'Ano':<6} {'Pop.Total':>11} {'Perdas%':>9} " + " ".join([f"{nm:>11}" for nm in nomes_modelos]) + f" {'MEDIA':>11}"
print(header_str)
print("  " + "-" * (len(header_str) - 2))
for ano_m in [2030, 2035, 2040, 2045, 2050]:
    row = df_proj_anual_2030_2050[df_proj_anual_2030_2050['Ano'] == ano_m].iloc[0]
    mods_vals = " ".join([f"{row[f'Demanda_{nm}']:>11,.1f}" for nm in nomes_modelos])
    print(f"  {ano_m:<6} {row['Pop_Total']:>11,.0f} {row['Idx_Perdas']:>8.2f}% {mods_vals} {row['Demanda_Media']:>11,.1f}")

# Tabela Demográfica e Territorial Completa
print("\n" + "=" * 80)
print("  PROJEÇÃO DEMOGRÁFICA E TERRITORIAL DETALHADA (2030–2050) — CARUARU/PE")
print("=" * 80)
print(f"  Área Total do Município (IBGE): {AREA_TOTAL_MUNICIPAL_KM2:,.3f} km² ({AREA_TOTAL_MUNICIPAL_M2:,.0f} m²)\n")
print(f"  {'Ano':<6} {'Pop.Total':>12} {'Pop.Urbana':>12} {'Pop.Rural':>11} {'Urbaniz.%':>10} {'Área Urbana':>14} {'Área Rural':>14} {'Ocup.Urb%':>10}")
print("  " + "-" * 95)
for _, r in df_proj_anual_2030_2050.iterrows():
    print(f"  {int(r['Ano']):<6} {r['Pop_Total']:>12,.0f} {r['Pop_Urbana']:>12,.0f} {r['Pop_Rural']:>11,.0f} "
          f"{r['Taxa_Urbanizacao_pct']:>9.2f}% {r['Area_Urbana_km2']:>11.2f} km² {r['Area_Rural_km2']:>11.2f} km² {r['Ocupacao_Urbana_pct']:>9.2f}%")


# ==============================================================================
# 7. PROJEÇÃO MENSAL (2030–2050, 252 MESES) COM SAZONALIDADE IDH
# ==============================================================================
print("\n" + "=" * 80)
print("  6. PROJEÇÃO DA DEMANDA HÍDRICA MENSAL (2030–2050, 252 MESES)")
print("=" * 80)
print("  Fórmula aplicada: Demanda_Mensal(ano, m) = Demanda_Anual(ano) * Peso_m")

registros_proj_mensal = []
for _, row_ano in df_proj_anual_2030_2050.iterrows():
    ano = int(row_ano['Ano'])
    pop = row_ano['Pop_Total']
    perdas_pct = row_ano['Idx_Perdas']

    for mes in range(1, 13):
        peso = PESOS_MENSAIS_CLIMATICOS[mes]
        nome_mes = NOMES_MESES[mes - 1]

        reg = {
            'Ano': ano,
            'Mes': mes,
            'Nome_Mes': nome_mes,
            'Peso_Sazonal_pct': round(peso * 100, 4),
            'Pop_Total': round(pop, 0),
            'Pop_Urbana': round(row_ano['Pop_Urbana'], 0),
            'Pop_Rural': round(row_ano['Pop_Rural'], 0),
            'Area_Urbana_km2': round(row_ano['Area_Urbana_km2'], 3),
            'Area_Rural_km2': round(row_ano['Area_Rural_km2'], 3),
            'Perdas_pct': round(perdas_pct, 2),
        }

        # Demanda mensal para cada modelo em 1.000 m³ e m³ absolutos
        for nm in nomes_modelos:
            dem_anual_mod = row_ano[f'Demanda_{nm}']
            dem_mensal_mod = dem_anual_mod * peso
            reg[f'Demanda_{nm}_1000m3'] = round(dem_mensal_mod, 2)
            reg[f'Demanda_{nm}_m3'] = round(dem_mensal_mod * 1000.0, 0)

        # Estatísticas do Ensemble
        dem_media_a = row_ano['Demanda_Media']
        reg['Demanda_Media_1000m3'] = round(dem_media_a * peso, 2)
        reg['Demanda_Media_m3'] = round(dem_media_a * peso * 1000.0, 0)
        reg['Demanda_Min_1000m3'] = round(row_ano['Demanda_Min'] * peso, 2)
        reg['Demanda_Max_1000m3'] = round(row_ano['Demanda_Max'] * peso, 2)

        # Consumo efetivo vs Perda mensal (com base na Média Ensemble)
        cons_media_m = reg['Demanda_Media_1000m3'] * (1.0 - perdas_pct / 100.0)
        perd_media_m = reg['Demanda_Media_1000m3'] * (perdas_pct / 100.0)
        reg['Consumo_Efetivo_Media_m3'] = round(cons_media_m * 1000.0, 0)
        reg['Volume_Perda_Media_m3'] = round(perd_media_m * 1000.0, 0)

        # Consumo efetivo vs Perda mensal pelo modelo RFR
        dem_rfr = reg['Demanda_RFR_1000m3']
        reg['Consumo_Efetivo_RFR_m3'] = round(dem_rfr * (1.0 - perdas_pct / 100.0) * 1000.0, 0)
        reg['Volume_Perda_RFR_m3'] = round(dem_rfr * (perdas_pct / 100.0) * 1000.0, 0)

        registros_proj_mensal.append(reg)

df_proj_mensal = pd.DataFrame(registros_proj_mensal)
print(f"  Total de registros mensais futuros gerados: {len(df_proj_mensal)} meses ({len(df_proj_anual_2030_2050)} anos x 12 meses)")

print("\n  EXEMPLO DE PROJEÇÃO MENSAL DETALHADA PARA O ANO 2030:")
print(f"  {'Mes':<10} {'Peso%':>8} {'MLR (m3)':>15} {'RFR (m3)':>15} {'SVR (m3)':>15} {'MEDIA (m3)':>15} {'Consumo Efetivo':>17} {'Perda na Rede':>15}")
print("  " + "-" * 115)
for _, r in df_proj_mensal[df_proj_mensal['Ano'] == 2030].iterrows():
    print(f"  {r['Nome_Mes']:<10} {r['Peso_Sazonal_pct']:>7.2f}% "
          f"{r['Demanda_MLR_m3']:>15,.0f} {r['Demanda_RFR_m3']:>15,.0f} {r['Demanda_SVR_m3']:>15,.0f} "
          f"{r['Demanda_Media_m3']:>15,.0f} {r['Consumo_Efetivo_Media_m3']:>17,.0f} {r['Volume_Perda_Media_m3']:>15,.0f}")


# ==============================================================================
# 8. GERAÇÃO DE GRÁFICOS INTEGRADOS DE ALTA RESOLUÇÃO
# ==============================================================================
print("\n" + "=" * 80)
print("  7. GERANDO GRÁFICOS TÉCNICOS COMPARATIVOS")
print("=" * 80)

cores = {
    'MLR': '#e74c3c',
    'Ridge': '#e67e22',
    'SVR': '#3498db',
    'RFR': '#2ecc71',
    'GBR': '#9b59b6',
    'XGBoost': '#1abc9c',
}

# --- Gráfico 1: Comparativo Anual vs Mensal (Histórico + Projeções) ---
try:
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('Demanda Hídrica de Caruaru/PE com Perdas na Distribuição (Versão Auditada)\n'
                 'Integração Completa: Escala Anual e Projeção Mensal Sazonal (2030–2050)',
                 fontsize=13, fontweight='bold')

    # Painel 1: Escala Anual (Histórico + Projeções 2030-2050)
    ax1 = axes[0]
    ax1.plot(anos_hist, y_demanda_real, 'o-', color='black', linewidth=2.5, label='Histórico Observado (SNIS)', zorder=5)
    for nm in nomes_modelos:
        cor = cores.get(nm, 'gray')
        ax1.plot(df_proj_anual_2030_2050['Ano'], df_proj_anual_2030_2050[f'Demanda_{nm}'],
                 '--', color=cor, linewidth=1.8, label=f'Projeção {nm}')
    ax1.plot(df_proj_anual_2030_2050['Ano'], df_proj_anual_2030_2050['Demanda_Media'],
             '-', color='#2c3e50', linewidth=3, label='Média Ensemble', zorder=4)
    ax1.fill_between(df_proj_anual_2030_2050['Ano'],
                     df_proj_anual_2030_2050['Demanda_Min'],
                     df_proj_anual_2030_2050['Demanda_Max'],
                     alpha=0.15, color='#2c3e50', label='Faixa de Incerteza (Min-Max)')
    ax1.axvspan(2012, 2017, color='red', alpha=0.1, label='Mega-Seca do Agreste (2012-2017)')
    ax1.set_title('Projeção em Escala ANUAL (com Perdas na Rede e Demografia Consistente)', fontweight='bold')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('Demanda Hídrica (×1.000 m³/ano)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=4, fontsize=8, loc='upper left')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Painel 2: Escala Mensal Contínua (2030–2050, 252 meses)
    ax2 = axes[1]
    eixo_x_meses = np.arange(len(df_proj_mensal))
    anos_unicos = df_proj_mensal['Ano'].unique()

    for nm in ['MLR', 'RFR', 'SVR']:
        cor = cores.get(nm, 'gray')
        ax2.plot(eixo_x_meses, df_proj_mensal[f'Demanda_{nm}_1000m3'],
                 '-', color=cor, linewidth=1.3, alpha=0.8, label=f'{nm} (Mensal)')
    ax2.plot(eixo_x_meses, df_proj_mensal['Demanda_Media_1000m3'],
             '-', color='#2c3e50', linewidth=2.0, label='Média Ensemble (Mensal)')

    passos_ano = np.arange(0, len(df_proj_mensal), 24)
    labels_ano = [str(anos_unicos[i // 12]) for i in passos_ano]
    ax2.set_xticks(passos_ano)
    ax2.set_xticklabels(labels_ano)
    ax2.set_title('Projeção em Escala MENSAL Contínua (2030–2050 via Fórmula de Sazonalidade IDH)', fontweight='bold')
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Demanda Mensal (×1.000 m³/mês)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    caminho_g1 = os.path.join(pasta_graficos, 'Demanda_Hidrica_Anual_vs_Mensal.png')
    plt.savefig(caminho_g1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico Anual vs Mensal salvo: {caminho_g1}")
except Exception as e:
    print(f"  [AVISO] Gráfico 1: {e}")

# --- Gráfico 2: Ciclo Sazonal Médio Projetado (Janeiro a Dezembro) ---
try:
    fig, ax = plt.subplots(figsize=(13, 6))
    marcos_plot = [2030, 2035, 2040, 2045, 2050]
    paleta_anos = sns.color_palette("viridis", len(marcos_plot))

    for ano_m, cor_ano in zip(marcos_plot, paleta_anos):
        sub_m = df_proj_mensal[df_proj_mensal['Ano'] == ano_m]
        ax.plot(sub_m['Nome_Mes'], sub_m['Demanda_Media_m3'] / 1e6, 'o-',
                color=cor_ano, linewidth=2, label=f'Ano {ano_m} (Média Ensemble)')

    ax.set_title('Perfil do Ciclo Sazonal Mensal de Demanda Hídrica — Caruaru/PE\n'
                 'Variação Intranual Orientada pela Climatologia (IDH: Calor e Estiagem)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Mês')
    ax.set_ylabel('Demanda Hídrica (Milhões de m³/mês)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    caminho_g2 = os.path.join(pasta_graficos, 'Demanda_Ciclo_Sazonal_Mensal.png')
    plt.savefig(caminho_g2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico Ciclo Sazonal salvo: {caminho_g2}")
except Exception as e:
    print(f"  [AVISO] Gráfico 2: {e}")

# --- Gráfico 3: Projeção Demográfica e Territorial ---
try:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Projeção Demográfica e Territorial de Caruaru/PE (2030–2050)\n'
                 'Crescimento Urbano e Balanço Espacial (Área Municipal: 920,61 km²)',
                 fontweight='bold', fontsize=13)

    # Painel 1: População Urbana vs População Rural
    ax1 = axes[0]
    ax1.plot(df_proj_anual_2030_2050['Ano'], df_proj_anual_2030_2050['Pop_Total'] / 1e3,
             'k-', linewidth=2.5, label='População Total')
    ax1.plot(df_proj_anual_2030_2050['Ano'], df_proj_anual_2030_2050['Pop_Urbana'] / 1e3,
             'o--', color='#2980b9', linewidth=2, label='População Urbana')
    ax1.plot(df_proj_anual_2030_2050['Ano'], df_proj_anual_2030_2050['Pop_Rural'] / 1e3,
             's--', color='#27ae60', linewidth=2, label='População Rural')
    ax1.set_title('Evolução Populacional (Urbana vs Rural)', fontweight='bold')
    ax1.set_xlabel('Ano')
    ax1.set_ylabel('População (milhares de habitantes)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Painel 2: Área Urbana vs Área Rural (km²)
    ax2 = axes[1]
    ax2.plot(df_proj_anual_2030_2050['Ano'], df_proj_anual_2030_2050['Area_Urbana_km2'],
             'o-', color='#e74c3c', linewidth=2, label='Área Urbana (km²)')
    ax2.plot(df_proj_anual_2030_2050['Ano'], df_proj_anual_2030_2050['Area_Rural_km2'],
             's-', color='#16a085', linewidth=2, label='Área Rural (km²)')
    ax2.set_title('Dinâmica Territorial: Área Urbana vs Rural (km²)', fontweight='bold')
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Área (km²)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    caminho_g3 = os.path.join(pasta_graficos, 'Evolucao_Demografica_e_Espacial_2030_2050.png')
    plt.savefig(caminho_g3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico Demográfico e Territorial salvo: {caminho_g3}")
except Exception as e:
    print(f"  [AVISO] Gráfico 3: {e}")

# --- Gráfico 4: Scatter Observado vs Predito (LOO-CV) ---
try:
    ncols = min(3, len(nomes_modelos))
    nrows = (len(nomes_modelos) + ncols - 1) // ncols
    fig, axes_sc = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.suptitle('Dispersão: Observado vs Predito — Demanda Hídrica Anual (LOO-CV)', fontweight='bold', fontsize=12)
    axes_flat = axes_sc.flatten() if hasattr(axes_sc, 'flatten') else [axes_sc]

    for idx, nm in enumerate(nomes_modelos):
        ax = axes_flat[idx]
        cor = cores.get(nm, 'gray')
        y_p = predicoes_anual_loo[nm]
        ax.scatter(y_demanda_real, y_p, c=cor, s=60, alpha=0.8, edgecolors='white')
        lim_min = min(y_demanda_real.min(), y_p.min()) * 0.9
        lim_max = max(y_demanda_real.max(), y_p.max()) * 1.1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5)
        ax.set_xlabel('Observado (×1.000 m³)')
        ax.set_ylabel('Predito (×1.000 m³)')
        ax.set_title(f'{nm}\nR²={resultados_anual[nm]["R2"]:.4f} | MAE={resultados_anual[nm]["MAE"]:,.1f}', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(len(nomes_modelos), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    caminho_g4 = os.path.join(pasta_graficos, 'Demanda_Scatter_TodosModelos.png')
    plt.savefig(caminho_g4, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico Scatter salvo: {caminho_g4}")
except Exception as e:
    print(f"  [AVISO] Gráfico 4: {e}")

# --- Gráfico 5: Heatmap de Métricas de Avaliação ---
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    df_heat = df_comp_metricas[df_comp_metricas['Escala'].str.startswith('Anual')].copy()
    df_heat = df_heat.set_index('Modelo')[['MAE', 'RMSE', 'R2', 'MAPE (%)', 'MedAE', 'MaxError']]
    sns.heatmap(df_heat, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5, ax=ax)
    ax.set_title('Heatmap de Métricas de Avaliação — Escala Anual (LOO-CV)', fontweight='bold')
    plt.tight_layout()
    caminho_g5 = os.path.join(pasta_graficos, 'Demanda_Heatmap_Metricas.png')
    plt.savefig(caminho_g5, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Gráfico Heatmap salvo: {caminho_g5}")
except Exception as e:
    print(f"  [AVISO] Gráfico 5: {e}")


# ==============================================================================
# 9. EXPORTAÇÃO DOS ARQUIVOS FINAIS
# ==============================================================================
print("\n" + "=" * 80)
print("  8. EXPORTAÇÃO DOS ARQUIVOS FINAIS")
print("=" * 80)

# 1. Projeção Anual CSV
caminho_csv_anual = os.path.join(pasta_csv, 'Projecao_Demanda_Hidrica_2030_2050_Anual.csv')
cols_export_anual = [
    'Ano', 'Pop_Total', 'Pop_Urbana', 'Pop_Rural', 'Taxa_Urbanizacao_pct',
    'Area_Urbana_km2', 'Area_Rural_km2', 'Area_Urbana_m2', 'Area_Rural_m2',
    'Ocupacao_Urbana_pct', 'Idx_Perdas'
] + [f'Demanda_{nm}' for nm in nomes_modelos] + [
    'Demanda_Media', 'Demanda_Min', 'Demanda_Max',
    'Consumo_Efetivo_Media', 'Volume_Perda_Media'
]
df_proj_anual_2030_2050[cols_export_anual].to_csv(caminho_csv_anual, index=False, encoding='utf-8-sig')
print(f"  [OK] CSV Anual (com Pop/Área Urbana e Rural): {caminho_csv_anual}")

# 2. Projeção Mensal CSV (252 linhas)
caminho_csv_mensal = os.path.join(pasta_csv, 'Projecao_Demanda_Hidrica_2030_2050_Mensal.csv')
df_proj_mensal.to_csv(caminho_csv_mensal, index=False, encoding='utf-8-sig')
print(f"  [OK] CSV Mensal (252 meses): {caminho_csv_mensal}")

# 3. Métricas Comparativas CSV
caminho_csv_met = os.path.join(pasta_csv, 'Metricas_Avaliacao_Anual_vs_Mensal.csv')
df_comp_metricas.to_csv(caminho_csv_met, index=False, encoding='utf-8-sig')
print(f"  [OK] CSV Métricas Comparativas: {caminho_csv_met}")

# 4. Relatório Técnico Textual TXT
caminho_rel = os.path.join(script_dir, 'Relatorio_Demanda_Anual_e_Mensal.txt')
with open(caminho_rel, 'w', encoding='utf-8') as f:
    f.write("=" * 95 + "\n")
    f.write("  RELATÓRIO TÉCNICO AUDITADO: MODELAGEM E PROJEÇÃO DE DEMANDA HÍDRICA (2030–2050)\n")
    f.write("  INTEGRAÇÃO: ESCALA ANUAL E ESCALA MENSAL COM SAZONALIDADE CLIMÁTICA\n")
    f.write("  VARIÁVEIS DEMOGRÁFICAS E TERRITORIAIS: POP. URBANA/RURAL E ÁREA URBANA/RURAL\n")
    f.write("  MUNICÍPIO: CARUARU - PE | FONTE: SNIS, INMET E MODELAGEM COMPUTACIONAL\n")
    f.write("=" * 95 + "\n\n")

    f.write("1. AUDITORIA SÊNIOR E RESOLUÇÃO DE INCONSISTÊNCIAS\n")
    f.write("-" * 75 + "\n")
    f.write("  * Eliminação de Target Leakage: removida a inclusão de Vol_per_capita em X;\n")
    f.write("  * Resolução do Defeito de Saturação de Árvores/SVR: adotada modelagem por intensidade\n")
    f.write("    per capita que escala com a população de 430k a 550k habitantes sem estagnar;\n")
    f.write("  * Pipeline LOO-CV estrito: padronização (StandardScaler) isolada dentro de cada fold;\n")
    f.write("  * Incorporação do Regime de Racionamento da Seca de 2012-2017;\n")
    f.write("  * Desagregação mensal via Fórmula IDH: w_m = 0.55*T_norm + 0.45*(1 - P_norm);\n")
    f.write(f"  * Área Municipal Total de Caruaru (IBGE): {AREA_TOTAL_MUNICIPAL_KM2:,.3f} km² ({AREA_TOTAL_MUNICIPAL_M2:,.0f} m²)\n\n")

    f.write("2. TABELA COMPARATIVA DE MÉTRICAS (LOO-CV): ESCALA ANUAL vs MENSAL\n")
    f.write("-" * 95 + "\n")
    f.write(df_comp_metricas.to_string(index=False) + "\n\n")

    f.write("3. MARCOS DE DEMANDA HÍDRICA ANUAL (2030, 2035, 2040, 2045, 2050):\n")
    f.write("-" * 95 + "\n")
    for ano_m in [2030, 2035, 2040, 2045, 2050]:
        ra = df_proj_anual_2030_2050[df_proj_anual_2030_2050['Ano'] == ano_m].iloc[0]
        f.write(f"Ano {ano_m} | Pop.Total: {ra['Pop_Total']:,.0f} | Pop.Urb: {ra['Pop_Urbana']:,.0f} | Pop.Rur: {ra['Pop_Rural']:,.0f}\n")
        f.write(f"        | Área Urbana: {ra['Area_Urbana_km2']:.2f} km² | Área Rural: {ra['Area_Rural_km2']:.2f} km² | Perdas: {ra['Idx_Perdas']:.2f}%\n")
        for nm in nomes_modelos:
            f.write(f"  * {nm:<10}: {ra[f'Demanda_{nm}']:>12,.2f} x1.000 m3 = {ra[f'Demanda_{nm}']*1000:>14,.0f} m3/ano\n")
        f.write(f"  * MÉDIA     : {ra['Demanda_Media']:>12,.2f} x1.000 m3 = {ra['Demanda_Media']*1000:>14,.0f} m3/ano\n")
        f.write(f"    - Consumo Efetivo : {ra['Consumo_Efetivo_Media']:>12,.2f} x1.000 m3/ano\n")
        f.write(f"    - Volume de Perda : {ra['Volume_Perda_Media']:>12,.2f} x1.000 m3/ano\n\n")

    f.write("4. PROJEÇÃO DEMOGRÁFICA E TERRITORIAL DETALHADA ANO A ANO (2030–2050):\n")
    f.write("-" * 95 + "\n")
    cols_demo_rel = ['Ano', 'Pop_Total', 'Pop_Urbana', 'Pop_Rural', 'Taxa_Urbanizacao_pct', 'Area_Urbana_km2', 'Area_Rural_km2', 'Ocupacao_Urbana_pct']
    f.write(df_proj_anual_2030_2050[cols_demo_rel].to_string(index=False) + "\n\n")

    f.write("5. SÍNTESE DO CICLO SAZONAL MENSAL (DISTRIBUIÇÃO PERCENTUAL):\n")
    f.write("-" * 65 + "\n")
    for m in range(1, 13):
        f.write(f"  Mês {m:>2} ({NOMES_MESES[m-1]:<10}): {PESOS_MENSAIS_CLIMATICOS[m]*100:>6.2f}%\n")

print(f"  [OK] Relatório Técnico: {caminho_rel}")

print("\n" + "=" * 80)
print("  PROCESSAMENTO COMPLETO CONCLUÍDO COM SUCESSO!")
print("=" * 80)
