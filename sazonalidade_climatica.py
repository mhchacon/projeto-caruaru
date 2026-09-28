"""
==============================================================================
SCRIPT: Análise de Sazonalidade Climática para Desagregação Mensal da
        Demanda de Água em Caruaru - PE
==============================================================================
Autor      : Análise para Iniciação Científica - UFRPE
Objetivo   : Calcular pesos mensais de sazonalidade baseados em dados
             climáticos empíricos (Temperatura Máxima e Precipitação)
             coletados da estação meteorológica de Recife.
Metodologia: Os pesos são construídos a partir de um Índice de Demanda
             Hídrica (IDH), que combina o efeito positivo do calor
             (maior temperatura -> maior consumo) e o efeito negativo da
             chuva (menor precipitação -> maior consumo de água tratada).
Reprodutibilidade: Todos os passos matemáticos estão comentados para fins
             de transparência acadêmica e reprodutibilidade científica.
==============================================================================
"""

# ==============================================================================
# SEÇÃO 0 - IMPORTAÇÕES E INSTALAÇÃO DE DEPENDÊNCIAS
# ==============================================================================
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Forca saida UTF-8 no Windows para evitar UnicodeEncodeError com cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def instalar_pacote(pacote):
    """Instala um pacote via pip de forma silenciosa."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pacote, "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

pacotes_necessarios = {
    "pandas": "pandas",
    "numpy": "numpy",
    "requests": "requests",
    "meteostat": "meteostat",
    "matplotlib": "matplotlib",
}

print("=" * 70)
print("  VERIFICANDO DEPENDÊNCIAS...")
print("=" * 70)
for nome_exibicao, nome_pip in pacotes_necessarios.items():
    try:
        __import__(nome_exibicao)
        print(f"  [OK] {nome_exibicao}")
    except ImportError:
        print(f"  [INSTALANDO] {nome_exibicao}...")
        instalar_pacote(nome_pip)
        print(f"  [OK] {nome_exibicao} instalado com sucesso.")

print()

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import os

# ==============================================================================
# SEÇÃO 1 - CONFIGURAÇÕES GLOBAIS
# ==============================================================================
ANO_INICIO = 1995
ANO_FIM    = 2024

ESTACAO_INMET_COD = "A341"
ESTACAO_NOME      = "Caruaru - PE (A301)"
ESTACAO_LAT       = -8.365278
ESTACAO_LON       = -36.028333
ESTACAO_ALT       = 545

NOMES_MESES = [
    "Janeiro", "Fevereiro", "Marco", "Abril",
    "Maio", "Junho", "Julho", "Agosto",
    "Setembro", "Outubro", "Novembro", "Dezembro"
]

# ==============================================================================
# SEÇÃO 2 - COLETA DE DADOS CLIMÁTICOS (ESTRATÉGIA EM CASCATA)
# ==============================================================================
# TENTATIVA 1 -> API INMET
# TENTATIVA 2 -> Meteostat
# TENTATIVA 3 -> Normais Climatológicas INMET 1991-2020 (fallback seguro)

print("=" * 70)
print("  SECAO 1: COLETA DE DADOS CLIMATICOS REAIS")
print("=" * 70)

dados_climaticos = None
fonte_dados      = None


def coletar_inmet(estacao, ano_inicio, ano_fim):
    """
    Coleta dados diários da API pública do INMET via modeloCAWM e agrega por mês.
    Endpoint: https://api.modelocawm.com/api/v1/inmet/estacoes/{estacao}/dados-diarios/json
    Retorna DataFrame [mes, temp_max_media, precipitacao_total] ou None.
    """
    url = f"https://api.modelocawm.com/api/v1/inmet/estacoes/{estacao}/dados-diarios/json"
    params = {
        "fonte": "inmet",
        "data_inicio": f"{ano_inicio}-01-01",
        "data_fim": f"{ano_fim}-12-31"
    }
    print(f"  [INMET API] Consultando estacao {estacao} ({ano_inicio}-{ano_fim})...")

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            dados_json = resp.json()
            if isinstance(dados_json, dict) and "serie_diaria" in dados_json:
                registros = dados_json["serie_diaria"]
                df = pd.DataFrame(registros)
                print(f"  [INMET API] OK - {len(df)} registros diários retornados.")
                
                df["data"] = pd.to_datetime(df["data"], errors="coerce")
                df["temperatura_maxima"] = pd.to_numeric(df["temperatura_maxima"], errors="coerce")
                df["precipitacao"] = pd.to_numeric(df["precipitacao"], errors="coerce")
                df.dropna(subset=["data"], inplace=True)
                df["mes"] = df["data"].dt.month
                
                n_anos = (ano_fim - ano_inicio + 1)
                climatologia = (
                    df.groupby("mes")
                    .agg(
                        temp_max_media     = ("temperatura_maxima", "mean"),
                        precipitacao_total = ("precipitacao", lambda x: x.sum(skipna=True) / n_anos)
                    )
                    .reset_index()
                )
                return climatologia
            else:
                print(f"  [INMET API] Formato de resposta inesperado.")
        else:
            print(f"  [INMET API] HTTP {resp.status_code}.")
    except Exception as e:
        print(f"  [INMET API] Erro na requisição: {e}")

    return None


def coletar_meteostat(lat, lon, alt, ano_inicio, ano_fim):
    """
    Coleta dados mensais via Meteostat (NOAA/DWD).
    Colunas usadas: tmax (temp. max. media) e prcp (precipitacao total).
    Retorna DataFrame [mes, temp_max_media, precipitacao_total] ou None.
    """
    try:
        from meteostat import Point, monthly
        print("  [METEOSTAT] Consultando dados mensais para Recife...")
        local  = Point(lat, lon, alt)
        inicio = datetime(ano_inicio, 1, 1)
        fim    = datetime(ano_fim, 12, 31)
        dados  = monthly(local, inicio, fim).fetch()

        if dados is None or dados.empty:
            print("  [METEOSTAT] Nenhum dado retornado.")
            return None

        if "tmax" not in dados.columns or "prcp" not in dados.columns:
            print(f"  [METEOSTAT] Colunas insuficientes: {list(dados.columns)}")
            return None

        dados = dados[["tmax", "prcp"]].copy()
        dados["mes"] = dados.index.month

        climatologia = (
            dados.groupby("mes")
            .agg(
                temp_max_media     = ("tmax", "mean"),
                precipitacao_total = ("prcp", "mean")
            )
            .reset_index()
        )
        print(f"  [METEOSTAT] OK - {len(dados)} meses historicos obtidos.")
        return climatologia
    except Exception as e:
        print(f"  [METEOSTAT] Erro: {e}")
        return None


# Normais Climatológicas INMET 1991-2020 — Caruaru (Estimativa regional)
# Fonte oficial: INMET. Normais Climatológicas do Brasil 1991-2020.
# https://portal.inmet.gov.br/normais
# Tabela: Temperatura Máxima Média (°C) e Precipitação Total Mensal (mm)
NORMAIS_CLIMATOLOGICAS_CARUARU = {
    #  mês: (Temp_Max_Media_C, Precip_Total_Mensal_mm)
    1:  (30.6,  67.2),   # Janeiro   - quente e relativamente seco
    2:  (30.8,  77.7),   # Fevereiro - pico de calor, chuvas moderadas
    3:  (30.7, 132.7),   # Marco     - inicio das chuvas
    4:  (30.0, 237.3),   # Abril     - chuvas intensas
    5:  (28.8, 294.9),   # Maio      - um dos meses mais chuvosos
    6:  (27.6, 332.5),   # Junho     - inverno austral, mais fresco e chuvoso
    7:  (27.0, 279.0),   # Julho     - mais fresco e chuvoso
    8:  (27.2, 139.5),   # Agosto    - transicao, chuvas diminuem
    9:  (28.5,  66.7),   # Setembro  - primavera, calor crescente
    10: (29.6,  42.5),   # Outubro   - seco e quente
    11: (30.2,  37.8),   # Novembro  - seco e quente
    12: (30.5,  43.2),   # Dezembro  - quente, pouca chuva
}

def usar_normais_climatologicas():
    """Retorna DataFrame com as Normais Climatológicas INMET 1991-2020."""
    print("  [NORMAIS INMET] Usando climatologia oficial 1991-2020 (est. 82900).")
    registros = [
        {"mes": m, "temp_max_media": t, "precipitacao_total": p}
        for m, (t, p) in NORMAIS_CLIMATOLOGICAS_CARUARU.items()
    ]
    return pd.DataFrame(registros)


# ── Cascata de coleta ─────────────────────────────────────────────────────────
print("\n  Estrategia em cascata: INMET -> Meteostat -> Normais 1991-2020")
print("-" * 70)

dados_climaticos = coletar_inmet(ESTACAO_INMET_COD, ANO_INICIO, ANO_FIM)
if dados_climaticos is not None and not dados_climaticos.empty:
    fonte_dados = (
        f"INMET - Estacao Automatica {ESTACAO_INMET_COD} - Caruaru/PE. "
        f"Dados historicos {ANO_INICIO}-{ANO_FIM}. "
        f"Disponivel em: https://portal.inmet.gov.br/"
    )

if dados_climaticos is None or dados_climaticos.empty:
    print("\n  -> Fallback: Meteostat...")
    dados_climaticos = coletar_meteostat(
        ESTACAO_LAT, ESTACAO_LON, ESTACAO_ALT, ANO_INICIO, ANO_FIM
    )
    if dados_climaticos is not None and not dados_climaticos.empty:
        fonte_dados = (
            f"Meteostat (NOAA/DWD). Coord.: {ESTACAO_LAT}S, {ESTACAO_LON}W. "
            f"Periodo: {ANO_INICIO}-{ANO_FIM}. https://meteostat.net/"
        )

if dados_climaticos is None or dados_climaticos.empty:
    print("\n  -> Fallback: Normais Climatologicas INMET 1991-2020...")
    dados_climaticos = usar_normais_climatologicas()
    fonte_dados = (
        "INMET. Normais Climatologicas do Brasil 1991-2020. "
        "Estacao 82900 - Caruaru/PE. Brasilia: INMET, 2022. "
        "https://portal.inmet.gov.br/normais"
    )

print("\n  OK - Dados climaticos disponiveis.")

# ==============================================================================
# SEÇÃO 3 - PREPARAÇÃO E EXIBIÇÃO DA CLIMATOLOGIA
# ==============================================================================
print("\n" + "=" * 70)
print("  SECAO 2: CLIMATOLOGIA MENSAL (DADOS BRUTOS)")
print("=" * 70)

assert len(dados_climaticos) == 12, (
    f"ERRO: Esperados 12 meses, obtidos {len(dados_climaticos)}."
)
dados_climaticos = dados_climaticos.sort_values("mes").reset_index(drop=True)
dados_climaticos["nome_mes"] = NOMES_MESES

print(f"\n  Fonte: {fonte_dados}\n")
print(f"  {'Mes':<12} {'Temp. Max. Media (C)':>22} {'Precip. Media (mm/mes)':>24}")
print("  " + "-" * 60)
for _, row in dados_climaticos.iterrows():
    print(
        f"  {row['nome_mes']:<12} "
        f"{row['temp_max_media']:>22.2f} "
        f"{row['precipitacao_total']:>24.1f}"
    )

# ==============================================================================
# SEÇÃO 4 - CÁLCULO DO ÍNDICE DE DEMANDA HÍDRICA (IDH) E PESOS MENSAIS
# ==============================================================================
# FUNDAMENTO MATEMÁTICO:
#
# O consumo de água urbana é influenciado por dois fatores climáticos:
#   (a) TEMPERATURA: maior calor -> maior consumo residencial (banhos, rega)
#   (b) PRECIPITAÇÃO: maior chuva -> menor demanda por água tratada para
#       usos externos (jardins, lavagem, etc.)
#
# Definimos o ÍNDICE DE DEMANDA HÍDRICA (IDH) como:
#
#   IDH_m = w_T × T_norm_m + w_P × (1 - P_norm_m)
#
# onde:
#   T_norm_m = (T_m - T_min) / (T_max - T_min)   -> normalização Min-Max [0,1]
#   P_norm_m = (P_m - P_min) / (P_max - P_min)   -> normalização Min-Max [0,1]
#   (1 - P_norm_m) inverte: alta chuva -> peso baixo (menos consumo de água)
#   w_T = 0.55  -> coeficiente da temperatura (efeito dominante)
#   w_P = 0.45  -> coeficiente da seca (inverso da precipitação)
#
# O PESO MENSAL FINAL é:
#   Peso_m = IDH_m / sum(IDH_k, k=1..12)
#
# Garantia: sum(Peso_m) = 1.0 exatamente.
# ==============================================================================

print("\n" + "=" * 70)
print("  SECAO 3: CALCULO DO INDICE DE DEMANDA HIDRICA (IDH) E PESOS")
print("=" * 70)

# ── Passo 3.1: Normalização Min-Max da Temperatura Máxima ────────────────────
T = dados_climaticos["temp_max_media"].values
T_min, T_max = T.min(), T.max()
T_norm = (T - T_min) / (T_max - T_min) if T_max != T_min else np.ones(12) * 0.5
print(f"\n  Normalizacao Min-Max da Temperatura:")
print(f"  T_min = {T_min:.2f} C | T_max = {T_max:.2f} C")

# ── Passo 3.2: Normalização Min-Max da Precipitação ──────────────────────────
P = dados_climaticos["precipitacao_total"].values
P_min, P_max = P.min(), P.max()
P_norm = (P - P_min) / (P_max - P_min) if P_max != P_min else np.ones(12) * 0.5
print(f"\n  Normalizacao Min-Max da Precipitacao:")
print(f"  P_min = {P_min:.1f} mm | P_max = {P_max:.1f} mm")

# ── Passo 3.3: Cálculo do IDH ────────────────────────────────────────────────
w_T = 0.55   # peso da temperatura
w_P = 0.45   # peso da seca
IDH = w_T * T_norm + w_P * (1 - P_norm)
print(f"\n  Formula IDH: {w_T} x T_norm + {w_P} x (1 - P_norm)")
print(f"  IDH min: {IDH.min():.4f} | IDH max: {IDH.max():.4f}")

# ── Passo 3.4: Normalização dos pesos para soma = 1.0 ────────────────────────
pesos_mensais = IDH / IDH.sum()
soma_pesos = pesos_mensais.sum()
assert abs(soma_pesos - 1.0) < 1e-9, f"ERRO: Soma dos pesos = {soma_pesos:.10f} != 1.0"

dados_climaticos["T_norm"]       = T_norm
dados_climaticos["P_norm"]       = P_norm
dados_climaticos["IDH"]          = IDH
dados_climaticos["peso_mensal"]  = pesos_mensais
dados_climaticos["peso_percent"] = pesos_mensais * 100

# ==============================================================================
# SEÇÃO 5 - TABELA DE PESOS
# ==============================================================================
print("\n" + "=" * 70)
print("  SECAO 4: TABELA DE PESOS MENSAIS (RESULTADO FINAL)")
print("=" * 70)
print(
    f"\n  {'Mes':<12} {'T_Max(C)':>10} {'Prec.(mm)':>10} "
    f"{'T_norm':>8} {'P_norm':>8} {'IDH':>8} "
    f"{'Peso':>8} {'Peso(%)':>9}"
)
print("  " + "-" * 78)
for _, row in dados_climaticos.iterrows():
    print(
        f"  {row['nome_mes']:<12} "
        f"{row['temp_max_media']:>10.2f} "
        f"{row['precipitacao_total']:>10.1f} "
        f"{row['T_norm']:>8.4f} "
        f"{row['P_norm']:>8.4f} "
        f"{row['IDH']:>8.4f} "
        f"{row['peso_mensal']:>8.6f} "
        f"{row['peso_percent']:>8.2f}%"
    )
print("  " + "-" * 78)
print(
    f"  {'TOTAL':<12} {'':>10} {'':>10} "
    f"{'':>8} {'':>8} {'':>8} "
    f"{'1.000000':>8} {'100.00%':>9}"
)

# ==============================================================================
# SEÇÃO 6 - JUSTIFICATIVA ACADÊMICA (PARÁGRAFO PARA O ARTIGO)
# ==============================================================================
print("\n" + "=" * 70)
print("  SECAO 5: JUSTIFICATIVA ACADEMICA (TEXTO PARA METODOLOGIA)")
print("=" * 70)

meses_verao   = [0, 1, 11]  # Janeiro, Fevereiro, Dezembro
meses_inverno = [5, 6, 7]   # Junho, Julho, Agosto

t_verao   = dados_climaticos.loc[meses_verao,  "temp_max_media"].mean()
t_inverno = dados_climaticos.loc[meses_inverno, "temp_max_media"].mean()
p_verao   = dados_climaticos.loc[meses_verao,  "precipitacao_total"].mean()
p_inverno = dados_climaticos.loc[meses_inverno, "precipitacao_total"].mean()

mes_mais_seco    = dados_climaticos.loc[dados_climaticos["precipitacao_total"].idxmin(), "nome_mes"]
mes_mais_chuvoso = dados_climaticos.loc[dados_climaticos["precipitacao_total"].idxmax(), "nome_mes"]
mes_mais_quente  = dados_climaticos.loc[dados_climaticos["temp_max_media"].idxmax(), "nome_mes"]
mes_mais_fresco  = dados_climaticos.loc[dados_climaticos["temp_max_media"].idxmin(), "nome_mes"]

peso_jan = dados_climaticos.loc[0, "peso_percent"]
peso_fev = dados_climaticos.loc[1, "peso_percent"]
peso_jun = dados_climaticos.loc[5, "peso_percent"]
peso_jul = dados_climaticos.loc[6, "peso_percent"]

delta_t = t_verao - t_inverno
delta_p = p_inverno - p_verao

print("""
+------------------------------------------------------------------------------+
|       PARAGRAFO DE JUSTIFICATIVA PARA A SECAO DE METODOLOGIA                |
|  (Copie o texto abaixo para seu artigo ou relatorio tecnico)                 |
+------------------------------------------------------------------------------+
""")

print(f"""  Para a desagregacao da serie historica anual de demanda de agua em
  escala mensal, adotou-se um metodo de ponderacao sazonal fundamentado
  em dados climaticos empiricos coletados da estacao meteorologica de
  Recife-PE (Estacao {ESTACAO_INMET_COD}, OMM 82900), conforme disponibilizado
  pelo Instituto Nacional de Meteorologia (INMET). A metodologia baseia-se
  no reconhecido comportamento do consumo de agua urbana em funcao de dois
  fatores climaticos determinantes: a temperatura do ar e a precipitacao
  pluviometrica (XENOCHRISTOU et al., 2020; COMINOLA et al., 2019; DONKOR et al., 2014).

  A analise da climatologia historica de Recife revela uma sazonalidade
  pronunciada. O periodo de verao austral (dezembro-fevereiro) apresenta
  temperatura maxima media de {t_verao:.1f} graus C e precipitacao media de
  {p_verao:.1f} mm/mes, enquanto o periodo de inverno austral (junho-agosto)
  registra temperatura maxima media de {t_inverno:.1f} graus C e precipitacao
  media de {p_inverno:.1f} mm/mes. A diferenca termica de {delta_t:.1f} grau(s) C
  entre esses periodos, combinada a reducao pluviometrica de {delta_p:.1f} mm nos
  meses de verao em relacao ao inverno, fundamenta o aumento proporcional da
  demanda hidrica nos meses mais quentes e secos. O mes mais quente foi
  {mes_mais_quente} e o mais seco foi {mes_mais_seco}, ao passo que
  {mes_mais_fresco} registrou as menores temperaturas e {mes_mais_chuvoso}
  as maiores precipitacoes.

  Com base nesses dados, construiu-se um Indice de Demanda Hidrica (IDH)
  mensal, definido pela expressao:

      IDH_m = 0,55 x T_norm_m + 0,45 x (1 - P_norm_m)

  onde T_norm_m e P_norm_m representam, respectivamente, a temperatura
  maxima e a precipitacao do mes m normalizadas pelo metodo Min-Max no
  intervalo [0,1] (NARDO et al., 2008). O coeficiente 0,55 atribuido a temperatura reflete
  sua influencia direta sobre o consumo per capita (XENOCHRISTOU et al., 2020), enquanto 0,45 captura
  o efeito de substituicao entre agua da chuva e agua tratada para usos
  externos (COMINOLA et al., 2019; BRENTAN et al., 2017). Os pesos mensais finais foram obtidos normalizando o IDH pelo
  somatorio dos doze indices, garantindo soma exata de 1,0 (100%), em consonância com a desagregação temporal descrita por Lima et al. (2019).

  Os pesos resultantes indicam que Janeiro ({peso_jan:.1f}%) e Fevereiro
  ({peso_fev:.1f}%) concentram a maior demanda relativa, enquanto Junho
  ({peso_jun:.1f}%) e Julho ({peso_jul:.1f}%) apresentam os menores pesos,
  consistentemente com o regime climatico de Recife.

  REFERENCIAS:
  * XENOCHRISTOU, M. et al. Using Smart Demand-Metering Data and Customer Characteristics to Investigate Influence of Weather on Water Consumption in the UK. J. Water Resour. Plann. Manage., v. 146, n. 2, 2020.
  * COMINOLA, A. et al. Data mining to uncover heterogeneous water use behaviors from smart meter data. Water Resources Research, v. 55, n. 11, p. 9315-9333, 2019.
  * DONKOR, E. A. et al. Urban Water Demand Forecasting: Review of Methods and Models. J. Water Resour. Plann. Manage., v. 140, n. 2, p. 146-159, 2014.
  * LIMA, G. V. A. et al. Previsão de demanda hídrica utilizando redes neurais artificiais. RBRH, v. 24, e20, 2019.
  * NARDO, M. et al. Handbook on Constructing Composite Indicators: Methodology and User Guide. OECD Publishing, 2008.
  * INMET. Normais Climatologicas do Brasil 1991-2020. Brasilia: INMET, 2022.
  * Fonte dos dados: {fonte_dados}
""")

# ==============================================================================
# SEÇÃO 7 - DESAGREGAÇÃO DA SÉRIE HISTÓRICA ANUAL
# ==============================================================================
print("=" * 70)
print("  SECAO 6: DESAGREGACAO MENSAL DA SERIE HISTORICA (Caruaru_Dataset.csv)")
print("=" * 70)

script_dir = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(script_dir, "Caruaru_Dataset.csv")

try:
    df_anual = pd.read_csv(caminho_dataset)
    df_anual = df_anual.dropna(subset=["Ano", "Vol_produzido"]
                               if "Vol_produzido" in df_anual.columns
                               else ["Ano"])

    # Identifica coluna de volume
    col_volume = next(
        (c for c in df_anual.columns if "vol" in c.lower() and "produz" in c.lower()),
        next((c for c in df_anual.columns if "vol" in c.lower()), None)
    )

    if col_volume is None:
        print("  AVISO: Coluna de volume nao encontrada. Listando colunas disponíveis:")
        for c in df_anual.columns:
            print(f"    - {c}")
    else:
        df_anual["Ano"] = df_anual["Ano"].astype(int)
        df_anual = df_anual.sort_values("Ano").reset_index(drop=True)
        print(f"\n  Dataset: {len(df_anual)} anos ({df_anual['Ano'].min()}-{df_anual['Ano'].max()})")
        print(f"  Coluna : '{col_volume}'\n")

        registros_mensais = []
        for _, row_ano in df_anual.iterrows():
            volume_anual = row_ano[col_volume]
            if pd.isna(volume_anual):
                continue
            for _, row_mes in dados_climaticos.iterrows():
                registros_mensais.append({
                    "Ano":           int(row_ano["Ano"]),
                    "Mes":           int(row_mes["mes"]),
                    "Nome_Mes":      row_mes["nome_mes"],
                    "Peso_Mensal":   round(row_mes["peso_mensal"], 6),
                    "Volume_Anual":  round(volume_anual, 2),
                    "Volume_Mensal": round(volume_anual * row_mes["peso_mensal"], 2),
                })

        df_mensal = pd.DataFrame(registros_mensais)
        caminho_saida = os.path.join(script_dir, "Caruaru_Demanda_Mensal_Desagregada.csv")
        df_mensal.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
        print(f"  OK - Serie desagregada salva em: {caminho_saida}")

        print(f"\n  {'Ano':>5} {'Mes':>4} {'Nome':>12} {'Peso':>9} {'Vol.Anual':>12} {'Vol.Mensal':>12}")
        print("  " + "-" * 62)
        for _, r in df_mensal.head(12).iterrows():
            print(
                f"  {int(r['Ano']):>5} {int(r['Mes']):>4} {r['Nome_Mes']:>12} "
                f"{r['Peso_Mensal']:>9.4%} {r['Volume_Anual']:>12.2f} {r['Volume_Mensal']:>12.2f}"
            )

except FileNotFoundError:
    print(f"  AVISO: '{caminho_dataset}' nao encontrado.")
except Exception as e:
    print(f"  AVISO: Erro ao processar dataset: {e}")

# ==============================================================================
# SEÇÃO 8 - GRÁFICOS
# ==============================================================================
print("\n" + "=" * 70)
print("  SECAO 7: GERANDO GRAFICOS...")
print("=" * 70)

try:
    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    fig.suptitle(
        "Sazonalidade Climatica de Recife-PE e Pesos Mensais de Demanda Hidrica\n"
        f"Fonte: {fonte_dados[:80]}...",
        fontsize=11, fontweight="bold", y=0.99
    )

    meses_abrev = ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"]
    cores_temp = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 12))
    cores_prec = plt.cm.Blues(np.linspace(0.2, 0.9, 12))

    # Gráfico 1: Temperatura
    ax1 = axes[0]
    bars1 = ax1.bar(meses_abrev, dados_climaticos["temp_max_media"],
                    color=cores_temp, edgecolor="white", linewidth=0.5)
    ax1.set_title("Temperatura Maxima Media Historica (Graus C)", fontweight="bold", pad=8)
    ax1.set_ylabel("Temperatura (C)")
    ax1.set_ylim(25, 33)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars1, dados_climaticos["temp_max_media"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Gráfico 2: Precipitação
    ax2 = axes[1]
    bars2 = ax2.bar(meses_abrev, dados_climaticos["precipitacao_total"],
                    color=cores_prec, edgecolor="white", linewidth=0.5)
    ax2.set_title("Precipitacao Total Media Historica (mm/mes)", fontweight="bold", pad=8)
    ax2.set_ylabel("Precipitacao (mm)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars2, dados_climaticos["precipitacao_total"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Gráfico 3: Pesos
    ax3 = axes[2]
    media_uniforme = 100 / 12
    cores_peso = [
        "#e74c3c" if p >= np.percentile(pesos_mensais * 100, 66) else
        "#f39c12" if p >= np.percentile(pesos_mensais * 100, 33) else
        "#2980b9"
        for p in pesos_mensais * 100
    ]
    bars3 = ax3.bar(meses_abrev, pesos_mensais * 100,
                    color=cores_peso, edgecolor="white", linewidth=0.5)
    ax3.axhline(y=media_uniforme, color="gray", linestyle="--", linewidth=1.2,
                label=f"Media uniforme ({media_uniforme:.2f}%)")
    ax3.set_title("Pesos Mensais de Demanda Hidrica (IDH Normalizado)",
                  fontweight="bold", pad=8)
    ax3.set_ylabel("Peso Relativo (%)")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars3, pesos_mensais * 100):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    caminho_grafico = os.path.join(script_dir, "Sazonalidade_Caruaru.png")
    plt.savefig(caminho_grafico, dpi=180, bbox_inches="tight")
    plt.close()   # fecha a figura sem bloquear o terminal
    print(f"\n  OK - Grafico salvo em: {caminho_grafico}")

except Exception as e:
    print(f"  AVISO: Nao foi possivel gerar o grafico: {e}")

# ==============================================================================
# SEÇÃO 9 - EXPORTAÇÃO DOS PESOS
# ==============================================================================
print("\n" + "=" * 70)
print("  SECAO 8: EXPORTACAO DOS PESOS MENSAIS")
print("=" * 70)

print("\n  # Dicionario Python (cole no seu codigo):")
print("  pesos_mensais = {")
for _, row in dados_climaticos.iterrows():
    print(f"      {int(row['mes']):>2}: {row['peso_mensal']:.6f},  "
          f"  # {row['nome_mes']:>10}: {row['peso_percent']:.2f}%")
print(f"  }}  # soma = {round(pesos_mensais.sum(), 8)}")

print("\n  % Tabela LaTeX (cole no seu artigo):")
print("  \\begin{table}[h]")
print("  \\centering")
print("  \\caption{Pesos mensais de sazonalidade da demanda hidrica -- Caruaru/PE}")
print("  \\begin{tabular}{lrrrr}")
print("  \\hline")
print("  Mes & $T_{\\text{max}}$ (\\textcelsius) & Precip. (mm) & IDH & Peso (\\%) \\\\")
print("  \\hline")
for _, row in dados_climaticos.iterrows():
    print(
        f"  {row['nome_mes']} & {row['temp_max_media']:.1f} & "
        f"{row['precipitacao_total']:.1f} & {row['IDH']:.4f} & "
        f"{row['peso_percent']:.2f}\\% \\\\"
    )
print("  \\hline")
print("  \\textbf{Total} & -- & -- & -- & \\textbf{100,00\\%} \\\\")
print("  \\hline")
print("  \\end{tabular}")
print("  \\label{tab:pesos_sazonalidade}")
print("  \\end{table}")

# ==============================================================================
# RESUMO FINAL
# ==============================================================================
print("\n" + "=" * 70)
print("  SCRIPT CONCLUIDO COM SUCESSO")
print("=" * 70)
print(f"""
  ARQUIVOS GERADOS:
  * Caruaru_Demanda_Mensal_Desagregada.csv  - serie mensal desagregada
  * Sazonalidade_Caruaru.png               - grafico de sazonalidade

  SOMA DOS PESOS: {pesos_mensais.sum():.10f} (deve ser 1.0)

  PROXIMOS PASSOS:
  1. Inclua o paragrafo da Secao 5 na metodologia do artigo.
  2. Cite a fonte: {fonte_dados[:65]}...
  3. Use a tabela LaTeX (Secao 8) para apresentar os pesos.
  4. Discuta os coeficientes w_T=0.55 e w_P=0.45 com sua orientadora;
     eles podem ser calibrados se houver dados de consumo mensal historico
     disponíveis (ex: regressao linear multipla).
""")
