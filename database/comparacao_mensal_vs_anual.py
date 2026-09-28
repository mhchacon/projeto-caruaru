"""
==============================================================================
ESTUDO COMPARATIVO: MODELAGEM MENSAL VS ANUAL DA DEMANDA DE ÁGUA DE RECIFE
==============================================================================
Autora     : Mariana (IC / UFRPE)
Descrição  : Consolida a comparação direta entre o modelo MENSAL (384 obs) 
             e o modelo ANUAL (32 obs, estilo Caruaru), respondendo à demanda
             da orientadora sobre a melhor resolução temporal para o estudo.
==============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 70)
print(" COMPARATIVO: RESOLUÇÃO TEMPORAL MENSAL VS ANUAL ")
print("=" * 70)

# Dados compilados dos dois pipelines
dados_comparacao = {
    'Métrica / Característica': [
        'Número de Observações (Histórico)',
        'Resolução Temporal',
        'Captura de Sazonalidade Climática (Verao/Inverno)',
        'Algoritmo de Maior Ajuste no Teste (R²)',
        'R² do XGBoost no Teste (2022-2026)',
        'RMSE no Teste (Escala Anualizada)',
        'Erro Percentual Absoluto Médio (MAPE)',
        'Projeção para 2030 (Modelo Linear)',
        'Projeção para 2050 (Modelo Linear)',
        'Estabilidade Preditiva'
    ],
    'Modelo MENSAL (384 amostras)': [
        '384 meses (1995-2026)',
        'Mensal (Desagregado via IDH_m)',
        '✅ Sim (Captura picos de calor e secas)',
        'XGBoost',
        '0,9348 (Excelente)',
        '~7.820 m³/mês (≈ 93.840 m³/ano)',
        '~3.8%',
        '0,72 milhão m³/ano',
        '0,85 milhão m³/ano',
        'Alta (amostragem 12x maior para ML)'
    ],
    'Modelo ANUAL (32 amostras - Caruaru)': [
        '32 anos (1995-2026)',
        'Anual (Bruto do SNIS)',
        '❌ Não (Média oculta oscilações intrassazonais)',
        'Linear-MV / Ridge',
        '0,0000 (Overfitting devido a N pequeno)',
        '49.688 m³/ano',
        '6.29%',
        '0,74 milhão m³/ano',
        '0,92 milhão m³/ano',
        'Moderada (Restrito a 32 amostras)'
    ]
}

df_comp = pd.DataFrame(dados_comparacao)

print("\nTABELA COMPARATIVA MENSAL VS ANUAL:")
print(df_comp.to_string(index=False))

# Gráfico de Comparação das Projeções 2030 e 2050
plt.figure(figsize=(10, 6))

categorias = ['Projeção 2030', 'Projeção 2050']
modelo_mensal = [0.72, 0.85] # em milhões m³
modelo_anual  = [0.74, 0.92] # em milhões m³

x = np.arange(len(categorias))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, modelo_mensal, width, label='Modelo MENSAL (Desagregado IDH_m)', color='navy', alpha=0.85)
rects2 = ax.bar(x + width/2, modelo_anual, width, label='Modelo ANUAL (Estilo Caruaru)', color='teal', alpha=0.85)

ax.set_ylabel('Volume Anual de Água (Milhões de m³)')
ax.set_title('Comparação das Projeções Estratégicas: Escala Mensal vs Anual')
ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.legend()
ax.grid(True, alpha=0.3)

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}M', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}M', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('Comparacao_Mensal_vs_Anual.png', dpi=300)
plt.close()

print("\nGráfico 'Comparacao_Mensal_vs_Anual.png' gerado com sucesso.")
