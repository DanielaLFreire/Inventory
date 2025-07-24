# processors/cubagem.py
'''
Processamento de dados de cubagem usando método de Smalian
'''

import pandas as pd
import numpy as np


def processar_cubagem_smalian(df_cubagem):
    '''
    Processa dados de cubagem usando o método de Smalian

    Args:
        df_cubagem: DataFrame com dados de cubagem

    Returns:
        DataFrame com volumes calculados por árvore
    '''
    df = df_cubagem.copy()

    # Converter para numérico
    colunas_num = ['d_cm', 'h_m', 'D_cm', 'H_m']
    for col in colunas_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calcular área seccional (π * d²/4 em m²)
    df['a_m2'] = np.pi * (df['d_cm'] ** 2 / 40000)  # /40000 para converter cm² para m²

    # Ordenar por árvore e altura
    df = df.sort_values(['arv', 'talhao', 'h_m']).reset_index(drop=True)

    # Aplicar método de Smalian
    volumes_list = []

    for (talhao, arv), grupo in df.groupby(['talhao', 'arv']):
        grupo = grupo.sort_values('h_m').reset_index(drop=True)

        for i in range(len(grupo)):
            row = grupo.iloc[i].copy()

            if i > 0:
                # Calcular volume da seção usando Smalian
                row['a1'] = grupo.iloc[i - 1]['a_m2']  # Área da seção inferior
                row['h1'] = grupo.iloc[i - 1]['h_m']  # Altura da seção inferior
                row['a2'] = grupo.iloc[i]['a_m2']  # Área da seção superior
                row['h2'] = grupo.iloc[i]['h_m']  # Altura da seção superior
                row['delta_h'] = row['h2'] - row['h1']  # Comprimento da seção

                # Fórmula de Smalian: V = ((A1 + A2) / 2) * L
                row['va_m3'] = ((row['a1'] + row['a2']) / 2) * row['delta_h']
            else:
                # Primeira seção (toco) - não tem volume anterior
                row['va_m3'] = np.nan

            volumes_list.append(row)

    df_volumes = pd.DataFrame(volumes_list)

    # Identificar seções do toco
    df_volumes['secao_tipo'] = df_volumes['h_m'].apply(
        lambda x: 'Toco' if abs(x - 0.1) < 0.05 else 'Seção'
    )

    # Calcular volume total por árvore (excluindo toco)
    volumes_arvore = df_volumes[
        (df_volumes['va_m3'].notna()) &
        (df_volumes['secao_tipo'] != 'Toco')
        ].groupby(['arv', 'talhao', 'D_cm', 'H_m']).agg({
        'va_m3': 'sum'
    }).reset_index()

    # Renomear coluna de volume
    volumes_arvore['V'] = volumes_arvore['va_m3']
    volumes_arvore = volumes_arvore.drop('va_m3', axis=1)

    # Limpar valores inválidos
    volumes_arvore = volumes_arvore[
        (volumes_arvore['V'] > 0) &
        (volumes_arvore['D_cm'] > 0) &
        (volumes_arvore['H_m'] > 1.3)
        ]

    return volumes_arvore


def calcular_estatisticas_cubagem(volumes_arvore):
    '''
    Calcula estatísticas da cubagem

    Args:
        volumes_arvore: DataFrame com volumes por árvore

    Returns:
        dict: Estatísticas da cubagem
    '''
    stats = {
        'total_arvores': len(volumes_arvore),
        'volume_total': volumes_arvore['V'].sum(),
        'volume_medio': volumes_arvore['V'].mean(),
        'volume_min': volumes_arvore['V'].min(),
        'volume_max': volumes_arvore['V'].max(),
        'dap_medio': volumes_arvore['D_cm'].mean(),
        'altura_media': volumes_arvore['H_m'].mean(),
        'cv_volume': (volumes_arvore['V'].std() / volumes_arvore['V'].mean()) * 100
    }

    return stats


def validar_dados_cubagem(df_cubagem):
    '''
    Valida dados de cubagem antes do processamento

    Args:
        df_cubagem: DataFrame com dados de cubagem

    Returns:
        tuple: (bool válido, list mensagens)
    '''
    mensagens = []
    valido = True

    # Verificar colunas obrigatórias
    colunas_obrigatorias = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']
    colunas_faltantes = [col for col in colunas_obrigatorias if col not in df_cubagem.columns]

    if colunas_faltantes:
        mensagens.append(f"Colunas obrigatórias faltantes: {colunas_faltantes}")
        valido = False

    # Verificar se existem dados
    if len(df_cubagem) == 0:
        mensagens.append("Arquivo de cubagem está vazio")
        valido = False
        return valido, mensagens

    # Verificar dados numéricos
    colunas_numericas = ['d_cm', 'h_m', 'D_cm', 'H_m']
    for col in colunas_numericas:
        if col in df_cubagem.columns:
            try:
                pd.to_numeric(df_cubagem[col], errors='coerce')
            except:
                mensagens.append(f"Coluna {col} contém dados não numéricos")
                valido = False

    # Verificar árvores com múltiplas medições
    if 'arv' in df_cubagem.columns and 'talhao' in df_cubagem.columns:
        arvores_medicoes = df_cubagem.groupby(['talhao', 'arv']).size()
        arvores_unicas = (arvores_medicoes == 1).sum()

        if arvores_unicas > len(arvores_medicoes) * 0.8:
            mensagens.append("Muitas árvores com apenas uma medição - verificar dados")

    # Verificar valores extremos
    if 'd_cm' in df_cubagem.columns:
        d_cm_numeric = pd.to_numeric(df_cubagem['d_cm'], errors='coerce')
        if d_cm_numeric.max() > 200 or d_cm_numeric.min() < 0:
            mensagens.append("Diâmetros com valores extremos detectados")

    if 'h_m' in df_cubagem.columns:
        h_m_numeric = pd.to_numeric(df_cubagem['h_m'], errors='coerce')
        if h_m_numeric.max() > 50 or h_m_numeric.min() < 0:
            mensagens.append("Alturas com valores extremos detectados")

    return valido, mensagens


def gerar_relatorio_cubagem(volumes_arvore):
    '''
    Gera relatório detalhado da cubagem

    Args:
        volumes_arvore: DataFrame com volumes calculados

    Returns:
        str: Relatório formatado
    '''
    stats = calcular_estatisticas_cubagem(volumes_arvore)

    relatorio = f'''
## RELATÓRIO DE CUBAGEM - MÉTODO DE SMALIAN

### Resumo Geral
- **Total de árvores cubadas**: {stats['total_arvores']}
- **Volume total**: {stats['volume_total']:.3f} m³
- **Volume médio por árvore**: {stats['volume_medio']:.4f} m³

### Estatísticas de Volume
- **Mínimo**: {stats['volume_min']:.4f} m³
- **Máximo**: {stats['volume_max']:.4f} m³
- **Coeficiente de variação**: {stats['cv_volume']:.1f}%

### Características Dendrométricas
- **DAP médio**: {stats['dap_medio']:.1f} cm
- **Altura média**: {stats['altura_media']:.1f} m

### Método Utilizado
**Fórmula de Smalian**: V = ((A₁ + A₂) / 2) × L

Onde:
- A₁ = Área da seção inferior
- A₂ = Área da seção superior  
- L = Comprimento da seção

### Observações
- Volumes calculados excluindo a seção do toco (0,1 m)
- Áreas seccionais calculadas a partir dos diâmetros medidos
- Dados validados para consistência antes do processamento
'''

    return relatorio


def exportar_volumes_detalhados(df_volumes_detalhado, volumes_arvore):
    '''
    Prepara dados detalhados para exportação

    Args:
        df_volumes_detalhado: DataFrame com detalhes por seção
        volumes_arvore: DataFrame com volumes totais por árvore

    Returns:
        tuple: (DataFrame seções, DataFrame árvores)
    '''
    # Preparar dados de seções
    df_secoes = df_volumes_detalhado[
        ['arv', 'talhao', 'h_m', 'd_cm', 'a_m2', 'va_m3', 'secao_tipo']
    ].copy()

    df_secoes = df_secoes.rename(columns={
        'arv': 'Árvore',
        'talhao': 'Talhão',
        'h_m': 'Altura (m)',
        'd_cm': 'Diâmetro (cm)',
        'a_m2': 'Área Seccional (m²)',
        'va_m3': 'Volume Seção (m³)',
        'secao_tipo': 'Tipo Seção'
    })

    # Preparar dados de árvores
    df_arvores = volumes_arvore.copy()
    df_arvores = df_arvores.rename(columns={
        'arv': 'Árvore',
        'talhao': 'Talhão',
        'D_cm': 'DAP (cm)',
        'H_m': 'Altura Total (m)',
        'V': 'Volume Total (m³)'
    })

    return df_secoes, df_arvores