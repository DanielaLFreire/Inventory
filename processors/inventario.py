# processors/inventario.py
'''
Processamento completo do inventário florestal
'''

import pandas as pd
import numpy as np
from utils.formatacao import formatar_brasileiro, formatar_numero_inteligente


def processar_inventario_completo(df_inventario, config, melhor_modelo_hip, melhor_modelo_vol):
    '''
    Processa o inventário completo aplicando os melhores modelos

    Args:
        df_inventario: DataFrame do inventário
        config: Configurações de processamento
        melhor_modelo_hip: Nome do melhor modelo hipsométrico
        melhor_modelo_vol: Nome do melhor modelo volumétrico

    Returns:
        dict: Resultados completos do inventário
    '''
    # Filtrar dados
    df_filtrado = aplicar_filtros_inventario(df_inventario, config)

    # Estimar alturas
    df_com_alturas = estimar_alturas_inventario(df_filtrado, melhor_modelo_hip, config)

    # Estimar volumes
    df_com_volumes = estimar_volumes_inventario(df_com_alturas, melhor_modelo_vol)

    # Processar áreas
    df_final = processar_areas_inventario(df_com_volumes, config)

    # Calcular resumos
    resumo_parcelas = calcular_resumo_por_parcela(df_final)
    resumo_talhoes = calcular_resumo_por_talhao(resumo_parcelas)
    estatisticas_gerais = calcular_estatisticas_gerais(resumo_parcelas)

    return {
        'inventario_completo': df_final,
        'resumo_parcelas': resumo_parcelas,
        'resumo_talhoes': resumo_talhoes,
        'estatisticas_gerais': estatisticas_gerais,
        'modelos_utilizados': {
            'hipsometrico': melhor_modelo_hip,
            'volumetrico': melhor_modelo_vol
        }
    }


def aplicar_filtros_inventario(df, config):
    '''
    Aplica filtros aos dados do inventário

    Args:
        df: DataFrame do inventário
        config: Configurações de filtros

    Returns:
        DataFrame filtrado
    '''
    df_filtrado = df.copy()

    # Filtrar talhões excluídos
    if config.get('talhoes_excluir'):
        df_filtrado = df_filtrado[~df_filtrado['talhao'].isin(config['talhoes_excluir'])]

    # Filtrar por diâmetro mínimo
    df_filtrado = df_filtrado[df_filtrado['D_cm'] >= config.get('diametro_min', 4.0)]

    # Filtrar códigos excluídos
    if config.get('codigos_excluir'):
        df_filtrado = df_filtrado[~df_filtrado['cod'].isin(config['codigos_excluir'])]

    # Remover valores inválidos
    df_filtrado = df_filtrado[
        (df_filtrado['D_cm'].notna()) &
        (df_filtrado['D_cm'] > 0)
        ]

    return df_filtrado


def estimar_alturas_inventario(df, melhor_modelo, config):
    '''
    Estima alturas usando o melhor modelo hipsométrico

    Args:
        df: DataFrame com dados do inventário
        melhor_modelo: Nome do melhor modelo hipsométrico
        config: Configurações

    Returns:
        DataFrame com alturas estimadas
    '''
    df = df.copy()

    # Função para estimar altura individual
    def estimar_altura_arvore(row):
        # Se já tem altura, usar a existente
        if pd.notna(row['H_m']) and row['H_m'] > 1.3:
            return row['H_m']

        # Estimar altura baseada no modelo
        try:
            if melhor_modelo == "Curtis":
                # ln(H) = β₀ + β₁ * (1/D)
                return np.exp(3.2 - 8.5 / row['D_cm'])

            elif melhor_modelo == "Campos":
                # ln(H) = β₀ + β₁ * (1/D) + β₂ * ln(H_dom)
                h_dom = row.get('H_dom', 25.0)
                return np.exp(2.8 - 7.2 / row['D_cm'] + 0.7 * np.log(h_dom))

            elif melhor_modelo == "Henri":
                # H = β₀ + β₁ * ln(D)
                return 1.3 + 8.5 * np.log(row['D_cm'])

            elif melhor_modelo == "Prodan":
                # Fórmula inversa baseada na produtividade
                prod = 0.8 * row['D_cm'] + 0.002 * (row['D_cm'] ** 2)
                return (row['D_cm'] ** 2) / prod + 1.3

            elif melhor_modelo == "Chapman":
                # H = b₀ * (1 - exp(-b₁ * D))^b₂
                return 30 * (1 - np.exp(-0.08 * row['D_cm'])) ** 1.2

            elif melhor_modelo == "Weibull":
                # H = a * (1 - exp(-b * D^c))
                return 32 * (1 - np.exp(-0.05 * (row['D_cm'] ** 1.1)))

            elif melhor_modelo == "Mononuclear":
                # H = a * (1 - b * exp(-c * D))
                return 28 * (1 - 0.9 * np.exp(-0.12 * row['D_cm']))

            else:
                # Modelo padrão simples
                return 1.3 + 0.8 * row['D_cm']

        except:
            # Fallback: relação linear simples
            return 1.3 + 0.8 * row['D_cm']

    # Aplicar estimativa de altura
    df['H_est'] = df.apply(estimar_altura_arvore, axis=1)

    # Garantir valores mínimos
    df['H_est'] = df['H_est'].clip(lower=1.5)

    return df


def estimar_volumes_inventario(df, melhor_modelo):
    '''
    Estima volumes usando o melhor modelo volumétrico

    Args:
        df: DataFrame com alturas estimadas
        melhor_modelo: Nome do melhor modelo volumétrico

    Returns:
        DataFrame com volumes estimados
    '''
    df = df.copy()

    # Função para estimar volume individual
    def estimar_volume_arvore(row):
        try:
            D = row['D_cm']
            H = row['H_est']

            if D <= 0 or H <= 1.3:
                return 0.0

            if melhor_modelo == 'Schumacher':
                # ln(V) = β₀ + β₁*ln(D) + β₂*ln(H)
                return np.exp(-9.5 + 1.8 * np.log(D) + 1.1 * np.log(H))

            elif melhor_modelo == 'G1':
                # ln(V) = β₀ + β₁*ln(D) + β₂*(1/D)
                return np.exp(-8.8 + 2.2 * np.log(D) - 1.2 / D)

            elif melhor_modelo == 'G2':
                # V = β₀ + β₁*D² + β₂*D²H + β₃*H
                D2 = D ** 2
                return -0.05 + 0.0008 * D2 + 0.000045 * D2 * H + 0.008 * H

            elif melhor_modelo == 'G3':
                # ln(V) = β₀ + β₁*ln(D²H)
                D2H = (D ** 2) * H
                return np.exp(-10.2 + 0.95 * np.log(D2H))

            else:
                # Fórmula básica de volume
                return 0.0008 * (D ** 2) * H

        except:
            # Fallback: fórmula básica
            return 0.0008 * (row['D_cm'] ** 2) * row['H_est']

    # Aplicar estimativa de volume
    df['V_est'] = df.apply(estimar_volume_arvore, axis=1)

    # Garantir valores positivos
    df['V_est'] = df['V_est'].clip(lower=0.001)

    return df


def processar_areas_inventario(df, config):
    '''
    Processa áreas dos talhões baseado na configuração

    Args:
        df: DataFrame com volumes estimados
        config: Configurações de área

    Returns:
        DataFrame com áreas dos talhões
    '''
    df = df.copy()

    metodo_area = config.get('metodo_area', 'Simular automaticamente')

    if metodo_area == 'Valores informados manualmente':
        # Usar áreas manuais da configuração
        areas_manuais = config.get('areas_manuais', {})
        df['area_ha'] = df['talhao'].map(areas_manuais).fillna(25.0)

    elif metodo_area == 'Upload shapefile':
        # Usar áreas do shapefile (já processadas)
        areas_shapefile = config.get('areas_shapefile')
        if areas_shapefile is not None:
            df = df.merge(areas_shapefile, on='talhao', how='left')
            df['area_ha'] = df['area_ha'].fillna(25.0)
        else:
            # Fallback para simulação
            df = simular_areas_talhoes(df)

    elif metodo_area == 'Coordenadas das parcelas':
        # Usar áreas calculadas das coordenadas
        areas_coordenadas = config.get('areas_coordenadas')
        if areas_coordenadas is not None:
            df = df.merge(areas_coordenadas, on='talhao', how='left')
            df['area_ha'] = df['area_ha'].fillna(25.0)
        else:
            # Fallback para simulação
            df = simular_areas_talhoes(df)

    else:
        # Simular automaticamente
        df = simular_areas_talhoes(df)

    return df


def simular_areas_talhoes(df):
    '''
    Simula áreas realistas para os talhões

    Args:
        df: DataFrame do inventário

    Returns:
        DataFrame com áreas simuladas
    '''
    talhoes_unicos = sorted(df['talhao'].unique())

    # Gerar áreas realistas baseadas no número de parcelas por talhão
    areas_simuladas = {}

    for talhao in talhoes_unicos:
        parcelas_talhao = df[df['talhao'] == talhao]['parcela'].nunique()

        # Área baseada no número de parcelas (assumindo distribuição sistemática)
        # Cada parcela representa aproximadamente 2-4 hectares
        area_base = parcelas_talhao * np.random.uniform(2.5, 4.0)

        # Adicionar variação realística
        variacao = np.random.uniform(0.8, 1.3)
        area_final = area_base * variacao

        # Arredondar para valores realísticos
        areas_simuladas[talhao] = round(area_final, 1)

    # Aplicar áreas simuladas
    df['area_ha'] = df['talhao'].map(areas_simuladas)

    return df


def calcular_resumo_por_parcela(df):
    '''
    Calcula resumo por parcela

    Args:
        df: DataFrame completo do inventário

    Returns:
        DataFrame com resumo por parcela
    '''
    area_parcela_m2 = 400  # Área padrão da parcela em m²

    resumo = df.groupby(['talhao', 'parcela']).agg({
        'area_ha': 'first',
        'idade_anos': lambda x: x.mean() if 'idade_anos' in df.columns else 5.0,
        'D_cm': 'mean',
        'H_est': 'mean',
        'V_est': 'sum',
        'cod': 'count'  # Número de árvores
    }).reset_index()

    # Renomear colunas
    resumo = resumo.rename(columns={
        'cod': 'n_arvores',
        'D_cm': 'dap_medio',
        'H_est': 'altura_media',
        'V_est': 'volume_parcela'
    })

    # Calcular volume por hectare
    resumo['vol_ha'] = resumo['volume_parcela'] * (10000 / area_parcela_m2)

    # Calcular IMA (se idade disponível)
    if 'idade_anos' in resumo.columns:
        resumo['ima'] = resumo['vol_ha'] / resumo['idade_anos']
    else:
        resumo['ima'] = resumo['vol_ha'] / 5.0  # Idade padrão

    return resumo


def calcular_resumo_por_talhao(resumo_parcelas):
    '''
    Calcula resumo por talhão

    Args:
        resumo_parcelas: DataFrame com resumo por parcela

    Returns:
        DataFrame com resumo por talhão
    '''
    resumo_talhao = resumo_parcelas.groupby('talhao').agg({
        'area_ha': 'first',
        'vol_ha': ['mean', 'std', 'count'],
        'dap_medio': 'mean',
        'altura_media': 'mean',
        'idade_anos': 'mean',
        'n_arvores': 'mean',
        'ima': 'mean'
    }).round(2)

    # Achatar colunas multi-nível
    resumo_talhao.columns = [
        'area_ha', 'vol_medio_ha', 'vol_desvio', 'n_parcelas',
        'dap_medio', 'altura_media', 'idade_media', 'arvores_por_parcela', 'ima_medio'
    ]

    resumo_talhao = resumo_talhao.reset_index()

    # Calcular estoque total por talhão
    resumo_talhao['estoque_total_m3'] = resumo_talhao['area_ha'] * resumo_talhao['vol_medio_ha']

    # Calcular CV
    resumo_talhao['cv_volume'] = (resumo_talhao['vol_desvio'] / resumo_talhao['vol_medio_ha']) * 100

    return resumo_talhao


def calcular_estatisticas_gerais(resumo_parcelas):
    '''
    Calcula estatísticas gerais do inventário

    Args:
        resumo_parcelas: DataFrame com resumo por parcela

    Returns:
        dict: Estatísticas gerais
    '''
    stats = {
        'total_parcelas': len(resumo_parcelas),
        'total_talhoes': resumo_parcelas['talhao'].nunique(),
        'area_total_ha': resumo_parcelas['area_ha'].sum(),
        'vol_medio_ha': resumo_parcelas['vol_ha'].mean(),
        'vol_min_ha': resumo_parcelas['vol_ha'].min(),
        'vol_max_ha': resumo_parcelas['vol_ha'].max(),
        'cv_volume': (resumo_parcelas['vol_ha'].std() / resumo_parcelas['vol_ha'].mean()) * 100,
        'dap_medio': resumo_parcelas['dap_medio'].mean(),
        'altura_media': resumo_parcelas['altura_media'].mean(),
        'idade_media': resumo_parcelas['idade_anos'].mean(),
        'ima_medio': resumo_parcelas['ima'].mean(),
        'arvores_por_parcela': resumo_parcelas['n_arvores'].mean()
    }

    # Calcular estoque total
    stats['estoque_total_m3'] = stats['area_total_ha'] * stats['vol_medio_ha']

    # Classificação de produtividade
    q25 = resumo_parcelas['vol_ha'].quantile(0.25)
    q75 = resumo_parcelas['vol_ha'].quantile(0.75)

    stats['classe_alta'] = (resumo_parcelas['vol_ha'] >= q75).sum()
    stats['classe_media'] = ((resumo_parcelas['vol_ha'] >= q25) & (resumo_parcelas['vol_ha'] < q75)).sum()
    stats['classe_baixa'] = (resumo_parcelas['vol_ha'] < q25).sum()
    stats['q25_volume'] = q25
    stats['q75_volume'] = q75

    return stats


def gerar_relatorio_inventario(resultados):
    '''
    Gera relatório executivo do inventário

    Args:
        resultados: Resultados completos do inventário

    Returns:
        str: Relatório em formato markdown
    '''
    stats = resultados['estatisticas_gerais']
    modelos = resultados['modelos_utilizados']

    relatorio = f'''
# RELATÓRIO EXECUTIVO - INVENTÁRIO FLORESTAL

## 🏆 MODELOS SELECIONADOS
- **Hipsométrico**: {modelos['hipsometrico']}
- **Volumétrico**: {modelos['volumetrico']}

## 🌲 RESUMO EXECUTIVO
- **Parcelas avaliadas**: {stats['total_parcelas']}
- **Talhões**: {stats['total_talhoes']}
- **Área total**: {formatar_brasileiro(stats['area_total_ha'], 1)} ha
- **Estoque total**: {formatar_numero_inteligente(stats['estoque_total_m3'], "m³")}
- **Produtividade média**: {formatar_brasileiro(stats['vol_medio_ha'], 1)} m³/ha
- **IMA médio**: {formatar_brasileiro(stats['ima_medio'], 1)} m³/ha/ano

## 📊 CLASSIFICAÇÃO DE PRODUTIVIDADE
- **Classe Alta** (≥ {formatar_brasileiro(stats['q75_volume'], 1)} m³/ha): {stats['classe_alta']} parcelas
- **Classe Média** ({formatar_brasileiro(stats['q25_volume'], 1)} - {formatar_brasileiro(stats['q75_volume'], 1)} m³/ha): {stats['classe_media']} parcelas
- **Classe Baixa** (< {formatar_brasileiro(stats['q25_volume'], 1)} m³/ha): {stats['classe_baixa']} parcelas

## 📊 ESTATÍSTICAS DENDROMÉTRICAS
- **DAP médio**: {formatar_brasileiro(stats['dap_medio'], 1)} cm
- **Altura média**: {formatar_brasileiro(stats['altura_media'], 1)} m
- **Idade média**: {formatar_brasileiro(stats['idade_media'], 1)} anos
- **Árvores por parcela**: {formatar_brasileiro(stats['arvores_por_parcela'], 0)}

## 📈 VARIABILIDADE
- **CV produtividade**: {formatar_brasileiro(stats['cv_volume'], 1)}%
- **Amplitude volume**: {formatar_brasileiro(stats['vol_min_ha'], 1)} - {formatar_brasileiro(stats['vol_max_ha'], 1)} m³/ha

---
*Relatório gerado pelo Sistema Modular de Inventário Florestal*
'''

    return relatorio


def validar_consistencia_inventario(resultados):
    '''
    Valida a consistência dos resultados do inventário

    Args:
        resultados: Resultados do inventário

    Returns:
        dict: Resultado da validação
    '''
    validacao = {
        'valido': True,
        'alertas': [],
        'erros': []
    }

    stats = resultados['estatisticas_gerais']

    # Verificar valores extremos
    if stats['vol_medio_ha'] > 1000:
        validacao['alertas'].append("Produtividade muito alta detectada")

    if stats['vol_medio_ha'] < 50:
        validacao['alertas'].append("Produtividade muito baixa detectada")

    if stats['cv_volume'] > 50:
        validacao['alertas'].append("Alta variabilidade entre parcelas")

    if stats['dap_medio'] > 50:
        validacao['alertas'].append("DAP médio muito alto")

    if stats['altura_media'] > 50:
        validacao['alertas'].append("Altura média muito alta")

    # Verificar dados faltantes
    resumo = resultados['resumo_parcelas']
    if resumo['vol_ha'].isna().any():
        validacao['erros'].append("Existem parcelas sem volume calculado")
        validacao['valido'] = False

    return validacao