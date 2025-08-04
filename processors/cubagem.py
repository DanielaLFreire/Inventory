# processors/cubagem.py - VERSÃO USANDO FUNÇÕES DE formatacao.py
'''
Processamento de dados de cubagem usando método de Smalian
CORRIGIDO: Usa funções de conversão existentes em formatacao.py
'''

import pandas as pd
import numpy as np

# USAR FUNÇÃO EXISTENTE do formatacao.py
from utils.formatacao import validar_dados_numericos


def converter_coluna_formato_brasileiro(serie, nome_coluna="coluna"):
    """
    Converte coluna do formato brasileiro usando validação existente

    Args:
        serie: Série pandas
        nome_coluna: Nome da coluna para logs

    Returns:
        Series: Série convertida
    """

    def converter_valor_individual(valor):
        """Converte valor individual do formato brasileiro"""
        if pd.isna(valor):
            return np.nan

        # Se já for numérico, retornar
        if isinstance(valor, (int, float)):
            return float(valor)

        # Se for string, processar
        if isinstance(valor, str):
            valor = valor.strip()
            if valor == '' or valor.lower() == 'nan':
                return np.nan

            try:
                # Substituir vírgula por ponto (formato brasileiro → internacional)
                valor_convertido = valor.replace(',', '.')
                return float(valor_convertido)
            except (ValueError, TypeError):
                return np.nan

        return np.nan

    # Aplicar conversão
    serie_convertida = serie.apply(converter_valor_individual)

    # Usar validação existente do formatacao.py
    relatorio = validar_dados_numericos(serie_convertida, nome_coluna)

    # Log dos resultados da validação
    if relatorio['valida']:
        print(
            f"✅ {nome_coluna}: {relatorio['estatisticas']['validos']}/{relatorio['estatisticas']['total']} valores convertidos")
    else:
        print(f"⚠️ {nome_coluna}: Problemas na conversão")
        for problema in relatorio['problemas']:
            print(f"    • {problema}")

    return serie_convertida


def processar_cubagem_smalian(df_cubagem):
    '''
    Processa dados de cubagem usando o método de Smalian
    VERSÃO USANDO FUNÇÕES DE formatacao.py

    Args:
        df_cubagem: DataFrame com dados de cubagem

    Returns:
        DataFrame com volumes calculados por árvore
    '''

    print("🌲 Iniciando processamento da cubagem (Método Smalian)...")

    df = df_cubagem.copy()

    # NOVO: Converter colunas usando função existente
    colunas_numericas = ['d_cm', 'h_m', 'D_cm', 'H_m']

    print("🔄 Convertendo dados do formato brasileiro...")
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = converter_coluna_formato_brasileiro(df[col], col)

    # Verificar se conversão foi bem-sucedida
    df_valido = df.dropna(subset=colunas_numericas)

    if len(df_valido) < 5:
        print(f"❌ Poucos dados válidos após conversão: {len(df_valido)}")
        return None

    print(f"✅ {len(df_valido)} registros válidos para processamento")

    # Calcular área seccional (π * d²/4 em m²)
    df_valido['a_m2'] = np.pi * (df_valido['d_cm'] ** 2 / 40000)

    # Ordenar por árvore e altura
    df_valido = df_valido.sort_values(['arv', 'talhao', 'h_m']).reset_index(drop=True)

    # Aplicar método de Smalian
    volumes_list = []

    print("🔄 Aplicando método de Smalian...")

    for (talhao, arv), grupo in df_valido.groupby(['talhao', 'arv']):
        grupo = grupo.sort_values('h_m').reset_index(drop=True)

        for i in range(len(grupo)):
            row = grupo.iloc[i].copy()

            if i > 0:
                # Calcular volume da seção usando Smalian
                row['a1'] = grupo.iloc[i - 1]['a_m2']
                row['h1'] = grupo.iloc[i - 1]['h_m']
                row['a2'] = grupo.iloc[i]['a_m2']
                row['h2'] = grupo.iloc[i]['h_m']
                row['delta_h'] = row['h2'] - row['h1']

                # Fórmula de Smalian: V = ((A1 + A2) / 2) * L
                row['va_m3'] = ((row['a1'] + row['a2']) / 2) * row['delta_h']
            else:
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

    print(f"✅ {len(volumes_arvore)} árvores com volumes calculados")

    return volumes_arvore


def calcular_estatisticas_cubagem(volumes_arvore):
    '''
    Calcula estatísticas da cubagem usando formatação brasileira

    Args:
        volumes_arvore: DataFrame com volumes por árvore

    Returns:
        dict: Estatísticas da cubagem formatadas
    '''
    if volumes_arvore is None or len(volumes_arvore) == 0:
        return {
            'total_arvores': 0,
            'volume_total': 0,
            'volume_medio': 0,
            'volume_min': 0,
            'volume_max': 0,
            'dap_medio': 0,
            'altura_media': 0,
            'cv_volume': 0
        }

    # USAR FUNÇÃO EXISTENTE do formatacao.py
    from utils.formatacao import formatar_estatisticas_resumo

    # Calcular estatísticas básicas
    stats_resumo = formatar_estatisticas_resumo(volumes_arvore, ['V', 'D_cm', 'H_m'])

    # Montar estatísticas finais
    stats = {
        'total_arvores': len(volumes_arvore),
        'volume_total': volumes_arvore['V'].sum(),
        'volume_medio': volumes_arvore['V'].mean(),
        'volume_min': volumes_arvore['V'].min(),
        'volume_max': volumes_arvore['V'].max(),
        'dap_medio': volumes_arvore['D_cm'].mean(),
        'altura_media': volumes_arvore['H_m'].mean(),
        'cv_volume': (volumes_arvore['V'].std() / volumes_arvore['V'].mean()) * 100 if volumes_arvore[
                                                                                           'V'].mean() > 0 else 0,
        'stats_formatadas': stats_resumo  # Adicionar estatísticas formatadas
    }

    return stats


def validar_dados_cubagem(df_cubagem):
    '''
    Valida dados de cubagem usando validação existente de formatacao.py

    Args:
        df_cubagem: DataFrame com dados de cubagem

    Returns:
        tuple: (bool válido, list mensagens)
    '''
    mensagens = []
    valido = True

    # Usar validação existente
    from utils.formatacao import validar_dados_numericos

    # Verificar colunas obrigatórias
    colunas_obrigatorias = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']
    colunas_faltantes = [col for col in colunas_obrigatorias if col not in df_cubagem.columns]

    if colunas_faltantes:
        mensagens.append(f"Colunas obrigatórias faltantes: {colunas_faltantes}")
        valido = False

    if len(df_cubagem) == 0:
        mensagens.append("Arquivo de cubagem está vazio")
        valido = False
        return valido, mensagens

    # Validar colunas numéricas usando função existente
    colunas_numericas = ['d_cm', 'h_m', 'D_cm', 'H_m']

    for col in colunas_numericas:
        if col in df_cubagem.columns:
            # Definir limites apropriados
            limites = {}
            if col in ['d_cm', 'D_cm']:
                limites = {'min': 1, 'max': 200}  # DAP entre 1 e 200 cm
            elif col in ['h_m', 'H_m']:
                limites = {'min': 0.1, 'max': 50}  # Altura entre 0.1 e 50 m

            # Usar validação existente
            relatorio = validar_dados_numericos(df_cubagem[col], col, limites)

            if not relatorio['valida']:
                valido = False
                mensagens.extend(relatorio['problemas'])

            # Adicionar avisos como mensagens informativas
            mensagens.extend(relatorio['avisos'])

    return valido, mensagens


def gerar_relatorio_cubagem(volumes_arvore):
    '''
    Gera relatório detalhado usando formatação brasileira

    Args:
        volumes_arvore: DataFrame com volumes calculados

    Returns:
        str: Relatório formatado
    '''
    if volumes_arvore is None or len(volumes_arvore) == 0:
        return """
## RELATÓRIO DE CUBAGEM - MÉTODO DE SMALIAN

### ❌ ERRO NO PROCESSAMENTO
- Não foi possível calcular volumes
- Verifique os dados de entrada
- Confirme se o formato numérico está correto
"""

    # USAR FUNÇÕES EXISTENTES do formatacao.py
    from utils.formatacao import (
        formatar_cabecalho_relatorio,
        formatar_brasileiro,
        formatar_percentual,
        classificar_qualidade_modelo
    )

    stats = calcular_estatisticas_cubagem(volumes_arvore)

    # Gerar cabeçalho formatado
    relatorio = formatar_cabecalho_relatorio(
        "Relatório de Cubagem",
        "Método de Smalian",
        True
    )

    relatorio += f'''
## 📊 Resumo Geral
- **Total de árvores cubadas**: {stats['total_arvores']}
- **Volume total**: {formatar_brasileiro(stats['volume_total'], 3)} m³
- **Volume médio por árvore**: {formatar_brasileiro(stats['volume_medio'], 4)} m³

## 📈 Estatísticas de Volume
- **Mínimo**: {formatar_brasileiro(stats['volume_min'], 4)} m³
- **Máximo**: {formatar_brasileiro(stats['volume_max'], 4)} m³
- **Coeficiente de variação**: {formatar_percentual(stats['cv_volume'] / 100, 1)}

## 🌲 Características Dendrométricas
- **DAP médio**: {formatar_brasileiro(stats['dap_medio'], 1)} cm
- **Altura média**: {formatar_brasileiro(stats['altura_media'], 1)} m

## ⚙️ Método Utilizado
**Fórmula de Smalian**: V = ((A₁ + A₂) / 2) × L

**Onde:**
- A₁ = Área da seção inferior
- A₂ = Área da seção superior  
- L = Comprimento da seção

## 📋 Observações
- ✅ Volumes calculados excluindo a seção do toco (0,1 m)
- ✅ Áreas seccionais calculadas a partir dos diâmetros medidos
- ✅ **Dados convertidos do formato brasileiro automaticamente**
- ✅ Validação numérica aplicada usando funções existentes do sistema
- ✅ Formatação brasileira aplicada em todos os números

---
*Relatório gerado usando funções de formatação centralizadas do sistema*
'''

    return relatorio


def exportar_volumes_detalhados(df_volumes_detalhado, volumes_arvore):
    '''
    Prepara dados detalhados para exportação com formatação brasileira

    Args:
        df_volumes_detalhado: DataFrame com detalhes por seção
        volumes_arvore: DataFrame com volumes totais por árvore

    Returns:
        tuple: (DataFrame seções, DataFrame árvores) formatados
    '''
    if volumes_arvore is None or len(volumes_arvore) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # USAR FUNÇÃO EXISTENTE do formatacao.py
    from utils.formatacao import formatar_dataframe_brasileiro

    # Preparar dados de seções
    if df_volumes_detalhado is not None and len(df_volumes_detalhado) > 0:
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

        # Aplicar formatação brasileira
        colunas_numericas = ['Altura (m)', 'Diâmetro (cm)', 'Área Seccional (m²)', 'Volume Seção (m³)']
        df_secoes = formatar_dataframe_brasileiro(df_secoes, colunas_numericas, 4)

    else:
        df_secoes = pd.DataFrame()

    # Preparar dados de árvores
    df_arvores = volumes_arvore.copy()
    df_arvores = df_arvores.rename(columns={
        'arv': 'Árvore',
        'talhao': 'Talhão',
        'D_cm': 'DAP (cm)',
        'H_m': 'Altura Total (m)',
        'V': 'Volume Total (m³)'
    })

    # Aplicar formatação brasileira
    colunas_numericas_arvores = ['DAP (cm)', 'Altura Total (m)', 'Volume Total (m³)']
    df_arvores = formatar_dataframe_brasileiro(df_arvores, colunas_numericas_arvores, 3)

    return df_secoes, df_arvores