# utils/formatacao.py
'''
Funções para formatação de números e dados no padrão brasileiro - VERSÃO MELHORADA
Mantém compatibilidade com código existente e adiciona novas funcionalidades
'''

import pandas as pd
import numpy as np
from datetime import datetime


def formatar_brasileiro(valor, decimais=2):
    '''
    Formata números no padrão brasileiro (. para milhares, , para decimais)
    VERSÃO ORIGINAL MANTIDA PARA COMPATIBILIDADE

    Args:
        valor: Número a ser formatado
        decimais: Número de casas decimais

    Returns:
        str: Número formatado no padrão brasileiro
    '''
    try:
        if pd.isna(valor) or valor is None:
            return "N/A"

        if isinstance(valor, (int, float, np.number)):
            formatado = f"{valor:.{decimais}f}".replace('.', ',')
            partes = formatado.split(',')

            if len(partes[0]) > 3:
                inteiro = partes[0]
                inteiro_formatado = ""

                for i, digito in enumerate(reversed(inteiro)):
                    if i > 0 and i % 3 == 0:
                        inteiro_formatado = "." + inteiro_formatado
                    inteiro_formatado = digito + inteiro_formatado

                formatado = inteiro_formatado + "," + partes[1]

            return formatado
        return str(valor)
    except:
        return str(valor)


def formatar_dataframe_brasileiro(df, colunas_numericas=None, decimais=2):
    '''
    Formata dataframe no padrão brasileiro
    VERSÃO ORIGINAL MANTIDA PARA COMPATIBILIDADE

    Args:
        df: DataFrame a ser formatado
        colunas_numericas: Lista de colunas numéricas (None = auto-detectar)
        decimais: Número de casas decimais

    Returns:
        DataFrame formatado
    '''
    if df is None or len(df) == 0:
        return df

    df_formatado = df.copy()

    if colunas_numericas is None:
        # Detectar colunas numéricas automaticamente
        colunas_numericas = df.select_dtypes(include=[np.number]).columns

    for col in colunas_numericas:
        if col in df.columns:
            df_formatado[col] = df[col].apply(lambda x: formatar_brasileiro(x, decimais))

    return df_formatado


def formatar_numero_inteligente(valor, unidade="", decimais_max=2):
    '''
    Formata números com unidades apropriadas (mil, milhão, etc.)
    para evitar números muito grandes na tela
    VERSÃO ORIGINAL MANTIDA PARA COMPATIBILIDADE

    Args:
        valor: Número a ser formatado
        unidade: Unidade de medida
        decimais_max: Máximo de casas decimais

    Returns:
        str: Número formatado com unidade apropriada
    '''
    try:
        if pd.isna(valor) or valor is None:
            return "N/A"

        if valor >= 1000000:
            valor_formatado = valor / 1000000
            if unidade:
                return f"{formatar_brasileiro(valor_formatado, 1)} milhões {unidade}"
            else:
                return f"{formatar_brasileiro(valor_formatado, 1)} milhões"

        elif valor >= 1000:
            valor_formatado = valor / 1000
            if unidade:
                return f"{formatar_brasileiro(valor_formatado, 1)} mil {unidade}"
            else:
                return f"{formatar_brasileiro(valor_formatado, 1)} mil"

        else:
            if unidade:
                return f"{formatar_brasileiro(valor, decimais_max)} {unidade}"
            else:
                return formatar_brasileiro(valor, decimais_max)

    except:
        return str(valor)


def classificar_qualidade_modelo(r2):
    '''
    Classifica a qualidade do modelo baseado no R²
    VERSÃO ORIGINAL MANTIDA PARA COMPATIBILIDADE

    Args:
        r2: Coeficiente de determinação

    Returns:
        str: Classificação da qualidade
    '''
    try:
        if pd.isna(r2) or r2 is None:
            return "Indeterminado"

        if r2 >= 0.9:
            return "Excelente"
        elif r2 >= 0.8:
            return "Muito Bom"
        elif r2 >= 0.7:
            return "*** Bom"
        elif r2 >= 0.6:
            return "Regular"
        else:
            return "Fraco"
    except:
        return "Erro na classificação"


# NOVAS FUNÇÕES ADICIONADAS

def formatar_percentual(valor, decimais=1):
    '''
    Formata valores como percentual

    Args:
        valor: Valor a ser formatado (0.85 ou 85)
        decimais: Casas decimais

    Returns:
        str: Percentual formatado (ex: "85,0%")
    '''
    try:
        if pd.isna(valor) or valor is None:
            return "N/A"

        # Se valor <= 1, assume que está em decimal (0.85)
        # Se valor > 1, assume que já está em percentual (85)
        if valor <= 1:
            percentual = valor * 100
        else:
            percentual = valor

        return f"{formatar_brasileiro(percentual, decimais)}%"
    except:
        return str(valor)


def formatar_moeda(valor, decimais=2):
    '''
    Formata valores monetários em Reais

    Args:
        valor: Valor monetário
        decimais: Casas decimais

    Returns:
        str: Valor formatado (ex: "R$ 1.234,56")
    '''
    try:
        if pd.isna(valor) or valor is None:
            return "N/A"

        return f"R$ {formatar_brasileiro(valor, decimais)}"
    except:
        return str(valor)


def formatar_timestamp_brasileiro(timestamp=None):
    """
    Formata timestamp no padrão brasileiro

    Args:
        timestamp: Timestamp (usa atual se None)

    Returns:
        str: Data/hora formatada (ex: "25/12/2024 às 14:30:15")
    """
    if timestamp is None:
        timestamp = datetime.now()

    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        return timestamp.strftime("%d/%m/%Y às %H:%M:%S")

    except Exception:
        return str(timestamp)


def classificar_qualidade_modelo_detalhado(r2, tipo='hipsometrico'):
    """
    Classificação detalhada por tipo de modelo

    Args:
        r2: Coeficiente de determinação
        tipo: Tipo do modelo ('hipsometrico', 'volumetrico', 'geral')

    Returns:
        str: Classificação detalhada
    """
    try:
        if pd.isna(r2) or r2 is None:
            return "❓ Indeterminado"

        r2 = float(r2)

        if tipo == 'hipsometrico':
            if r2 >= 0.95:
                return "⭐⭐⭐⭐⭐ Excepcional"
            elif r2 >= 0.90:
                return "⭐⭐⭐⭐ Excelente"
            elif r2 >= 0.80:
                return "⭐⭐⭐ Muito Bom"
            elif r2 >= 0.70:
                return "⭐⭐ Bom"
            elif r2 >= 0.60:
                return "⭐ Regular"
            else:
                return "❌ Insatisfatório"

        elif tipo == 'volumetrico':
            if r2 >= 0.98:
                return "⭐⭐⭐⭐⭐ Excepcional"
            elif r2 >= 0.95:
                return "⭐⭐⭐⭐ Excelente"
            elif r2 >= 0.90:
                return "⭐⭐⭐ Muito Bom"
            elif r2 >= 0.85:
                return "⭐⭐ Bom"
            elif r2 >= 0.75:
                return "⭐ Regular"
            else:
                return "❌ Insatisfatório"

        else:
            # Classificação geral
            return classificar_qualidade_modelo(r2)

    except Exception:
        return "❓ Erro na classificação"


def classificar_produtividade_florestal(volume_ha, especie='eucalipto'):
    """
    Classifica produtividade florestal por espécie

    Args:
        volume_ha: Volume por hectare (m³/ha)
        especie: Espécie florestal

    Returns:
        str: Classificação de produtividade
    """
    try:
        if pd.isna(volume_ha) or volume_ha is None:
            return "❓ Indeterminado"

        volume = float(volume_ha)

        if especie.lower() == 'eucalipto':
            if volume >= 250:
                return "Excelente (≥250 m³/ha)"
            elif volume >= 200:
                return "Muito Boa (200-249 m³/ha)"
            elif volume >= 150:
                return "Boa (150-199 m³/ha)"
            elif volume >= 100:
                return "Regular (100-149 m³/ha)"
            elif volume >= 50:
                return "Baixa (50-99 m³/ha)"
            else:
                return "⚫ Muito Baixa (<50 m³/ha)"

        elif especie.lower() == 'pinus':
            if volume >= 300:
                return "Excelente (≥300 m³/ha)"
            elif volume >= 250:
                return "Muito Boa (250-299 m³/ha)"
            elif volume >= 200:
                return "Boa (200-249 m³/ha)"
            elif volume >= 150:
                return "🟠 Regular (150-199 m³/ha)"
            elif volume >= 100:
                return "Baixa (100-149 m³/ha)"
            else:
                return "Muito Baixa (<100 m³/ha)"

        else:
            # Classificação genérica
            if volume >= 200:
                return "🟢 Alta (≥200 m³/ha)"
            elif volume >= 150:
                return "🟡 Média (150-199 m³/ha)"
            elif volume >= 100:
                return "🟠 Regular (100-149 m³/ha)"
            else:
                return "🔴 Baixa (<100 m³/ha)"

    except Exception:
        return "❓ Erro na classificação"


def formatar_estatisticas_resumo(df, colunas_numericas=None):
    """
    Cria resumo estatístico formatado

    Args:
        df: DataFrame
        colunas_numericas: Lista de colunas (auto-detecta se None)

    Returns:
        dict: Estatísticas formatadas
    """
    if df is None or len(df) == 0:
        return {"erro": "DataFrame vazio"}

    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    estatisticas = {}

    for coluna in colunas_numericas:
        if coluna in df.columns:
            serie = df[coluna].dropna()

            if len(serie) > 0:
                estatisticas[coluna] = {
                    'count': len(serie),
                    'count_str': f"{len(serie):,}".replace(',', '.'),
                    'media': formatar_brasileiro(serie.mean(), 2),
                    'mediana': formatar_brasileiro(serie.median(), 2),
                    'std': formatar_brasileiro(serie.std(), 2),
                    'min': formatar_brasileiro(serie.min(), 2),
                    'max': formatar_brasileiro(serie.max(), 2),
                    'cv_pct': formatar_percentual((serie.std() / serie.mean()) if serie.mean() != 0 else 0)
                }
            else:
                estatisticas[coluna] = {"erro": "Sem dados válidos"}

    return estatisticas


def criar_relatorio_modelo(resultado, nome_modelo, tipo='hipsometrico'):
    """
    Cria relatório formatado de um modelo

    Args:
        resultado: Dict com resultados do modelo
        nome_modelo: Nome do modelo
        tipo: Tipo do modelo

    Returns:
        str: Relatório formatado
    """
    if not resultado:
        return f"❌ **{nome_modelo}**: Falha no ajuste"

    try:
        # Obter métricas principais
        r2 = resultado.get('r2', resultado.get('r2g', 0))
        rmse = resultado.get('rmse', 0)

        # Classificar qualidade
        qualidade = classificar_qualidade_modelo_detalhado(r2, tipo)

        # Montar relatório
        relatorio = f"""
📊 **{nome_modelo}**
• **R²:** {formatar_brasileiro(r2, 4)}
• **RMSE:** {formatar_brasileiro(rmse, 3)}
• **Qualidade:** {qualidade}
"""

        # Adicionar métricas extras se disponíveis
        if 'aic' in resultado:
            relatorio += f"• **AIC:** {formatar_brasileiro(resultado['aic'], 2)}\n"

        if 'bic' in resultado:
            relatorio += f"• **BIC:** {formatar_brasileiro(resultado['bic'], 2)}\n"

        if 'significancia' in resultado:
            sig = "✅ Significativo" if resultado['significancia'] else "⚠️ Não significativo"
            relatorio += f"• **Significância:** {sig}\n"

        return relatorio.strip()

    except Exception as e:
        return f"❌ Erro ao formatar relatório do {nome_modelo}: {e}"


def formatar_cabecalho_relatorio(titulo, subtitulo="", incluir_timestamp=True):
    """
    Formata cabeçalho padrão para relatórios

    Args:
        titulo: Título principal
        subtitulo: Subtítulo (opcional)
        incluir_timestamp: Se deve incluir data/hora

    Returns:
        str: Cabeçalho formatado em markdown
    """
    cabecalho = f"# 🌲 {titulo}\n"

    if subtitulo:
        cabecalho += f"### {subtitulo}\n"

    if incluir_timestamp:
        timestamp = formatar_timestamp_brasileiro()
        cabecalho += f"**Data/Hora:** {timestamp}\n"

    cabecalho += "\n---\n"

    return cabecalho


def validar_dados_numericos(serie, nome_coluna="coluna", limites=None):
    """
    Valida série numérica e gera relatório

    Args:
        serie: Série pandas
        nome_coluna: Nome da coluna
        limites: Dict com 'min' e 'max' opcionais

    Returns:
        dict: Relatório de validação
    """
    relatorio = {
        'valida': True,
        'problemas': [],
        'avisos': [],
        'estatisticas': {}
    }

    try:
        # Estatísticas básicas
        total = len(serie)
        nulos = serie.isna().sum()
        validos = total - nulos

        relatorio['estatisticas'] = {
            'total': total,
            'validos': validos,
            'nulos': nulos,
            'pct_nulos': (nulos / total) * 100 if total > 0 else 0
        }

        # Validações críticas
        if nulos == total:
            relatorio['valida'] = False
            relatorio['problemas'].append(f"{nome_coluna}: Todos os valores são nulos")
            return relatorio

        if nulos > total * 0.8:
            relatorio['valida'] = False
            relatorio['problemas'].append(f"{nome_coluna}: Mais de 80% dos valores são nulos")
        elif nulos > total * 0.5:
            relatorio['avisos'].append(f"{nome_coluna}: Mais de 50% dos valores são nulos")

        # Análise de valores válidos
        if validos > 0:
            valores_validos = serie.dropna()

            # Verificar valores negativos onde não deveria
            if nome_coluna.lower() in ['d_cm', 'dap', 'h_m', 'altura', 'volume', 'area']:
                negativos = (valores_validos < 0).sum()
                if negativos > 0:
                    relatorio['problemas'].append(f"{nome_coluna}: {negativos} valores negativos detectados")

            # Verificar limites específicos
            if limites:
                if 'min' in limites:
                    abaixo_min = (valores_validos < limites['min']).sum()
                    if abaixo_min > 0:
                        relatorio['avisos'].append(
                            f"{nome_coluna}: {abaixo_min} valores abaixo do mínimo ({limites['min']})")

                if 'max' in limites:
                    acima_max = (valores_validos > limites['max']).sum()
                    if acima_max > 0:
                        relatorio['avisos'].append(
                            f"{nome_coluna}: {acima_max} valores acima do máximo ({limites['max']})")

            # Detectar outliers extremos
            Q1 = valores_validos.quantile(0.25)
            Q3 = valores_validos.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:  # Evitar divisão por zero
                outliers_inf = valores_validos < (Q1 - 3 * IQR)
                outliers_sup = valores_validos > (Q3 + 3 * IQR)
                total_outliers = outliers_inf.sum() + outliers_sup.sum()

                if total_outliers > validos * 0.1:  # Mais de 10% outliers
                    relatorio['avisos'].append(
                        f"{nome_coluna}: {total_outliers} outliers extremos ({(total_outliers / validos) * 100:.1f}%)")

            # Estatísticas adicionais
            relatorio['estatisticas'].update({
                'media': valores_validos.mean(),
                'media_str': formatar_brasileiro(valores_validos.mean(), 2),
                'mediana': valores_validos.median(),
                'std': valores_validos.std(),
                'min': valores_validos.min(),
                'max': valores_validos.max(),
                'cv': (valores_validos.std() / valores_validos.mean()) * 100 if valores_validos.mean() != 0 else 0,
                'outliers': total_outliers if 'total_outliers' in locals() else 0
            })

    except Exception as e:
        relatorio['valida'] = False
        relatorio['problemas'].append(f"Erro na validação de {nome_coluna}: {e}")

    return relatorio


def formatar_relatorio_validacao(relatorio_validacao, nome_dataset="dados"):
    """
    Formata relatório de validação para exibição

    Args:
        relatorio_validacao: Dict do relatório de validação
        nome_dataset: Nome do conjunto de dados

    Returns:
        str: Relatório formatado em markdown
    """
    if not relatorio_validacao:
        return f"❌ Relatório de validação indisponível para {nome_dataset}"

    relatorio = f"""
## 📊 Relatório de Validação - {nome_dataset.title()}

**Status Geral:** {"✅ Aprovado" if relatorio_validacao.get('pode_prosseguir', False) else "❌ Reprovado"}
**Qualidade:** {relatorio_validacao.get('qualidade_geral', 'Indeterminada')}

### 📈 Estatísticas Gerais
• **Total de registros:** {relatorio_validacao.get('total_registros', 0):,}
• **Total de colunas:** {relatorio_validacao.get('total_colunas', 0)}
• **Colunas numéricas:** {len(relatorio_validacao.get('colunas_numericas', []))}
• **Colunas categóricas:** {len(relatorio_validacao.get('colunas_categoricas', []))}
"""

    # Problemas críticos
    problemas = relatorio_validacao.get('problemas_criticos', [])
    if problemas:
        relatorio += "\n### ❌ Problemas Críticos\n"
        for problema in problemas:
            relatorio += f"• {problema}\n"

    # Avisos
    avisos = relatorio_validacao.get('avisos', [])
    if avisos:
        relatorio += "\n### ⚠️ Avisos\n"
        for aviso in avisos[:5]:  # Mostrar apenas os 5 primeiros
            relatorio += f"• {aviso}\n"

        if len(avisos) > 5:
            relatorio += f"• ... e mais {len(avisos) - 5} avisos\n"

    # Recomendações
    recomendacoes = relatorio_validacao.get('recomendacoes', [])
    if recomendacoes:
        relatorio += "\n### 💡 Recomendações\n"
        for rec in recomendacoes:
            relatorio += f"• {rec}\n"

    return relatorio


def criar_sumario_executivo_inventario(stats_gerais, melhor_hip=None, melhor_vol=None):
    """
    Cria sumário executivo do inventário

    Args:
        stats_gerais: Dict com estatísticas gerais
        melhor_hip: Nome do melhor modelo hipsométrico
        melhor_vol: Nome do melhor modelo volumétrico

    Returns:
        str: Sumário executivo formatado
    """
    timestamp = formatar_timestamp_brasileiro()

    sumario = formatar_cabecalho_relatorio(
        "Sumário Executivo do Inventário Florestal",
        "Resultados Principais",
        True
    )

    sumario += """
## 🎯 Resultados Principais

### 📊 Estatísticas do Povoamento
"""

    # Adicionar estatísticas se disponíveis
    if stats_gerais:
        volume_total = stats_gerais.get('volume_total_m3', 0)
        volume_ha = stats_gerais.get('volume_medio_ha', 0)
        area_total = stats_gerais.get('area_total_ha', 0)
        num_talhoes = stats_gerais.get('num_talhoes', 0)

        sumario += f"""
• **Volume Total:** {formatar_numero_inteligente(volume_total, 'm³')}
• **Produtividade Média:** {formatar_brasileiro(volume_ha, 1)} m³/ha
• **Área Total:** {formatar_brasileiro(area_total, 1)} ha
• **Número de Talhões:** {num_talhoes}
"""

        # Classificar produtividade
        if volume_ha > 0:
            classe_prod = classificar_produtividade_florestal(volume_ha)
            sumario += f"• **Classificação:** {classe_prod}\n"

    # Modelos utilizados
    if melhor_hip or melhor_vol:
        sumario += "\n### 🔬 Modelos Utilizados\n"

        if melhor_hip:
            sumario += f"• **Hipsométrico:** {melhor_hip}\n"

        if melhor_vol:
            sumario += f"• **Volumétrico:** {melhor_vol}\n"

    sumario += "\n---\n*Relatório gerado automaticamente pelo Sistema de Inventário Florestal*"

    return sumario


def formatar_tabela_talhoes(df_talhoes, incluir_classificacao=True):
    """
    Formata tabela de resultados por talhão

    Args:
        df_talhoes: DataFrame com dados por talhão
        incluir_classificacao: Se deve incluir classificação de produtividade

    Returns:
        DataFrame formatado para exibição
    """
    if df_talhoes is None or len(df_talhoes) == 0:
        return pd.DataFrame()

    df_formatado = df_talhoes.copy()

    # Mapear nomes de colunas para exibição
    mapeamento_colunas = {
        'talhao': 'Talhão',
        'area_ha': 'Área (ha)',
        'num_parcelas': 'Parcelas',
        'vol_total_m3': 'Volume Total (m³)',
        'vol_medio_ha': 'Volume/ha (m³)',
        'dap_medio': 'DAP Médio (cm)',
        'altura_media': 'Altura Média (m)',
        'idade_anos': 'Idade (anos)',
        'densidade_arv_ha': 'Densidade (árv/ha)'
    }

    # Renomear colunas existentes
    colunas_renomear = {k: v for k, v in mapeamento_colunas.items() if k in df_formatado.columns}
    df_formatado = df_formatado.rename(columns=colunas_renomear)

    # Formatar colunas numéricas
    for coluna in df_formatado.columns:
        if coluna in ['Área (ha)', 'Volume/ha (m³)', 'DAP Médio (cm)', 'Altura Média (m)', 'Idade (anos)']:
            df_formatado[coluna] = df_formatado[coluna].apply(lambda x: formatar_brasileiro(x, 1))
        elif coluna in ['Volume Total (m³)', 'Densidade (árv/ha)']:
            df_formatado[coluna] = df_formatado[coluna].apply(lambda x: formatar_brasileiro(x, 0))

    # Adicionar classificação de produtividade se solicitado
    if incluir_classificacao and 'Volume/ha (m³)' in df_formatado.columns:
        # Converter de volta para numérico para classificação
        vol_ha_numeric = df_talhoes['vol_medio_ha'] if 'vol_medio_ha' in df_talhoes.columns else df_talhoes.get(
            'vol_ha', pd.Series([0]))
        df_formatado['Classificação'] = vol_ha_numeric.apply(classificar_produtividade_florestal)

    return df_formatado


def gerar_metricas_card_streamlit(valor, titulo, delta=None, help_text=None):
    """
    Gera dados formatados para st.metric do Streamlit

    Args:
        valor: Valor principal
        titulo: Título da métrica
        delta: Valor de comparação (opcional)
        help_text: Texto de ajuda (opcional)

    Returns:
        dict: Dados formatados para st.metric
    """
    return {
        'label': titulo,
        'value': formatar_brasileiro(valor, 2) if isinstance(valor, (int, float)) else str(valor),
        'delta': formatar_brasileiro(delta, 2) if delta is not None and isinstance(delta, (int, float)) else delta,
        'help': help_text
    }


# Função utilitária para detectar tipo de coluna
def detectar_tipo_coluna(serie, nome_coluna=""):
    """
    Detecta o tipo/categoria de uma coluna baseado no nome e valores

    Args:
        serie: Série pandas
        nome_coluna: Nome da coluna

    Returns:
        str: Tipo detectado
    """
    nome_lower = nome_coluna.lower()

    # Detectar por nome
    if any(palavra in nome_lower for palavra in ['dap', 'd_cm', 'diametro']):
        return 'diametro'
    elif any(palavra in nome_lower for palavra in ['altura', 'h_m', 'height']):
        return 'altura'
    elif any(palavra in nome_lower for palavra in ['volume', 'vol']):
        return 'volume'
    elif any(palavra in nome_lower for palavra in ['area', 'área']):
        return 'area'
    elif any(palavra in nome_lower for palavra in ['idade', 'age', 'anos']):
        return 'idade'
    elif any(palavra in nome_lower for palavra in ['talhao', 'talhão', 'stand']):
        return 'talhao'
    elif any(palavra in nome_lower for palavra in ['parcela', 'plot']):
        return 'parcela'
    elif any(palavra in nome_lower for palavra in ['cod', 'codigo', 'code']):
        return 'codigo'

    # Detectar por tipo de dados
    if serie.dtype in ['object', 'string']:
        return 'categorico'
    elif serie.dtype in ['int64', 'float64', 'Int64', 'Float64']:
        return 'numerico'
    else:
        return 'desconhecido'


# Compatibilidade total com código existente
def formatar_numero_brasileiro_compat(valor, decimais=2):
    """Alias para manter compatibilidade total"""
    return formatar_brasileiro(valor, decimais)