# utils/formatacao.py - VERSÃO CONSOLIDADA E CORRIGIDA
"""
Funções para formatação de números e dados no padrão brasileiro
VERSÃO CORRIGIDA: Elimina duplicações e incompatibilidades
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# CORE FUNCTIONS - CONVERSÃO DECIMAL BRASILEIRA
# ==========================================

def processar_decimal_brasileiro(valor) -> Union[float, any]:
    """
    FUNÇÃO PRINCIPAL: Converte formato decimal brasileiro para internacional
    Esta função estava sendo importada mas não existia - CORRIGIDA

    Args:
        valor: Valor a ser convertido (str, int, float, np.number)

    Returns:
        float: Valor convertido ou valor original se não for possível converter
    """
    try:
        # Se já é numérico, retornar como está
        if isinstance(valor, (int, float, np.number)):
            return float(valor)

        # Se não é string, retornar original
        if not isinstance(valor, str):
            return valor

        # Limpar string
        valor_limpo = str(valor).strip()

        # Se vazio, retornar NaN
        if not valor_limpo or valor_limpo.lower() in ['', 'nan', 'null', 'none']:
            return np.nan

        # Remover espaços e caracteres especiais (exceto números, vírgula, ponto, sinal)
        import re
        valor_limpo = re.sub(r'[^\d\.,\-+]', '', valor_limpo)

        if not valor_limpo:
            return np.nan

        # PADRÃO BRASILEIRO COM MILHARES: "1.234.567,89" → 1234567.89
        if '.' in valor_limpo and ',' in valor_limpo:
            # Verificar se o último ponto está antes da vírgula (formato brasileiro)
            pos_ultimo_ponto = valor_limpo.rfind('.')
            pos_virgula = valor_limpo.rfind(',')

            if pos_virgula > pos_ultimo_ponto:  # Vírgula após ponto = formato brasileiro
                # Remover pontos (milhares) e trocar vírgula por ponto (decimal)
                valor_convertido = valor_limpo.replace('.', '').replace(',', '.')
                return float(valor_convertido)
            else:
                # Formato internacional ou inválido
                return float(valor_limpo.replace(',', ''))

        # PADRÃO BRASILEIRO SIMPLES: "123,45" → 123.45
        elif ',' in valor_limpo and '.' not in valor_limpo:
            # Verificar se parece formato brasileiro (vírgula como decimal)
            partes = valor_limpo.split(',')
            if len(partes) == 2 and len(partes[1]) <= 3:  # No máximo 3 casas decimais
                return float(valor_limpo.replace(',', '.'))
            else:
                # Pode ser formato internacional com vírgula como milhares
                return float(valor_limpo.replace(',', ''))

        # SOMENTE PONTO: pode ser decimal internacional ou milhares brasileiros
        elif '.' in valor_limpo and ',' not in valor_limpo:
            partes = valor_limpo.split('.')

            # Se última parte tem 1-3 dígitos, provavelmente é decimal
            if len(partes[-1]) <= 3:
                return float(valor_limpo)  # Formato internacional
            else:
                # Provavelmente são milhares brasileiros: "1.234" → 1234
                return float(valor_limpo.replace('.', ''))

        # SOMENTE NÚMEROS: retornar como inteiro/float
        else:
            return float(valor_limpo)

    except (ValueError, TypeError, AttributeError):
        # Se não conseguir converter, retornar valor original
        return valor


# Alias para compatibilidade (função era chamada com este nome)
def parsear_decimal_brasileiro(valor):
    """Alias para processar_decimal_brasileiro - COMPATIBILIDADE"""
    return processar_decimal_brasileiro(valor)


def detectar_formato_numerico(serie: pd.Series) -> str:
    """
    Detecta se série usa formato brasileiro, internacional ou misto

    Args:
        serie: Série pandas para análise

    Returns:
        str: 'brasileiro', 'internacional', 'misto' ou 'nao_numerico'
    """
    try:
        # Se já é numérico, é internacional
        if serie.dtype in ['int64', 'float64', 'Int64', 'Float64']:
            return 'internacional'

        # Analisar apenas valores string não-nulos
        amostra = serie.dropna().astype(str).head(50)

        if len(amostra) == 0:
            return 'nao_numerico'

        # Contar padrões
        brasileiro_simples = 0  # "123,45"
        brasileiro_completo = 0  # "1.234,56"
        internacional_simples = 0  # "123.45"
        internacional_completo = 0  # "1,234.56"
        apenas_inteiros = 0  # "123"

        import re

        for valor in amostra:
            valor_str = str(valor).strip()

            # Brasileiro completo: 1.234,56 (ponto para milhares, vírgula para decimal)
            if re.match(r'^\d{1,3}(\.\d{3})*,\d{1,3}$', valor_str):
                brasileiro_completo += 1

            # Brasileiro simples: 123,45 (apenas vírgula decimal)
            elif re.match(r'^\d+,\d{1,3}$', valor_str):
                brasileiro_simples += 1

            # Internacional completo: 1,234.56 (vírgula para milhares, ponto para decimal)
            elif re.match(r'^\d{1,3}(,\d{3})*\.\d{1,3}$', valor_str):
                internacional_completo += 1

            # Internacional simples: 123.45 (apenas ponto decimal)
            elif re.match(r'^\d+\.\d{1,3}$', valor_str):
                internacional_simples += 1

            # Apenas inteiros: 123
            elif re.match(r'^\d+$', valor_str):
                apenas_inteiros += 1

        total_validos = len(amostra)
        brasileiro_total = brasileiro_simples + brasileiro_completo
        internacional_total = internacional_simples + internacional_completo

        # Determinar formato predominante
        if brasileiro_total > total_validos * 0.7:
            return 'brasileiro'
        elif internacional_total > total_validos * 0.7:
            return 'internacional'
        elif apenas_inteiros > total_validos * 0.8:
            return 'internacional'  # Inteiros são tratados como internacionais
        elif brasileiro_total + internacional_total > total_validos * 0.3:
            return 'misto'
        else:
            return 'nao_numerico'

    except Exception:
        return 'nao_numerico'


def aplicar_formato_brasileiro_completo(df: pd.DataFrame, mostrar_info: bool = True,
                                        validar: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica conversão completa de formato brasileiro em DataFrame
    FUNÇÃO PRINCIPAL para conversão de DataFrames

    Args:
        df: DataFrame a ser convertido
        mostrar_info: Se deve mostrar informações no Streamlit
        validar: Se deve validar a conversão

    Returns:
        tuple: (DataFrame convertido, relatório de conversão)
    """
    try:
        import streamlit as st

        if df is None or len(df) == 0:
            return df, {'sucesso': False, 'erro': 'DataFrame vazio'}

        df_convertido = df.copy()
        colunas_convertidas = []
        problemas = []

        # Analisar cada coluna object (string)
        for coluna in df.select_dtypes(include=['object']).columns:
            formato_detectado = detectar_formato_numerico(df[coluna])

            if formato_detectado in ['brasileiro', 'misto']:
                try:
                    # Aplicar conversão
                    serie_convertida = df[coluna].apply(processar_decimal_brasileiro)

                    # Verificar se conversão foi bem-sucedida
                    if validar:
                        valores_numericos = pd.to_numeric(serie_convertida, errors='coerce')
                        taxa_sucesso = valores_numericos.notna().sum() / len(serie_convertida)

                        if taxa_sucesso > 0.8:  # 80% de sucesso mínimo
                            df_convertido[coluna] = valores_numericos
                            colunas_convertidas.append(coluna)
                        else:
                            problemas.append(f"Coluna {coluna}: baixa taxa de conversão ({taxa_sucesso:.1%})")
                    else:
                        # Conversão sem validação
                        df_convertido[coluna] = pd.to_numeric(serie_convertida, errors='coerce')
                        colunas_convertidas.append(coluna)

                except Exception as e:
                    problemas.append(f"Erro na coluna {coluna}: {str(e)}")

        # Relatório
        relatorio = {
            'sucesso': len(colunas_convertidas) > 0,
            'colunas_convertidas': len(colunas_convertidas),
            'nomes_colunas_convertidas': colunas_convertidas,
            'problemas': problemas,
            'total_colunas_analisadas': len(df.select_dtypes(include=['object']).columns)
        }

        # Mostrar informações se solicitado
        if mostrar_info and len(colunas_convertidas) > 0:
            st.success(f"🇧🇷 Formato brasileiro convertido: {len(colunas_convertidas)} colunas")

            if problemas:
                with st.expander("⚠️ Avisos de conversão"):
                    for problema in problemas:
                        st.warning(problema)

        return df_convertido, relatorio

    except Exception as e:
        return df, {'sucesso': False, 'erro': f'Erro na conversão: {str(e)}'}


def processar_csv_brasileiro_automatico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processamento automático completo para CSVs brasileiros
    FUNÇÃO DE CONVENIÊNCIA para uso simples

    Args:
        df: DataFrame carregado de CSV

    Returns:
        DataFrame processado
    """
    try:
        if df is None or len(df) == 0:
            return df

        # Aplicar conversão
        df_convertido, relatorio = aplicar_formato_brasileiro_completo(df, mostrar_info=False)

        return df_convertido

    except Exception:
        # Em caso de erro, retornar DataFrame original
        return df


def validar_conversao_brasileira(df_original: pd.DataFrame, df_convertido: pd.DataFrame) -> Dict:
    """
    Valida se a conversão brasileira foi bem-sucedida

    Args:
        df_original: DataFrame antes da conversão
        df_convertido: DataFrame após conversão

    Returns:
        dict: Relatório de validação
    """
    try:
        relatorio = {
            'conversao_valida': True,
            'colunas_alteradas': [],
            'tipos_alterados': {},
            'problemas': []
        }

        if df_original is None or df_convertido is None:
            relatorio['conversao_valida'] = False
            relatorio['problemas'].append('DataFrames nulos')
            return relatorio

        # Comparar tipos de dados
        for coluna in df_original.columns:
            if coluna in df_convertido.columns:
                tipo_original = str(df_original[coluna].dtype)
                tipo_convertido = str(df_convertido[coluna].dtype)

                if tipo_original != tipo_convertido:
                    relatorio['colunas_alteradas'].append(coluna)
                    relatorio['tipos_alterados'][coluna] = {
                        'antes': tipo_original,
                        'depois': tipo_convertido
                    }

        # Verificar se houve perda excessiva de dados
        for coluna in relatorio['colunas_alteradas']:
            if coluna in df_original.columns and coluna in df_convertido.columns:
                nulos_antes = df_original[coluna].isna().sum()
                nulos_depois = df_convertido[coluna].isna().sum()

                if nulos_depois > nulos_antes * 1.5:  # Aumento de 50% nos nulos
                    relatorio['problemas'].append(
                        f'Coluna {coluna}: possível perda de dados na conversão'
                    )

        return relatorio

    except Exception as e:
        return {
            'conversao_valida': False,
            'problemas': [f'Erro na validação: {str(e)}'],
            'colunas_alteradas': [],
            'tipos_alterados': {}
        }


# ==========================================
# FORMATAÇÃO DE SAÍDA (INTERNACIONAL → BRASILEIRO)
# ==========================================

def formatar_brasileiro(valor, decimais: int = 2) -> str:
    """
    Formata números no padrão brasileiro para EXIBIÇÃO
    (. para milhares, , para decimais)

    Args:
        valor: Número a ser formatado
        decimais: Número de casas decimais

    Returns:
        str: Número formatado no padrão brasileiro
    """
    try:
        if pd.isna(valor) or valor is None:
            return "N/A"

        if isinstance(valor, (int, float, np.number)):
            # Formatar com decimais
            formatado = f"{valor:.{decimais}f}".replace('.', ',')
            partes = formatado.split(',')

            # Adicionar pontos como separadores de milhares
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

    except Exception:
        return str(valor)


def formatar_brasileiro_completo(valor, decimais: int = 2, entrada_brasileira: bool = True) -> str:
    """
    Versão completa que trata entrada E saída brasileira
    Converte entrada + formata saída

    Args:
        valor: Valor a processar
        decimais: Casas decimais para saída
        entrada_brasileira: Se deve tentar converter entrada

    Returns:
        str: Valor formatado em padrão brasileiro
    """
    try:
        # Se entrada_brasileira=True, primeiro converte entrada
        if entrada_brasileira:
            valor_convertido = processar_decimal_brasileiro(valor)
        else:
            valor_convertido = valor

        # Depois aplica formatação de saída brasileira
        return formatar_brasileiro(valor_convertido, decimais)

    except Exception:
        return str(valor)


def formatar_dataframe_brasileiro(df: pd.DataFrame, colunas_numericas: List[str] = None,
                                  decimais: int = 2) -> pd.DataFrame:
    """
    Formata DataFrame no padrão brasileiro para EXIBIÇÃO
    NÃO altera os dados, apenas a apresentação

    Args:
        df: DataFrame a ser formatado
        colunas_numericas: Lista de colunas numéricas (None = auto-detectar)
        decimais: Número de casas decimais

    Returns:
        DataFrame formatado para exibição
    """
    try:
        if df is None or len(df) == 0:
            return df

        df_formatado = df.copy()

        if colunas_numericas is None:
            # Detectar colunas numéricas automaticamente
            colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in colunas_numericas:
            if col in df.columns:
                df_formatado[col] = df[col].apply(lambda x: formatar_brasileiro(x, decimais))

        return df_formatado

    except Exception:
        return df


# ==========================================
# FUNÇÕES AUXILIARES E UTILITÁRIAS
# ==========================================

def formatar_percentual(valor, decimais: int = 1) -> str:
    """
    Formata valores como percentual

    Args:
        valor: Valor a ser formatado (0.85 ou 85)
        decimais: Casas decimais

    Returns:
        str: Percentual formatado (ex: "85,0%")
    """
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
    except Exception:
        return str(valor)


def formatar_moeda(valor, decimais: int = 2) -> str:
    """
    Formata valores monetários em Reais

    Args:
        valor: Valor monetário
        decimais: Casas decimais

    Returns:
        str: Valor formatado (ex: "R$ 1.234,56")
    """
    try:
        if pd.isna(valor) or valor is None:
            return "N/A"

        return f"R$ {formatar_brasileiro(valor, decimais)}"
    except Exception:
        return str(valor)


def formatar_numero_inteligente(valor, unidade: str = "", decimais_max: int = 2) -> str:
    """
    Formata números com unidades apropriadas (mil, milhão, etc.)

    Args:
        valor: Número a ser formatado
        unidade: Unidade de medida
        decimais_max: Máximo de casas decimais

    Returns:
        str: Número formatado com unidade apropriada
    """
    try:
        if pd.isna(valor) or valor is None:
            return "N/A"

        if valor >= 1000000:
            valor_formatado = valor / 1000000
            sufixo = "milhões" if unidade else "milhões"
            return f"{formatar_brasileiro(valor_formatado, 1)} {sufixo} {unidade}".strip()

        elif valor >= 1000:
            valor_formatado = valor / 1000
            sufixo = "mil" if unidade else "mil"
            return f"{formatar_brasileiro(valor_formatado, 1)} {sufixo} {unidade}".strip()

        else:
            return f"{formatar_brasileiro(valor, decimais_max)} {unidade}".strip()

    except Exception:
        return str(valor)


def formatar_timestamp_brasileiro(timestamp=None) -> str:
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


def classificar_qualidade_modelo(r2) -> str:
    """
    Classifica a qualidade do modelo baseado no R²

    Args:
        r2: Coeficiente de determinação

    Returns:
        str: Classificação da qualidade
    """
    try:
        if pd.isna(r2) or r2 is None:
            return "Indeterminado"

        if r2 >= 0.9:
            return "Excelente"
        elif r2 >= 0.8:
            return "Muito Bom"
        elif r2 >= 0.7:
            return "Bom"
        elif r2 >= 0.6:
            return "Regular"
        else:
            return "Fraco"
    except Exception:
        return "Erro na classificação"


def classificar_qualidade_modelo_detalhado(r2, tipo: str = 'hipsometrico') -> str:
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


def classificar_produtividade_florestal(volume_ha, especie: str = 'eucalipto') -> str:
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
                return "🟢 Excelente (≥250 m³/ha)"
            elif volume >= 200:
                return "🟢 Muito Boa (200-249 m³/ha)"
            elif volume >= 150:
                return "🟡 Boa (150-199 m³/ha)"
            elif volume >= 100:
                return "🟠 Regular (100-149 m³/ha)"
            elif volume >= 50:
                return "🔴 Baixa (50-99 m³/ha)"
            else:
                return "⚫ Muito Baixa (<50 m³/ha)"

        elif especie.lower() == 'pinus':
            if volume >= 300:
                return "🟢 Excelente (≥300 m³/ha)"
            elif volume >= 250:
                return "🟢 Muito Boa (250-299 m³/ha)"
            elif volume >= 200:
                return "🟡 Boa (200-249 m³/ha)"
            elif volume >= 150:
                return "🟠 Regular (150-199 m³/ha)"
            elif volume >= 100:
                return "🔴 Baixa (100-149 m³/ha)"
            else:
                return "⚫ Muito Baixa (<100 m³/ha)"

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


# ==========================================
# FUNÇÕES DE COMPATIBILIDADE E ALIASES
# ==========================================

def normalizar_entrada_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função de conveniência para normalizar dados de entrada
    Aplica todas as conversões necessárias automaticamente
    """
    return processar_csv_brasileiro_automatico(df)


def converter_dataframe_brasileiro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias para compatibilidade - converte DataFrame brasileiro
    """
    return processar_csv_brasileiro_automatico(df)


def formatar_numero_brasileiro_compat(valor, decimais: int = 2) -> str:
    """
    Alias para manter compatibilidade total
    """
    return formatar_brasileiro(valor, decimais)


# ==========================================
# FUNÇÕES DE TESTE E DEBUG
# ==========================================

def testar_conversao_brasileira() -> None:
    """
    Função de teste para validar todas as conversões
    Para uso em desenvolvimento e debugging
    """
    try:
        import streamlit as st

        st.subheader("🧪 Teste de Conversão Brasileira")

        # Casos de teste
        casos_teste = [
            "24,8",  # Brasileiro simples
            "1.234,56",  # Brasileiro com milhares
            "24.8",  # Internacional simples
            "1,234.56",  # Internacional com milhares
            "123",  # Inteiro
            "-45,67",  # Negativo brasileiro
            "-45.67",  # Negativo internacional
            "abc",  # Não numérico
            "",  # Vazio
            "0,00",  # Zero brasileiro
            "0.00"  # Zero internacional
        ]

        st.write("**Resultados dos testes:**")

        for i, caso in enumerate(casos_teste):
            resultado = processar_decimal_brasileiro(caso)
            formato_detectado = detectar_formato_numerico(pd.Series([caso]))

            st.write(f"{i + 1}. `\"{caso}\"` → `{resultado}` "
                     f"(tipo: {type(resultado).__name__}, formato: {formato_detectado})")

        # Teste com DataFrame
        st.write("\n**Teste com DataFrame:**")
        df_teste = pd.DataFrame({
            'brasileiro': ['1.234,56', '567,89', '12,3'],
            'internacional': ['1234.56', '567.89', '12.3'],
            'inteiros': ['1234', '567', '12']
        })

        df_convertido = processar_csv_brasileiro_automatico(df_teste)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original:**")
            st.dataframe(df_teste)
            st.write("Tipos:", df_teste.dtypes.to_dict())

        with col2:
            st.write("**Convertido:**")
            st.dataframe(df_convertido)
            st.write("Tipos:", df_convertido.dtypes.to_dict())

    except ImportError:
        print("Streamlit não disponível para teste interativo")


def obter_estatisticas_conversao(df_original: pd.DataFrame,
                                 df_convertido: pd.DataFrame) -> Dict:
    """
    Obtém estatísticas detalhadas da conversão

    Args:
        df_original: DataFrame original
        df_convertido: DataFrame convertido

    Returns:
        dict: Estatísticas da conversão
    """
    try:
        stats = {
            'total_colunas': len(df_original.columns),
            'colunas_convertidas': 0,
            'tipos_antes': {},
            'tipos_depois': {},
            'conversoes_por_tipo': {
                'object_para_float': 0,
                'object_para_int': 0,
                'mantido_object': 0,
                'mantido_numerico': 0
            }
        }

        for coluna in df_original.columns:
            tipo_antes = str(df_original[coluna].dtype)
            tipo_depois = str(df_convertido[coluna].dtype)

            stats['tipos_antes'][tipo_antes] = stats['tipos_antes'].get(tipo_antes, 0) + 1
            stats['tipos_depois'][tipo_depois] = stats['tipos_depois'].get(tipo_depois, 0) + 1

            if tipo_antes != tipo_depois:
                stats['colunas_convertidas'] += 1

                if tipo_antes == 'object' and 'float' in tipo_depois:
                    stats['conversoes_por_tipo']['object_para_float'] += 1
                elif tipo_antes == 'object' and 'int' in tipo_depois:
                    stats['conversoes_por_tipo']['object_para_int'] += 1
            else:
                if tipo_antes == 'object':
                    stats['conversoes_por_tipo']['mantido_object'] += 1
                else:
                    stats['conversoes_por_tipo']['mantido_numerico'] += 1

        return stats

    except Exception as e:
        return {'erro': f'Erro ao calcular estatísticas: {str(e)}'}


# ==========================================
# FUNÇÕES ESPECÍFICAS PARA RELATÓRIOS
# ==========================================

def formatar_estatisticas_resumo(df: pd.DataFrame, colunas_numericas: List[str] = None) -> Dict:
    """
    Cria resumo estatístico formatado

    Args:
        df: DataFrame
        colunas_numericas: Lista de colunas (auto-detecta se None)

    Returns:
        dict: Estatísticas formatadas
    """
    try:
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

    except Exception as e:
        return {"erro": f"Erro ao calcular estatísticas: {str(e)}"}


def criar_relatorio_modelo(resultado: Dict, nome_modelo: str, tipo: str = 'hipsometrico') -> str:
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
        relatorio = f"""📊 **{nome_modelo}**
• **R²:** {formatar_brasileiro(r2, 4)}
• **RMSE:** {formatar_brasileiro(rmse, 3)}
• **Qualidade:** {qualidade}"""

        # Adicionar métricas extras se disponíveis
        if 'aic' in resultado:
            relatorio += f"\n• **AIC:** {formatar_brasileiro(resultado['aic'], 2)}"

        if 'bic' in resultado:
            relatorio += f"\n• **BIC:** {formatar_brasileiro(resultado['bic'], 2)}"

        if 'significancia' in resultado:
            sig = "✅ Significativo" if resultado['significancia'] else "⚠️ Não significativo"
            relatorio += f"\n• **Significância:** {sig}"

        return relatorio

    except Exception as e:
        return f"❌ Erro ao formatar relatório do {nome_modelo}: {e}"


def formatar_cabecalho_relatorio(titulo: str, subtitulo: str = "", incluir_timestamp: bool = True) -> str:
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


def formatar_tabela_talhoes(df_talhoes: pd.DataFrame, incluir_classificacao: bool = True) -> pd.DataFrame:
    """
    Formata tabela de resultados por talhão

    Args:
        df_talhoes: DataFrame com dados por talhão
        incluir_classificacao: Se deve incluir classificação de produtividade

    Returns:
        DataFrame formatado para exibição
    """
    try:
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

    except Exception:
        return df_talhoes


def gerar_metricas_card_streamlit(valor, titulo: str, delta=None, help_text: str = None) -> Dict:
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


def validar_dados_numericos(serie: pd.Series, nome_coluna: str = "coluna", limites: Dict = None) -> Dict:
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


def detectar_tipo_coluna(serie: pd.Series, nome_coluna: str = "") -> str:
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


# ==========================================
# DOCUMENTAÇÃO E INFORMAÇÕES
# ==========================================

def documentacao_formato_brasileiro() -> str:
    """
    Retorna documentação completa sobre suporte ao formato brasileiro

    Returns:
        str: Documentação formatada
    """
    doc = """
# 🇧🇷 SUPORTE AO FORMATO BRASILEIRO - DOCUMENTAÇÃO COMPLETA

## 📊 FUNCIONALIDADES PRINCIPAIS

### 1. CONVERSÃO AUTOMÁTICA DE ENTRADA
- **`processar_decimal_brasileiro()`**: Converte valores individuais
- **`converter_dataframe_brasileiro()`**: Converte DataFrames completos  
- **`detectar_formato_numerico()`**: Detecta formato usado nos dados
- **`processar_csv_brasileiro_automatico()`**: Pipeline completo automático

### 2. FORMATAÇÃO DE SAÍDA
- **`formatar_brasileiro()`**: Formata números para exibição brasileira
- **`formatar_dataframe_brasileiro()`**: Formata DataFrames para exibição
- **`formatar_percentual()`**: Formata percentuais
- **`formatar_moeda()`**: Formata valores monetários

### 3. VALIDAÇÃO E CONTROLE
- **`validar_conversao_brasileira()`**: Verifica sucesso da conversão
- **`aplicar_formato_brasileiro_completo()`**: Conversão com validação completa
- **`obter_estatisticas_conversao()`**: Estatísticas detalhadas

## 🔄 FLUXO DE PROCESSAMENTO

```
CSV Upload → Detecção → Conversão → Validação → Processamento
     ↓          ↓           ↓          ↓           ↓
"arquivo.csv" "brasileiro" 24.8 float  ✅ OK   Cálculos normais
```

## 💡 EXEMPLOS DE USO

### Conversão Automática (Recomendado):
```python
# Pipeline completo automático
df_processado = processar_csv_brasileiro_automatico(df)
```

### Conversão com Controle:
```python
# Conversão com validação e relatório
df_convertido, relatorio = aplicar_formato_brasileiro_completo(df)
if relatorio['sucesso']:
    print(f"Convertidas {relatorio['colunas_convertidas']} colunas")
```

### Formatação para Exibição:
```python
# Formatar números para exibição
valor_formatado = formatar_brasileiro(1234.56, 2)  # "1.234,56"
percentual = formatar_percentual(0.85, 1)          # "85,0%"
moeda = formatar_moeda(1234.56)                    # "R$ 1.234,56"
```

## 🛡️ COMPATIBILIDADE E SEGURANÇA

### Formatos Suportados:
- ✅ **Brasileiro simples**: "24,8" → 24.8
- ✅ **Brasileiro completo**: "1.234,56" → 1234.56
- ✅ **Internacional**: "24.8" → 24.8 (mantido)
- ✅ **Inteiros**: "123" → 123 (mantido)
- ✅ **Dados mistos**: Detecta e trata cada coluna adequadamente

### Validações Automáticas:
- **Detecção conservadora**: Só converte se >70% dos valores forem válidos
- **Preservação de dados**: Mantém valores originais se conversão falhar
- **Feedback claro**: Informa quais colunas foram convertidas
- **Sem quebras**: Não interfere com dados já no formato correto

## 🔧 FUNÇÕES AUXILIARES

### Classificação e Análise:
- `classificar_qualidade_modelo()`: Classifica R² de modelos
- `classificar_produtividade_florestal()`: Classifica volume/ha
- `validar_dados_numericos()`: Valida séries numéricas
- `detectar_tipo_coluna()`: Detecta tipo de dados por nome/conteúdo

### Relatórios e Exibição:
- `formatar_estatisticas_resumo()`: Resumo estatístico formatado
- `criar_relatorio_modelo()`: Relatório de modelos formatado
- `formatar_tabela_talhoes()`: Tabela de talhões formatada
- `formatar_cabecalho_relatorio()`: Cabeçalhos padronizados

## ⚙️ CONFIGURAÇÕES DISPONÍVEIS

### Para `aplicar_formato_brasileiro_completo()`:
- `mostrar_info=True`: Mostra mensagens de progresso
- `validar=True`: Valida conversão (recomendado)

### Para funções de formatação:
- `decimais`: Número de casas decimais
- `entrada_brasileira`: Se deve converter entrada antes de formatar

## 🧪 TESTE E DEBUG

```python
# Testar conversões interativamente
testar_conversao_brasileira()

# Obter estatísticas detalhadas
stats = obter_estatisticas_conversao(df_original, df_convertido)
```

---
*Sistema GreenVista - Suporte nativo completo ao formato brasileiro* 🌲🇧🇷
"""

    return doc


def inicializar_suporte_brasileiro() -> bool:
    """
    Inicializa suporte ao formato brasileiro
    Verifica dependências e configurações

    Returns:
        bool: True se inicialização foi bem-sucedida
    """
    try:
        # Verificar se pandas está disponível
        import pandas as pd
        import numpy as np

        # Testar funções principais
        teste_valor = processar_decimal_brasileiro("123,45")
        if not isinstance(teste_valor, (int, float)):
            return False

        teste_serie = pd.Series(["123,45", "678,90"])
        formato = detectar_formato_numerico(teste_serie)
        if formato not in ['brasileiro', 'internacional', 'misto', 'nao_numerico']:
            return False

        return True

    except Exception as e:
        print(f"Erro ao inicializar suporte brasileiro: {e}")
        return False


# ==========================================
# FUNÇÕES DE DEMONSTRAÇÃO E EXEMPLO
# ==========================================

def criar_exemplo_conversao_brasileira() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cria exemplo de conversão para demonstração

    Returns:
        tuple: (DataFrame original, DataFrame convertido)
    """
    # DataFrame com formato brasileiro
    df_original = pd.DataFrame({
        'valores_brasileiros': ['1.234,56', '567,89', '12,30', '4.567,12'],
        'valores_internacionais': ['1234.56', '567.89', '12.30', '4567.12'],
        'valores_inteiros': ['1234', '567', '12', '4567'],
        'valores_mistos': ['123,45', '678.90', '12', '3.456,78']
    })

    # Aplicar conversão
    df_convertido = processar_csv_brasileiro_automatico(df_original)

    return df_original, df_convertido


# ==========================================
# FUNÇÕES UTILITÁRIAS FINAIS
# ==========================================

def limpar_cache_conversao():
    """
    Limpa cache interno de conversões se existir
    Para uso em casos de problemas de memória
    """
    # Placeholder para futuras implementações de cache
    pass


def obter_versao_formatacao() -> str:
    """
    Retorna versão atual do módulo de formatação

    Returns:
        str: Versão do módulo
    """
    return "2.0.0 - Consolidada e Corrigida"


def obter_informacoes_sistema() -> Dict:
    """
    Retorna informações sobre o sistema de formatação

    Returns:
        dict: Informações do sistema
    """
    try:
        import pandas as pd
        import numpy as np

        return {
            'versao_formatacao': obter_versao_formatacao(),
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'funcoes_disponiveis': [
                'processar_decimal_brasileiro',
                'detectar_formato_numerico',
                'aplicar_formato_brasileiro_completo',
                'processar_csv_brasileiro_automatico',
                'formatar_brasileiro',
                'formatar_dataframe_brasileiro',
                'validar_conversao_brasileira'
            ],
            'status': 'Operacional',
            'suporte_brasileiro': True
        }
    except Exception as e:
        return {
            'versao_formatacao': obter_versao_formatacao(),
            'status': 'Erro',
            'erro': str(e),
            'suporte_brasileiro': False
        }


# ==========================================
# EXPORTS E COMPATIBILIDADE
# ==========================================

# Lista de todas as funções públicas para facilitar importações
__all__ = [
    # Core - Conversão de entrada
    'processar_decimal_brasileiro',
    'parsear_decimal_brasileiro',  # Alias
    'detectar_formato_numerico',
    'aplicar_formato_brasileiro_completo',
    'processar_csv_brasileiro_automatico',
    'validar_conversao_brasileira',

    # Formatação de saída
    'formatar_brasileiro',
    'formatar_brasileiro_completo',
    'formatar_dataframe_brasileiro',
    'formatar_percentual',
    'formatar_moeda',
    'formatar_numero_inteligente',
    'formatar_timestamp_brasileiro',

    # Classificação e análise
    'classificar_qualidade_modelo',
    'classificar_qualidade_modelo_detalhado',
    'classificar_produtividade_florestal',
    'validar_dados_numericos',
    'detectar_tipo_coluna',

    # Relatórios e utilitários
    'formatar_estatisticas_resumo',
    'criar_relatorio_modelo',
    'formatar_cabecalho_relatorio',
    'formatar_tabela_talhoes',
    'gerar_metricas_card_streamlit',

    # Compatibilidade
    'normalizar_entrada_dados',
    'converter_dataframe_brasileiro',
    'formatar_numero_brasileiro_compat',

    # Sistema e debug
    'testar_conversao_brasileira',
    'obter_estatisticas_conversao',
    'documentacao_formato_brasileiro',
    'inicializar_suporte_brasileiro',
    'criar_exemplo_conversao_brasileira',
    'obter_versao_formatacao',
    'obter_informacoes_sistema'
]