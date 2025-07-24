# utils/formatacao.py
'''
Funções para formatação de números e dados no padrão brasileiro
'''

import pandas as pd
import numpy as np


def formatar_brasileiro(valor, decimais=2):
    '''
    Formata números no padrão brasileiro (. para milhares, , para decimais)

    Args:
        valor: Número a ser formatado
        decimais: Número de casas decimais

    Returns:
        str: Número formatado no padrão brasileiro
    '''
    try:
        if isinstance(valor, (int, float)):
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

    Args:
        df: DataFrame a ser formatado
        colunas_numericas: Lista de colunas numéricas (None = auto-detectar)
        decimais: Número de casas decimais

    Returns:
        DataFrame formatado
    '''
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

    Args:
        valor: Número a ser formatado
        unidade: Unidade de medida
        decimais_max: Máximo de casas decimais

    Returns:
        str: Número formatado com unidade apropriada
    '''
    try:
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

    Args:
        r2: Coeficiente de determinação

    Returns:
        str: Classificação da qualidade
    '''
    if r2 >= 0.9:
        return "***** Excelente"
    elif r2 >= 0.8:
        return "**** Muito Bom"
    elif r2 >= 0.7:
        return "*** Bom"
    elif r2 >= 0.6:
        return "** Regular"
    else:
        return "* Fraco"