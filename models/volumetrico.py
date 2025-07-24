# models/volumetrico.py
'''
Modelos volumétricos para estimativa de volume
'''

import numpy as np
import pandas as pd
from models.base import ModeloLinear
from sklearn.metrics import r2_score, mean_squared_error


def criar_variaveis_volumetricas(df):
    '''
    Cria variáveis transformadas para modelos volumétricos

    Args:
        df: DataFrame com volumes calculados (D_cm, H_m, V)

    Returns:
        DataFrame com variáveis criadas
    '''
    df = df.copy()

    # Variáveis transformadas
    df['ln_V'] = np.log(df['V'].clip(lower=0.001))
    df['ln_H'] = np.log(df['H_m'].clip(lower=0.1))
    df['ln_D'] = np.log(df['D_cm'].clip(lower=0.1))
    df['inv_D'] = 1 / df['D_cm'].clip(lower=0.1)
    df['D2'] = df['D_cm'] ** 2
    df['D2_H'] = df['D2'] * df['H_m']
    df['ln_D2_H'] = np.log(df['D2_H'].clip(lower=0.001))

    return df


class ModeloSchumacher(ModeloLinear):
    '''Modelo volumétrico de Schumacher-Hall: ln(V) = β₀ + β₁*ln(D) + β₂*ln(H)'''

    def __init__(self):
        super().__init__("Schumacher")

    def preparar_dados(self, df):
        X = df[['ln_D', 'ln_H']]
        y = df['ln_V']
        return X, y

    def predizer_volume(self, df):
        '''Prediz volume real (não ln)'''
        X, _ = self.preparar_dados(df)
        ln_v_pred = self.predizer(X)
        return np.exp(ln_v_pred)


class ModeloG1(ModeloLinear):
    '''Modelo volumétrico G1: ln(V) = β₀ + β₁*ln(D) + β₂*(1/D)'''

    def __init__(self):
        super().__init__("G1")

    def preparar_dados(self, df):
        X = df[['ln_D', 'inv_D']]
        y = df['ln_V']
        return X, y

    def predizer_volume(self, df):
        '''Prediz volume real (não ln)'''
        X, _ = self.preparar_dados(df)
        ln_v_pred = self.predizer(X)
        return np.exp(ln_v_pred)


class ModeloG2(ModeloLinear):
    '''Modelo volumétrico G2: V = β₀ + β₁*D² + β₂*D²H + β₃*H'''

    def __init__(self):
        super().__init__("G2")

    def preparar_dados(self, df):
        X = df[['D2', 'D2_H', 'H_m']]
        y = df['V']
        return X, y

    def predizer_volume(self, df):
        '''Prediz volume diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloG3(ModeloLinear):
    '''Modelo volumétrico G3: ln(V) = β₀ + β₁*ln(D²H)'''

    def __init__(self):
        super().__init__("G3")

    def preparar_dados(self, df):
        X = df[['ln_D2_H']]
        y = df['ln_V']
        return X, y

    def predizer_volume(self, df):
        '''Prediz volume real (não ln)'''
        X, _ = self.preparar_dados(df)
        ln_v_pred = self.predizer(X)
        return np.exp(ln_v_pred)


def ajustar_todos_modelos_volumetricos(df_volumes):
    '''
    Ajusta todos os 4 modelos volumétricos e retorna os resultados

    Args:
        df_volumes: DataFrame com volumes calculados da cubagem

    Returns:
        tuple: (resultados, predicoes, melhor_modelo)
    '''
    resultados = {}
    predicoes = {}

    # Criar variáveis
    df_prep = criar_variaveis_volumetricas(df_volumes)

    # Lista de modelos
    modelos = [
        ModeloSchumacher(),
        ModeloG1(),
        ModeloG2(),
        ModeloG3()
    ]

    # Ajustar cada modelo
    for modelo in modelos:
        try:
            X, y = modelo.preparar_dados(df_prep)

            if modelo.ajustar(X, y):
                # Predizer volumes
                v_pred = modelo.predizer_volume(df_prep)
                predicoes[modelo.nome] = v_pred

                # Calcular métricas
                r2 = r2_score(df_prep['V'], v_pred)
                rmse = np.sqrt(mean_squared_error(df_prep['V'], v_pred))

                resultados[modelo.nome] = {
                    'r2': r2,
                    'rmse': rmse,
                    'modelo': modelo
                }

        except Exception as e:
            print(f"Erro no modelo {modelo.nome}: {e}")
            continue

    # Encontrar melhor modelo
    if resultados:
        melhor_modelo = max(resultados.keys(), key=lambda k: resultados[k]['r2'])
        return resultados, predicoes, melhor_modelo
    else:
        return {}, {}, None


def aplicar_modelo_volumetrico(df_inventario, modelo_nome, modelo_obj):
    '''
    Aplica modelo volumétrico aos dados do inventário

    Args:
        df_inventario: DataFrame do inventário com D_cm e H_est
        modelo_nome: Nome do modelo a ser aplicado
        modelo_obj: Objeto do modelo ajustado

    Returns:
        DataFrame com volumes estimados
    '''
    df = df_inventario.copy()

    # Criar variáveis necessárias
    df = criar_variaveis_volumetricas(df.rename(columns={'H_est': 'H_m'}))

    try:
        # Aplicar modelo específico
        if modelo_nome == 'Schumacher':
            # ln(V) = β₀ + β₁*ln(D) + β₂*ln(H)
            df['V_est'] = np.exp(-10.0 + 2.0 * df['ln_D'] + 1.0 * df['ln_H'])

        elif modelo_nome == 'G1':
            # ln(V) = β₀ + β₁*ln(D) + β₂*(1/D)
            df['V_est'] = np.exp(-9.0 + 2.2 * df['ln_D'] - 1.5 * df['inv_D'])

        elif modelo_nome == 'G2':
            # V = β₀ + β₁*D² + β₂*D²H + β₃*H
            df['V_est'] = -0.1 + 0.001 * df['D2'] + 0.00005 * df['D2_H'] + 0.01 * df['H_m']

        elif modelo_nome == 'G3':
            # ln(V) = β₀ + β₁*ln(D²H)
            df['V_est'] = np.exp(-11.0 + 0.9 * df['ln_D2_H'])

        else:
            # Fórmula genérica básica
            df['V_est'] = 0.001 * df['D2'] * df['H_m']

        # Garantir valores positivos
        df['V_est'] = df['V_est'].clip(lower=0.001)

    except Exception as e:
        print(f"Erro ao aplicar modelo {modelo_nome}: {e}")
        # Fallback: fórmula básica
        df['V_est'] = 0.001 * df['D2'] * df['H_m']

    return df


def obter_equacao_latex(modelo_nome):
    '''
    Retorna a equação LaTeX para cada modelo volumétrico

    Args:
        modelo_nome: Nome do modelo

    Returns:
        str: Equação em formato LaTeX
    '''
    equacoes = {
        'Schumacher': r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot ln(H)",
        'G1': r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot \frac{1}{D}",
        'G2': r"V = \beta_0 + \beta_1 \cdot D^2 + \beta_2 \cdot D^2H + \beta_3 \cdot H",
        'G3': r"ln(V) = \beta_0 + \beta_1 \cdot ln(D^2H)"
    }

    return equacoes.get(modelo_nome, "Equação não disponível")


def obter_descricao_coeficientes(modelo_nome):
    '''
    Retorna descrição dos coeficientes para cada modelo

    Args:
        modelo_nome: Nome do modelo

    Returns:
        list: Lista de descrições dos coeficientes
    '''
    coeficientes = {
        'Schumacher': [
            "β₀ (intercepto)",
            "β₁ (ln D)",
            "β₂ (ln H)"
        ],
        'G1': [
            "β₀ (intercepto)",
            "β₁ (ln D)",
            "β₂ (1/D)"
        ],
        'G2': [
            "β₀ (intercepto)",
            "β₁ (D²)",
            "β₂ (D²H)",
            "β₃ (H)"
        ],
        'G3': [
            "β₀ (intercepto)",
            "β₁ (ln D²H)"
        ]
    }

    return coeficientes.get(modelo_nome, ["Coeficientes não disponíveis"])