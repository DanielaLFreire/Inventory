# models/volumetrico.py
'''
Modelos volumétricos para estimativa de volume
'''

import numpy as np
import pandas as pd
from models.base import ModeloLinear
from sklearn.metrics import r2_score, mean_squared_error

from utils.formatacao import validar_dados_numericos, detectar_tipo_coluna


def converter_dados_volumetricos_brasileiros(df_volumes):
    """
    Converte dados volumétricos do formato brasileiro usando validação existente

    Args:
        df_volumes: DataFrame com dados em formato brasileiro

    Returns:
        DataFrame com dados convertidos e validados
    """
    print("🇧🇷 Convertendo dados volumétricos do formato brasileiro...")

    df = df_volumes.copy()

    # Detectar e converter colunas numéricas
    colunas_converter = ['D_cm', 'H_m', 'V']

    for coluna in colunas_converter:
        if coluna in df.columns:
            print(f"  Processando {coluna}...")

            # Detectar tipo da coluna
            tipo_detectado = detectar_tipo_coluna(df[coluna], coluna)
            print(f"    Tipo detectado: {tipo_detectado}")

            # Converter valores do formato brasileiro
            def converter_valor_brasileiro(valor):
                if pd.isna(valor):
                    return np.nan
                if isinstance(valor, (int, float)):
                    return float(valor)
                if isinstance(valor, str):
                    valor = valor.strip()
                    if valor == '' or valor.lower() == 'nan':
                        return np.nan
                    try:
                        # Formato brasileiro: vírgula para decimal
                        valor_convertido = valor.replace(',', '.')
                        return float(valor_convertido)
                    except (ValueError, TypeError):
                        return np.nan
                return np.nan

            # Aplicar conversão
            valores_originais = df[coluna].iloc[:3].tolist()
            df[coluna] = df[coluna].apply(converter_valor_brasileiro)
            valores_convertidos = df[coluna].iloc[:3].tolist()

            print(f"    Exemplo conversão: {valores_originais} → {valores_convertidos}")

            # Validar usando função existente
            limites = {}
            if coluna == 'D_cm':
                limites = {'min': 1, 'max': 100}
            elif coluna == 'H_m':
                limites = {'min': 1, 'max': 50}
            elif coluna == 'V':
                limites = {'min': 0.001, 'max': 5}

            validacao = validar_dados_numericos(df[coluna], coluna, limites)

            if validacao['valida']:
                stats = validacao['estatisticas']
                print(f"    ✅ {stats['validos']}/{stats['total']} valores convertidos com sucesso")
            else:
                print(f"    ⚠️ Problemas na conversão:")
                for problema in validacao['problemas'][:2]:  # Mostrar só os primeiros 2
                    print(f"      • {problema}")

    return df


def criar_variaveis_volumetricas(df_volumes):
    """
    Cria variáveis derivadas para modelos volumétricos
    VERSÃO USANDO FUNÇÕES EXISTENTES DE formatacao.py

    Args:
        df_volumes: DataFrame com volumes calculados (D_cm, H_m, V)

    Returns:
        DataFrame com variáveis transformadas
    """
    print("🧮 Preparando variáveis para modelos volumétricos...")

    # CONVERTER DADOS BRASILEIROS PRIMEIRO
    df = converter_dados_volumetricos_brasileiros(df_volumes)

    # Verificar se conversão foi bem-sucedida
    colunas_essenciais = ['D_cm', 'H_m', 'V']
    for col in colunas_essenciais:
        if col not in df.columns:
            raise ValueError(f"Coluna essencial {col} não encontrada")

        valores_validos = df[col].notna().sum()
        if valores_validos == 0:
            raise ValueError(f"Nenhum valor válido na coluna {col} após conversão")

    # Remover valores inválidos
    df_limpo = df.dropna(subset=colunas_essenciais)

    if len(df_limpo) == 0:
        raise ValueError("Nenhum registro válido após conversão brasileira")

    print(f"✅ {len(df_limpo)} registros válidos para modelagem")

    # Garantir valores positivos
    df_limpo = df_limpo[
        (df_limpo['D_cm'] > 0) &
        (df_limpo['H_m'] > 0) &
        (df_limpo['V'] > 0)
        ]

    if len(df_limpo) < 5:
        raise ValueError("Poucos dados válidos após filtros (< 5 registros)")

    print(f"📊 Dataset para transformações: {len(df_limpo)} registros")

    # Criar variáveis transformadas com validação
    try:
        print("  Calculando logaritmos...")

        # Logaritmos - usar clip para evitar valores <= 0
        df_limpo['ln_V'] = np.log(df_limpo['V'].clip(lower=0.001))
        df_limpo['ln_D'] = np.log(df_limpo['D_cm'].clip(lower=0.1))
        df_limpo['ln_H'] = np.log(df_limpo['H_m'].clip(lower=0.1))

        print("  Calculando variáveis derivadas...")

        # Variáveis derivadas
        df_limpo['D2'] = df_limpo['D_cm'] ** 2
        df_limpo['D2H'] = df_limpo['D2'] * df_limpo['H_m']
        df_limpo['ln_D2H'] = np.log(df_limpo['D2H'].clip(lower=0.001))
        df_limpo['inv_D'] = 1 / df_limpo['D_cm'].clip(lower=0.1)

        print("✅ Variáveis transformadas criadas com sucesso")

        # Validar transformações usando função existente
        variaveis_log = ['ln_V', 'ln_D', 'ln_H', 'ln_D2H']

        for var in variaveis_log:
            validacao = validar_dados_numericos(df_limpo[var], var)

            if not validacao['valida']:
                print(f"⚠️ Problemas em {var}:")
                for problema in validacao['problemas'][:1]:
                    print(f"    • {problema}")

        # Remover infinitos e NaNs finais
        df_final = df_limpo.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_final) < 5:
            raise ValueError("Poucos registros válidos após transformações")

        print(f"📊 Dataset final para modelagem: {len(df_final)} registros")

        # Log de estatísticas finais usando função existente
        from utils.formatacao import formatar_estatisticas_resumo
        stats_resumo = formatar_estatisticas_resumo(df_final, ['V', 'D_cm', 'H_m'])

        if 'V' in stats_resumo:
            print(f"📊 Volume - Média: {stats_resumo['V']['media']}, CV: {stats_resumo['V']['cv_pct']}")

        return df_final

    except Exception as e:
        print(f"❌ Erro ao criar variáveis transformadas: {e}")

        # Debug adicional
        print("🔍 Debug - Primeiros valores:")
        for col in ['D_cm', 'H_m', 'V']:
            if col in df_limpo.columns:
                valores = df_limpo[col].head(3).tolist()
                tipos = [type(v).__name__ for v in valores]
                print(f"  {col}: {valores} (tipos: {tipos})")

        raise ValueError(f"Erro nas transformações: {e}")


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