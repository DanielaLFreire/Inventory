# models/hipsometrico.py - VERSÃO MODIFICADA COM CONFIGURAÇÕES
'''
Modelos hipsométricos para estimativa de altura
ATUALIZADO: Agora aceita parâmetros de configuração
'''

import numpy as np
import pandas as pd
from models.base import ModeloLinear, ModeloNaoLinear, ajustar_modelo_seguro, calcular_r2_generalizado
from sklearn.metrics import mean_squared_error


# MANTER TODAS AS CLASSES EXISTENTES INALTERADAS
# (Copiando suas classes exatamente como estão)

def calcular_altura_dominante(df):
    '''
    Calcula altura dominante por parcela com múltiplas estratégias

    Args:
        df: DataFrame com dados do inventário

    Returns:
        DataFrame com parcela e H_dom
    '''
    dominantes_list = []

    # Estratégia 1: Usar árvores marcadas como dominantes (cod = 'D')
    arvores_dominantes = df[df['cod'] == 'D']

    if len(arvores_dominantes) > 0:
        dominantes_parcela = arvores_dominantes.groupby('parcela')['H_m'].mean().reset_index()
        dominantes_parcela.columns = ['parcela', 'H_dom']
        dominantes_list.extend(dominantes_parcela.to_dict('records'))
        parcelas_com_dominantes = set(dominantes_parcela['parcela'])
    else:
        parcelas_com_dominantes = set()

    # Estratégia 2: Para parcelas sem dominantes, calcular automaticamente
    todas_parcelas = set(df['parcela'].unique())
    parcelas_sem_dominantes = todas_parcelas - parcelas_com_dominantes

    if parcelas_sem_dominantes:
        for parcela in parcelas_sem_dominantes:
            dados_parcela = df[df['parcela'] == parcela]

            if len(dados_parcela) > 0:
                # Pegar as 3 maiores árvores em diâmetro (ou todas se menos de 3)
                n_arvores = min(3, len(dados_parcela))
                maiores_arvores = dados_parcela.nlargest(n_arvores, 'D_cm')
                h_dom = maiores_arvores['H_m'].mean()

                dominantes_list.append({
                    'parcela': parcela,
                    'H_dom': h_dom
                })

    # Converter para DataFrame
    if dominantes_list:
        dominantes_df = pd.DataFrame(dominantes_list)
        dominantes_df['H_dom'] = dominantes_df['H_dom'].fillna(df['H_m'].mean())
        return dominantes_df
    else:
        # Fallback extremo
        parcelas_unicas = df['parcela'].unique()
        h_media = df['H_m'].mean()

        dominantes_df = pd.DataFrame({
            'parcela': parcelas_unicas,
            'H_dom': [h_media] * len(parcelas_unicas)
        })

        return dominantes_df


def criar_variaveis_hipsometricas(df, dominantes):
    '''
    Cria variáveis transformadas para modelos hipsométricos

    Args:
        df: DataFrame com dados básicos
        dominantes: DataFrame com altura dominante por parcela

    Returns:
        DataFrame com variáveis criadas
    '''
    # Fazer merge com dominantes
    if 'H_dom' not in df.columns:
        df = df.merge(dominantes, on='parcela', how='left')

    # Verificar e corrigir H_dom
    h_dom_medio = df['H_dom'].mean()
    if pd.isna(h_dom_medio) or h_dom_medio <= 0:
        h_dom_medio = df['H_m'].mean()

    df['H_dom'] = df['H_dom'].fillna(h_dom_medio)

    # Criar variáveis transformadas com clipping
    df['ln_H'] = np.log(df['H_m'].clip(lower=0.1))
    df['inv_D'] = 1 / df['D_cm'].clip(lower=0.1)
    df['D2'] = df['D_cm'] ** 2
    df['ln_D'] = np.log(df['D_cm'].clip(lower=0.1))
    df['ln_H_dom'] = np.log(df['H_dom'].clip(lower=0.1))

    # Produtividade (Prod)
    h_adjusted = (df['H_m'] - 1.3).clip(lower=0.1)
    df['Prod'] = df['D2'] / h_adjusted

    # Idade (se disponível)
    if 'idade_anos' in df.columns:
        idade_media = df['idade_anos'].mean()
        if pd.isna(idade_media) or idade_media <= 0:
            idade_media = 5.0

        df['idade_anos'] = df['idade_anos'].fillna(idade_media)
        df['DI'] = df['D_cm'] * df['idade_anos']

    return df


class ModeloCurtis(ModeloLinear):
    '''Modelo hipsométrico de Curtis: ln(H) = β₀ + β₁ * (1/D)'''

    def __init__(self):
        super().__init__("Curtis")

    def preparar_dados(self, df):
        X = df[['inv_D']]
        y = df['ln_H']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura real (não ln)'''
        X, _ = self.preparar_dados(df)
        ln_h_pred = self.predizer(X)
        return np.exp(ln_h_pred)


class ModeloCampos(ModeloLinear):
    '''Modelo hipsométrico de Campos: ln(H) = β₀ + β₁ * (1/D) + β₂ * ln(H_dom)'''

    def __init__(self):
        super().__init__("Campos")

    def preparar_dados(self, df):
        X = df[['inv_D', 'ln_H_dom']]
        y = df['ln_H']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura real (não ln)'''
        X, _ = self.preparar_dados(df)
        ln_h_pred = self.predizer(X)
        return np.exp(ln_h_pred)


class ModeloHenri(ModeloLinear):
    '''Modelo hipsométrico de Henri: H = β₀ + β₁ * ln(D)'''

    def __init__(self):
        super().__init__("Henri")

    def preparar_dados(self, df):
        X = df[['ln_D']]
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloProdan(ModeloLinear):
    '''Modelo hipsométrico de Prodan: D²/(H-1.3) = β₀ + β₁*D + β₂*D² + β₃*D*Idade'''

    def __init__(self):
        super().__init__("Prodan")
        self.tem_idade = False

    def preparar_dados(self, df):
        # Verificar disponibilidade de idade
        self.tem_idade = 'idade_anos' in df.columns and df['idade_anos'].notna().sum() > 10

        if self.tem_idade:
            colunas = ['D_cm', 'D2', 'DI']
        else:
            colunas = ['D_cm', 'D2']

        X = df[colunas]
        y = df['Prod']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura através da produtividade'''
        X, _ = self.preparar_dados(df)
        prod_pred = self.predizer(X)
        return (df['D2'] / np.clip(prod_pred, 0.1, None)) + 1.3


class ModeloChapman(ModeloNaoLinear):
    '''Modelo hipsométrico de Chapman: H = b₀ * (1 - exp(-b₁ * D))^b₂'''

    def __init__(self, altura_max, max_iter=5000, tolerancia=0.01):
        def chapman_func(D, b0, b1, b2):
            return b0 * (1 - np.exp(-b1 * D)) ** b2

        super().__init__("Chapman", chapman_func, [altura_max, 0.01, 1.0])
        # NOVO: Configurar parâmetros de otimização
        self.max_iter = max_iter
        self.tolerancia = tolerancia

    def preparar_dados(self, df):
        X = df['D_cm']
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloWeibull(ModeloNaoLinear):
    '''Modelo hipsométrico de Weibull: H = a * (1 - exp(-b * D^c))'''

    def __init__(self, altura_max, max_iter=5000, tolerancia=0.01):
        def weibull_func(D, a, b, c):
            return a * (1 - np.exp(-b * D ** c))

        super().__init__("Weibull", weibull_func, [altura_max, 0.01, 1.0])
        # NOVO: Configurar parâmetros de otimização
        self.max_iter = max_iter
        self.tolerancia = tolerancia

    def preparar_dados(self, df):
        X = df['D_cm']
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloMononuclear(ModeloNaoLinear):
    '''Modelo hipsométrico Mononuclear: H = a * (1 - b * exp(-c * D))'''

    def __init__(self, altura_max, max_iter=5000, tolerancia=0.01):
        def mono_func(D, a, b, c):
            return a * (1 - b * np.exp(-c * D))

        super().__init__("Mononuclear", mono_func, [altura_max, 1.0, 0.1])
        # NOVO: Configurar parâmetros de otimização
        self.max_iter = max_iter
        self.tolerancia = tolerancia

    def preparar_dados(self, df):
        X = df['D_cm']
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


# FUNÇÃO ORIGINAL (mantida para compatibilidade)
def ajustar_todos_modelos_hipsometricos_original(df):
    '''
    FUNÇÃO ORIGINAL - mantida para compatibilidade
    Ajusta todos os 7 modelos hipsométricos e retorna os resultados

    Args:
        df: DataFrame com dados preparados

    Returns:
        tuple: (resultados, predicoes, melhor_modelo)
    '''
    resultados = {}
    predicoes = {}

    # Calcular altura dominante
    dominantes = calcular_altura_dominante(df)

    # Criar variáveis
    df_prep = criar_variaveis_hipsometricas(df, dominantes)

    altura_max = df_prep['H_m'].max() * 1.2

    # Lista de modelos
    modelos = [
        ModeloCurtis(),
        ModeloCampos(),
        ModeloHenri(),
        ModeloProdan(),
        ModeloChapman(altura_max),
        ModeloWeibull(altura_max),
        ModeloMononuclear(altura_max)
    ]

    # Ajustar cada modelo
    for modelo in modelos:
        try:
            X, y = modelo.preparar_dados(df_prep)

            if modelo.ajustar(X, y):
                # Predizer alturas
                h_pred = modelo.predizer_altura(df_prep)
                predicoes[modelo.nome] = h_pred

                # Calcular métricas
                r2g = calcular_r2_generalizado(df_prep['H_m'], h_pred)
                rmse = np.sqrt(mean_squared_error(df_prep['H_m'], h_pred))

                resultados[modelo.nome] = {
                    'r2g': r2g,
                    'rmse': rmse,
                    'modelo': modelo
                }

        except Exception as e:
            print(f"Erro no modelo {modelo.nome}: {e}")
            continue

    # Encontrar melhor modelo
    if resultados:
        melhor_modelo = max(resultados.keys(), key=lambda k: resultados[k]['r2g'])
        return resultados, predicoes, melhor_modelo
    else:
        return {}, {}, None


# NOVA FUNÇÃO COM CONFIGURAÇÕES
def ajustar_todos_modelos_hipsometricos(df, config=None):
    '''
    NOVA VERSÃO: Ajusta todos os modelos hipsométricos com configurações opcionais

    Args:
        df: DataFrame com dados preparados
        config: Dict com configurações opcionais (NOVO)
               - incluir_nao_lineares: bool (default: True)
               - max_iteracoes: int (default: 5000)
               - tolerancia_ajuste: float (default: 0.01)

    Returns:
        tuple: (resultados, predicoes, melhor_modelo)
    '''
    # NOVO: Processar configurações
    if config is None:
        config = {}

    incluir_nao_lineares = config.get('incluir_nao_lineares', True)
    max_iteracoes = config.get('max_iteracoes', 5000)
    tolerancia_ajuste = config.get('tolerancia_ajuste', 0.01)

    resultados = {}
    predicoes = {}

    # Calcular altura dominante
    dominantes = calcular_altura_dominante(df)

    # Criar variáveis
    df_prep = criar_variaveis_hipsometricas(df, dominantes)

    altura_max = df_prep['H_m'].max() * 1.2

    # Lista de modelos - SEPARAR lineares e não-lineares
    modelos_lineares = [
        ModeloCurtis(),
        ModeloCampos(),
        ModeloHenri(),
        ModeloProdan()
    ]

    modelos_nao_lineares = []
    if incluir_nao_lineares:
        modelos_nao_lineares = [
            ModeloChapman(altura_max, max_iteracoes, tolerancia_ajuste),
            ModeloWeibull(altura_max, max_iteracoes, tolerancia_ajuste),
            ModeloMononuclear(altura_max, max_iteracoes, tolerancia_ajuste)
        ]

    # Combinar modelos conforme configuração
    todos_modelos = modelos_lineares + modelos_nao_lineares

    print(
        f"Ajustando {len(todos_modelos)} modelos (Lineares: {len(modelos_lineares)}, Não-lineares: {len(modelos_nao_lineares)})")

    # Ajustar cada modelo
    for modelo in todos_modelos:
        try:
            X, y = modelo.preparar_dados(df_prep)

            if modelo.ajustar(X, y):
                # Predizer alturas
                h_pred = modelo.predizer_altura(df_prep)
                predicoes[modelo.nome] = h_pred

                # Calcular métricas
                r2g = calcular_r2_generalizado(df_prep['H_m'], h_pred)
                rmse = np.sqrt(mean_squared_error(df_prep['H_m'], h_pred))

                resultados[modelo.nome] = {
                    'r2g': r2g,
                    'rmse': rmse,
                    'modelo': modelo
                }

                print(f"✅ {modelo.nome}: R²={r2g:.4f}, RMSE={rmse:.4f}")

        except Exception as e:
            print(f"❌ Erro no modelo {modelo.nome}: {e}")
            continue

    # Encontrar melhor modelo
    if resultados:
        melhor_modelo = max(resultados.keys(), key=lambda k: resultados[k]['r2g'])
        print(f"🏆 Melhor modelo: {melhor_modelo} (R²={resultados[melhor_modelo]['r2g']:.4f})")
        return resultados, predicoes, melhor_modelo
    else:
        print("❌ Nenhum modelo foi ajustado com sucesso")
        return {}, {}, None


# FUNÇÃO WRAPPER PARA COMPATIBILIDADE TOTAL
def ajustar_modelos_hipsometricos_compativel(df, config=None):
    '''
    WRAPPER: Garantia de compatibilidade total
    Detecta automaticamente se pode usar configurações
    '''
    try:
        # Tentar primeira com configurações
        return ajustar_todos_modelos_hipsometricos(df, config)
    except TypeError as e:
        if 'config' in str(e):
            # Se erro é relacionado ao parâmetro config, usar função original
            print("⚠️ Usando função original (sem configurações)")
            return ajustar_todos_modelos_hipsometricos_original(df)
        else:
            # Outro tipo de erro, repassar
            raise e
    except Exception as e:
        # Outros erros, repassar
        raise e


class ModeloCurtis(ModeloLinear):
    '''Modelo hipsométrico de Curtis: ln(H) = β₀ + β₁ * (1/D)'''

    def __init__(self):
        super().__init__("Curtis")

    def preparar_dados(self, df):
        X = df[['inv_D']]
        y = df['ln_H']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura real (não ln)'''
        X, _ = self.preparar_dados(df)
        ln_h_pred = self.predizer(X)
        return np.exp(ln_h_pred)


class ModeloCampos(ModeloLinear):
    '''Modelo hipsométrico de Campos: ln(H) = β₀ + β₁ * (1/D) + β₂ * ln(H_dom)'''

    def __init__(self):
        super().__init__("Campos")

    def preparar_dados(self, df):
        X = df[['inv_D', 'ln_H_dom']]
        y = df['ln_H']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura real (não ln)'''
        X, _ = self.preparar_dados(df)
        ln_h_pred = self.predizer(X)
        return np.exp(ln_h_pred)


class ModeloHenri(ModeloLinear):
    '''Modelo hipsométrico de Henri: H = β₀ + β₁ * ln(D)'''

    def __init__(self):
        super().__init__("Henri")

    def preparar_dados(self, df):
        X = df[['ln_D']]
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloProdan(ModeloLinear):
    '''Modelo hipsométrico de Prodan: D²/(H-1.3) = β₀ + β₁*D + β₂*D² + β₃*D*Idade'''

    def __init__(self):
        super().__init__("Prodan")
        self.tem_idade = False

    def preparar_dados(self, df):
        # Verificar disponibilidade de idade
        self.tem_idade = 'idade_anos' in df.columns and df['idade_anos'].notna().sum() > 10

        if self.tem_idade:
            colunas = ['D_cm', 'D2', 'DI']
        else:
            colunas = ['D_cm', 'D2']

        X = df[colunas]
        y = df['Prod']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura através da produtividade'''
        X, _ = self.preparar_dados(df)
        prod_pred = self.predizer(X)
        return (df['D2'] / np.clip(prod_pred, 0.1, None)) + 1.3


class ModeloChapman(ModeloNaoLinear):
    '''Modelo hipsométrico de Chapman: H = b₀ * (1 - exp(-b₁ * D))^b₂'''

    def __init__(self, altura_max):
        def chapman_func(D, b0, b1, b2):
            return b0 * (1 - np.exp(-b1 * D)) ** b2

        super().__init__("Chapman", chapman_func, [altura_max, 0.01, 1.0])

    def preparar_dados(self, df):
        X = df['D_cm']
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloWeibull(ModeloNaoLinear):
    '''Modelo hipsométrico de Weibull: H = a * (1 - exp(-b * D^c))'''

    def __init__(self, altura_max):
        def weibull_func(D, a, b, c):
            return a * (1 - np.exp(-b * D ** c))

        super().__init__("Weibull", weibull_func, [altura_max, 0.01, 1.0])

    def preparar_dados(self, df):
        X = df['D_cm']
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloMononuclear(ModeloNaoLinear):
    '''Modelo hipsométrico Mononuclear: H = a * (1 - b * exp(-c * D))'''

    def __init__(self, altura_max):
        def mono_func(D, a, b, c):
            return a * (1 - b * np.exp(-c * D))

        super().__init__("Mononuclear", mono_func, [altura_max, 1.0, 0.1])

    def preparar_dados(self, df):
        X = df['D_cm']
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


def ajustar_todos_modelos_hipsometricos(df):
    '''
    Ajusta todos os 7 modelos hipsométricos e retorna os resultados

    Args:
        df: DataFrame com dados preparados

    Returns:
        tuple: (resultados, predicoes, melhor_modelo)
    '''
    resultados = {}
    predicoes = {}

    # Calcular altura dominante
    dominantes = calcular_altura_dominante(df)

    # Criar variáveis
    df_prep = criar_variaveis_hipsometricas(df, dominantes)

    altura_max = df_prep['H_m'].max() * 1.2

    # Lista de modelos
    modelos = [
        ModeloCurtis(),
        ModeloCampos(),
        ModeloHenri(),
        ModeloProdan(),
        ModeloChapman(altura_max),
        ModeloWeibull(altura_max),
        ModeloMononuclear(altura_max)
    ]

    # Ajustar cada modelo
    for modelo in modelos:
        try:
            X, y = modelo.preparar_dados(df_prep)

            if modelo.ajustar(X, y):
                # Predizer alturas
                h_pred = modelo.predizer_altura(df_prep)
                predicoes[modelo.nome] = h_pred

                # Calcular métricas
                r2g = calcular_r2_generalizado(df_prep['H_m'], h_pred)
                rmse = np.sqrt(mean_squared_error(df_prep['H_m'], h_pred))

                resultados[modelo.nome] = {
                    'r2g': r2g,
                    'rmse': rmse,
                    'modelo': modelo
                }

        except Exception as e:
            print(f"Erro no modelo {modelo.nome}: {e}")
            continue

    # Encontrar melhor modelo
    if resultados:
        melhor_modelo = max(resultados.keys(), key=lambda k: resultados[k]['r2g'])
        return resultados, predicoes, melhor_modelo
    else:
        return {}, {}, None