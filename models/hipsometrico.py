# models/hipsometrico.py - VERSÃO ADAPTADA PARA CONFIGURAÇÕES GLOBAIS
'''
Modelos hipsométricos para estimativa de altura
ATUALIZADO: Usa parâmetros iniciais das configurações globais
'''

import numpy as np
import pandas as pd
from models.base import ModeloLinear, ModeloNaoLinear, ajustar_modelo_seguro, calcular_r2_generalizado
from sklearn.metrics import mean_squared_error


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

    def __init__(self, parametros_config=None):
        """
        NOVO: Aceita parâmetros das configurações globais

        Args:
            parametros_config: Dict com parâmetros das configurações
        """

        def chapman_func(D, b0, b1, b2):
            return b0 * (1 - np.exp(-b1 * D)) ** b2

        # NOVO: Usar parâmetros das configurações ou padrão
        if parametros_config:
            params_iniciais = [
                parametros_config.get('b0', 42.12),
                parametros_config.get('b1', 0.01),
                parametros_config.get('b2', 1.00)
            ]
        else:
            params_iniciais = [42.12, 0.01, 1.0]  # Valores padrão

        super().__init__("Chapman", chapman_func, params_iniciais)

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

    def __init__(self, parametros_config=None):
        """
        NOVO: Aceita parâmetros das configurações globais

        Args:
            parametros_config: Dict com parâmetros das configurações
        """

        def weibull_func(D, a, b, c):
            return a * (1 - np.exp(-b * D ** c))

        # NOVO: Usar parâmetros das configurações ou padrão
        if parametros_config:
            params_iniciais = [
                parametros_config.get('a', 42.12),
                parametros_config.get('b', 0.01),
                parametros_config.get('c', 1.00)
            ]
        else:
            params_iniciais = [42.12, 0.01, 1.0]  # Valores padrão

        super().__init__("Weibull", weibull_func, params_iniciais)

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

    def __init__(self, parametros_config=None):
        """
        NOVO: Aceita parâmetros das configurações globais

        Args:
            parametros_config: Dict com parâmetros das configurações
        """

        def mono_func(D, a, b, c):
            return a * (1 - b * np.exp(-c * D))

        # NOVO: Usar parâmetros das configurações ou padrão
        if parametros_config:
            params_iniciais = [
                parametros_config.get('a', 42.12),
                parametros_config.get('b', 1.00),
                parametros_config.get('c', 0.10)
            ]
        else:
            params_iniciais = [42.12, 1.0, 0.1]  # Valores padrão

        super().__init__("Mononuclear", mono_func, params_iniciais)

    def preparar_dados(self, df):
        X = df['D_cm']
        y = df['H_m']
        return X, y

    def predizer_altura(self, df):
        '''Prediz altura diretamente'''
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


def ajustar_todos_modelos_hipsometricos(df, config=None):
    '''
    VERSÃO ATUALIZADA: Ajusta todos os modelos hipsométricos usando configurações globais

    Args:
        df: DataFrame com dados preparados
        config: Dict com configurações globais (NOVO)

    Returns:
        tuple: (resultados, predicoes, melhor_modelo)
    '''
    # NOVO: Processar configurações
    if config is None:
        config = {}

    incluir_nao_lineares = config.get('incluir_nao_lineares', True)
    max_iteracoes = config.get('max_iteracoes', 5000)
    tolerancia_ajuste = config.get('tolerancia_ajuste', 0.01)

    # NOVO: Obter parâmetros específicos dos modelos não-lineares
    parametros_chapman = config.get('parametros_chapman', {'b0': 42.12, 'b1': 0.01, 'b2': 1.00})
    parametros_weibull = config.get('parametros_weibull', {'a': 42.12, 'b': 0.01, 'c': 1.00})
    parametros_mononuclear = config.get('parametros_mononuclear', {'a': 42.12, 'b': 1.00, 'c': 0.10})

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
        # NOVO: Criar modelos não-lineares com parâmetros das configurações
        modelos_nao_lineares = [
            ModeloChapman(parametros_chapman),
            ModeloWeibull(parametros_weibull),
            ModeloMononuclear(parametros_mononuclear)
        ]

    # Combinar modelos conforme configuração
    todos_modelos = modelos_lineares + modelos_nao_lineares

    print(
        f"Ajustando {len(todos_modelos)} modelos (Lineares: {len(modelos_lineares)}, Não-lineares: {len(modelos_nao_lineares)})")

    # NOVO: Configurar parâmetros de otimização nos modelos não-lineares
    for modelo in modelos_nao_lineares:
        if hasattr(modelo, 'max_iter'):
            modelo.max_iter = max_iteracoes
        if hasattr(modelo, 'tolerancia'):
            modelo.tolerancia = tolerancia_ajuste

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

                # NOVO: Mostrar parâmetros utilizados para modelos não-lineares
                if isinstance(modelo, ModeloNaoLinear):
                    print(f"✅ {modelo.nome}: R²={r2g:.4f}, RMSE={rmse:.4f}")
                    print(f"   Parâmetros iniciais: {modelo.params_iniciais}")
                    if hasattr(modelo, 'parametros') and modelo.parametros is not None:
                        print(f"   Parâmetros finais: {modelo.parametros}")
                else:
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


def obter_parametros_modelo_das_configuracoes(nome_modelo, config):
    """
    NOVA: Obtém parâmetros iniciais para um modelo específico das configurações

    Args:
        nome_modelo: 'Chapman', 'Weibull', ou 'Mononuclear'
        config: Configurações globais

    Returns:
        dict: Parâmetros iniciais para o modelo
    """
    parametros_map = {
        'Chapman': config.get('parametros_chapman', {'b0': 42.12, 'b1': 0.01, 'b2': 1.00}),
        'Weibull': config.get('parametros_weibull', {'a': 42.12, 'b': 0.01, 'c': 1.00}),
        'Mononuclear': config.get('parametros_mononuclear', {'a': 42.12, 'b': 1.00, 'c': 0.10})
    }

    return parametros_map.get(nome_modelo, {})


def validar_parametros_configuracao(config):
    """
    NOVA: Valida parâmetros de configuração antes do ajuste

    Args:
        config: Configurações globais

    Returns:
        dict: {'valido': bool, 'avisos': list, 'erros': list}
    """
    avisos = []
    erros = []

    if config.get('incluir_nao_lineares', True):
        # Validar Chapman
        chapman = config.get('parametros_chapman', {})
        if chapman.get('b0', 0) < 10:
            avisos.append("Chapman: Altura assintótica muito baixa (< 10m)")
        if chapman.get('b1', 0) > 0.5:
            avisos.append("Chapman: Taxa de crescimento muito alta (> 0.5)")

        # Validar Weibull
        weibull = config.get('parametros_weibull', {})
        if weibull.get('a', 0) < 10:
            avisos.append("Weibull: Altura assintótica muito baixa (< 10m)")
        if weibull.get('c', 0) > 3:
            avisos.append("Weibull: Parâmetro de forma muito alto (> 3)")

        # Validar Mononuclear
        mono = config.get('parametros_mononuclear', {})
        if mono.get('a', 0) < 10:
            avisos.append("Mononuclear: Altura assintótica muito baixa (< 10m)")
        if mono.get('b', 0) < 0.5:
            avisos.append("Mononuclear: Parâmetro de intercepto muito baixo (< 0.5)")

        # Validar parâmetros de otimização
        max_iter = config.get('max_iteracoes', 5000)
        if max_iter < 1000:
            avisos.append("Poucas iterações para modelos não-lineares (< 1000)")

        tolerancia = config.get('tolerancia_ajuste', 0.01)
        if tolerancia > 0.1:
            avisos.append("Tolerância muito alta (> 0.1)")

    return {
        'valido': len(erros) == 0,
        'avisos': avisos,
        'erros': erros
    }


def validar_parametros_configuracao(config):
    """
    NOVA: Valida parâmetros de configuração antes do ajuste

    Args:
        config: Configurações globais

    Returns:
        dict: {'valido': bool, 'avisos': list, 'erros': list}
    """
    avisos = []
    erros = []

    if config.get('incluir_nao_lineares', True):
        # Validar Chapman
        chapman = config.get('parametros_chapman', {})
        if chapman.get('b0', 0) < 10:
            avisos.append("Chapman: Altura assintótica muito baixa (< 10m)")
        if chapman.get('b0', 0) > 60:
            avisos.append("Chapman: Altura assintótica muito alta (> 60m)")
        if chapman.get('b1', 0) > 0.5:
            avisos.append("Chapman: Taxa de crescimento muito alta (> 0.5)")
        if chapman.get('b1', 0) <= 0:
            erros.append("Chapman: Taxa de crescimento deve ser > 0")

        # Validar Weibull
        weibull = config.get('parametros_weibull', {})
        if weibull.get('a', 0) < 10:
            avisos.append("Weibull: Altura assintótica muito baixa (< 10m)")
        if weibull.get('a', 0) > 60:
            avisos.append("Weibull: Altura assintótica muito alta (> 60m)")
        if weibull.get('c', 0) > 3:
            avisos.append("Weibull: Parâmetro de forma muito alto (> 3)")
        if weibull.get('b', 0) <= 0:
            erros.append("Weibull: Parâmetro b deve ser > 0")

        # Validar Mononuclear
        mono = config.get('parametros_mononuclear', {})
        if mono.get('a', 0) < 10:
            avisos.append("Mononuclear: Altura assintótica muito baixa (< 10m)")
        if mono.get('a', 0) > 60:
            avisos.append("Mononuclear: Altura assintótica muito alta (> 60m)")
        if mono.get('b', 0) < 0.5:
            avisos.append("Mononuclear: Parâmetro de intercepto muito baixo (< 0.5)")
        if mono.get('b', 0) > 2.0:
            avisos.append("Mononuclear: Parâmetro de intercepto muito alto (> 2.0)")
        if mono.get('c', 0) <= 0:
            erros.append("Mononuclear: Taxa de decaimento deve ser > 0")

        # Validar parâmetros de otimização
        max_iter = config.get('max_iteracoes', 5000)
        if max_iter < 1000:
            avisos.append("Poucas iterações para modelos não-lineares (< 1000)")
        if max_iter > 20000:
            avisos.append("Muitas iterações configuradas (> 20000) - pode ser lento")

        tolerancia = config.get('tolerancia_ajuste', 0.01)
        if tolerancia > 0.1:
            avisos.append("Tolerância muito alta (> 0.1)")
        if tolerancia < 0.001:
            avisos.append("Tolerância muito baixa (< 0.001) - pode não convergir")

    return {
        'valido': len(erros) == 0,
        'avisos': avisos,
        'erros': erros
    }


def gerar_relatorio_parametros_utilizados(config, resultados):
    """
    NOVA: Gera relatório dos parâmetros utilizados nos modelos

    Args:
        config: Configurações aplicadas
        resultados: Resultados dos modelos

    Returns:
        str: Relatório em formato markdown
    """
    relatorio = "# PARÂMETROS UTILIZADOS - MODELOS HIPSOMÉTRICOS\n\n"

    if config.get('incluir_nao_lineares', True):
        relatorio += "## Parâmetros Iniciais dos Modelos Não-Lineares\n\n"

        # Chapman
        chapman = config.get('parametros_chapman', {})
        relatorio += f"### Chapman\n"
        relatorio += f"- b₀ (altura assintótica): {chapman.get('b0', 42.12)}\n"
        relatorio += f"- b₁ (taxa de crescimento): {chapman.get('b1', 0.01)}\n"
        relatorio += f"- b₂ (parâmetro de forma): {chapman.get('b2', 1.00)}\n\n"

        # Weibull
        weibull = config.get('parametros_weibull', {})
        relatorio += f"### Weibull\n"
        relatorio += f"- a (altura assintótica): {weibull.get('a', 42.12)}\n"
        relatorio += f"- b (parâmetro de escala): {weibull.get('b', 0.01)}\n"
        relatorio += f"- c (parâmetro de forma): {weibull.get('c', 1.00)}\n\n"

        # Mononuclear
        mono = config.get('parametros_mononuclear', {})
        relatorio += f"### Mononuclear\n"
        relatorio += f"- a (altura assintótica): {mono.get('a', 42.12)}\n"
        relatorio += f"- b (parâmetro de intercepto): {mono.get('b', 1.00)}\n"
        relatorio += f"- c (taxa de decaimento): {mono.get('c', 0.10)}\n\n"

        # Parâmetros de otimização
        relatorio += "## Parâmetros de Otimização\n\n"
        relatorio += f"- Máximo de iterações: {config.get('max_iteracoes', 5000)}\n"
        relatorio += f"- Tolerância: {config.get('tolerancia_ajuste', 0.01)}\n\n"

    # Resultados obtidos
    relatorio += "## Resultados Obtidos\n\n"
    for modelo, resultado in resultados.items():
        r2g = resultado['r2g']
        rmse = resultado['rmse']
        relatorio += f"- **{modelo}**: R² = {r2g:.4f}, RMSE = {rmse:.4f}\n"

    relatorio += f"\n**Total de modelos ajustados**: {len(resultados)}\n"
    relatorio += f"**Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    return relatorio