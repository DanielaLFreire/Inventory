# utils/validacao.py
'''
Funções para validação de dados e colunas
'''

from config.config import COLUNAS_INVENTARIO, COLUNAS_CUBAGEM


def verificar_colunas_inventario(df):
    '''
    Verifica colunas obrigatórias do inventário

    Args:
        df: DataFrame do inventário

    Returns:
        list: Lista de colunas faltantes
    '''
    faltantes = [col for col in COLUNAS_INVENTARIO if col not in df.columns]
    return faltantes


def verificar_colunas_cubagem(df):
    '''
    Verifica colunas obrigatórias da cubagem

    Args:
        df: DataFrame da cubagem

    Returns:
        list: Lista de colunas faltantes
    '''
    faltantes = [col for col in COLUNAS_CUBAGEM if col not in df.columns]
    return faltantes


def validar_dados_numericos(df, colunas):
    '''
    Valida se as colunas especificadas são numéricas

    Args:
        df: DataFrame a ser validado
        colunas: Lista de colunas que devem ser numéricas

    Returns:
        dict: Dicionário com status de cada coluna
    '''
    status = {}

    for col in colunas:
        if col in df.columns:
            # Tentar converter para numérico
            try:
                pd.to_numeric(df[col], errors='coerce')
                status[col] = {'valido': True, 'nulos': df[col].isna().sum()}
            except:
                status[col] = {'valido': False, 'erro': 'Conversão falhou'}
        else:
            status[col] = {'valido': False, 'erro': 'Coluna não encontrada'}

    return status


def filtrar_dados_inventario(df, config):
    '''
    Aplica filtros aos dados do inventário

    Args:
        df: DataFrame do inventário
        config: Dicionário com configurações de filtro

    Returns:
        DataFrame filtrado
    '''
    df_filtrado = df.copy()

    # Filtrar talhões excluídos
    if config.get('talhoes_excluir'):
        df_filtrado = df_filtrado[~df_filtrado['talhao'].isin(config['talhoes_excluir'])]

    # Filtrar por diâmetro mínimo
    if config.get('diametro_min'):
        df_filtrado = df_filtrado[df_filtrado['D_cm'] >= config['diametro_min']]

    # Filtrar códigos excluídos
    if config.get('codigos_excluir'):
        df_filtrado = df_filtrado[~df_filtrado['cod'].isin(config['codigos_excluir'])]

    # Remover valores nulos essenciais
    df_filtrado = df_filtrado[
        (df_filtrado['D_cm'].notna()) &
        (df_filtrado['H_m'].notna()) &
        (df_filtrado['D_cm'] > 0) &
        (df_filtrado['H_m'] > 1.3)
        ]

    return df_filtrado


def verificar_qualidade_dados(df, tipo='inventario'):
    '''
    Verifica a qualidade geral dos dados

    Args:
        df: DataFrame a ser analisado
        tipo: Tipo de dados ('inventario' ou 'cubagem')

    Returns:
        dict: Relatório de qualidade
    '''
    relatorio = {
        'total_registros': len(df),
        'registros_validos': 0,
        'campos_problematicos': [],
        'alertas': []
    }

    if tipo == 'inventario':
        colunas_criticas = ['D_cm', 'H_m']
    else:
        colunas_criticas = ['d_cm', 'h_m', 'D_cm', 'H_m']

    # Verificar campos críticos
    for col in colunas_criticas:
        if col in df.columns:
            nulos = df[col].isna().sum()
            zeros = (df[col] == 0).sum() if df[col].dtype in ['int64', 'float64'] else 0

            if nulos > 0:
                relatorio['campos_problematicos'].append(f"{col}: {nulos} valores nulos")
            if zeros > 0:
                relatorio['campos_problematicos'].append(f"{col}: {zeros} valores zero")

    # Verificar registros completamente válidos
    if tipo == 'inventario':
        mask_validos = (
                df['D_cm'].notna() &
                df['H_m'].notna() &
                (df['D_cm'] > 0) &
                (df['H_m'] > 1.3)
        )
    else:
        mask_validos = (
                df['d_cm'].notna() &
                df['h_m'].notna() &
                df['D_cm'].notna() &
                df['H_m'].notna() &
                (df['D_cm'] > 0) &
                (df['H_m'] > 1.3)
        )

    relatorio['registros_validos'] = mask_validos.sum()

    # Gerar alertas
    if relatorio['registros_validos'] < len(df) * 0.9:
        relatorio['alertas'].append("Menos de 90% dos registros são válidos")

    if len(relatorio['campos_problematicos']) > 0:
        relatorio['alertas'].append("Existem campos com dados problemáticos")

    return relatorio