# utils/__init__.py
"""
Módulo de utilitários do Sistema de Inventário Florestal
Contém funções auxiliares para formatação, manipulação de arquivos e processamento de dados
"""

# Importações principais para facilitar o uso
from utils.formatacao import (
    formatar_brasileiro,
    formatar_numero_inteligente,
    classificar_qualidade_modelo,
    formatar_dataframe_brasileiro,
    formatar_percentual,
    formatar_moeda,
    formatar_timestamp_brasileiro,
    classificar_qualidade_modelo_detalhado,
    classificar_produtividade_florestal,
    formatar_estatisticas_resumo,
    criar_relatorio_modelo,
    validar_dados_numericos,
    gerar_metricas_card_streamlit
)

from utils.arquivo_handler import (
    carregar_arquivo,
    validar_estrutura_arquivo,
    exportar_dataframe,
    criar_template_csv,
    verificar_qualidade_dados,
    normalizar_nomes_colunas,
    detectar_separador_csv,
    detectar_encoding_arquivo
)

# Versão do módulo
__version__ = "1.0.0"

# Informações do módulo
__author__ = "Sistema GreenVista"
__description__ = "Utilitários para o Sistema Integrado de Inventário Florestal"

# Lista de funções exportadas
__all__ = [
    # Formatação
    'formatar_brasileiro',
    'formatar_numero_inteligente',
    'classificar_qualidade_modelo',
    'formatar_dataframe_brasileiro',
    'formatar_percentual',
    'formatar_moeda',
    'formatar_timestamp_brasileiro',
    'classificar_qualidade_modelo_detalhado',
    'classificar_produtividade_florestal',
    'formatar_estatisticas_resumo',
    'criar_relatorio_modelo',
    'validar_dados_numericos',
    'gerar_metricas_card_streamlit',

    # Manipulação de arquivos
    'carregar_arquivo',
    'validar_estrutura_arquivo',
    'exportar_dataframe',
    'criar_template_csv',
    'verificar_qualidade_dados',
    'normalizar_nomes_colunas',
    'detectar_separador_csv',
    'detectar_encoding_arquivo'
]

# Configurações padrão do módulo
DEFAULT_DECIMAL_PLACES = 2
DEFAULT_ENCODING = 'utf-8'
DEFAULT_CSV_SEPARATOR = ';'

# Constantes úteis
COLUNAS_INVENTARIO_OBRIGATORIAS = ['D_cm', 'H_m', 'talhao', 'parcela']
COLUNAS_CUBAGEM_OBRIGATORIAS = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']

EXTENSOES_SUPORTADAS = {
    'dados': ['.csv', '.xlsx', '.xls', '.xlsb'],
    'shapefile': ['.shp', '.zip'],
    'coordenadas': ['.csv', '.xlsx', '.xls']
}

LIMITES_VALIDACAO_PADRAO = {
    'dap_min': 1.0,  # cm
    'dap_max': 100.0,  # cm
    'altura_min': 1.3,  # m
    'altura_max': 60.0,  # m
    'idade_min': 0.5,  # anos
    'idade_max': 50.0  # anos
}

# Mensagens padrão
MENSAGENS_ERRO = {
    'arquivo_vazio': "Arquivo está vazio ou não pôde ser carregado",
    'colunas_faltantes': "Colunas obrigatórias não encontradas",
    'dados_invalidos': "Dados contêm valores inválidos",
    'formato_nao_suportado': "Formato de arquivo não suportado"
}

MENSAGENS_SUCESSO = {
    'arquivo_carregado': "Arquivo carregado com sucesso",
    'dados_validados': "Dados validados com sucesso",
    'processamento_concluido': "Processamento concluído"
}


# Funções auxiliares de configuração
def configurar_modulo(decimal_places=None, encoding=None, csv_separator=None):
    """
    Configura parâmetros padrão do módulo utils

    Args:
        decimal_places: Número padrão de casas decimais
        encoding: Encoding padrão para arquivos
        csv_separator: Separador padrão para CSV
    """
    global DEFAULT_DECIMAL_PLACES, DEFAULT_ENCODING, DEFAULT_CSV_SEPARATOR

    if decimal_places is not None:
        DEFAULT_DECIMAL_PLACES = decimal_places

    if encoding is not None:
        DEFAULT_ENCODING = encoding

    if csv_separator is not None:
        DEFAULT_CSV_SEPARATOR = csv_separator


def obter_configuracao_modulo():
    """
    Retorna configuração atual do módulo

    Returns:
        dict: Configurações atuais
    """
    return {
        'decimal_places': DEFAULT_DECIMAL_PLACES,
        'encoding': DEFAULT_ENCODING,
        'csv_separator': DEFAULT_CSV_SEPARATOR,
        'version': __version__
    }


def validar_ambiente():
    """
    Valida se o ambiente possui as dependências necessárias

    Returns:
        dict: Status da validação
    """
    resultado = {
        'valido': True,
        'dependencias_faltantes': [],
        'avisos': []
    }

    # Verificar dependências essenciais
    dependencias_essenciais = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('streamlit', 'st')
    ]

    for nome_pacote, alias in dependencias_essenciais:
        try:
            __import__(nome_pacote)
        except ImportError:
            resultado['valido'] = False
            resultado['dependencias_faltantes'].append(nome_pacote)

    # Verificar dependências opcionais
    dependencias_opcionais = [
        ('openpyxl', 'Para arquivos Excel .xlsx'),
        ('xlrd', 'Para arquivos Excel .xls'),
        ('pyxlsb', 'Para arquivos Excel .xlsb'),
        ('geopandas', 'Para shapefiles'),
        ('chardet', 'Para detecção de encoding')
    ]

    for nome_pacote, descricao in dependencias_opcionais:
        try:
            __import__(nome_pacote)
        except ImportError:
            resultado['avisos'].append(f"{nome_pacote}: {descricao}")

    return resultado


# Função de diagnóstico
def diagnostico_completo():
    """
    Executa diagnóstico completo do módulo utils

    Returns:
        dict: Relatório de diagnóstico
    """
    import sys
    import platform

    diagnostico = {
        'timestamp': formatar_timestamp_brasileiro(),
        'sistema': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture()
        },
        'modulo': {
            'version': __version__,
            'configuracao': obter_configuracao_modulo()
        },
        'dependencias': validar_ambiente()
    }

    return diagnostico


# Inicialização automática
def _inicializar_modulo():
    """Inicialização automática do módulo"""
    try:
        # Configurar pandas para melhor exibição
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)

        # Configurar warnings
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)

    except ImportError:
        pass  # Continuar mesmo se pandas não estiver disponível


# Executar inicialização
_inicializar_modulo()