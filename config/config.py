# config/config.py
'''
Configurações globais do Sistema de Inventário Florestal
'''

# Configurações da página Streamlit
PAGE_CONFIG = {
    'page_title': "Sistema Integrado de Inventário Florestal",
    'page_icon': "🌲",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Colunas obrigatórias para cada tipo de arquivo
COLUNAS_INVENTARIO = ['D_cm', 'H_m', 'talhao', 'parcela', 'cod']
COLUNAS_CUBAGEM = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']

# Configurações padrão
DEFAULTS = {
    'diametro_min': 4.0,
    'area_parcela': 400,  # m²
    'raio_parcela': 11.28,  # m
    'codigos_excluir': ['C', 'I'],
    'idade_padrao': 5.0,
    'altura_padrao': 25.0
}

# Extensões de arquivo suportadas
EXTENSOES_SUPORTADAS = {
    'dados': ['csv', 'xlsx', 'xls', 'xlsb'],
    'shapefile': ['shp', 'zip']
}

# Separadores para CSV
SEPARADORES_CSV = [';', ',', '\t']

# Engines para Excel
ENGINES_EXCEL = ['openpyxl', 'xlrd', 'pyxlsb']

# Nomes alternativos para colunas
NOMES_ALTERNATIVOS = {
    'talhao': ['talhao', 'talhão', 'talh', 'plot', 'stand'],
    'coordenadas': [['x', 'y'], ['X', 'Y'], ['lon', 'lat'], ['longitude', 'latitude']],
    'area': ['area_ha', 'area', 'hectares', 'ha', 'area_m2']
}

# Classificação de qualidade dos modelos
CLASSIFICACAO_R2 = {
    0.9: "***** Excelente",
    0.8: "**** Muito Bom", 
    0.7: "*** Bom",
    0.6: "** Regular",
    0.0: "* Fraco"
}

# Cores para gráficos
CORES_MODELOS = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']