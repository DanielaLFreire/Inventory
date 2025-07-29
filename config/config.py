# config/config.py - VERSÃO ATUALIZADA COM SUPORTE LIDAR
'''
Configurações globais do Sistema de Inventário Florestal
NOVO: Suporte para integração com dados LiDAR
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

# NOVO: Colunas esperadas do LiDAR (métricas extraídas do R)
COLUNAS_LIDAR_OBRIGATORIAS = ['talhao', 'parcela']
COLUNAS_LIDAR_METRICAS = [
    'zmean',  # Altura média
    'zmax',  # Altura máxima
    'zmin',  # Altura mínima
    'zsd',  # Desvio padrão altura
    'zkurt',  # Curtose
    'zskew',  # Assimetria
    'pzabove2',  # % pontos acima 2m
    'pzabove5',  # % pontos acima 5m
    'zentropy',  # Entropia vertical
    'VCI'  # Índice de Complexidade Vertical
]

# NOVO: Mapeamento de nomes alternativos para métricas LiDAR
NOMES_ALTERNATIVOS_LIDAR = {
    'altura_media': ['zmean', 'z_mean', 'altura_media', 'height_mean'],
    'altura_maxima': ['zmax', 'z_max', 'altura_max', 'height_max'],
    'altura_minima': ['zmin', 'z_min', 'altura_min', 'height_min'],
    'desvio_altura': ['zsd', 'z_sd', 'height_sd', 'std_height'],
    'curtose': ['zkurt', 'z_kurt', 'kurtosis'],
    'assimetria': ['zskew', 'z_skew', 'skewness'],
    'cobertura': ['pzabove2', 'cover_2m', 'cobertura_2m'],
    'densidade': ['pzabove5', 'density_5m', 'densidade_5m'],
    'entropia': ['zentropy', 'z_entropy', 'entropy'],
    'complexidade': ['VCI', 'vci', 'complexity_index']
}

# Configurações padrão
DEFAULTS = {
    'diametro_min': 4.0,
    'area_parcela': 400,  # m²
    'raio_parcela': 11.28,  # m
    'codigos_excluir': ['C', 'I'],
    'idade_padrao': 5.0,
    'altura_padrao': 25.0,
    # NOVO: Configurações LiDAR
    'altura_min_lidar': 2.0,  # Altura mínima para análise LiDAR
    'altura_max_lidar': 45.0,  # Altura máxima para análise LiDAR
    'tolerancia_altura': 2.0,  # Tolerância para comparação altura (m)
    'usar_lidar_calibracao': True,  # Usar LiDAR para calibrar modelos
}

# Extensões de arquivo suportadas
EXTENSOES_SUPORTADAS = {
    'dados': ['csv', 'xlsx', 'xls', 'xlsb'],
    'shapefile': ['shp', 'zip'],
    'lidar': ['csv', 'xlsx', 'xls']  # NOVO: Extensões para dados LiDAR
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

# NOVO: Classificação de qualidade da calibração LiDAR
CLASSIFICACAO_CALIBRACAO_LIDAR = {
    0.95: "***** Excelente",
    0.90: "**** Muito Bom",
    0.80: "*** Bom",
    0.70: "** Regular",
    0.0: "* Fraco"
}

# Cores para gráficos
CORES_MODELOS = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']

# NOVO: Cores específicas para gráficos LiDAR
CORES_LIDAR = {
    'campo': '#2E8B57',  # Verde floresta para dados de campo
    'lidar': '#1E90FF',  # Azul dodger para dados LiDAR
    'calibrado': '#FF6347',  # Tomate para modelos calibrados
    'residuos': '#FFD700',  # Ouro para análise de resíduos
    'correlacao': '#9370DB'  # Violeta médio para correlações
}

# NOVO: Configurações específicas para análise LiDAR
CONFIGURACOES_LIDAR = {
    'metricas_principais': ['zmean', 'zmax', 'zsd', 'pzabove2'],
    'metricas_estruturais': ['zkurt', 'zskew', 'zentropy', 'VCI'],
    'limites_validacao': {
        'altura_min': 1.3,
        'altura_max': 60.0,
        'diferenca_maxima': 10.0,  # Diferença máxima aceitável entre campo e LiDAR
        'r2_minimo': 0.7  # R² mínimo para validação
    },
    'parametros_calibracao': {
        'usar_pesos': True,  # Usar pesos baseados na precisão LiDAR
        'metodo_calibracao': 'ols',  # 'ols', 'robust', 'weighted'
        'validacao_cruzada': True,  # Aplicar validação cruzada
        'k_folds': 5  # Número de folds para validação cruzada
    }
}

# NOVO: Mensagens de ajuda para dados LiDAR
MENSAGENS_AJUDA_LIDAR = {
    'upload': """
    **📊 Arquivo de Métricas LiDAR**

    Arquivo CSV/Excel gerado pelo script R com métricas das parcelas.

    **Colunas obrigatórias:**
    - talhao, parcela

    **Métricas principais:**
    - zmean: Altura média (m)
    - zmax: Altura máxima (m) 
    - zsd: Desvio padrão altura
    - pzabove2: % cobertura acima 2m

    **Opcional:**
    - zkurt, zskew, zentropy, VCI
    """,

    'calibracao': """
    **🔧 Calibração com LiDAR**

    O LiDAR pode ser usado para:
    - ✅ Validar modelos hipsométricos
    - ✅ Calibrar estimativas de altura
    - ✅ Detectar outliers
    - ✅ Melhorar precisão geral

    **Processo:**
    1. Comparação altura campo vs LiDAR
    2. Identificação de padrões de erro
    3. Ajuste de coeficientes
    4. Validação cruzada
    """,

    'interpretacao': """
    **📈 Interpretação das Métricas**

    **Estruturais:**
    - zmean: Altura média do dossel
    - zmax: Árvores dominantes
    - zsd: Variabilidade estrutural

    **Cobertura:**
    - pzabove2: Densidade do sub-bosque
    - pzabove5: Cobertura do dossel

    **Complexidade:**
    - zentropy: Diversidade vertical
    - VCI: Índice de complexidade
    """
}

# NOVO: Configurações para relatórios integrados
CONFIGURACOES_RELATORIO_LIDAR = {
    'incluir_comparacoes': True,
    'incluir_mapas_altura': True,
    'incluir_analise_estrutural': True,
    'incluir_calibracao': True,
    'formato_coordenadas': 'UTM',  # 'UTM' ou 'LatLon'
    'precisao_decimal': 4
}

# NOVO: Thresholds para alertas automáticos
THRESHOLDS_ALERTAS_LIDAR = {
    'diferenca_altura_critica': 5.0,  # Diferença >5m entre campo e LiDAR
    'r2_baixo': 0.6,  # R² <0.6 indica problemas
    'cv_alto': 50.0,  # CV >50% indica alta variabilidade
    'cobertura_baixa': 70.0,  # Cobertura <70% pode indicar falhas
    'outlier_percentil': 95.0  # Percentil para detecção de outliers
}