# config/config_las.py
"""
Configurações específicas para processamento de arquivos LAS/LAZ
"""

# Configurações padrão do processador LAS
CONFIGURACOES_LAS_PADRAO = {
    'chunk_size': 500_000,
    'min_points_per_plot': 5,
    'height_threshold': 1.3,
    'max_height': 80.0,
    'buffer_parcela': 11.28,
    'max_file_size_mb': 500,
    'max_points_total': 50_000_000
}

# Configurações por tipo de floresta
CONFIGURACOES_POR_ESPECIE = {
    'eucalipto': {
        'height_threshold': 1.3,
        'max_height': 60.0,
        'buffer_parcela': 11.28,
        'min_points_per_plot': 5
    },
    'pinus': {
        'height_threshold': 1.3,
        'max_height': 80.0,
        'buffer_parcela': 11.28,
        'min_points_per_plot': 5
    },
    'nativa': {
        'height_threshold': 1.3,
        'max_height': 100.0,
        'buffer_parcela': 15.96,  # 800m²
        'min_points_per_plot': 10
    }
}

# Limites de validação para métricas LAS
LIMITES_VALIDACAO_LAS = {
    'altura_media': {'min': 0.5, 'max': 80.0},
    'altura_maxima': {'min': 1.0, 'max': 100.0},
    'desvio_altura': {'min': 0.0, 'max': 30.0},
    'cobertura': {'min': 0.0, 'max': 100.0},
    'densidade_pontos': {'min': 0.1, 'max': 1000.0},
    'n_pontos': {'min': 5, 'max': 100000}
}

# Thresholds para alertas automáticos
THRESHOLDS_ALERTAS_LAS = {
    'cobertura_baixa': 30.0,
    'densidade_baixa': 1.0,
    'altura_inconsistente': 5.0,
    'pontos_insuficientes': 5,
    'cv_altura_alto': 50.0
}

# Cores para visualizações LAS
CORES_LAS = {
    'pontos_validos': '#2E8B57',
    'pontos_filtrados': '#CD5C5C',
    'altura_baixa': '#FFD700',
    'altura_media': '#32CD32',
    'altura_alta': '#006400',
    'densidade': '#4169E1',
    'cobertura': '#228B22'
}

# Mensagens de ajuda para interface LAS
MENSAGENS_AJUDA_LAS = {
    'upload': """
    **Formatos aceitos:** .las, .laz
    **Tamanho máximo:** 500MB
    **Pontos máximos:** 50 milhões

    **Importante:** Arquivos devem estar normalizados (altura relativa ao solo)
    """,

    'chunk_size': """
    **Tamanho do Chunk:** Número de pontos processados por vez
    - Menor: Usa menos memória, mais lento
    - Maior: Usa mais memória, mais rápido
    """,

    'buffer_parcela': """
    **Raio da Parcela:** Raio em metros para extração de pontos
    - 11.28m = 400m² (padrão florestal)
    - 15.96m = 800m² (parcelas maiores)
    """,

    'height_threshold': """
    **Altura Mínima:** Altura mínima para considerar vegetação
    - 1.3m é padrão para medições florestais
    - Pontos abaixo são considerados solo/sub-bosque
    """
}

# Perfis pré-configurados
PERFIS_PROCESSAMENTO = {
    'rapido': {
        'chunk_size': 1_000_000,
        'min_points_per_plot': 3,
        'descricao': 'Processamento rápido com menor precisão'
    },

    'balanceado': {
        'chunk_size': 500_000,
        'min_points_per_plot': 5,
        'descricao': 'Equilíbrio entre velocidade e precisão'
    },

    'preciso': {
        'chunk_size': 200_000,
        'min_points_per_plot': 10,
        'descricao': 'Máxima precisão, processamento mais lento'
    },

    'memoria_limitada': {
        'chunk_size': 100_000,
        'min_points_per_plot': 3,
        'descricao': 'Para sistemas com pouca memória RAM'
    }
}

# Configurações de qualidade por densidade de pontos
QUALIDADE_POR_DENSIDADE = {
    'muito_baixa': {'min_densidade': 0.0, 'max_densidade': 0.5, 'min_points': 3},
    'baixa': {'min_densidade': 0.5, 'max_densidade': 2.0, 'min_points': 5},
    'media': {'min_densidade': 2.0, 'max_densidade': 10.0, 'min_points': 10},
    'alta': {'min_densidade': 10.0, 'max_densidade': 50.0, 'min_points': 20},
    'muito_alta': {'min_densidade': 50.0, 'max_densidade': 1000.0, 'min_points': 50}
}

# Métricas padrão calculadas
METRICAS_LAS_PADRAO = [
    'altura_media',
    'altura_maxima',
    'altura_minima',
    'desvio_altura',
    'altura_p95',
    'altura_p75',
    'altura_p50',
    'altura_p25',
    'cv_altura',
    'amplitude_altura',
    'densidade_pontos',
    'cobertura',
    'rugosidade',
    'shannon_height'
]

# Métricas opcionais (dependem de dados disponíveis)
METRICAS_LAS_OPCIONAIS = [
    'intensidade_media',
    'intensidade_desvio',
    'prop_primeiro_retorno',
    'retornos_medio'
]

# Configurações de exportação
CONFIGURACOES_EXPORTACAO = {
    'csv': {
        'separador': ';',
        'encoding': 'utf-8',
        'decimal': ','
    },
    'excel': {
        'engine': 'openpyxl',
        'sheet_name': 'Metricas_LiDAR'
    }
}


# Funções utilitárias para configuração

def obter_configuracao_por_especie(especie: str) -> dict:
    """
    Obtém configuração específica por espécie

    Args:
        especie: Nome da espécie ('eucalipto', 'pinus', 'nativa')

    Returns:
        dict: Configurações específicas
    """
    especie_lower = especie.lower()

    if especie_lower in CONFIGURACOES_POR_ESPECIE:
        config = CONFIGURACOES_LAS_PADRAO.copy()
        config.update(CONFIGURACOES_POR_ESPECIE[especie_lower])
        return config
    else:
        return CONFIGURACOES_LAS_PADRAO.copy()


def obter_perfil_processamento(perfil: str) -> dict:
    """
    Obtém configuração de perfil de processamento

    Args:
        perfil: Nome do perfil

    Returns:
        dict: Configurações do perfil
    """
    if perfil in PERFIS_PROCESSAMENTO:
        config = CONFIGURACOES_LAS_PADRAO.copy()
        config.update(PERFIS_PROCESSAMENTO[perfil])
        return config
    else:
        return CONFIGURACOES_LAS_PADRAO.copy()


def validar_configuracao_las(config: dict) -> tuple:
    """
    Valida configuração LAS

    Args:
        config: Dicionário de configuração

    Returns:
        tuple: (valida, lista_erros)
    """
    erros = []

    # Validações básicas
    if config.get('chunk_size', 0) < 10_000:
        erros.append("Chunk size muito pequeno (mínimo: 10.000)")

    if config.get('chunk_size', 0) > 5_000_000:
        erros.append("Chunk size muito grande (máximo: 5.000.000)")

    if config.get('min_points_per_plot', 0) < 1:
        erros.append("Mínimo de pontos deve ser pelo menos 1")

    if config.get('height_threshold', 0) < 0:
        erros.append("Altura mínima deve ser positiva")

    if config.get('max_height', 0) <= config.get('height_threshold', 1):
        erros.append("Altura máxima deve ser maior que altura mínima")

    if config.get('buffer_parcela', 0) <= 0:
        erros.append("Raio da parcela deve ser positivo")

    return len(erros) == 0, erros


def otimizar_configuracao_para_arquivo(tamanho_mb: float, num_pontos: int) -> dict:
    """
    Otimiza configuração baseada no tamanho do arquivo

    Args:
        tamanho_mb: Tamanho do arquivo em MB
        num_pontos: Número total de pontos

    Returns:
        dict: Configuração otimizada
    """
    config = CONFIGURACOES_LAS_PADRAO.copy()

    # Ajustar chunk_size baseado no tamanho
    if tamanho_mb > 200:  # Arquivos grandes
        config['chunk_size'] = 200_000
    elif tamanho_mb > 100:  # Arquivos médios
        config['chunk_size'] = 300_000
    elif tamanho_mb > 50:  # Arquivos pequenos-médios
        config['chunk_size'] = 500_000
    else:  # Arquivos pequenos
        config['chunk_size'] = 1_000_000

    # Ajustar baseado no número de pontos
    if num_pontos > 10_000_000:  # Muitos pontos
        config['min_points_per_plot'] = 10
    elif num_pontos > 1_000_000:  # Pontos médios
        config['min_points_per_plot'] = 5
    else:  # Poucos pontos
        config['min_points_per_plot'] = 3

    return config


def gerar_relatorio_configuracao(config: dict) -> str:
    """
    Gera relatório das configurações utilizadas

    Args:
        config: Configuração utilizada

    Returns:
        str: Relatório formatado
    """
    relatorio = f"""
# RELATÓRIO DE CONFIGURAÇÃO LAS/LAZ

## Configurações de Processamento
- **Chunk Size:** {config.get('chunk_size', 'N/A'):,} pontos
- **Pontos Mínimos/Parcela:** {config.get('min_points_per_plot', 'N/A')}
- **Altura Mínima:** {config.get('height_threshold', 'N/A')} m
- **Altura Máxima:** {config.get('max_height', 'N/A')} m
- **Raio da Parcela:** {config.get('buffer_parcela', 'N/A')} m

## Área da Parcela
- **Raio:** {config.get('buffer_parcela', 0):.2f} m
- **Área:** {3.14159 * (config.get('buffer_parcela', 0) ** 2):.0f} m²

## Limites de Arquivo
- **Tamanho Máximo:** {config.get('max_file_size_mb', 'N/A')} MB
- **Pontos Máximos:** {config.get('max_points_total', 'N/A'):,}

## Métricas Calculadas
### Básicas:
{', '.join(METRICAS_LAS_PADRAO)}

### Opcionais (se disponíveis):
{', '.join(METRICAS_LAS_OPCIONAIS)}

---
*Configuração gerada automaticamente pelo Sistema GreenVista*
"""

    return relatorio