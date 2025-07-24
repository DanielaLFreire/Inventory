# processors/__init__.py
'''
Módulo de processadores para o Sistema de Inventário Florestal
'''

from .cubagem import (
    processar_cubagem_smalian,
    calcular_estatisticas_cubagem,
    validar_dados_cubagem
)

from .inventario import (
    processar_inventario_completo,
    gerar_relatorio_inventario,
    validar_consistencia_inventario
)

from .areas import (
    processar_areas_por_metodo,
    validar_areas_processadas,
    gerar_resumo_areas
)

__all__ = [
    'processar_cubagem_smalian',
    'calcular_estatisticas_cubagem',
    'validar_dados_cubagem',
    'processar_inventario_completo',
    'gerar_relatorio_inventario',
    'validar_consistencia_inventario',
    'processar_areas_por_metodo',
    'validar_areas_processadas',
    'gerar_resumo_areas'
]

