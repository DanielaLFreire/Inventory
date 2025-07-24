# models/__init__.py
'''
Módulo de modelos para o Sistema de Inventário Florestal
'''

from .hipsometrico import (
    ajustar_todos_modelos_hipsometricos,
    calcular_altura_dominante,
    criar_variaveis_hipsometricas
)

from .volumetrico import (
    ajustar_todos_modelos_volumetricos,
    aplicar_modelo_volumetrico,
    criar_variaveis_volumetricas
)

__all__ = [
    'ajustar_todos_modelos_hipsometricos',
    'calcular_altura_dominante',
    'criar_variaveis_hipsometricas',
    'ajustar_todos_modelos_volumetricos',
    'aplicar_modelo_volumetrico',
    'criar_variaveis_volumetricas'
]

