# utils/utils_gerais.py
"""
Utilitários gerais para o Sistema Integrado de Inventário Florestal
Funções auxiliares, validações e operações comuns
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
import json
from datetime import datetime, timedelta
import re
import warnings

warnings.filterwarnings('ignore')


def gerar_id_unico(prefixo="", incluir_timestamp=True):
    """
    Gera ID único para componentes Streamlit

    Args:
        prefixo: Prefixo para o ID
        incluir_timestamp: Se deve incluir timestamp

    Returns:
        str: ID único
    """
    if incluir_timestamp:
        timestamp = int(time.time() * 1000)
        return f"{prefixo}_{timestamp}" if prefixo else str(timestamp)
    else:
        import random
        return f"{prefixo}_{random.randint(1000, 9999)}" if prefixo else str(random.randint(1000, 9999))


def criar_hash_dados(df):
    """
    Cria hash dos dados para verificar mudanças

    Args:
        df: DataFrame

    Returns:
        str: Hash MD5 dos dados
    """
    try:
        if df is None or len(df) == 0:
            return "empty_data"

        # Criar string representativa dos dados
        data_string = f"{len(df)}_{df.columns.tolist()}_{df.dtypes.tolist()}"

        # Adicionar sample dos dados se não for muito grande
        if len(df) <= 1000:
            data_string += str(df.values.tobytes())
        else:
            # Para datasets grandes, usar sample
            sample_df = df.sample(n=100, random_state=42)
            data_string += str(sample_df.values.tobytes())

        return hashlib.md5(data_string.encode()).hexdigest()[:12]

    except Exception:
        return f"hash_error_{int(time.time())}"


def verificar_mudanca_dados(df, key_prefix="dados"):
    """
    Verifica se dados mudaram desde a última verificação

    Args:
        df: DataFrame atual
        key_prefix: Prefixo para a chave no session_state

    Returns:
        bool: True se dados mudaram
    """
    if df is None:
        return False

    hash_key = f"{key_prefix}_hash"
    hash_atual = criar_hash_dados(df)

    if hash_key not in st.session_state:
        st.session_state[hash_key] = hash_atual
        return True

    hash_anterior = st.session_state[hash_key]

    if hash_atual != hash_anterior:
        st.session_state[hash_key] = hash_atual
        return True

    return False


def limpar_cache_resultados(tipos=None):
    """
    Limpa cache de resultados específicos

    Args:
        tipos: Lista de tipos a limpar ou None para todos
    """
    if tipos is None:
        tipos = ['hipsometricos', 'volumetricos', 'inventario', 'lidar']

    keys_para_limpar = []

    for tipo in tipos:
        if tipo == 'hipsometricos':
            keys_para_limpar.extend(['resultados_hipsometricos', 'melhor_modelo_hip'])
        elif tipo == 'volumetricos':
            keys_para_limpar.extend(['resultados_volumetricos', 'melhor_modelo_vol'])
        elif tipo == 'inventario':
            keys_para_limpar.extend(['inventario_processado', 'resumo_talhoes'])
        elif tipo == 'lidar':
            keys_para_limpar.extend(['dados_lidar', 'calibracao_lidar'])

    # Remover keys do session_state
    for key in keys_para_limpar:
        if key in st.session_state:
            del st.session_state[key]


def cronometrar_operacao(nome_operacao="Operação"):
    """
    Context manager para cronometrar operações

    Args:
        nome_operacao: Nome da operação sendo cronometrada

    Usage:
        with cronometrar_operacao("Carregamento de dados"):
            df = carregar_arquivo(arquivo)
    """

    class CronometroOperacao:
        def __init__(self, nome):
            self.nome = nome
            self.inicio = None

        def __enter__(self):
            self.inicio = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            fim = time.time()
            duracao = fim - self.inicio

            if duracao < 1:
                st.info(f"⏱️ {self.nome}: {duracao:.2f}s")
            elif duracao < 60:
                st.info(f"⏱️ {self.nome}: {duracao:.1f}s")
            else:
                minutos = int(duracao // 60)
                segundos = duracao % 60
                st.info(f"⏱️ {self.nome}: {minutos}m {segundos:.1f}s")

    return CronometroOperacao(nome_operacao)


def salvar_progresso_etapa(etapa, dados, sucesso=True):
    """
    Salva progresso de uma etapa no session_state

    Args:
        etapa: Nome da etapa
        dados: Dados a salvar
        sucesso: Se a etapa foi bem-sucedida
    """
    key_progresso = f"progresso_{etapa}"

    progresso = {
        'dados': dados,
        'sucesso': sucesso,
        'timestamp': datetime.now(),
        'hash_dados': criar_hash_dados(dados) if isinstance(dados, pd.DataFrame) else None
    }

    st.session_state[key_progresso] = progresso


def obter_progresso_etapa(etapa):
    """
    Obtém progresso salvo de uma etapa

    Args:
        etapa: Nome da etapa

    Returns:
        dict: Progresso da etapa ou None
    """
    key_progresso = f"progresso_{etapa}"
    return st.session_state.get(key_progresso)


def validar_integridade_session_state():
    """
    Valida integridade dos dados no session_state

    Returns:
        dict: Relatório de integridade
    """
    relatorio = {
        'timestamp': datetime.now(),
        'keys_encontradas': [],
        'keys_problematicas': [],
        'tamanho_total_mb': 0,
        'recomendacoes': []
    }

    try:
        import sys

        for key, value in st.session_state.items():
            relatorio['keys_encontradas'].append(key)

            # Verificar tamanho
            try:
                tamanho_bytes = sys.getsizeof(value)
                relatorio['tamanho_total_mb'] += tamanho_bytes / (1024 * 1024)

                # Identificar keys problemáticas (muito grandes)
                if tamanho_bytes > 50 * 1024 * 1024:  # 50MB
                    relatorio['keys_problematicas'].append({
                        'key': key,
                        'tamanho_mb': tamanho_bytes / (1024 * 1024),
                        'tipo': type(value).__name__
                    })

            except Exception:
                pass

        # Gerar recomendações
        if relatorio['tamanho_total_mb'] > 200:
            relatorio['recomendacoes'].append("Session state muito grande (>200MB)")

        if len(relatorio['keys_problematicas']) > 0:
            relatorio['recomendacoes'].append("Existem objetos muito grandes no cache")

        if len(relatorio['keys_encontradas']) > 50:
            relatorio['recomendacoes'].append("Muitas keys no session_state - considere limpeza")

    except Exception as e:
        relatorio['erro'] = str(e)

    return relatorio


def otimizar_dataframe_memoria(df):
    """
    Otimiza DataFrame para usar menos memória

    Args:
        df: DataFrame a otimizar

    Returns:
        DataFrame otimizado
    """
    if df is None or len(df) == 0:
        return df

    df_otimizado = df.copy()

    try:
        # Otimizar colunas numéricas
        for col in df_otimizado.select_dtypes(include=[np.number]).columns:
            # Verificar se pode ser int
            if df_otimizado[col].dtype == 'float64':
                if df_otimizado[col].notna().all() and (df_otimizado[col] % 1 == 0).all():
                    # Pode ser convertido para int
                    max_val = df_otimizado[col].max()
                    min_val = df_otimizado[col].min()

                    if min_val >= 0 and max_val < 2 ** 32:
                        df_otimizado[col] = df_otimizado[col].astype('uint32')
                    elif min_val >= -2 ** 31 and max_val < 2 ** 31:
                        df_otimizado[col] = df_otimizado[col].astype('int32')
                else:
                    # Manter como float mas usar float32 se possível
                    if df_otimizado[col].max() < 3.4e38 and df_otimizado[col].min() > -3.4e38:
                        df_otimizado[col] = df_otimizado[col].astype('float32')

        # Otimizar colunas categóricas
        for col in df_otimizado.select_dtypes(include=['object']).columns:
            num_unique = df_otimizado[col].nunique()
            total_values = len(df_otimizado[col])

            # Se menos de 50% dos valores são únicos, converter para category
            if num_unique / total_values < 0.5:
                df_otimizado[col] = df_otimizado[col].astype('category')

    except Exception:
        # Se otimização falhar, retornar original
        return df

    return df_otimizado


def verificar_performance_sistema():
    """
    Verifica performance atual do sistema

    Returns:
        dict: Métricas de performance
    """
    import psutil
    import gc

    metricas = {
        'timestamp': datetime.now(),
        'memoria_sistema': {},
        'streamlit_info': {},
        'recomendacoes': []
    }

    try:
        # Informações de memória do sistema
        memoria = psutil.virtual_memory()
        metricas['memoria_sistema'] = {
            'total_gb': memoria.total / (1024 ** 3),
            'disponivel_gb': memoria.available / (1024 ** 3),
            'percentual_uso': memoria.percent,
            'livre_gb': memoria.free / (1024 ** 3)
        }

        # Informações do Streamlit
        session_state_info = validar_integridade_session_state()
        metricas['streamlit_info'] = {
            'keys_no_state': len(session_state_info['keys_encontradas']),
            'tamanho_state_mb': session_state_info['tamanho_total_mb'],
            'keys_grandes': len(session_state_info['keys_problematicas'])
        }

        # Forçar garbage collection
        objetos_coletados = gc.collect()
        metricas['gc_objetos_coletados'] = objetos_coletados

        # Gerar recomendações
        if memoria.percent > 85:
            metricas['recomendacoes'].append("Memória do sistema alta (>85%)")

        if session_state_info['tamanho_total_mb'] > 100:
            metricas['recomendacoes'].append("Session state muito grande")

        if objetos_coletados > 1000:
            metricas['recomendacoes'].append("Muitos objetos não referenciados")

    except Exception as e:
        metricas['erro'] = str(e)

    return metricas


def criar_backup_session_state(incluir_dados=False):
    """
    Cria backup do session_state

    Args:
        incluir_dados: Se deve incluir DataFrames grandes

    Returns:
        dict: Backup serializado
    """
    backup = {
        'timestamp': datetime.now().isoformat(),
        'versao_backup': '1.0',
        'dados': {}
    }

    try:
        for key, value in st.session_state.items():
            try:
                # Verificar se é serializável
                if isinstance(value, (str, int, float, bool, list, dict)):
                    backup['dados'][key] = value
                elif isinstance(value, pd.DataFrame):
                    if incluir_dados:
                        backup['dados'][key] = {
                            'tipo': 'DataFrame',
                            'shape': value.shape,
                            'columns': value.columns.tolist(),
                            'dtypes': value.dtypes.to_dict(),
                            'data': value.to_dict() if len(value) < 1000 else "too_large"
                        }
                    else:
                        backup['dados'][key] = {
                            'tipo': 'DataFrame',
                            'shape': value.shape,
                            'columns': value.columns.tolist(),
                            'info': 'DataFrame não incluído no backup'
                        }
                else:
                    backup['dados'][key] = {
                        'tipo': str(type(value)),
                        'info': 'Objeto não serializável'
                    }

            except Exception:
                backup['dados'][key] = 'erro_serialização'

    except Exception as e:
        backup['erro_geral'] = str(e)

    return backup


def restaurar_backup_session_state(backup_data):
    """
    Restaura backup do session_state

    Args:
        backup_data: Dados do backup

    Returns:
        bool: Sucesso da operação
    """
    try:
        if not isinstance(backup_data, dict) or 'dados' not in backup_data:
            return False

        keys_restauradas = 0

        for key, value in backup_data['dados'].items():
            try:
                if isinstance(value, dict) and value.get('tipo') == 'DataFrame':
                    # Pular DataFrames por enquanto
                    continue
                elif not isinstance(value, dict) or 'tipo' not in value:
                    # Valor simples que pode ser restaurado
                    st.session_state[key] = value
                    keys_restauradas += 1

            except Exception:
                continue

        st.success(f"✅ Backup restaurado: {keys_restauradas} configurações")
        return True

    except Exception as e:
        st.error(f"❌ Erro ao restaurar backup: {e}")
        return False


def detectar_anomalias_dados(df, colunas_numericas=None):
    """
    Detecta anomalias nos dados usando métodos estatísticos

    Args:
        df: DataFrame a analisar
        colunas_numericas: Colunas a analisar (auto-detecta se None)

    Returns:
        dict: Relatório de anomalias
    """
    if df is None or len(df) == 0:
        return {'erro': 'DataFrame vazio'}

    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    relatorio = {
        'timestamp': datetime.now(),
        'total_registros': len(df),
        'anomalias_por_coluna': {},
        'registros_anomalos': set(),
        'resumo': {
            'total_anomalias': 0,
            'colunas_problematicas': [],
            'percentual_afetado': 0
        }
    }

    for coluna in colunas_numericas:
        if coluna not in df.columns:
            continue

        serie = df[coluna].dropna()

        if len(serie) < 10:  # Muito poucos dados
            continue

        # Método IQR para detectar outliers
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Encontrar outliers
        outliers_mask = (serie < limite_inferior) | (serie > limite_superior)
        outliers_indices = serie[outliers_mask].index.tolist()

        # Método Z-score para outliers extremos
        z_scores = np.abs((serie - serie.mean()) / serie.std())
        outliers_extremos = serie[z_scores > 3].index.tolist()

        relatorio['anomalias_por_coluna'][coluna] = {
            'outliers_iqr': len(outliers_indices),
            'outliers_extremos': len(outliers_extremos),
            'indices_outliers': outliers_indices,
            'indices_extremos': outliers_extremos,
            'limite_inferior': limite_inferior,
            'limite_superior': limite_superior,
            'z_score_max': z_scores.max() if len(z_scores) > 0 else 0
        }

        # Adicionar aos registros anômalos
        relatorio['registros_anomalos'].update(outliers_indices)
        relatorio['registros_anomalos'].update(outliers_extremos)

        # Verificar se coluna é problemática
        if len(outliers_indices) > len(serie) * 0.1:  # Mais de 10% outliers
            relatorio['resumo']['colunas_problematicas'].append(coluna)

    # Calcular resumo
    relatorio['resumo']['total_anomalias'] = len(relatorio['registros_anomalos'])
    relatorio['resumo']['percentual_afetado'] = (len(relatorio['registros_anomalos']) / len(df)) * 100

    # Converter set para list para serialização
    relatorio['registros_anomalos'] = list(relatorio['registros_anomalos'])

    return relatorio


def gerar_relatorio_sistema():
    """
    Gera relatório completo do sistema

    Returns:
        str: Relatório formatado em markdown
    """
    relatorio = f"""# 🔧 Relatório do Sistema - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## 📊 Status Geral do Sistema
"""

    try:
        # Verificar dados carregados
        dados_inventario = hasattr(st.session_state,
                                   'dados_inventario') and st.session_state.dados_inventario is not None
        dados_cubagem = hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None

        relatorio += f"""
### 📁 Dados Carregados
- **Inventário:** {'✅ Carregado' if dados_inventario else '❌ Não carregado'}
- **Cubagem:** {'✅ Carregada' if dados_cubagem else '❌ Não carregada'}
"""

        if dados_inventario:
            df_inv = st.session_state.dados_inventario
            relatorio += f"  - Registros: {len(df_inv):,}\n"
            relatorio += f"  - Talhões: {df_inv['talhao'].nunique()}\n"

        if dados_cubagem:
            df_cub = st.session_state.dados_cubagem
            relatorio += f"  - Árvores cubadas: {df_cub['arv'].nunique()}\n"

        # Verificar configuração
        try:
            from config.configuracoes_globais import obter_configuracao_global
            config = obter_configuracao_global()
            configurado = config.get('configurado', False)

            relatorio += f"""
### ⚙️ Configuração
- **Status:** {'✅ Configurado' if configurado else '❌ Não configurado'}
"""
            if configurado:
                relatorio += f"- **Diâmetro mínimo:** {config.get('diametro_min', 'N/A')} cm\n"
                relatorio += f"- **Método de área:** {config.get('metodo_area', 'N/A')}\n"

        except Exception:
            relatorio += "### ⚙️ Configuração\n- **Status:** ❌ Erro ao verificar\n"

        # Verificar etapas executadas
        etapas = {
            'Hipsométricos': hasattr(st.session_state,
                                     'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None,
            'Volumétricos': hasattr(st.session_state,
                                    'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None,
            'Inventário': hasattr(st.session_state,
                                  'inventario_processado') and st.session_state.inventario_processado is not None,
            'LiDAR': hasattr(st.session_state, 'dados_lidar') and st.session_state.dados_lidar is not None
        }

        relatorio += "\n### 🔄 Etapas Executadas\n"
        for etapa, executada in etapas.items():
            status = "✅ Concluída" if executada else "⏳ Pendente"
            relatorio += f"- **{etapa}:** {status}\n"

        # Performance do sistema
        performance = verificar_performance_sistema()
        relatorio += f"""
### ⚡ Performance
- **Uso de memória:** {performance['memoria_sistema']['percentual_uso']:.1f}%
- **Memória disponível:** {performance['memoria_sistema']['disponivel_gb']:.1f} GB
- **Keys no session_state:** {performance['streamlit_info']['keys_no_state']}
- **Tamanho do cache:** {performance['streamlit_info']['tamanho_state_mb']:.1f} MB
"""

        if performance['recomendacoes']:
            relatorio += "\n### ⚠️ Recomendações\n"
            for rec in performance['recomendacoes']:
                relatorio += f"- {rec}\n"

        # Integridade dos dados
        if dados_inventario:
            anomalias = detectar_anomalias_dados(st.session_state.dados_inventario)
            relatorio += f"""
### 🔍 Qualidade dos Dados (Inventário)
- **Total de registros:** {anomalias['total_registros']:,}
- **Anomalias detectadas:** {anomalias['resumo']['total_anomalias']}
- **Percentual afetado:** {anomalias['resumo']['percentual_afetado']:.2f}%
"""

            if anomalias['resumo']['colunas_problematicas']:
                relatorio += f"- **Colunas problemáticas:** {', '.join(anomalias['resumo']['colunas_problematicas'])}\n"

    except Exception as e:
        relatorio += f"\n### ❌ Erro na Geração do Relatório\n{str(e)}\n"

    relatorio += "\n---\n*Relatório gerado automaticamente pelo Sistema GreenVista*"

    return relatorio


def limpar_sistema_completo():
    """
    Limpa completamente o sistema (reset total)
    """
    keys_para_manter = [
        'FormSubmitter:pages/0_⚙️_Configurações.py-',  # Manter forms do Streamlit
        'FileUploader:'  # Manter upload states básicos
    ]

    keys_para_remover = []

    for key in st.session_state.keys():
        deve_manter = False
        for key_manter in keys_para_manter:
            if key.startswith(key_manter):
                deve_manter = True
                break

        if not deve_manter:
            keys_para_remover.append(key)

    # Remover keys
    for key in keys_para_remover:
        try:
            del st.session_state[key]
        except:
            pass

    # Forçar garbage collection
    import gc
    gc.collect()


def exportar_configuracao_completa():
    """
    Exporta configuração completa do sistema

    Returns:
        dict: Configuração exportável
    """
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'versao_sistema': '4.0',
        'configuracao_global': {},
        'status_etapas': {},
        'metadados': {}
    }

    try:
        # Configuração global
        from config.configuracoes_globais import obter_configuracao_global
        config = obter_configuracao_global()
        export_data['configuracao_global'] = config

        # Status das etapas
        etapas = ['resultados_hipsometricos', 'resultados_volumetricos', 'inventario_processado']
        for etapa in etapas:
            if hasattr(st.session_state, etapa):
                resultado = getattr(st.session_state, etapa)
                if isinstance(resultado, dict):
                    # Exportar apenas metadados, não os dados completos
                    export_data['status_etapas'][etapa] = {
                        'executado': True,
                        'melhor_modelo': resultado.get('melhor_modelo'),
                        'timestamp': resultado.get('timestamp'),
                        'num_modelos': len(resultado.get('resultados', {}))
                    }
                else:
                    export_data['status_etapas'][etapa] = {'executado': True}
            else:
                export_data['status_etapas'][etapa] = {'executado': False}

        # Metadados do sistema
        if hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
            df_inv = st.session_state.dados_inventario
            export_data['metadados']['inventario'] = {
                'num_registros': len(df_inv),
                'num_talhoes': df_inv['talhao'].nunique(),
                'colunas': df_inv.columns.tolist(),
                'dap_range': [float(df_inv['D_cm'].min()), float(df_inv['D_cm'].max())],
                'altura_range': [float(df_inv['H_m'].min()), float(df_inv['H_m'].max())]
            }

    except Exception as e:
        export_data['erro_export'] = str(e)

    return export_data


def importar_configuracao_completa(config_data):
    """
    Importa configuração completa do sistema

    Args:
        config_data: Dados de configuração a importar

    Returns:
        bool: Sucesso da operação
    """
    try:
        if not isinstance(config_data, dict):
            return False

        # Importar configuração global
        if 'configuracao_global' in config_data:
            try:
                from config.configuracoes_globais import inicializar_configuracoes_globais
                inicializar_configuracoes_globais()

                # Atualizar configuração
                st.session_state.config_global.update(config_data['configuracao_global'])
                st.success("✅ Configuração global importada")

            except Exception as e:
                st.warning(f"⚠️ Erro ao importar configuração: {e}")

        return True

    except Exception as e:
        st.error(f"❌ Erro na importação: {e}")
        return False


def monitorar_memoria_streamlit():
    """
    Monitora uso de memória do Streamlit em tempo real

    Returns:
        dict: Informações de memória
    """
    import psutil
    import sys

    info = {
        'timestamp': datetime.now(),
        'processo_atual': {},
        'sistema': {},
        'streamlit_specific': {}
    }

    try:
        # Informações do processo atual
        processo = psutil.Process()
        memoria_processo = processo.memory_info()

        info['processo_atual'] = {
            'rss_mb': memoria_processo.rss / (1024 ** 2),
            'vms_mb': memoria_processo.vms / (1024 ** 2),
            'percent': processo.memory_percent(),
            'num_threads': processo.num_threads()
        }

        # Informações do sistema
        memoria_sistema = psutil.virtual_memory()
        info['sistema'] = {
            'total_gb': memoria_sistema.total / (1024 ** 3),
            'disponivel_gb': memoria_sistema.available / (1024 ** 3),
            'usado_percent': memoria_sistema.percent
        }

        # Informações específicas do Streamlit
        session_state_size = sys.getsizeof(st.session_state) / (1024 ** 2)

        info['streamlit_specific'] = {
            'session_state_mb': session_state_size,
            'num_keys': len(st.session_state.keys()),
            'cache_hits': getattr(st, '_cache_hits', 0),
            'cache_misses': getattr(st, '_cache_misses', 0)
        }

    except Exception as e:
        info['erro'] = str(e)

    return info


def otimizar_performance_automatica():
    """
    Aplica otimizações automáticas de performance

    Returns:
        list: Lista de otimizações aplicadas
    """
    otimizacoes = []

    try:
        # 1. Garbage collection
        import gc
        objetos_coletados = gc.collect()
        if objetos_coletados > 0:
            otimizacoes.append(f"Garbage collection: {objetos_coletados} objetos coletados")

        # 2. Otimizar DataFrames grandes
        for key, value in st.session_state.items():
            if isinstance(value, pd.DataFrame) and len(value) > 10000:
                try:
                    original_memory = value.memory_usage(deep=True).sum() / (1024 ** 2)
                    optimized_df = otimizar_dataframe_memoria(value)
                    new_memory = optimized_df.memory_usage(deep=True).sum() / (1024 ** 2)

                    if new_memory < original_memory * 0.8:  # Reduziu pelo menos 20%
                        st.session_state[key] = optimized_df
                        reducao = original_memory - new_memory
                        otimizacoes.append(f"DataFrame {key}: -{reducao:.1f}MB")

                except Exception:
                    pass

        # 3. Limpar keys antigas (baseado em timestamp)
        keys_antigas = []
        agora = datetime.now()

        for key, value in st.session_state.items():
            if isinstance(value, dict) and 'timestamp' in value:
                try:
                    timestamp = value['timestamp']
                    if isinstance(timestamp, datetime):
                        idade = agora - timestamp
                        if idade > timedelta(hours=2):  # Mais de 2 horas
                            keys_antigas.append(key)
                except Exception:
                    pass

        for key in keys_antigas:
            try:
                del st.session_state[key]
                otimizacoes.append(f"Removida key antiga: {key}")
            except Exception:
                pass

        # 4. Limitar número total de keys
        if len(st.session_state.keys()) > 100:
            # Manter apenas keys essenciais
            keys_essenciais = [
                'dados_inventario', 'dados_cubagem', 'config_global',
                'resultados_hipsometricos', 'resultados_volumetricos', 'inventario_processado'
            ]

            keys_para_remover = []
            for key in st.session_state.keys():
                if key not in keys_essenciais and not any(essential in key for essential in keys_essenciais):
                    keys_para_remover.append(key)

            # Remover apenas algumas (não todas de uma vez)
            for key in keys_para_remover[:20]:  # Máximo 20 por vez
                try:
                    del st.session_state[key]
                    otimizacoes.append(f"Removida key não-essencial: {key}")
                except Exception:
                    pass

    except Exception as e:
        otimizacoes.append(f"Erro na otimização: {e}")

    return otimizacoes


def verificar_compatibilidade_navegador():
    """
    Verifica compatibilidade do navegador com o sistema

    Returns:
        dict: Informações de compatibilidade
    """
    # Esta função seria mais útil com JavaScript, mas podemos fazer verificações básicas
    compatibilidade = {
        'timestamp': datetime.now(),
        'streamlit_version': st.__version__,
        'recomendacoes': [],
        'warnings': []
    }

    try:
        # Verificar versão do Streamlit
        version_parts = st.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        if major < 1:
            compatibilidade['warnings'].append("Versão do Streamlit muito antiga")
        elif major == 1 and minor < 25:
            compatibilidade['warnings'].append("Recomendada versão mais recente do Streamlit")

        # Verificar funcionalidades específicas
        funcionalidades = {
            'file_uploader': hasattr(st, 'file_uploader'),
            'columns': hasattr(st, 'columns'),
            'tabs': hasattr(st, 'tabs'),
            'sidebar': hasattr(st, 'sidebar'),
            'session_state': hasattr(st, 'session_state')
        }

        compatibilidade['funcionalidades'] = funcionalidades

        funcionalidades_faltantes = [k for k, v in funcionalidades.items() if not v]
        if funcionalidades_faltantes:
            compatibilidade['warnings'].extend([
                f"Funcionalidade não disponível: {func}" for func in funcionalidades_faltantes
            ])

        # Recomendações gerais
        compatibilidade['recomendacoes'].extend([
            "Use Chrome ou Firefox para melhor experiência",
            "Mantenha o navegador atualizado",
            "Evite fechar a aba durante processamentos longos",
            "Use resolução mínima de 1024x768"
        ])

    except Exception as e:
        compatibilidade['erro'] = str(e)

    return compatibilidade


# Função utilitária para debug
def debug_session_state(mostrar_valores=False):
    """
    Função de debug para examinar o session_state

    Args:
        mostrar_valores: Se deve mostrar os valores das keys
    """
    st.subheader("🔍 Debug - Session State")

    if not st.session_state:
        st.info("Session state vazio")
        return

    # Mostrar resumo
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de Keys", len(st.session_state.keys()))

    with col2:
        # Calcular tamanho aproximado
        import sys
        tamanho_total = sum(sys.getsizeof(v) for v in st.session_state.values()) / (1024 ** 2)
        st.metric("Tamanho Aprox.", f"{tamanho_total:.1f} MB")

    with col3:
        dataframes = sum(1 for v in st.session_state.values() if isinstance(v, pd.DataFrame))
        st.metric("DataFrames", dataframes)

    # Lista de keys
    st.subheader("📋 Keys no Session State")

    for i, (key, value) in enumerate(st.session_state.items()):
        with st.expander(f"{i + 1}. {key} ({type(value).__name__})"):
            if isinstance(value, pd.DataFrame):
                st.write(f"Shape: {value.shape}")
                st.write(f"Colunas: {list(value.columns)}")
                if mostrar_valores:
                    st.dataframe(value.head())
            elif isinstance(value, dict):
                st.write(f"Keys: {list(value.keys())}")
                if mostrar_valores:
                    st.json(value)
            elif isinstance(value, (list, tuple)):
                st.write(f"Tamanho: {len(value)}")
                if mostrar_valores and len(value) < 20:
                    st.write(value)
            else:
                if mostrar_valores:
                    st.write(repr(value)[:200] + "..." if len(repr(value)) > 200 else repr(value))