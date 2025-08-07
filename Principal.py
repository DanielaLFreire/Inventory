# Principal.py - VERSÃO COMPLETA COM LAS/LAZ
"""
Sistema Integrado de Inventário Florestal - GreenVista
Página principal do sistema com upload de dados e navegação
VERSÃO COMPLETA: Inclui processamento LAS/LAZ, persistência total, interface completa
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import warnings

warnings.filterwarnings('ignore')

# Importar componentes do sistema
from ui.sidebar import criar_sidebar_melhorada, obter_status_sistema_completo
from ui.components import (
    configurar_pagina_greenvista,
    criar_cabecalho_greenvista,
    criar_navegacao_rapida_botoes,
    criar_secao_instrucoes,
    mostrar_alertas_sistema,
    mostrar_empresa)

# Importar processadores CORRIGIDOS
from utils.arquivo_handler import carregar_arquivo_seguro
from config.configuracoes_globais import (
    inicializar_configuracoes_globais,
    obter_configuracao_global,
    aplicar_filtros_configuracao_global
)

# Importar processador LAS se disponível
try:
    from processors.las_processor_integrado import (
        ProcessadorLASIntegrado,
        integrar_com_pagina_lidar
    )

    PROCESSAMENTO_LAS_DISPONIVEL = True
except ImportError:
    PROCESSAMENTO_LAS_DISPONIVEL = False



# Configurar página
configurar_pagina_greenvista("Página Principal", "./images/logo.png")


def verificar_disponibilidade_las():
    """Verifica se processamento LAS está disponível no sistema"""
    if not PROCESSAMENTO_LAS_DISPONIVEL:
        return False, ["Módulo las_processor_integrado não encontrado"]

    try:
        return integrar_com_pagina_lidar(), []
    except Exception as e:
        return False, [str(e)]


def processar_dados_inventario(arquivo_inventario):
    """
    Processa e valida dados do inventário florestal
    VERSÃO CORRIGIDA: Usa função segura de carregamento

    Args:
        arquivo_inventario: Arquivo de inventário carregado OU DataFrame

    Returns:
        DataFrame processado ou None se erro
    """
    try:
        if arquivo_inventario is None:
            return None

        # CORREÇÃO: Usar função segura que trata DataFrames
        with st.spinner("🔄 Carregando dados de inventário..."):
            df_inventario = carregar_arquivo_seguro(arquivo_inventario, "inventário")

        if df_inventario is None:
            st.error("❌ Não foi possível carregar o arquivo de inventário")
            return None

        # Verificar se é DataFrame
        if not isinstance(df_inventario, pd.DataFrame):
            st.error("❌ Dados de inventário inválidos")
            return None

        # Validar estrutura básica
        colunas_obrigatorias = ['D_cm', 'H_m', 'talhao', 'parcela']
        colunas_faltantes = [col for col in colunas_obrigatorias if col not in df_inventario.columns]

        if colunas_faltantes:
            st.error(f"❌ Colunas obrigatórias faltantes: {colunas_faltantes}")
            mostrar_colunas_disponiveis(df_inventario)
            return None

        # Limpar e validar dados
        df_limpo = limpar_dados_inventario(df_inventario)

        if len(df_limpo) == 0:
            st.error("❌ Nenhum registro válido após limpeza dos dados")
            return None

        # Mostrar estatísticas
        mostrar_estatisticas_inventario(df_limpo, df_inventario)

        st.success(f"✅ Inventário processado: {len(df_limpo)} registros válidos")

        return df_limpo

    except Exception as e:
        st.error(f"❌ Erro ao processar inventário: {e}")
        with st.expander("🔍 Detalhes do erro"):
            st.code(traceback.format_exc())
        return None


def processar_dados_cubagem(arquivo_cubagem):
    """
    Processa e valida dados de cubagem
    VERSÃO CORRIGIDA: Usa função segura de carregamento

    Args:
        arquivo_cubagem: Arquivo de cubagem carregado OU DataFrame

    Returns:
        DataFrame processado ou None se erro
    """
    try:
        if arquivo_cubagem is None:
            return None

        # CORREÇÃO: Usar função segura que trata DataFrames
        with st.spinner("🔄 Carregando dados de cubagem..."):
            df_cubagem = carregar_arquivo_seguro(arquivo_cubagem, "cubagem")

        if df_cubagem is None:
            st.error("❌ Não foi possível carregar o arquivo de cubagem")
            return None

        # Verificar se é DataFrame
        if not isinstance(df_cubagem, pd.DataFrame):
            st.error("❌ Dados de cubagem inválidos")
            return None

        # Validar estrutura básica
        colunas_obrigatorias = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']
        colunas_faltantes = [col for col in colunas_obrigatorias if col not in df_cubagem.columns]

        if colunas_faltantes:
            st.error(f"❌ Colunas obrigatórias faltantes: {colunas_faltantes}")
            mostrar_colunas_disponiveis(df_cubagem)
            return None

        # Limpar e validar dados
        df_limpo = limpar_dados_cubagem(df_cubagem)

        if len(df_limpo) == 0:
            st.error("❌ Nenhum registro válido após limpeza dos dados")
            return None

        # Mostrar estatísticas
        mostrar_estatisticas_cubagem(df_limpo, df_cubagem)

        st.success(f"✅ Cubagem processada: {len(df_limpo)} registros válidos")

        return df_limpo

    except Exception as e:
        st.error(f"❌ Erro ao processar cubagem: {e}")
        with st.expander("🔍 Detalhes do erro"):
            st.code(traceback.format_exc())
        return None


def limpar_dados_inventario(df_inventario):
    """
    Limpa e valida dados do inventário
    VERSÃO CORRIGIDA: Melhor tratamento de erros

    Args:
        df_inventario: DataFrame bruto do inventário

    Returns:
        DataFrame limpo
    """
    if not isinstance(df_inventario, pd.DataFrame):
        st.error("❌ Dados de inventário não são um DataFrame válido")
        return pd.DataFrame()

    df_limpo = df_inventario.copy()

    # Converter colunas para tipos apropriados
    try:
        df_limpo['D_cm'] = pd.to_numeric(df_limpo['D_cm'], errors='coerce')
        df_limpo['H_m'] = pd.to_numeric(df_limpo['H_m'], errors='coerce')
        df_limpo['talhao'] = pd.to_numeric(df_limpo['talhao'], errors='coerce').astype('Int64')
        df_limpo['parcela'] = pd.to_numeric(df_limpo['parcela'], errors='coerce').astype('Int64')

        # Idade se disponível
        if 'idade_anos' in df_limpo.columns:
            df_limpo['idade_anos'] = pd.to_numeric(df_limpo['idade_anos'], errors='coerce')

        # Código se disponível
        if 'cod' in df_limpo.columns:
            df_limpo['cod'] = df_limpo['cod'].astype(str)

    except Exception as e:
        st.warning(f"⚠️ Problema na conversão de tipos: {e}")

    # Remover registros inválidos
    try:
        mask_valido = (
                df_limpo['D_cm'].notna() &
                df_limpo['H_m'].notna() &
                df_limpo['talhao'].notna() &
                df_limpo['parcela'].notna() &
                (df_limpo['D_cm'] > 0) &
                (df_limpo['H_m'] > 1.3)  # Altura mínima realística
        )

        df_limpo = df_limpo[mask_valido]

    except Exception as e:
        st.warning(f"⚠️ Problema na filtragem básica: {e}")

    # Remover outliers extremos
    try:
        # Outliers de DAP (usando IQR)
        if 'D_cm' in df_limpo.columns and len(df_limpo) > 0:
            Q1_dap = df_limpo['D_cm'].quantile(0.25)
            Q3_dap = df_limpo['D_cm'].quantile(0.75)
            IQR_dap = Q3_dap - Q1_dap
            limite_inf_dap = Q1_dap - 3 * IQR_dap
            limite_sup_dap = Q3_dap + 3 * IQR_dap

            # Outliers de altura
            Q1_h = df_limpo['H_m'].quantile(0.25)
            Q3_h = df_limpo['H_m'].quantile(0.75)
            IQR_h = Q3_h - Q1_h
            limite_inf_h = Q1_h - 3 * IQR_h
            limite_sup_h = Q3_h + 3 * IQR_h

            # Aplicar filtros de outliers
            mask_outliers = (
                    (df_limpo['D_cm'] >= limite_inf_dap) &
                    (df_limpo['D_cm'] <= limite_sup_dap) &
                    (df_limpo['H_m'] >= limite_inf_h) &
                    (df_limpo['H_m'] <= limite_sup_h)
            )

            df_limpo = df_limpo[mask_outliers]

    except Exception as e:
        st.warning(f"⚠️ Problema na remoção de outliers: {e}")

    return df_limpo


def limpar_dados_cubagem(df_cubagem):
    """
    Limpa e valida dados de cubagem
    VERSÃO CORRIGIDA: Melhor tratamento de erros

    Args:
        df_cubagem: DataFrame bruto da cubagem

    Returns:
        DataFrame limpo
    """
    if not isinstance(df_cubagem, pd.DataFrame):
        st.error("❌ Dados de cubagem não são um DataFrame válido")
        return pd.DataFrame()

    df_limpo = df_cubagem.copy()

    # Converter colunas para tipos apropriados
    try:
        df_limpo['arv'] = pd.to_numeric(df_limpo['arv'], errors='coerce').astype('Int64')
        df_limpo['talhao'] = pd.to_numeric(df_limpo['talhao'], errors='coerce').astype('Int64')
        df_limpo['d_cm'] = pd.to_numeric(df_limpo['d_cm'], errors='coerce')
        df_limpo['h_m'] = pd.to_numeric(df_limpo['h_m'], errors='coerce')
        df_limpo['D_cm'] = pd.to_numeric(df_limpo['D_cm'], errors='coerce')
        df_limpo['H_m'] = pd.to_numeric(df_limpo['H_m'], errors='coerce')

    except Exception as e:
        st.warning(f"⚠️ Problema na conversão de tipos: {e}")

    # Remover registros inválidos
    try:
        mask_valido = (
                df_limpo['arv'].notna() &
                df_limpo['talhao'].notna() &
                df_limpo['d_cm'].notna() &
                df_limpo['h_m'].notna() &
                df_limpo['D_cm'].notna() &
                df_limpo['H_m'].notna() &
                (df_limpo['d_cm'] > 0) &
                (df_limpo['h_m'] > 0) &
                (df_limpo['D_cm'] > 0) &
                (df_limpo['H_m'] > 1.3)
        )

        df_limpo = df_limpo[mask_valido]

    except Exception as e:
        st.warning(f"⚠️ Problema na filtragem básica: {e}")

    # Validar consistência (diâmetro da seção <= DAP)
    try:
        if len(df_limpo) > 0:
            mask_consistente = df_limpo['d_cm'] <= df_limpo['D_cm'] * 1.2  # Tolerância de 20%
            df_limpo = df_limpo[mask_consistente]

    except Exception as e:
        st.warning(f"⚠️ Problema na validação de consistência: {e}")

    return df_limpo


def mostrar_colunas_disponiveis(df):
    """Mostra colunas disponíveis no arquivo"""
    if isinstance(df, pd.DataFrame):
        st.info("📋 Colunas disponíveis no arquivo:")
        colunas_str = ", ".join(df.columns.tolist())
        st.code(colunas_str)
    else:
        st.warning("⚠️ Não foi possível exibir colunas - dados inválidos")


def mostrar_estatisticas_inventario(df_limpo, df_original):
    """Mostra estatísticas do inventário processado"""
    try:
        if not isinstance(df_limpo, pd.DataFrame) or not isinstance(df_original, pd.DataFrame):
            st.warning("⚠️ Não foi possível calcular estatísticas - dados inválidos")
            return

        with st.expander("📊 Estatísticas do Inventário"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Registros Originais", len(df_original))
                st.metric("Registros Válidos", len(df_limpo))

            with col2:
                try:
                    talhoes = df_limpo['talhao'].nunique()
                    parcelas = df_limpo.groupby(['talhao', 'parcela']).ngroups
                    st.metric("Talhões", talhoes)
                    st.metric("Parcelas", parcelas)
                except Exception:
                    st.metric("Talhões", "N/A")
                    st.metric("Parcelas", "N/A")

            with col3:
                try:
                    dap_medio = df_limpo['D_cm'].mean()
                    altura_media = df_limpo['H_m'].mean()
                    st.metric("DAP Médio", f"{dap_medio:.1f} cm")
                    st.metric("Altura Média", f"{altura_media:.1f} m")
                except Exception:
                    st.metric("DAP Médio", "N/A")
                    st.metric("Altura Média", "N/A")

            with col4:
                try:
                    dap_min, dap_max = df_limpo['D_cm'].min(), df_limpo['D_cm'].max()
                    alt_min, alt_max = df_limpo['H_m'].min(), df_limpo['H_m'].max()
                    st.metric("DAP Min-Max", f"{dap_min:.1f}-{dap_max:.1f} cm")
                    st.metric("Altura Min-Max", f"{alt_min:.1f}-{alt_max:.1f} m")
                except Exception:
                    st.metric("DAP Min-Max", "N/A")
                    st.metric("Altura Min-Max", "N/A")

            # Mostrar preview dos dados
            st.subheader("👀 Preview dos Dados")
            if len(df_limpo) > 0:
                st.dataframe(df_limpo.head(10), use_container_width=True)
            else:
                st.warning("⚠️ Nenhum dado para exibir")

    except Exception as e:
        st.error(f"❌ Erro ao mostrar estatísticas: {e}")


def mostrar_estatisticas_cubagem(df_limpo, df_original):
    """Mostra estatísticas da cubagem processada"""
    try:
        if not isinstance(df_limpo, pd.DataFrame) or not isinstance(df_original, pd.DataFrame):
            st.warning("⚠️ Não foi possível calcular estatísticas - dados inválidos")
            return

        with st.expander("📊 Estatísticas da Cubagem"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Registros Originais", len(df_original))
                st.metric("Registros Válidos", len(df_limpo))

            with col2:
                try:
                    arvores = df_limpo['arv'].nunique()
                    talhoes = df_limpo['talhao'].nunique()
                    st.metric("Árvores Cubadas", arvores)
                    st.metric("Talhões", talhoes)
                except Exception:
                    st.metric("Árvores Cubadas", "N/A")
                    st.metric("Talhões", "N/A")

            with col3:
                try:
                    seções_por_arvore = df_limpo.groupby(['talhao', 'arv']).size()
                    secoes_media = seções_por_arvore.mean()
                    dap_medio = df_limpo['D_cm'].mean()
                    st.metric("Seções/Árvore", f"{secoes_media:.1f}")
                    st.metric("DAP Médio", f"{dap_medio:.1f} cm")
                except Exception:
                    st.metric("Seções/Árvore", "N/A")
                    st.metric("DAP Médio", "N/A")

            with col4:
                try:
                    altura_media = df_limpo['H_m'].mean()
                    diam_secao_medio = df_limpo['d_cm'].mean()
                    st.metric("Altura Média", f"{altura_media:.1f} m")
                    st.metric("Diâm. Seção Médio", f"{diam_secao_medio:.1f} cm")
                except Exception:
                    st.metric("Altura Média", "N/A")
                    st.metric("Diâm. Seção Médio", "N/A")

            # Mostrar preview dos dados
            st.subheader("👀 Preview dos Dados")
            if len(df_limpo) > 0:
                st.dataframe(df_limpo.head(10), use_container_width=True)
            else:
                st.warning("⚠️ Nenhum dado para exibir")

    except Exception as e:
        st.error(f"❌ Erro ao mostrar estatísticas: {e}")


def mostrar_preview_dados_carregados():
    """Mostra preview dos dados já carregados incluindo LiDAR"""
    try:
        # === DADOS PRINCIPAIS ===
        if hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
            st.subheader("📋 Dados de Inventário Carregados")

            df_inv = st.session_state.dados_inventario

            if isinstance(df_inv, pd.DataFrame) and len(df_inv) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Registros", len(df_inv))
                with col2:
                    st.metric("Talhões", df_inv['talhao'].nunique())
                with col3:
                    try:
                        dap_medio = df_inv['D_cm'].mean()
                        st.metric("DAP Médio", f"{dap_medio:.1f} cm")
                    except Exception:
                        st.metric("DAP Médio", "N/A")
                with col4:
                    try:
                        altura_media = df_inv['H_m'].mean()
                        st.metric("Altura Média", f"{altura_media:.1f} m")
                    except Exception:
                        st.metric("Altura Média", "N/A")

                if st.checkbox("👀 Mostrar Preview do Inventário"):
                    st.dataframe(df_inv.head(), use_container_width=True)
            else:
                st.warning("⚠️ Dados de inventário inválidos ou vazios")

        if hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None:
            st.subheader("📏 Dados de Cubagem Carregados")

            df_cub = st.session_state.dados_cubagem

            if isinstance(df_cub, pd.DataFrame) and len(df_cub) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Registros", len(df_cub))
                with col2:
                    try:
                        arvores = df_cub['arv'].nunique()
                        st.metric("Árvores", arvores)
                    except Exception:
                        st.metric("Árvores", "N/A")
                with col3:
                    try:
                        dap_medio = df_cub['D_cm'].mean()
                        st.metric("DAP Médio", f"{dap_medio:.1f} cm")
                    except Exception:
                        st.metric("DAP Médio", "N/A")
                with col4:
                    try:
                        seções = df_cub.groupby(['talhao', 'arv']).size().mean()
                        st.metric("Seções/Árvore", f"{seções:.1f}")
                    except Exception:
                        st.metric("Seções/Árvore", "N/A")

                if st.checkbox("👀 Mostrar Preview da Cubagem"):
                    st.dataframe(df_cub.head(), use_container_width=True)
            else:
                st.warning("⚠️ Dados de cubagem inválidos ou vazios")

        # === DADOS LIDAR ===
        mostrar_preview_dados_lidar()

    except Exception as e:
        st.error(f"❌ Erro ao mostrar preview: {e}")


def mostrar_preview_dados_lidar():
    """Mostra preview específico dos dados LiDAR"""
    dados_lidar_encontrados = False

    # Verificar arquivo LAS/LAZ
    if hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None:
        st.subheader("🛩️ Arquivo LAS/LAZ Carregado")

        arquivo_las = st.session_state.arquivo_las

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Arquivo", "LAS/LAZ")
        with col2:
            try:
                nome = getattr(arquivo_las, 'name', 'arquivo.las')
                st.metric("Nome", nome[:20] + "..." if len(nome) > 20 else nome)
            except:
                st.metric("Nome", "N/A")
        with col3:
            try:
                tamanho_mb = getattr(arquivo_las, 'size', 0) / (1024 * 1024)
                st.metric("Tamanho", f"{tamanho_mb:.1f} MB")
            except:
                st.metric("Tamanho", "N/A")
        with col4:
            # Verificar se já foi processado
            processado = hasattr(st.session_state, 'dados_lidar_las') and st.session_state.dados_lidar_las is not None
            st.metric("Status", "✅ Processado" if processado else "⏳ Pendente")

        dados_lidar_encontrados = True

    # Verificar dados LAS processados
    if hasattr(st.session_state, 'dados_lidar_las') and st.session_state.dados_lidar_las is not None:
        st.subheader("📊 Dados LAS Processados")

        dados_las = st.session_state.dados_lidar_las

        if 'df_metricas' in dados_las:
            df_metricas = dados_las['df_metricas']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Parcelas", len(df_metricas))
            with col2:
                try:
                    altura_media = df_metricas['altura_media'].mean()
                    st.metric("Altura Média", f"{altura_media:.1f} m")
                except:
                    st.metric("Altura Média", "N/A")
            with col3:
                try:
                    pontos_total = df_metricas['n_pontos'].sum()
                    st.metric("Total Pontos", f"{pontos_total:,}")
                except:
                    st.metric("Total Pontos", "N/A")
            with col4:
                try:
                    cobertura_media = df_metricas['cobertura'].mean()
                    st.metric("Cobertura", f"{cobertura_media:.1f}%")
                except:
                    st.metric("Cobertura", "N/A")

            if st.checkbox("👀 Preview Métricas LAS"):
                st.dataframe(df_metricas.head(), use_container_width=True)

        dados_lidar_encontrados = True

    # Verificar métricas LiDAR carregadas
    if hasattr(st.session_state, 'arquivo_metricas_lidar') and st.session_state.arquivo_metricas_lidar is not None:
        st.subheader("📊 Métricas LiDAR Carregadas")

        arquivo_metricas = st.session_state.arquivo_metricas_lidar

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tipo", "Métricas CSV/Excel")
        with col2:
            try:
                nome = getattr(arquivo_metricas, 'name', 'metricas.csv')
                st.metric("Arquivo", nome[:20] + "..." if len(nome) > 20 else nome)
            except:
                st.metric("Arquivo", "N/A")
        with col3:
            # Verificar se foi integrado
            integrado = hasattr(st.session_state, 'dados_lidar') and st.session_state.dados_lidar is not None
            st.metric("Status", "✅ Integrado" if integrado else "⏳ Pendente")
        with col4:
            st.metric("Origem", "Pré-processado")

        dados_lidar_encontrados = True

    # Verificar dados LiDAR integrados
    if hasattr(st.session_state, 'dados_lidar') and st.session_state.dados_lidar is not None:
        st.subheader("🔗 Dados LiDAR Integrados")

        dados_lidar = st.session_state.dados_lidar

        if 'stats_comparacao' in dados_lidar and dados_lidar['stats_comparacao'] is not None:
            stats = dados_lidar['stats_comparacao']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                correlacao = stats.get('correlacao', 0)
                st.metric("Correlação", f"{correlacao:.3f}")
            with col2:
                r2 = stats.get('r2', 0)
                st.metric("R²", f"{r2:.3f}")
            with col3:
                rmse = stats.get('rmse', 0)
                st.metric("RMSE", f"{rmse:.2f} m")
            with col4:
                n_parcelas = stats.get('n_parcelas', 0)
                st.metric("Parcelas", n_parcelas)

        # Mostrar alertas se houver
        if 'alertas' in dados_lidar and dados_lidar['alertas']:
            st.info(f"⚠️ {len(dados_lidar['alertas'])} alertas gerados na integração")

        dados_lidar_encontrados = True

    # Mostrar botão de acesso ao LiDAR se há dados
    if dados_lidar_encontrados:
        st.info("🛩️ **Dados LiDAR disponíveis!** Acesse a Etapa 4 para análise completa.")
        #if st.button("🚀 Ir para Etapa 4 - LiDAR", type="primary"):
        #    st.switch_page("pages/4_🛩️_Dados_LiDAR.py")


def mostrar_status_sistema():
    """Mostra status geral do sistema incluindo LiDAR"""
    st.subheader("🔧 Status do Sistema")

    try:
        # Obter status completo do sistema
        status = obter_status_sistema_completo()

        # === LINHA 1: DADOS PRINCIPAIS ===
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if status['dados_inventario'] and status['dados_cubagem']:
                st.success("✅ **Dados Necessários**\nCarregados")
            elif status['dados_inventario'] or status['dados_cubagem']:
                st.warning("⚠️ **Dados Necessários**\nIncompletos")
            else:
                st.error("❌ **Dados**\nFaltantes")

        with col2:
            if status['configurado']:
                st.success("✅ **Configuração**\nOK")
            else:
                st.error("❌ **Configuração**\nNecessária")

        with col3:
            if status['hip_executado']:
                st.success("✅ **Hipsométricos**\nConcluída")
            else:
                st.info("⏳ **Hipsométricos**\nPendente")

        with col4:
            if status['vol_executado']:
                st.success("✅ **Volumétricos**\nConcluída")
            else:
                st.info("⏳ **Volumétricos**\nPendente")

        with col5:
            if status['inv_executado']:
                st.success("✅ **Inventário**\nConcluída")
            else:
                st.info("⏳ **Inventário**\nPendente")

        # === LINHA 2: DADOS LIDAR ===
        st.markdown("#### 🛩️ Status LiDAR")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if status['arquivo_las_disponivel']:
                st.success("✅ **Arquivo LAS**\nDisponível")
            else:
                st.info("⏳ **Arquivo LAS**\nNão carregado")

        with col2:
            if status['metricas_lidar_disponivel']:
                st.success("✅ **Métricas**\nDisponíveis")
            else:
                st.info("⏳ **Métricas**\nNão carregadas")

        with col3:
            if status['dados_lidar_processados']:
                st.success("✅ **LAS Processado**\nConcluído")
            elif status['arquivo_las_disponivel']:
                st.warning("⚠️ **LAS Processado**\nPendente")
            else:
                st.info("⏳ **LAS Processado**\nSem arquivo")

        with col4:
            if status['dados_lidar_integrados']:
                st.success("✅ **LiDAR Integrado**\nConcluído")
            elif status['metricas_lidar_disponivel'] or status['dados_lidar_processados']:
                st.warning("⚠️ **LiDAR Integrado**\nPendente")
            else:
                st.info("⏳ **LiDAR Integrado**\nSem dados")

        # Progresso geral
        st.markdown("#### 📊 Progresso Geral")

        # Barra de progresso principal (etapas obrigatórias)
        st.progress(status['progresso_total'], text=f"Etapas Principais: {status['progresso_total'] * 100:.0f}%")

        # Barra de progresso completo (incluindo LiDAR)
        if status['progresso_completo'] > status['progresso_total']:
            st.progress(status['progresso_completo'],
                        text=f"Progresso Completo: {status['progresso_completo'] * 100:.0f}% (inclui LiDAR)")

    except Exception as e:
        st.error(f"❌ Erro ao mostrar status do sistema: {e}")


def main():
    """Função principal da aplicação - VERSÃO COMPLETA COM LAS/LAZ"""
    try:
        # Inicializar configurações globais
        inicializar_configuracoes_globais()

        #st.image("./images/logo.png")

        # Criar cabeçalho
        criar_cabecalho_greenvista("Página Principal")

        # Criar sidebar com uploads - VERSÃO COMPLETA
        arquivos = criar_sidebar_melhorada()

        # Processar arquivos se carregados - COM TRATAMENTO SEGURO
        if arquivos['inventario'] is not None:
            dados_inventario = processar_dados_inventario(arquivos['inventario'])
            if dados_inventario is not None:
                st.session_state.dados_inventario = dados_inventario

        if arquivos['cubagem'] is not None:
            dados_cubagem = processar_dados_cubagem(arquivos['cubagem'])
            if dados_cubagem is not None:
                st.session_state.dados_cubagem = dados_cubagem

        # === SEÇÃO PRINCIPAL DA PÁGINA ===
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Instruções",
            "📊 Status do Sistema",
            "⚠️ Alertas",
            "👨🏻‍💻 Quem somos"
        ])

        with tab1:
            criar_secao_instrucoes()

        with tab2:
            mostrar_status_sistema()

        with tab3:
            mostrar_alertas_sistema()

        with tab4:
            mostrar_empresa()

        # Navegação rápida
        st.markdown("---")
        criar_navegacao_rapida_botoes()

        # === INFORMAÇÕES DE PERSISTÊNCIA ===
        mostrar_info_persistencia()

    except Exception as e:
        st.error("❌ Erro crítico na aplicação principal")

        with st.expander("🔍 Detalhes do Erro Crítico"):
            st.code(f"Erro: {str(e)}")
            st.code(traceback.format_exc())

        # Oferecer reset do sistema
        st.warning("🔄 **Solução:** Tente recarregar a página ou limpar o cache")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Recarregar Página", type="primary"):
                st.rerun()

        with col2:
            if st.button("🗑️ Limpar Cache", type="secondary"):
                # Limpar session_state
                keys_para_limpar = [k for k in st.session_state.keys()
                                    if not k.startswith('FormSubmitter')]
                for key in keys_para_limpar:
                    try:
                        del st.session_state[key]
                    except:
                        pass
                st.success("✅ Cache limpo! Recarregando...")
                st.rerun()


def mostrar_info_persistencia():
    """Mostra informações sobre persistência na sessão"""
    try:
        with st.expander("💾 Informações de Persistência"):
            st.markdown("""
            **🔄 Dados Persistentes na Sessão:**

            **✅ Permanecem ao navegar entre páginas:**
            - 📋 Dados de inventário processados
            - 📏 Dados de cubagem processados
            - 🛩️ Arquivos LAS/LAZ carregados
            - 📊 Métricas LiDAR carregadas
            - 🗺️ Shapefiles carregados
            - 📍 Coordenadas carregadas
            - ⚙️ Configurações do sistema
            - 📈 Resultados das análises

            **❌ São perdidos ao:**
            - Fechar o navegador
            - Recarregar a página (F5)
            - Limpar cache do sistema
            - Timeout da sessão (inatividade longa)

            **💡 Dica:** Faça download dos resultados importantes!
            """)

            # Mostrar arquivos atualmente persistidos
            st.markdown("**📂 Status Atual da Sessão:**")

            arquivos_persistidos = []

            # Verificar cada tipo de dado
            if hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
                arquivos_persistidos.append("✅ Inventário processado")

            if hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None:
                arquivos_persistidos.append("✅ Cubagem processada")

            if hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None:
                arquivos_persistidos.append("✅ Arquivo LAS/LAZ")

            if hasattr(st.session_state,
                       'arquivo_metricas_lidar') and st.session_state.arquivo_metricas_lidar is not None:
                arquivos_persistidos.append("✅ Métricas LiDAR")

            if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None:
                arquivos_persistidos.append("✅ Shapefile")

            if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None:
                arquivos_persistidos.append("✅ Coordenadas")

            # Verificar resultados
            resultados = ['resultados_hipsometricos', 'resultados_volumetricos', 'inventario_processado',
                          'dados_lidar_las', 'dados_lidar']

            for resultado in resultados:
                if hasattr(st.session_state, resultado) and getattr(st.session_state, resultado) is not None:
                    nome_amigavel = {
                        'resultados_hipsometricos': 'Modelos Hipsométricos',
                        'resultados_volumetricos': 'Modelos Volumétricos',
                        'inventario_processado': 'Inventário Final',
                        'dados_lidar_las': 'Processamento LAS',
                        'dados_lidar': 'Integração LiDAR'
                    }
                    arquivos_persistidos.append(f"✅ {nome_amigavel.get(resultado, resultado)}")

            if arquivos_persistidos:
                for arquivo in arquivos_persistidos:
                    st.success(arquivo)
            else:
                st.info("📭 Nenhum dado persistido na sessão")

    except Exception as e:
        st.error(f"❌ Erro ao mostrar informações de persistência: {e}")


if __name__ == "__main__":
    main()