# Principal.py - VERSÃO COMPLETA COM LAS/LAZ E PREVIEW EXPANDIDO
"""
Sistema Integrado de Inventário Florestal - GreenVista
Página principal do sistema com upload de dados e navegação
VERSÃO COMPLETA: Inclui processamento LAS/LAZ, persistência total, interface completa, preview expandido
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


# ================================
# FUNÇÕES DE PREVIEW EXPANDIDO
# ================================

def escolher_modo_preview():
    """Permite escolher entre preview resumido ou detalhado"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        modo_detalhado = st.toggle(
            "📊 **Análise Detalhada dos Dados**", 
            value=False,
            help="Ative para ver estatísticas completas de todos os arquivos carregados\n\n" +
                 "• **Desativado**: Resumo geral e status\n" +
                 "• **Ativado**: Análise completa com métricas, distribuições e qualidade"
        )
    
    return modo_detalhado


def mostrar_preview_inteligente():
    """Mostra preview adequado baseado na escolha do usuário"""
    modo_detalhado = escolher_modo_preview()
    
    if modo_detalhado:
        mostrar_preview_dados_carregados()  # Versão completa e detalhada
    else:
        mostrar_resumo_geral_dados()  # Versão resumida e concisa


def mostrar_resumo_geral_dados():
    """
    Mostra um resumo conciso e direto de todos os dados carregados
    VERSÃO OTIMIZADA: Para uso como modo 'simples' vs detalhado
    """
    st.subheader("📊 Resumo dos Dados Carregados")
    
    # Verificar dados disponíveis
    tem_inventario = hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None
    tem_cubagem = hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None
    tem_las = hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None
    tem_metricas_lidar = hasattr(st.session_state, 'arquivo_metricas_lidar') and st.session_state.arquivo_metricas_lidar is not None
    tem_shapefile = hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None
    tem_coordenadas = hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None
    
    # === STATUS GERAL ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if tem_inventario and tem_cubagem:
            st.success("✅ **Dados Core**\nCompletos")
            if tem_inventario:
                df_inv = st.session_state.dados_inventario
                st.caption(f"📋 {len(df_inv):,} registros")
        else:
            st.error("❌ **Dados Core**\nIncompletos")
            
    with col2:
        # Contar arquivos extras
        extras = sum([tem_las, tem_metricas_lidar, tem_shapefile, tem_coordenadas])
        if extras > 0:
            st.info(f"📁 **{extras} Arquivos**\nAdicionais")
            extras_lista = []
            if tem_las or tem_metricas_lidar:
                extras_lista.append("🛩️ LiDAR")
            if tem_shapefile:
                extras_lista.append("🗺️ SHP")
            if tem_coordenadas:
                extras_lista.append("📍 Coord")
            st.caption(" • ".join(extras_lista))
        else:
            st.warning("📁 **Sem Arquivos**\nAdicionais")
    
    with col3:
        # Status de processamento
        processados = 0
        total_processos = 3
        
        if hasattr(st.session_state, 'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None:
            processados += 1
        if hasattr(st.session_state, 'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None:
            processados += 1
        if hasattr(st.session_state, 'inventario_processado') and st.session_state.inventario_processado is not None:
            processados += 1
            
        if processados == total_processos:
            st.success("🎉 **Análises**\nCompletas")
        elif processados > 0:
            st.warning(f"⚠️ **{processados}/{total_processos} Etapas**\nConcluídas")
        else:
            st.info("⏳ **Análises**\nPendentes")
        
    with col4:
        # LiDAR específico
        if hasattr(st.session_state, 'dados_lidar') and st.session_state.dados_lidar is not None:
            st.success("🛩️ **LiDAR**\nIntegrado")
            stats = st.session_state.dados_lidar.get('stats_comparacao', {})
            if 'r2' in stats:
                r2 = stats['r2']
                st.caption(f"📊 R²: {r2:.3f}")
        elif tem_las or tem_metricas_lidar:
            st.warning("🛩️ **LiDAR**\nDisponível")
            st.caption("⏳ Não processado")
        else:
            st.info("🛩️ **LiDAR**\nOpcional")

    # === MÉTRICAS RÁPIDAS ===
    if tem_inventario and tem_cubagem:
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        try:
            df_inv = st.session_state.dados_inventario
            df_cub = st.session_state.dados_cubagem
            
            with col1:
                talhoes = df_inv['talhao'].nunique()
                st.metric("🌳 Talhões", talhoes)
                
            with col2:
                parcelas = df_inv.groupby(['talhao', 'parcela']).ngroups
                st.metric("📍 Parcelas", parcelas)
                
            with col3:
                arvores_cubadas = df_cub['arv'].nunique()
                st.metric("📏 Árvores Cubadas", arvores_cubadas)
                
            with col4:
                dap_medio = df_inv['D_cm'].mean()
                st.metric("📐 DAP Médio", f"{dap_medio:.1f} cm")
                
            with col5:
                altura_media = df_inv['H_m'].mean()
                st.metric("📏 Altura Média", f"{altura_media:.1f} m")
                
        except Exception:
            st.caption("⚠️ Erro ao calcular métricas rápidas")

    # === PRÓXIMOS PASSOS ===
    st.markdown("---")
    st.markdown("### 🚀 Próximos Passos")
    
    if not (tem_inventario and tem_cubagem):
        st.error("**1.** 📁 Carregue dados de Inventário e Cubagem na sidebar")
        return
    
    # Verificar configuração
    try:
        from config.configuracoes_globais import obter_configuracao_global
        config_global = obter_configuracao_global()
        configurado = config_global.get('configurado', False)
        
        if not configurado:
            st.warning("**1.** ⚙️ Configure o sistema na Etapa 0")
            st.info("**2.** 🔄 Execute etapas 1-3 em sequência")
            return
    except:
        st.warning("**1.** ⚙️ Configure o sistema na Etapa 0")
        return
    
    # Verificar etapas executadas
    hip_ok = hasattr(st.session_state, 'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None
    vol_ok = hasattr(st.session_state, 'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None
    inv_ok = hasattr(st.session_state, 'inventario_processado') and st.session_state.inventario_processado is not None
    
    if not hip_ok:
        st.info("**1.** 🌳 Execute Etapa 1 - Modelos Hipsométricos")
    elif not vol_ok:
        st.info("**1.** 📊 Execute Etapa 2 - Modelos Volumétricos")
    elif not inv_ok:
        st.info("**1.** 📈 Execute Etapa 3 - Inventário Final")
    else:
        st.success("🎉 **Todas as etapas principais concluídas!**")
        
        # Sugestões extras
        if tem_las or tem_metricas_lidar:
            if not hasattr(st.session_state, 'dados_lidar') or st.session_state.dados_lidar is None:
                st.info("💡 **Opcional:** Processe dados LiDAR na Análise LiDAR")
        else:
            st.info("💡 **Opcional:** Carregue dados LiDAR para análises avançadas")

    # === INFORMAÇÕES DE SESSÃO ===
    with st.expander("💾 Informações da Sessão"):
        st.markdown("""
        **✅ Dados Persistentes:**
        - Todos os arquivos permanecem na sessão
        - Navegue livremente entre páginas
        - Resultados são mantidos até fechar o navegador
        
        **⚠️ Dados são perdidos ao:**
        - Fechar/recarregar o navegador
        - Timeout por inatividade prolongada
        
        **💡 Dica:** Faça download dos resultados importantes!
        """)
        
        # Mostrar arquivos atualmente na sessão
        arquivos_na_sessao = []
        if tem_inventario:
            arquivos_na_sessao.append("📋 Inventário")
        if tem_cubagem:
            arquivos_na_sessao.append("📏 Cubagem")
        if tem_las:
            arquivos_na_sessao.append("🛩️ Arquivo LAS")
        if tem_metricas_lidar:
            arquivos_na_sessao.append("📊 Métricas LiDAR")
        if tem_shapefile:
            arquivos_na_sessao.append("🗺️ Shapefile")
        if tem_coordenadas:
            arquivos_na_sessao.append("📍 Coordenadas")
            
        if arquivos_na_sessao:
            st.success("**Na sessão:** " + " • ".join(arquivos_na_sessao))
        else:
            st.info("Nenhum arquivo na sessão")


def mostrar_preview_dados_carregados():
    """
    Mostra preview completo de todos os dados carregados
    VERSÃO COM EXPANDERS: Organizado como mostrar_estatisticas_cubagem
    """
    try:
        st.subheader("📊 Dados Carregados no Sistema")

        # === DADOS DE INVENTÁRIO ===
        if hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
            df_inventario = st.session_state.dados_inventario

            if isinstance(df_inventario, pd.DataFrame) and len(df_inventario) > 0:
                st.success(f"✅ **Inventário processado:** {len(df_inventario)} registros válidos")

                with st.expander("📊 Estatísticas do Inventário"):
                    # Métricas principais
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Registros", f"{len(df_inventario):,}")
                        talhoes = df_inventario['talhao'].nunique()
                        st.metric("🌳 Talhões", talhoes)
                    with col2:
                        parcelas = df_inventario.groupby(['talhao', 'parcela']).ngroups
                        st.metric("📍 Parcelas", parcelas)
                        try:
                            dap_medio = df_inventario['D_cm'].mean()
                            st.metric("📐 DAP Médio", f"{dap_medio:.1f} cm")
                        except:
                            st.metric("📐 DAP Médio", "N/A")
                    with col3:
                        try:
                            altura_media = df_inventario['H_m'].mean()
                            st.metric("📏 Altura Média", f"{altura_media:.1f} m")
                            dap_min, dap_max = df_inventario['D_cm'].min(), df_inventario['D_cm'].max()
                            st.metric("📊 DAP Min-Max", f"{dap_min:.1f}-{dap_max:.1f}")
                        except:
                            st.metric("📏 Altura Média", "N/A")
                            st.metric("📊 DAP Min-Max", "N/A")
                    with col4:
                        try:
                            alt_min, alt_max = df_inventario['H_m'].min(), df_inventario['H_m'].max()
                            st.metric("📏 Alt Min-Max", f"{alt_min:.1f}-{alt_max:.1f}")
                            # Área basal total
                            area_basal = (df_inventario['D_cm'] ** 2 * np.pi / 40000).sum()
                            st.metric("🎯 Área Basal", f"{area_basal:.1f} m²")
                        except:
                            st.metric("📏 Alt Min-Max", "N/A")
                            st.metric("🎯 Área Basal", "N/A")

                    # Informações de idade se disponível
                    if 'idade_anos' in df_inventario.columns:
                        try:
                            idade_info = df_inventario.groupby('talhao')['idade_anos'].agg(['mean', 'min', 'max'])
                            st.info(f"🕐 **Idade:** {idade_info['mean'].mean():.1f} anos (média geral)")
                        except Exception:
                            pass

                    # Preview dos dados
                    st.subheader("👀 Preview dos Dados")
                    if len(df_inventario) > 0:
                        st.dataframe(df_inventario.head(10), use_container_width=True)
                    else:
                        st.warning("⚠️ Nenhum dado para exibir")

                    # Distribuições opcionais
                    if st.checkbox("📊 Mostrar Distribuições", key="dist_inventario"):
                        col_dist1, col_dist2 = st.columns(2)
                        with col_dist1:
                            st.write("**Distribuição DAP**")
                            try:
                                hist_dap = df_inventario['D_cm'].value_counts().sort_index().head(20)
                                st.bar_chart(hist_dap)
                            except:
                                st.caption("⚠️ Erro ao gerar distribuição DAP")
                        with col_dist2:
                            st.write("**Árvores por Talhão**")
                            try:
                                arvores_talhao = df_inventario['talhao'].value_counts().sort_index()
                                st.bar_chart(arvores_talhao)
                            except:
                                st.caption("⚠️ Erro ao gerar distribuição por talhão")
            else:
                st.warning("⚠️ Inventário existe mas está vazio ou inválido")
        else:
            st.error("❌ **Dados de Inventário:** Não carregados")

        # === DADOS DE CUBAGEM ===
        if hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None:
            df_cubagem = st.session_state.dados_cubagem

            if isinstance(df_cubagem, pd.DataFrame) and len(df_cubagem) > 0:
                arvores_cubadas = df_cubagem['arv'].nunique()
                st.success(f"✅ **Cubagem processada:** {arvores_cubadas} árvores cubadas")

                with st.expander("📊 Estatísticas da Cubagem"):
                    # Métricas principais
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📏 Árvores", arvores_cubadas)
                        total_secoes = len(df_cubagem)
                        st.metric("📊 Seções", f"{total_secoes:,}")
                    with col2:
                        talhoes_cub = df_cubagem['talhao'].nunique()
                        st.metric("🌳 Talhões", talhoes_cub)
                        try:
                            secoes_media = df_cubagem.groupby(['talhao', 'arv']).size().mean()
                            st.metric("📐 Seções/Árvore", f"{secoes_media:.1f}")
                        except:
                            st.metric("📐 Seções/Árvore", "N/A")
                    with col3:
                        try:
                            dap_medio_cub = df_cubagem['D_cm'].mean()
                            st.metric("📊 DAP Médio", f"{dap_medio_cub:.1f} cm")
                            altura_media_cub = df_cubagem['H_m'].mean()
                            st.metric("📏 Alt. Média", f"{altura_media_cub:.1f} m")
                        except:
                            st.metric("📊 DAP Médio", "N/A")
                            st.metric("📏 Alt. Média", "N/A")
                    with col4:
                        try:
                            diam_secao_medio = df_cubagem['d_cm'].mean()
                            st.metric("🎯 Diâm. Seção", f"{diam_secao_medio:.1f} cm")
                            # Consistência d/DAP
                            df_cubagem['razao_d_D'] = df_cubagem['d_cm'] / df_cubagem['D_cm']
                            consistencia = df_cubagem['razao_d_D'].mean()
                            st.metric("⚖️ Consistência", f"{consistencia:.2f}")
                        except:
                            st.metric("🎯 Diâm. Seção", "N/A")
                            st.metric("⚖️ Consistência", "N/A")

                    # Análise de qualidade
                    try:
                        if 'razao_d_D' in df_cubagem.columns:
                            consistencia_pct = (df_cubagem['razao_d_D'] <= 1.0).mean() * 100
                            if consistencia_pct > 95:
                                st.success(
                                    f"🎯 **Excelente qualidade:** {consistencia_pct:.1f}% das seções consistentes")
                            elif consistencia_pct > 85:
                                st.info(f"👍 **Boa qualidade:** {consistencia_pct:.1f}% das seções consistentes")
                            else:
                                st.warning(
                                    f"⚠️ **Verificar qualidade:** {consistencia_pct:.1f}% das seções consistentes")
                    except:
                        pass

                    # Preview dos dados
                    st.subheader("👀 Preview dos Dados")
                    if len(df_cubagem) > 0:
                        st.dataframe(df_cubagem.head(10), use_container_width=True)
                    else:
                        st.warning("⚠️ Nenhum dado para exibir")

                    # Análise por árvore opcional
                    if st.checkbox("🌳 Análise Detalhada por Árvore", key="analise_arvore_detalhada"):
                        try:
                            arvore_stats = df_cubagem.groupby(['talhao', 'arv']).agg({
                                'D_cm': 'first',
                                'H_m': 'first',
                                'd_cm': ['count', 'mean', 'std'],
                                'h_m': 'max'
                            }).round(2)

                            arvore_stats.columns = ['DAP', 'Altura', 'N_Secoes', 'Diam_Medio', 'Diam_Std', 'Alt_Cubada']
                            st.dataframe(arvore_stats.head(15), use_container_width=True)
                        except Exception as e:
                            st.error(f"⚠️ Erro na análise por árvore: {e}")
            else:
                st.warning("⚠️ Cubagem existe mas está vazia ou inválida")
        else:
            st.error("❌ **Dados de Cubagem:** Não carregados")

        # === DADOS DE ÁREA (ARQUIVOS ESPACIAIS) ===
        arquivos_espaciais_encontrados = False

        # Shapefile
        if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None:
            arquivo_shapefile = st.session_state.arquivo_shapefile
            arquivos_espaciais_encontrados = True

            st.success("✅ **Shapefile carregado** - Disponível para cálculo preciso de áreas")

            with st.expander("🗺️ Informações do Shapefile"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    nome_arquivo = getattr(arquivo_shapefile, 'name', 'shapefile.zip')
                    st.metric("📁 Arquivo", nome_arquivo.split('.')[-1].upper())
                    st.metric("📄 Nome", nome_arquivo[:20] + "..." if len(nome_arquivo) > 20 else nome_arquivo)
                with col2:
                    try:
                        tamanho_kb = getattr(arquivo_shapefile, 'size', 0) / 1024
                        st.metric("💾 Tamanho", f"{tamanho_kb:.0f} KB")
                    except:
                        st.metric("💾 Tamanho", "N/A")
                    st.metric("📊 Status", "✅ Ativo")
                with col3:
                    st.metric("🎯 Uso", "Cálculo de áreas")
                    st.metric("⚙️ Config", "Método SHP")

                st.info(
                    "🗺️ **Uso recomendado:** Configure na Etapa 0 o método de área como 'Baseado em Shapefile' para cálculos precisos")

        # Coordenadas
        if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None:
            arquivo_coordenadas = st.session_state.arquivo_coordenadas
            arquivos_espaciais_encontrados = True

            st.success("✅ **Coordenadas carregadas** - Disponíveis para análises espaciais")

            with st.expander("📍 Informações das Coordenadas"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    nome_arquivo = getattr(arquivo_coordenadas, 'name', 'coordenadas.csv')
                    st.metric("📁 Arquivo", nome_arquivo.split('.')[-1].upper())
                    st.metric("📄 Nome", nome_arquivo[:20] + "..." if len(nome_arquivo) > 20 else nome_arquivo)
                with col2:
                    try:
                        tamanho_kb = getattr(arquivo_coordenadas, 'size', 0) / 1024
                        st.metric("💾 Tamanho", f"{tamanho_kb:.0f} KB")
                    except:
                        st.metric("💾 Tamanho", "N/A")
                    st.metric("📊 Status", "✅ Ativo")
                with col3:
                    st.metric("🎯 Uso", "Análises espaciais")
                    st.metric("🗺️ Tipo", "Coordenadas XY")

                # Tentar mostrar informações das coordenadas
                try:
                    df_coordenadas = carregar_arquivo_seguro(arquivo_coordenadas, "coordenadas")
                    if df_coordenadas is not None and len(df_coordenadas) > 0:
                        st.info(f"📍 **{len(df_coordenadas)} coordenadas** carregadas e prontas para uso")

                        # Preview opcional
                        if st.checkbox("👀 Preview das Coordenadas", key="preview_coord_expander"):
                            st.subheader("📊 Dados das Coordenadas")
                            st.dataframe(df_coordenadas.head(), use_container_width=True)

                            # Estatísticas básicas se houver colunas X, Y
                            try:
                                if 'X' in df_coordenadas.columns and 'Y' in df_coordenadas.columns:
                                    col_coord1, col_coord2 = st.columns(2)
                                    with col_coord1:
                                        x_min, x_max = df_coordenadas['X'].min(), df_coordenadas['X'].max()
                                        st.metric("🌐 X Min-Max", f"{x_min:.0f} - {x_max:.0f}")
                                    with col_coord2:
                                        y_min, y_max = df_coordenadas['Y'].min(), df_coordenadas['Y'].max()
                                        st.metric("🌐 Y Min-Max", f"{y_min:.0f} - {y_max:.0f}")
                            except:
                                pass
                except Exception:
                    st.warning("⚠️ Erro ao carregar preview das coordenadas")

        if not arquivos_espaciais_encontrados:
            st.info("📁 **Arquivos de Área:** Nenhum carregado (opcional)")

        # === DADOS LIDAR ===
        dados_lidar_encontrados = False

        # Arquivo LAS/LAZ
        if hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None:
            arquivo_las = st.session_state.arquivo_las
            dados_lidar_encontrados = True

            st.success("✅ **Arquivo LAS/LAZ carregado** - Pronto para processamento")

            with st.expander("🛩️ Informações do Arquivo LAS/LAZ"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    nome_arquivo = getattr(arquivo_las, 'name', 'arquivo.las')
                    st.metric("📁 Tipo", nome_arquivo.split('.')[-1].upper())
                    st.metric("📄 Nome", nome_arquivo[:15] + "..." if len(nome_arquivo) > 15 else nome_arquivo)
                with col2:
                    try:
                        tamanho_mb = getattr(arquivo_las, 'size', 0) / (1024 * 1024)
                        st.metric("💾 Tamanho", f"{tamanho_mb:.1f} MB")
                    except:
                        st.metric("💾 Tamanho", "N/A")
                    st.metric("🎯 Uso", "Processamento LiDAR")
                with col3:
                    processado = hasattr(st.session_state,
                                         'dados_lidar_las') and st.session_state.dados_lidar_las is not None
                    st.metric("📊 Status", "✅ Processado" if processado else "⏳ Pendente")
                    st.metric("🔄 Ação", "Concluído" if processado else "Processar")
                with col4:
                    st.metric("📍 Destino", "Análise LiDAR")
                    st.metric("⚙️ Método", "Automático")

                if not processado:
                    st.info(
                        "🚀 **Próximo passo:** Acesse a página 'Análise LiDAR' para processar o arquivo e extrair métricas")

        # Dados LAS processados
        if hasattr(st.session_state, 'dados_lidar_las') and st.session_state.dados_lidar_las is not None:
            dados_las = st.session_state.dados_lidar_las
            dados_lidar_encontrados = True

            if 'df_metricas' in dados_las:
                df_metricas = dados_las['df_metricas']
                st.success(f"✅ **Dados LAS processados:** {len(df_metricas)} parcelas com métricas")

                with st.expander("📊 Estatísticas do Processamento LAS"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📍 Parcelas", len(df_metricas))
                        try:
                            altura_media = df_metricas['altura_media'].mean()
                            st.metric("📏 Alt. Média", f"{altura_media:.1f} m")
                        except:
                            st.metric("📏 Alt. Média", "N/A")
                    with col2:
                        try:
                            pontos_total = df_metricas['n_pontos'].sum()
                            st.metric("🎯 Total Pontos", f"{pontos_total:,}")
                            densidade_media = df_metricas['densidade'].mean()
                            st.metric("📊 Densidade", f"{densidade_media:.1f} pts/m²")
                        except:
                            st.metric("🎯 Total Pontos", "N/A")
                            st.metric("📊 Densidade", "N/A")
                    with col3:
                        try:
                            cobertura_media = df_metricas['cobertura'].mean()
                            st.metric("🌳 Cobertura", f"{cobertura_media:.1f}%")
                            if 'altura_max' in df_metricas.columns:
                                altura_max = df_metricas['altura_max'].max()
                                st.metric("🔝 Alt. Máxima", f"{altura_max:.1f} m")
                        except:
                            st.metric("🌳 Cobertura", "N/A")
                            st.metric("🔝 Alt. Máxima", "N/A")
                    with col4:
                        try:
                            if 'altura_p95' in df_metricas.columns:
                                altura_p95 = df_metricas['altura_p95'].mean()
                                st.metric("📈 Alt. P95", f"{altura_p95:.1f} m")
                            if 'biomassa' in df_metricas.columns:
                                biomassa_total = df_metricas['biomassa'].sum()
                                st.metric("🌿 Biomassa", f"{biomassa_total:.0f} kg")
                        except:
                            st.metric("📈 Alt. P95", "N/A")
                            st.metric("🌿 Biomassa", "N/A")

                    # Análise de qualidade dos dados LAS
                    try:
                        if 'densidade' in df_metricas.columns:
                            densidade_min = df_metricas['densidade'].min()
                            densidade_media = df_metricas['densidade'].mean()

                            if densidade_min > 4:
                                st.success(
                                    f"🎯 **Excelente densidade:** mínimo {densidade_min:.1f} pts/m², média {densidade_media:.1f} pts/m²")
                            elif densidade_min > 2:
                                st.info(
                                    f"👍 **Boa densidade:** mínimo {densidade_min:.1f} pts/m², média {densidade_media:.1f} pts/m²")
                            else:
                                st.warning(
                                    f"⚠️ **Densidade baixa:** mínimo {densidade_min:.1f} pts/m², média {densidade_media:.1f} pts/m²")
                    except:
                        pass

                    # Preview das métricas
                    st.subheader("👀 Preview das Métricas LAS")
                    if len(df_metricas) > 0:
                        st.dataframe(df_metricas.head(10), use_container_width=True)
                    else:
                        st.warning("⚠️ Nenhuma métrica para exibir")

        # Métricas LiDAR pré-processadas
        elif hasattr(st.session_state,
                     'arquivo_metricas_lidar') and st.session_state.arquivo_metricas_lidar is not None:
            arquivo_metricas = st.session_state.arquivo_metricas_lidar
            dados_lidar_encontrados = True

            st.success("✅ **Métricas LiDAR carregadas** - Dados pré-processados disponíveis")

            with st.expander("📊 Informações das Métricas LiDAR"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    nome_arquivo = getattr(arquivo_metricas, 'name', 'metricas.csv')
                    st.metric("📁 Tipo", nome_arquivo.split('.')[-1].upper())
                    st.metric("📄 Arquivo", nome_arquivo[:20] + "..." if len(nome_arquivo) > 20 else nome_arquivo)
                with col2:
                    integrado = hasattr(st.session_state, 'dados_lidar') and st.session_state.dados_lidar is not None
                    st.metric("📊 Status", "✅ Integrado" if integrado else "⏳ Pendente")
                    st.metric("🎯 Origem", "Pré-processado")
                with col3:
                    st.metric("🔄 Próximo", "Integração" if not integrado else "Concluído")
                    st.metric("📍 Destino", "Análise LiDAR")

                if not integrado:
                    st.info(
                        "🚀 **Próximo passo:** Acesse a página 'Análise LiDAR' para integrar com os dados de inventário")

        # Dados LiDAR integrados
        if hasattr(st.session_state, 'dados_lidar') and st.session_state.dados_lidar is not None:
            dados_lidar = st.session_state.dados_lidar
            dados_lidar_encontrados = True

            st.success("✅ **LiDAR integrado com inventário** - Análise comparativa disponível")

            with st.expander("🔗 Estatísticas da Integração LiDAR"):
                if 'stats_comparacao' in dados_lidar and dados_lidar['stats_comparacao'] is not None:
                    stats = dados_lidar['stats_comparacao']

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        correlacao = stats.get('correlacao', 0)
                        st.metric("📊 Correlação", f"{correlacao:.3f}")
                        r2 = stats.get('r2', 0)
                        st.metric("📈 R²", f"{r2:.3f}")
                    with col2:
                        rmse = stats.get('rmse', 0)
                        st.metric("📏 RMSE", f"{rmse:.2f} m")
                        bias = stats.get('bias', 0)
                        st.metric("⚖️ Bias", f"{bias:+.2f} m")
                    with col3:
                        n_parcelas = stats.get('n_parcelas', 0)
                        st.metric("📍 Parcelas", n_parcelas)
                        try:
                            mae = stats.get('mae', 0)
                            st.metric("📊 MAE", f"{mae:.2f} m")
                        except:
                            st.metric("📊 MAE", "N/A")
                    with col4:
                        try:
                            mape = stats.get('mape', 0)
                            st.metric("📈 MAPE", f"{mape:.1f}%")
                            altura_campo_media = stats.get('altura_campo_media', 0)
                            st.metric("🌳 Alt. Campo", f"{altura_campo_media:.1f} m")
                        except:
                            st.metric("📈 MAPE", "N/A")
                            st.metric("🌳 Alt. Campo", "N/A")
                    with col5:
                        try:
                            altura_lidar_media = stats.get('altura_lidar_media', 0)
                            st.metric("🛩️ Alt. LiDAR", f"{altura_lidar_media:.1f} m")
                            diferenca_media = stats.get('diferenca_media', 0)
                            st.metric("📏 Diff. Média", f"{diferenca_media:+.2f} m")
                        except:
                            st.metric("🛩️ Alt. LiDAR", "N/A")
                            st.metric("📏 Diff. Média", "N/A")

                    # Interpretar qualidade da correlação
                    st.subheader("🎯 Qualidade da Integração")
                    if correlacao >= 0.8:
                        st.success(f"🎯 **Excelente correlação** entre dados de campo e LiDAR (r = {correlacao:.3f})")
                    elif correlacao >= 0.6:
                        st.info(f"👍 **Boa correlação** entre dados de campo e LiDAR (r = {correlacao:.3f})")
                    elif correlacao >= 0.4:
                        st.warning(f"⚠️ **Correlação moderada** entre dados de campo e LiDAR (r = {correlacao:.3f})")
                    else:
                        st.error(f"❌ **Correlação fraca** entre dados de campo e LiDAR (r = {correlacao:.3f})")

                    # Análise do R²
                    if r2 >= 0.7:
                        st.success(
                            f"📈 **Excelente ajuste:** R² = {r2:.3f} (modelo explica {r2 * 100:.1f}% da variação)")
                    elif r2 >= 0.5:
                        st.info(f"📊 **Bom ajuste:** R² = {r2:.3f} (modelo explica {r2 * 100:.1f}% da variação)")
                    else:
                        st.warning(
                            f"⚠️ **Ajuste moderado:** R² = {r2:.3f} (modelo explica {r2 * 100:.1f}% da variação)")

                # Mostrar alertas se houver
                if 'alertas' in dados_lidar and dados_lidar['alertas']:
                    alertas = dados_lidar['alertas']
                    if len(alertas) > 0:
                        st.subheader("⚠️ Alertas da Integração")
                        for i, alerta in enumerate(alertas[:5], 1):  # Mostrar até 5 alertas
                            st.warning(f"**{i}.** {alerta}")
                        if len(alertas) > 5:
                            st.caption(f"... e mais {len(alertas) - 5} alertas")

        if not dados_lidar_encontrados:
            st.info("🛩️ **Dados LiDAR:** Nenhum carregado (opcional)")

        # === RESUMO FINAL ===
        st.markdown("---")

        with st.expander("🎯 Resumo do Status Geral", expanded=True):
            # Verificar completude
            tem_inventario = hasattr(st.session_state,
                                     'dados_inventario') and st.session_state.dados_inventario is not None
            tem_cubagem = hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None
            dados_basicos_ok = tem_inventario and tem_cubagem

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("📊 Status Principal")
                if dados_basicos_ok:
                    st.success("🎉 **Dados principais completos!**")
                    st.caption("✅ Inventário e Cubagem carregados")
                    st.caption("🚀 Sistema pronto para análises")
                else:
                    st.error("❌ **Dados principais incompletos**")
                    if not tem_inventario:
                        st.caption("❌ Falta: Inventário")
                    if not tem_cubagem:
                        st.caption("❌ Falta: Cubagem")

            with col2:
                st.subheader("📁 Dados Adicionais")
                extras = []
                if arquivos_espaciais_encontrados:
                    extras.append("🗺️ Espaciais")
                if dados_lidar_encontrados:
                    extras.append("🛩️ LiDAR")

                if extras:
                    st.info(f"✨ **{len(extras)} tipo(s) extra(s)**")
                    for extra in extras:
                        st.caption(f"✅ {extra}")
                else:
                    st.warning("📭 **Nenhum dado adicional**")
                    st.caption("💡 Carregue LiDAR ou dados espaciais")

            with col3:
                st.subheader("🚀 Próximos Passos")
                if dados_basicos_ok:
                    try:
                        from config.configuracoes_globais import obter_configuracao_global
                        config_global = obter_configuracao_global()
                        configurado = config_global.get('configurado', False)

                        if not configurado:
                            st.warning("⚙️ **Configure o sistema**")
                            st.caption("📍 Vá para Etapa 0")
                        else:
                            # Verificar etapas executadas
                            hip_ok = hasattr(st.session_state,
                                             'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None
                            vol_ok = hasattr(st.session_state,
                                             'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None
                            inv_ok = hasattr(st.session_state,
                                             'inventario_processado') and st.session_state.inventario_processado is not None

                            if not hip_ok:
                                st.info("🌳 **Execute Etapa 1**")
                                st.caption("📍 Modelos Hipsométricos")
                            elif not vol_ok:
                                st.info("📊 **Execute Etapa 2**")
                                st.caption("📍 Modelos Volumétricos")
                            elif not inv_ok:
                                st.info("📈 **Execute Etapa 3**")
                                st.caption("📍 Inventário Final")
                            else:
                                st.success("✅ **Core completo!**")
                                if dados_lidar_encontrados:
                                    st.caption("🛩️ LiDAR disponível")
                                else:
                                    st.caption("💡 Carregue LiDAR")
                    except:
                        st.warning("⚙️ **Configure primeiro**")
                        st.caption("📍 Etapa 0 obrigatória")
                else:
                    st.error("📁 **Carregue dados**")
                    st.caption("📍 Inventário + Cubagem")

            # Informações de sessão
            st.markdown("---")
            st.markdown("**💾 Informações da Sessão:**")

            arquivos_na_sessao = []
            if tem_inventario:
                arquivos_na_sessao.append("📋 Inventário")
            if tem_cubagem:
                arquivos_na_sessao.append("📏 Cubagem")
            if dados_lidar_encontrados:
                arquivos_na_sessao.append("🛩️ LiDAR")
            if arquivos_espaciais_encontrados:
                arquivos_na_sessao.append("🗺️ Espaciais")

            # Verificar resultados processados
            resultados_na_sessao = []
            if hasattr(st.session_state,
                       'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None:
                resultados_na_sessao.append("🌳 Hipsométricos")
            if hasattr(st.session_state,
                       'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None:
                resultados_na_sessao.append("📊 Volumétricos")
            if hasattr(st.session_state,
                       'inventario_processado') and st.session_state.inventario_processado is not None:
                resultados_na_sessao.append("📈 Inventário Final")

            if arquivos_na_sessao:
                st.success("**Arquivos persistidos:** " + " • ".join(arquivos_na_sessao))

            if resultados_na_sessao:
                st.info("**Resultados salvos:** " + " • ".join(resultados_na_sessao))

            if not arquivos_na_sessao and not resultados_na_sessao:
                st.warning("📭 Nenhum dado na sessão")

            st.caption("💡 **Dica:** Dados permanecem ao navegar entre páginas, mas são perdidos ao fechar o navegador")

    except Exception as e:
        st.error(f"❌ Erro ao mostrar preview dos dados: {e}")
        with st.expander("🔍 Detalhes do erro"):
            st.code(traceback.format_exc())


def mostrar_preview_dados_lidar():
    """Mostra preview específico dos dados LiDAR - VERSÃO ORIGINAL"""
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


def main():
    """Função principal da aplicação - VERSÃO COMPLETA COM LAS/LAZ E PREVIEW EXPANDIDO"""
    try:
        # Inicializar configurações globais
        inicializar_configuracoes_globais()

        # Criar cabeçalho
        criar_cabecalho_greenvista("Página Principal")

        # Criar sidebar com uploads - VERSÃO COMPLETA
        arquivos = criar_sidebar_melhorada()

        st.markdown('''
        O sistema LiDAR do **GreenVista** representa uma solução completa e robusta para integração de dados 
        de sensoriamento remoto com inventários florestais tradicionais. Combina facilidade de uso com
         capacidades técnicas avançadas, oferecendo desde processamento básico até análises estruturais 
         sofisticadas.
         **Ideal para:** Empresas florestais que desejam modernizar seus inventários com tecnologia
          LiDAR sem complexidade técnica excessiva.
            
        '''  )

        with st.expander("🛩️ Saiba mais sobre a análise do sistema de processamento LiDAR do **GreenVista**"):
            st.markdown("""

            O sistema **GreenVista** integra processamento de dados LiDAR para análise florestal, oferecendo duas abordagens principais:
            
            1. **Processamento Direto LAS/LAZ** - Arquivos brutos do sensor
            2. **Integração de Métricas** - CSV/Excel pré-processados
            
            ## 🏗️ Arquitetura do Sistema
            
            ### Componentes Principais
            
            #### 1. **Configuração Central**
            - **Configurações por Espécie**: Eucalipto, Pinus, Nativa
            - **Perfis de Processamento**: Rápido, Balanceado, Preciso, Memória Limitada
            - **Validação de Parâmetros**: Limites automáticos para métricas
            - **Otimização Dinâmica**: Ajuste baseado no tamanho do arquivo
                    
            #### 2. **Processador LAS Integrado** 
            - **Gestão de Memória**: Processamento em chunks otimizado
            - **Validação Automática**: Verificação de estrutura e qualidade
            - **Interface Streamlit**: Feedback em tempo real
            - **Métricas Abrangentes**: 15+ métricas estruturais calculadas
            
            ### Funcionalidades Avançadas
            
            #### **Processamento Inteligente**
            - ✅ **Chunks Adaptativos**: 100K-2M pontos por chunk baseado na memória
            - ✅ **Validação Estrutural**: Verificação de coordenadas, alturas e geometria  
            - ✅ **Otimização de Memória**: Garbage collection automático
            - ✅ **Progress Tracking**: Monitoramento em tempo real
            
            #### **Métricas Calculadas**
            
            #### **Integração com Inventário**
            - 🎯 **Parcelas Georreferenciadas**: Usa coordenadas X,Y quando disponíveis
            - 🎯 **Grid Automático**: Cria malha quando coordenadas não existem
            - 🎯 **Validação Cruzada**: Comparação campo vs LiDAR
            - 🎯 **Calibração de Modelos**: Ajuste hipsométrico com dados LiDAR
            
            ## 🔄 Fluxo de Trabalho
            
            ### Cenário 1: Processamento LAS/LAZ
            ```mermaid
           
                A[Upload Arquivo LAS] --> B{Validar Estrutura}
                B -->|✅ Válido| C[Definir Parcelas]
                B -->|❌ Inválido| D[Erro/Instruções]
                C --> E{Tamanho Arquivo}
                E -->|Grande| F[Processamento Chunks]
                E -->|Pequeno| G[Processamento Direto]
                F --> H[Calcular Métricas]
                G --> H
                H --> I[Integrar com Inventário]
                I --> J[Análise Comparativa]
            ```
            
            ### Cenário 2: Métricas Pré-processadas
            ```mermaid
    
                A[Upload CSV/Excel] --> B[Validar Colunas]
                B --> C[Padronizar Nomes]
                C --> D[Limpar Dados]
                D --> E[Integrar com Inventário]
                E --> F[Comparação Campo-LiDAR]
                F --> G[Gerar Alertas]
            ```
            
            ## 📊 Interface de Usuário
            
            ### Página Principal
            
            #### **Recursos de Interface**
            - 🎨 **Identidade Visual**: Cabeçalho GreenVista consistente
            - 📱 **Layout Responsivo**: Tabs dinâmicas baseadas em dados disponíveis
            - 🔄 **Estado Persistente**: Dados salvos entre sessões
            - 📋 **Feedback Contextual**: Mensagens específicas por situação
            
            #### **Controle de Fluxo Inteligente**
            
            ## 🛡️ Robustez e Confiabilidade
            
            ### Validação Multinível
            
            #### **Nível 1: Arquivo**
            - Formato (.las/.laz)
            - Tamanho (máx 500MB)
            - Estrutura (coordenadas XYZ)
            
            #### **Nível 2: Dados**
            - Número de pontos (máx 50M)
            - Alturas realísticas (0.1-150m)
            - Geometria válida
            
            #### **Nível 3: Métricas**
            - Valores dentro de limites esperados
            - Detecção de outliers (IQR 3×)
            - Consistência entre parcelas
            
            ### Gestão de Erros
            
            ## 🔧 Recursos Técnicos Avançados
            
            ### Otimização de Performance
            
            #### **Processamento em Chunks**
            - **Tamanho Adaptativo**: 100K-2M pontos baseado na memória disponível
            - **Gestão de Memória**: Limpeza automática a cada 3 chunks
            - **Progress Tracking**: Feedback visual em tempo real
            
            #### **Algoritmos Otimizados**
            
            ### Integração Inteligente
            
            #### **Detecção Automática de Parcelas**
            1. **Com Coordenadas**: Parcelas circulares georreferenciadas
            2. **Sem Coordenadas**: Grid estimado baseado na distribuição
            3. **Grid Automático**: Células 20×20m para análise exploratória
            
            #### **Calibração de Modelos**
            
            ## 📈 Análises Disponíveis
            
            ### 1. **Comparação Campo vs LiDAR**
            - Correlação e R²
            - RMSE e bias sistemático
            - Detecção de outliers
            - Gráficos de dispersão e resíduos
            
            ### 2. **Análise Estrutural**
            - Distribuição de alturas por talhão
            - Métricas de complexidade estrutural
            - Índices de diversidade (Shannon)
            - Cobertura e densidade do dossel
            
            ### 3. **Calibração de Modelos**
            - Ajuste de modelos hipsométricos
            - Validação cruzada
            - Comparação pré/pós calibração
            - Métricas de melhoria
            
            ### 4. **Alertas Automáticos**
            - Correlação baixa (<0.6)
            - Outliers excessivos (>10%)
            - Bias sistemático (>2m)
            - Cobertura insuficiente (<30%)
            
            ## 💾 Sistema de Persistência
            
            ### Gerenciamento de Estado
            
            ### Downloads Disponíveis
            - 📊 **CSV/Excel**: Métricas completas
            - 📄 **Relatório MD**: Análise detalhada  
            - 📈 **Métricas JSON**: Validação técnica
            - 🎯 **Outliers CSV**: Parcelas problemáticas
            
            ## 🚀 Vantagens Competitivas
            
            ### ✅ **Facilidade de Uso**
            - Interface intuitiva sem conhecimento técnico
            - Processamento automático com configuração mínima
            - Feedback visual constante
            
            ### ✅ **Flexibilidade**
            - Suporte a LAS/LAZ e métricas pré-processadas
            - Configurações por espécie florestal
            - Integração com qualquer inventário
            
            ### ✅ **Robustez**
            - Validação multinível
            - Gestão inteligente de memória
            - Recuperação de erros graceful
            
            ### ✅ **Completude**
            - 15+ métricas estruturais
            - Análises comparativas automáticas
            - Relatórios prontos para uso
            
            ## 🎯 Casos de Uso Típicos
            
            ### 1. **Validação de Inventário**
            Empresa quer verificar se medições de campo são consistentes com dados LiDAR
            
            ### 2. **Calibração de Modelos**
            Melhorar modelos hipsométricos usando dados LiDAR como referência
            
            ### 3. **Mapeamento de Estrutura**
            Analisar heterogeneidade estrutural em diferentes talhões
            
            ### 4. **Detecção de Problemas**
            Identificar parcelas com medições inconsistentes ou problemáticas
                       
             """)

        # === SEÇÃO PRINCIPAL DA PÁGINA ===
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Status do Sistema",
            "📋 Instruções",
            "⚠️ Alertas",
            "👨🏻‍💻 Quem somos"
        ])

        with tab1:
            mostrar_status_sistema()
            st.markdown("---")
            mostrar_preview_inteligente()
            # === INFORMAÇÕES DE PERSISTÊNCIA ===
            mostrar_info_persistencia()

        with tab2:
            criar_secao_instrucoes()

        with tab3:
            mostrar_alertas_sistema()

        with tab4:
            mostrar_empresa()

        # Navegação rápida
        st.markdown("---")
        criar_navegacao_rapida_botoes()


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


if __name__ == "__main__":
    main()