# Principal.py
"""
Sistema Integrado de Inventário Florestal - GreenVista
Página principal do sistema com upload de dados e navegação
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import warnings

warnings.filterwarnings('ignore')

# Importar componentes do sistema
from ui.sidebar import criar_sidebar_melhorada
from ui.components import (
    configurar_pagina_greenvista,
    criar_cabecalho_greenvista,
    criar_navegacao_rapida_botoes,
    criar_secao_instrucoes,
    mostrar_alertas_sistema
)

# Importar processadores
from utils.arquivo_handler import carregar_arquivo
from config.configuracoes_globais import (
    inicializar_configuracoes_globais,
    obter_configuracao_global,
    aplicar_filtros_configuracao_global
)

# Configurar página
configurar_pagina_greenvista("Página Principal", "🌲")


def processar_dados_inventario(arquivo_inventario):
    """
    Processa e valida dados do inventário florestal

    Args:
        arquivo_inventario: Arquivo de inventário carregado

    Returns:
        DataFrame processado ou None se erro
    """
    try:
        if arquivo_inventario is None:
            return None

        # Carregar arquivo
        with st.spinner("🔄 Carregando dados de inventário..."):
            df_inventario = carregar_arquivo(arquivo_inventario)

        if df_inventario is None:
            st.error("❌ Não foi possível carregar o arquivo de inventário")
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

    Args:
        arquivo_cubagem: Arquivo de cubagem carregado

    Returns:
        DataFrame processado ou None se erro
    """
    try:
        if arquivo_cubagem is None:
            return None

        # Carregar arquivo
        with st.spinner("🔄 Carregando dados de cubagem..."):
            df_cubagem = carregar_arquivo(arquivo_cubagem)

        if df_cubagem is None:
            st.error("❌ Não foi possível carregar o arquivo de cubagem")
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

    Args:
        df_inventario: DataFrame bruto do inventário

    Returns:
        DataFrame limpo
    """
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
    mask_valido = (
            df_limpo['D_cm'].notna() &
            df_limpo['H_m'].notna() &
            df_limpo['talhao'].notna() &
            df_limpo['parcela'].notna() &
            (df_limpo['D_cm'] > 0) &
            (df_limpo['H_m'] > 1.3)  # Altura mínima realística
    )

    df_limpo = df_limpo[mask_valido]

    # Remover outliers extremos
    try:
        # Outliers de DAP (usando IQR)
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

    Args:
        df_cubagem: DataFrame bruto da cubagem

    Returns:
        DataFrame limpo
    """
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

    # Validar consistência (diâmetro da seção <= DAP)
    mask_consistente = df_limpo['d_cm'] <= df_limpo['D_cm'] * 1.2  # Tolerância de 20%
    df_limpo = df_limpo[mask_consistente]

    return df_limpo


def mostrar_colunas_disponiveis(df):
    """Mostra colunas disponíveis no arquivo"""
    st.info("📋 Colunas disponíveis no arquivo:")
    colunas_str = ", ".join(df.columns.tolist())
    st.code(colunas_str)


def mostrar_estatisticas_inventario(df_limpo, df_original):
    """Mostra estatísticas do inventário processado"""
    with st.expander("📊 Estatísticas do Inventário"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Registros Originais", len(df_original))
            st.metric("Registros Válidos", len(df_limpo))

        with col2:
            st.metric("Talhões", df_limpo['talhao'].nunique())
            st.metric("Parcelas", df_limpo.groupby(['talhao', 'parcela']).ngroups)

        with col3:
            st.metric("DAP Médio", f"{df_limpo['D_cm'].mean():.1f} cm")
            st.metric("Altura Média", f"{df_limpo['H_m'].mean():.1f} m")

        with col4:
            st.metric("DAP Min-Max", f"{df_limpo['D_cm'].min():.1f}-{df_limpo['D_cm'].max():.1f} cm")
            st.metric("Altura Min-Max", f"{df_limpo['H_m'].min():.1f}-{df_limpo['H_m'].max():.1f} m")

        # Mostrar preview dos dados
        st.subheader("👀 Preview dos Dados")
        st.dataframe(df_limpo.head(10), use_container_width=True)


def mostrar_estatisticas_cubagem(df_limpo, df_original):
    """Mostra estatísticas da cubagem processada"""
    with st.expander("📊 Estatísticas da Cubagem"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Registros Originais", len(df_original))
            st.metric("Registros Válidos", len(df_limpo))

        with col2:
            st.metric("Árvores Cubadas", df_limpo['arv'].nunique())
            st.metric("Talhões", df_limpo['talhao'].nunique())

        with col3:
            seções_por_arvore = df_limpo.groupby(['talhao', 'arv']).size()
            st.metric("Seções/Árvore", f"{seções_por_arvore.mean():.1f}")
            st.metric("DAP Médio", f"{df_limpo['D_cm'].mean():.1f} cm")

        with col4:
            st.metric("Altura Média", f"{df_limpo['H_m'].mean():.1f} m")
            st.metric("Diâm. Seção Médio", f"{df_limpo['d_cm'].mean():.1f} cm")

        # Mostrar preview dos dados
        st.subheader("👀 Preview dos Dados")
        st.dataframe(df_limpo.head(10), use_container_width=True)


def mostrar_preview_dados_carregados():
    """Mostra preview dos dados já carregados"""
    if hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
        st.subheader("📋 Dados de Inventário Carregados")

        df_inv = st.session_state.dados_inventario

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Registros", len(df_inv))
        with col2:
            st.metric("Talhões", df_inv['talhao'].nunique())
        with col3:
            st.metric("DAP Médio", f"{df_inv['D_cm'].mean():.1f} cm")
        with col4:
            st.metric("Altura Média", f"{df_inv['H_m'].mean():.1f} m")

        if st.checkbox("👀 Mostrar Preview do Inventário"):
            st.dataframe(df_inv.head(), use_container_width=True)

    if hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None:
        st.subheader("📏 Dados de Cubagem Carregados")

        df_cub = st.session_state.dados_cubagem

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Registros", len(df_cub))
        with col2:
            st.metric("Árvores", df_cub['arv'].nunique())
        with col3:
            st.metric("DAP Médio", f"{df_cub['D_cm'].mean():.1f} cm")
        with col4:
            seções = df_cub.groupby(['talhao', 'arv']).size().mean()
            st.metric("Seções/Árvore", f"{seções:.1f}")

        if st.checkbox("👀 Mostrar Preview da Cubagem"):
            st.dataframe(df_cub.head(), use_container_width=True)


def mostrar_status_sistema():
    """Mostra status geral do sistema"""
    st.subheader("🔧 Status do Sistema")

    # Verificar configuração
    config = obter_configuracao_global()
    configurado = config.get('configurado', False)

    # Verificar dados carregados
    dados_inventario = hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None
    dados_cubagem = hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None

    # Verificar etapas executadas
    hip_executado = hasattr(st.session_state,
                            'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None
    vol_executado = hasattr(st.session_state,
                            'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None
    inv_executado = hasattr(st.session_state,
                            'inventario_processado') and st.session_state.inventario_processado is not None

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if dados_inventario and dados_cubagem:
            st.success("✅ **Dados**\nCarregados")
        elif dados_inventario or dados_cubagem:
            st.warning("⚠️ **Dados**\nIncompletos")
        else:
            st.error("❌ **Dados**\nFaltantes")

    with col2:
        if configurado:
            st.success("✅ **Config**\nOK")
        else:
            st.error("❌ **Config**\nNecessária")

    with col3:
        if hip_executado:
            st.success("✅ **Etapa 1**\nConcluída")
        else:
            st.info("⏳ **Etapa 1**\nPendente")

    with col4:
        if vol_executado:
            st.success("✅ **Etapa 2**\nConcluída")
        else:
            st.info("⏳ **Etapa 2**\nPendente")

    with col5:
        if inv_executado:
            st.success("✅ **Etapa 3**\nConcluída")
        else:
            st.info("⏳ **Etapa 3**\nPendente")

    # Progresso geral
    etapas_completas = sum(
        [dados_inventario and dados_cubagem, configurado, hip_executado, vol_executado, inv_executado])
    progresso = etapas_completas / 5

    st.progress(progresso, text=f"Progresso Geral: {etapas_completas}/5 itens concluídos")

    # Status dos arquivos opcionais
    mostrar_status_arquivos_opcionais()


def mostrar_status_arquivos_opcionais():
    """Mostra status dos arquivos opcionais"""
    st.subheader("📁 Arquivos Opcionais")

    col1, col2 = st.columns(2)

    with col1:
        # Shapefile
        if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None:
            st.success("✅ **Shapefile**\nCarregado")
            st.caption(f"📄 {st.session_state.arquivo_shapefile.name}")
            st.info("🗺️ Método 'Upload shapefile' disponível nas configurações")
        else:
            st.warning("⚠️ **Shapefile**\nNão carregado")
            st.caption("Upload na sidebar para habilitar método avançado de área")

    with col2:
        # Coordenadas
        if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None:
            st.success("✅ **Coordenadas**\nCarregadas")
            st.caption(f"📄 {st.session_state.arquivo_coordenadas.name}")
            st.info("📍 Método 'Coordenadas das parcelas' disponível nas configurações")
        else:
            st.warning("⚠️ **Coordenadas**\nNão carregadas")
            st.caption("Upload na sidebar para habilitar método avançado de área")


def mostrar_proximos_passos():
    """Mostra próximos passos recomendados"""
    st.subheader("🚀 Próximos Passos")

    # Verificar estado atual
    dados_ok = (hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None and
                hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None)

    config = obter_configuracao_global()
    configurado = config.get('configurado', False)

    if not dados_ok:
        st.info("1️⃣ **Carregue os dados** - Upload dos arquivos de inventário e cubagem na sidebar")
    elif not configurado:
        st.info("2️⃣ **Configure o sistema** - Defina filtros e parâmetros na Etapa 0")
        if st.button("⚙️ Ir para Configurações", type="primary"):
            st.switch_page("pages/0_⚙️_Configurações.py")
    else:
        st.success("✅ **Sistema pronto!** Execute as etapas de análise:")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🌳 Etapa 1\nModelos Hipsométricos", use_container_width=True):
                st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")

        with col2:
            if st.button("📊 Etapa 2\nModelos Volumétricos", use_container_width=True):
                st.switch_page("pages/2_📊_Modelos_Volumétricos.py")

        with col3:
            if st.button("📈 Etapa 3\nInventário Final", use_container_width=True):
                st.switch_page("pages/3_📈_Inventário_Florestal.py")


def main():
    """Função principal da aplicação"""
    # Inicializar configurações globais
    inicializar_configuracoes_globais()

    # Criar cabeçalho
    criar_cabecalho_greenvista("Página Principal")

    # Criar sidebar com uploads
    arquivos = criar_sidebar_melhorada()

    # Processar arquivos se carregados
    if arquivos['inventario'] is not None:
        dados_inventario = processar_dados_inventario(arquivos['inventario'])
        if dados_inventario is not None:
            st.session_state.dados_inventario = dados_inventario

    if arquivos['cubagem'] is not None:
        dados_cubagem = processar_dados_cubagem(arquivos['cubagem'])
        if dados_cubagem is not None:
            st.session_state.dados_cubagem = dados_cubagem

    # Seção principal da página
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Status do Sistema",
        "👀 Preview dos Dados",
        "📋 Instruções",
        "⚠️ Alertas"
    ])

    with tab1:
        mostrar_status_sistema()
        st.markdown("---")
        mostrar_proximos_passos()

    with tab2:
        mostrar_preview_dados_carregados()

    with tab3:
        criar_secao_instrucoes()

    with tab4:
        mostrar_alertas_sistema()

    # Navegação rápida
    st.markdown("---")
    criar_navegacao_rapida_botoes()


if __name__ == "__main__":
    main()