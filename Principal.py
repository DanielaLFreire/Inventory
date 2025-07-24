# Principal.py - Aplicação Principal Simplificada
"""
Sistema Integrado de Inventário Florestal
Hub Central de Navegação
"""

import streamlit as st
import pandas as pd
from ui.sidebar import criar_sidebar, mostrar_status_arquivos
from utils.arquivo_handler import carregar_arquivo

# Configuração da página
st.set_page_config(
    page_title="Sistema de Inventário Florestal",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inicializar_session_state():
    """Inicializa estados do sistema"""
    estados_iniciais = {
        'dados_inventario': None,
        'dados_cubagem': None,
        'arquivos_carregados': False,
        'resultados_hipsometricos': None,
        'resultados_volumetricos': None,
        'inventario_processado': None,
        'config_sistema': {}
    }

    for estado, valor_inicial in estados_iniciais.items():
        if estado not in st.session_state:
            st.session_state[estado] = valor_inicial


def processar_uploads(arquivos):
    """Processa uploads de arquivos"""
    arquivos_processados = False

    # Processar inventário
    if arquivos['inventario'] is not None:
        df_inventario = carregar_arquivo(arquivos['inventario'])
        if df_inventario is not None:
            st.session_state.dados_inventario = df_inventario
            st.sidebar.success(f"✅ Inventário: {len(df_inventario)} registros")
            arquivos_processados = True

    # Processar cubagem
    if arquivos['cubagem'] is not None:
        df_cubagem = carregar_arquivo(arquivos['cubagem'])
        if df_cubagem is not None:
            st.session_state.dados_cubagem = df_cubagem
            st.sidebar.success(f"✅ Cubagem: {len(df_cubagem)} medições")
            if arquivos_processados:
                st.session_state.arquivos_carregados = True

    # NOVO: Armazenar arquivos opcionais no session_state
    st.session_state.arquivo_shapefile = arquivos.get('shapefile')
    st.session_state.arquivo_coordenadas = arquivos.get('coordenadas')

    # Mostrar status dos arquivos opcionais
    if arquivos.get('shapefile'):
        st.sidebar.info(f"📁 Shapefile: {arquivos['shapefile'].name}")
    if arquivos.get('coordenadas'):
        st.sidebar.info(f"📍 Coordenadas: {arquivos['coordenadas'].name}")

    return st.session_state.arquivos_carregados


def mostrar_progresso_sistema():
    """Mostra progresso do sistema na sidebar"""
    st.sidebar.subheader("🔄 Progresso das Etapas")

    # Status das etapas
    etapas = [
        ("Hipsométricos", st.session_state.resultados_hipsometricos, "🌳"),
        ("Volumétricos", st.session_state.resultados_volumetricos, "📊"),
        ("Inventário", st.session_state.inventario_processado, "📈")
    ]

    etapas_concluidas = 0

    for nome, resultado, icone in etapas:
        if resultado:
            st.sidebar.success(f"✅ {nome}")
            if isinstance(resultado, dict) and 'melhor_modelo' in resultado:
                st.sidebar.caption(f"🏆 {resultado['melhor_modelo']}")
            etapas_concluidas += 1
        else:
            st.sidebar.info(f"⏳ {nome}")

    # Barra de progresso
    if etapas_concluidas > 0:
        progresso = etapas_concluidas / 3
        st.sidebar.progress(progresso, text=f"Progresso: {etapas_concluidas}/3")


def mostrar_navegacao_principal():
    """Mostra navegação principal do sistema"""
    st.header("🧭 Sistema de Inventário Florestal")
    st.markdown("### Análise Completa em 3 Etapas")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌳 Etapa 1: Modelos Hipsométricos")
        st.markdown("""
        **Análise Altura-Diâmetro**
        - 7 modelos disponíveis
        - Lineares e não-lineares
        - Seleção automática do melhor
        - Análise de significância
        """)

        if st.button("🚀 Iniciar Hipsométricos", use_container_width=True, key="btn_hip_main"):
            st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")

    with col2:
        st.subheader("📊 Etapa 2: Modelos Volumétricos")
        st.markdown("""
        **Cubagem e Volume**
        - Método de Smalian
        - 4 modelos volumétricos
        - Coeficientes detalhados
        - Análise de resíduos
        """)

        if st.button("🚀 Iniciar Volumétricos", use_container_width=True, key="btn_vol_main"):
            st.switch_page("pages/2_📊_Modelos_Volumétricos.py")

    with col3:
        st.subheader("📈 Etapa 3: Inventário Final")
        st.markdown("""
        **Processamento Completo**
        - Aplicação dos melhores modelos
        - Relatórios executivos
        - Análise por talhão
        - Downloads organizados
        """)

        if st.button("🚀 Processar Inventário", use_container_width=True, key="btn_inv_main"):
            st.switch_page("pages/3_📈_Inventário_Florestal.py")


def mostrar_instrucoes():
    """Mostra instruções quando arquivos não estão carregados"""
    st.header("📋 Como Usar o Sistema")

    st.markdown("""
    ### 🎯 **Fluxo de Trabalho**
    1. **📁 Upload**: Carregue inventário e cubagem na barra lateral
    2. **🌳 Etapa 1**: Ajuste modelos hipsométricos
    3. **📊 Etapa 2**: Ajuste modelos volumétricos  
    4. **📈 Etapa 3**: Processe o inventário completo
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Arquivo de Inventário")
        st.markdown("""
        **Colunas obrigatórias:**
        - `D_cm`: Diâmetro (cm)
        - `H_m`: Altura (m) 
        - `talhao`: ID do talhão
        - `parcela`: ID da parcela
        - `cod`: Código da árvore (D/N/C/I)

        **Opcional:**
        - `idade_anos`: Idade do povoamento
        """)

        # Exemplo
        exemplo_inv = pd.DataFrame({
            'talhao': [1, 1, 2, 2],
            'parcela': [1, 1, 1, 2],
            'D_cm': [15.2, 18.5, 20.1, 16.8],
            'H_m': [18.5, 22.1, 24.3, 19.8],
            'cod': ['N', 'D', 'D', 'N']
        })
        st.dataframe(exemplo_inv, hide_index=True)

    with col2:
        st.subheader("📏 Arquivo de Cubagem")
        st.markdown("""
        **Colunas obrigatórias:**
        - `arv`: ID da árvore
        - `talhao`: ID do talhão
        - `d_cm`: Diâmetro da seção (cm)
        - `h_m`: Altura da seção (m)
        - `D_cm`: DAP da árvore (cm)
        - `H_m`: Altura total (m)
        """)

        # Exemplo
        exemplo_cub = pd.DataFrame({
            'arv': [1, 1, 2, 2],
            'talhao': [1, 1, 1, 1],
            'd_cm': [0, 15.2, 0, 18.5],
            'h_m': [0.1, 2.0, 0.1, 2.0],
            'D_cm': [15.2, 15.2, 18.5, 18.5],
            'H_m': [18.5, 18.5, 22.1, 22.1]
        })
        st.dataframe(exemplo_cub, hide_index=True)


def main():
    # Inicializar sistema
    inicializar_session_state()

    # Sidebar com uploads
    arquivos = criar_sidebar()
    mostrar_status_arquivos(arquivos)

    # Processar uploads
    arquivos_ok = processar_uploads(arquivos)

    if arquivos_ok:
        # Mostrar progresso
        mostrar_progresso_sistema()

        # Preview dos dados na sidebar
        if st.sidebar.checkbox("👀 Preview Dados", key="checkbox_preview_main"):
            with st.sidebar.expander("📊 Inventário"):
                st.dataframe(st.session_state.dados_inventario.head(3))
            with st.sidebar.expander("📏 Cubagem"):
                st.dataframe(st.session_state.dados_cubagem.head(3))

        # Navegação principal
        mostrar_navegacao_principal()

        # Resumo dos dados
        st.subheader("📊 Resumo dos Dados Carregados")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Registros Inventário", len(st.session_state.dados_inventario))
        with col2:
            st.metric("Registros Cubagem", len(st.session_state.dados_cubagem))
        with col3:
            st.metric("Talhões", st.session_state.dados_inventario['talhao'].nunique())
        with col4:
            st.metric("Parcelas", st.session_state.dados_inventario['parcela'].nunique())

    else:
        # Instruções de uso
        mostrar_instrucoes()

        # Características do sistema
        st.subheader("⭐ Características do Sistema")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🌳 Modelos Hipsométricos:**
            - Curtis, Campos, Henri
            - Prodan (com idade)
            - Chapman, Weibull, Mononuclear
            """)

        with col2:
            st.markdown("""
            **📊 Modelos Volumétricos:**
            - Schumacher-Hall
            - G1, G2, G3 (Goulding)
            - Cubagem por Smalian
            """)

        with col3:
            st.markdown("""
            **📈 Funcionalidades:**
            - Análise de significância
            - Gráficos interativos
            - Relatórios executivos
            - Downloads organizados
            """)


if __name__ == "__main__":
    main()