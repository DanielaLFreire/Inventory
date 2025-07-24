# ui/sidebar.py
'''
Interface da barra lateral para upload de arquivos - Versão melhorada com status das etapas
'''

import streamlit as st


def criar_sidebar():
    '''
    Cria a interface da barra lateral com uploads e status das etapas

    Returns:
        dict: Dicionário com os arquivos carregados
    '''
    st.sidebar.header("📁 Upload de Dados")

    # Upload do arquivo de inventário
    arquivo_inventario = st.sidebar.file_uploader(
        "📋 Arquivo de Inventário",
        type=['csv', 'xlsx', 'xls'],
        help="Dados de parcelas (D_cm, H_m, talhao, parcela, cod, idade_anos)"
    )

    # Upload do arquivo de cubagem
    arquivo_cubagem = st.sidebar.file_uploader(
        "📏 Arquivo de Cubagem",
        type=['csv', 'xlsx', 'xls'],
        help="Medições detalhadas (arv, talhao, d_cm, h_m, D_cm, H_m)"
    )

    # Upload opcional de shapefile para áreas
    arquivo_shapefile = st.sidebar.file_uploader(
        "🗺️ Shapefile Áreas (Opcional)",
        type=['shp', 'zip'],
        help="Arquivo shapefile com áreas dos talhões"
    )

    # Upload opcional de coordenadas
    arquivo_coordenadas = st.sidebar.file_uploader(
        "📍 Coordenadas Parcelas (Opcional)",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo com coordenadas X,Y das parcelas"
    )

    arquivos = {
        'inventario': arquivo_inventario,
        'cubagem': arquivo_cubagem,
        'shapefile': arquivo_shapefile,
        'coordenadas': arquivo_coordenadas
    }

    # Mostrar status dos arquivos
    mostrar_status_arquivos(arquivos)

    # NOVO: Mostrar progresso das etapas na sidebar
    mostrar_progresso_etapas_sidebar()

    # Mostrar informações adicionais
    mostrar_informacoes_adicionais()

    return arquivos


def mostrar_status_arquivos(arquivos):
    '''
    Mostra o status dos arquivos carregados

    Args:
        arquivos: Dict com os arquivos
    '''
    st.sidebar.subheader("📊 Status dos Arquivos")

    # Inventário
    if arquivos['inventario'] is not None:
        st.sidebar.success("✅ Inventário carregado")
        # Mostrar nome do arquivo
        st.sidebar.caption(f"📄 {arquivos['inventario'].name}")
    else:
        st.sidebar.error("❌ Inventário necessário")

    # Cubagem
    if arquivos['cubagem'] is not None:
        st.sidebar.success("✅ Cubagem carregada")
        # Mostrar nome do arquivo
        st.sidebar.caption(f"📄 {arquivos['cubagem'].name}")
    else:
        st.sidebar.error("❌ Cubagem necessária")

    # Opcionais
    if arquivos['shapefile'] is not None:
        st.sidebar.info("📁 Shapefile carregado")
        st.sidebar.caption(f"📄 {arquivos['shapefile'].name}")

    if arquivos['coordenadas'] is not None:
        st.sidebar.info("📁 Coordenadas carregadas")
        st.sidebar.caption(f"📄 {arquivos['coordenadas'].name}")


def mostrar_progresso_etapas_sidebar():
    '''NOVO: Mostra o progresso das etapas na sidebar'''
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Progresso das Etapas")

    # Verificar se session state está inicializado
    if not hasattr(st.session_state, 'etapa1_concluida'):
        st.sidebar.info("💡 Execute a análise para ver o progresso")
        return

    # Etapa 1
    if st.session_state.etapa1_concluida:
        st.sidebar.success("✅ **Etapa 1** - Hipsométricos")
        if hasattr(st.session_state, 'resultados_etapa1') and st.session_state.resultados_etapa1:
            melhor_hip = st.session_state.resultados_etapa1.get('melhor_modelo', 'N/A')
            st.sidebar.caption(f"🏆 Melhor: {melhor_hip}")
    else:
        st.sidebar.info("⏳ **Etapa 1** - Hipsométricos")

    # Etapa 2
    if st.session_state.etapa2_concluida:
        st.sidebar.success("✅ **Etapa 2** - Volumétricos")
        if hasattr(st.session_state, 'resultados_etapa2') and st.session_state.resultados_etapa2:
            melhor_vol = st.session_state.resultados_etapa2.get('melhor_modelo', 'N/A')
            st.sidebar.caption(f"🏆 Melhor: {melhor_vol}")
    else:
        st.sidebar.info("⏳ **Etapa 2** - Volumétricos")

    # Etapa 3
    if st.session_state.etapa3_concluida:
        st.sidebar.success("✅ **Etapa 3** - Inventário")
        if hasattr(st.session_state, 'resultados_etapa3') and st.session_state.resultados_etapa3:
            stats = st.session_state.resultados_etapa3.get('estatisticas_gerais', {})
            vol_medio = stats.get('vol_medio_ha', 0)
            if vol_medio > 0:
                st.sidebar.caption(f"📊 {vol_medio:.1f} m³/ha")
    else:
        st.sidebar.info("⏳ **Etapa 3** - Inventário")

    # Mostrar progresso geral
    etapas_concluidas = sum([
        st.session_state.etapa1_concluida,
        st.session_state.etapa2_concluida,
        st.session_state.etapa3_concluida
    ])

    if etapas_concluidas > 0:
        progresso = etapas_concluidas / 3
        st.sidebar.progress(progresso, text=f"Progresso: {etapas_concluidas}/3 etapas")

        if etapas_concluidas == 3:
            st.sidebar.balloons()  # Celebração quando completar todas as etapas


def mostrar_informacoes_adicionais():
    '''Mostra informações adicionais na sidebar'''
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Informações")

    st.sidebar.markdown('''
    **Formatos aceitos:**
    - CSV (separadores: ; , tab)
    - Excel (.xlsx, .xls, .xlsb)
    - Shapefile (.shp ou .zip)

    **Tamanho máximo:**
    - 200MB por arquivo

    **Encoding:**
    - UTF-8 recomendado
    ''')

    # NOVO: Mostrar informações sobre persistência
    if hasattr(st.session_state, 'analise_executada') and st.session_state.analise_executada:
        st.sidebar.markdown("---")
        st.sidebar.info('''
        💾 **Resultados Salvos**

        Os resultados das 3 etapas estão salvos e permanecerão disponíveis até você:
        - Carregar novos arquivos
        - Reexecutar a análise
        - Fechar o navegador
        ''')

    # NOVO: Botões de ação rápida
    if hasattr(st.session_state, 'analise_executada') and st.session_state.analise_executada:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Ações Rápidas")

        if st.sidebar.button("🔄 Limpar Resultados", use_container_width=True):
            # Limpar apenas os resultados, mantendo os dados carregados
            st.session_state.analise_executada = False
            st.session_state.resultados_analise = None
            st.session_state.etapa1_concluida = False
            st.session_state.etapa2_concluida = False
            st.session_state.etapa3_concluida = False
            st.session_state.resultados_etapa1 = None
            st.session_state.resultados_etapa2 = None
            st.session_state.resultados_etapa3 = None
            st.sidebar.success("✅ Resultados limpos!")
            st.rerun()

        # Botão para download rápido (se resultados disponíveis)
        if (hasattr(st.session_state, 'resultados_analise') and
                st.session_state.resultados_analise is not None):

            # Preparar dados para download rápido
            resumo_parcelas = st.session_state.resultados_analise.get('resumo_parcelas')
            if resumo_parcelas is not None:
                csv_dados = resumo_parcelas.to_csv(index=False)
                st.sidebar.download_button(
                    label="📥 Download Resumo",
                    data=csv_dados,
                    file_name="resumo_inventario.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download rápido do resumo por parcelas"
                )

    st.sidebar.markdown("---")
    st.sidebar.markdown('''
    **🚀 Dica:**
    Comece carregando os dois arquivos obrigatórios (Inventário + Cubagem) para ativar o sistema.

    **💡 Nova funcionalidade:**
    Os resultados de cada etapa ficam sempre disponíveis após a execução!
    ''')


def mostrar_resumo_sidebar():
    '''NOVO: Mostra resumo dos resultados na sidebar'''
    if not (hasattr(st.session_state, 'analise_executada') and st.session_state.analise_executada):
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Resumo Rápido")

    try:
        resultados = st.session_state.resultados_analise
        if resultados is None:
            return

        stats = resultados.get('estatisticas_gerais', {})

        # Métricas principais na sidebar
        st.sidebar.metric(
            "🌲 Produtividade",
            f"{stats.get('vol_medio_ha', 0):.1f} m³/ha"
        )

        st.sidebar.metric(
            "📏 Área Total",
            f"{stats.get('area_total_ha', 0):.1f} ha"
        )

        st.sidebar.metric(
            "🌳 Estoque",
            f"{stats.get('estoque_total_m3', 0):,.0f} m³"
        )

        # Classificação rápida
        vol_medio = stats.get('vol_medio_ha', 0)
        if vol_medio >= 150:
            st.sidebar.success("🌟 Alta Produtividade")
        elif vol_medio >= 100:
            st.sidebar.info("📊 Média Produtividade")
        else:
            st.sidebar.warning("📈 Baixa Produtividade")

    except Exception as e:
        st.sidebar.error(f"Erro no resumo: {e}")


def criar_sidebar_melhorada():
    '''
    Versão melhorada da sidebar com todas as funcionalidades

    Returns:
        dict: Dicionário com os arquivos carregados
    '''
    # Criar sidebar principal
    arquivos = criar_sidebar()

    # Adicionar resumo se análise foi executada
    mostrar_resumo_sidebar()

    return arquivos