# ui/sidebar.py
'''
Interface da barra lateral para upload de arquivos
'''

import streamlit as st


def criar_sidebar():
    '''
    Cria a interface da barra lateral com uploads

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

    return {
        'inventario': arquivo_inventario,
        'cubagem': arquivo_cubagem,
        'shapefile': arquivo_shapefile,
        'coordenadas': arquivo_coordenadas
    }


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
    else:
        st.sidebar.error("❌ Inventário necessário")

    # Cubagem
    if arquivos['cubagem'] is not None:
        st.sidebar.success("✅ Cubagem carregada")
    else:
        st.sidebar.error("❌ Cubagem necessária")

    # Opcionais
    if arquivos['shapefile'] is not None:
        st.sidebar.info("📁 Shapefile carregado")

    if arquivos['coordenadas'] is not None:
        st.sidebar.info("📁 Coordenadas carregadas")


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

    st.sidebar.markdown("---")
    st.sidebar.markdown('''
    **🚀 Dica:**
    Comece carregando os dois arquivos obrigatórios (Inventário + Cubagem) para ativar o sistema.
    ''')