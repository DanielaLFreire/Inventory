# Principal.py - VERSÃO ATUALIZADA COM CONFIGURAÇÕES CENTRALIZADAS
"""
Sistema Integrado de Inventário Florestal
Hub Central de Navegação com Configurações Centralizadas
"""

import streamlit as st
import pandas as pd
from ui.sidebar import criar_sidebar, mostrar_status_arquivos
from utils.arquivo_handler import carregar_arquivo
from ui.sidebar import criar_sidebar_melhorada

# NOVO: Importar configurações centralizadas
from config.configuracoes_globais import (
    inicializar_configuracoes_globais,
    mostrar_status_configuracao_sidebar,
    obter_configuracao_global
)

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

    # NOVO: Inicializar configurações globais
    inicializar_configuracoes_globais()


def processar_uploads_com_persistencia(arquivos):
    """
    Processa uploads mantendo persistência - VERSÃO CORRIGIDA
    """
    arquivos_processados = False

    # Processar inventário (sempre reprocessar se upload novo)
    if arquivos['inventario'] is not None:
        try:
            df_inventario = carregar_arquivo(arquivos['inventario'])
            if df_inventario is not None:
                st.session_state.dados_inventario = df_inventario
                st.sidebar.success(f"✅ Inventário: {len(df_inventario)} registros")
                arquivos_processados = True
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar inventário: {e}")

    # Processar cubagem (sempre reprocessar se upload novo)
    if arquivos['cubagem'] is not None:
        try:
            df_cubagem = carregar_arquivo(arquivos['cubagem'])
            if df_cubagem is not None:
                st.session_state.dados_cubagem = df_cubagem
                st.sidebar.success(f"✅ Cubagem: {len(df_cubagem)} medições")
                if arquivos_processados:
                    st.session_state.arquivos_carregados = True
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar cubagem: {e}")

    # CORREÇÃO: Verificar se dados ainda existem mesmo sem upload novo
    if (not arquivos_processados and
            hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None and
            hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None):
        # Dados já estão carregados de sessão anterior
        st.session_state.arquivos_carregados = True
        arquivos_processados = True

        # Mostrar status dos dados existentes
        st.sidebar.info(f"📊 Inventário ativo: {len(st.session_state.dados_inventario)} registros")
        st.sidebar.info(f"📏 Cubagem ativa: {len(st.session_state.dados_cubagem)} medições")

    # Arquivos opcionais já são gerenciados pela sidebar melhorada
    # Verificar se estão disponíveis no session_state
    shapefile_disponivel = (
            hasattr(st.session_state, 'arquivo_shapefile') and
            st.session_state.arquivo_shapefile is not None
    )

    coordenadas_disponivel = (
            hasattr(st.session_state, 'arquivo_coordenadas') and
            st.session_state.arquivo_coordenadas is not None
    )

    # Mostrar status dos arquivos persistidos sem reprocessar
    if shapefile_disponivel:
        st.sidebar.info(f"📁 Shapefile ativo: {st.session_state.arquivo_shapefile.name}")

    if coordenadas_disponivel:
        st.sidebar.info(f"📍 Coordenadas ativas: {st.session_state.arquivo_coordenadas.name}")

    return st.session_state.get('arquivos_carregados', False)


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
    """Mostra navegação principal do sistema com status de configuração"""
    st.header("🌲 Sistema de Inventário Florestal")
    st.markdown("### Análise Completa em 4 Etapas")

    # NOVO: Verificar status de configuração
    config_global = obter_configuracao_global()
    configurado = config_global.get('configurado', False)

    # NOVO: Etapa 0 - Configurações
    if not configurado:
        st.warning("⚠️ **Sistema não configurado** - Configure primeiro antes de executar as análises")

    col0, col1, col2, col3 = st.columns(4)

    with col0:
        st.subheader("⚙️ Etapa 0: Configurações")
        st.markdown("""
        **Setup Centralizado**
        - Filtros globais de dados
        - Configuração de áreas
        - Parâmetros florestais
        - Configurações de modelos
        """)

        config_status = "✅ Configurado" if configurado else "⚠️ Pendente"
        st.info(f"Status: {config_status}")

        if st.button("⚙️ Configurar Sistema", use_container_width=True, key="btn_config_main"):
            st.switch_page("pages/0_⚙️_Configurações.py")

    with col1:
        st.subheader("🌳 Etapa 1: Modelos Hipsométricos")
        st.markdown("""
        **Análise Altura-Diâmetro**
        - 7 modelos disponíveis
        - Lineares e não-lineares
        - Seleção automática do melhor
        - Análise de significância
        """)

        disabled_hip = not configurado
        if st.button("🚀 Iniciar Hipsométricos", use_container_width=True,
                     key="btn_hip_main", disabled=disabled_hip):
            st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")

        if disabled_hip:
            st.caption("Configure o sistema primeiro")

    with col2:
        st.subheader("📊 Etapa 2: Modelos Volumétricos")
        st.markdown("""
        **Cubagem e Volume**
        - Método de Smalian
        - 4 modelos volumétricos
        - Coeficientes detalhados
        - Análise de resíduos
        """)

        disabled_vol = not configurado
        if st.button("🚀 Iniciar Volumétricos", use_container_width=True,
                     key="btn_vol_main", disabled=disabled_vol):
            st.switch_page("pages/2_📊_Modelos_Volumétricos.py")

        if disabled_vol:
            st.caption("Configure o sistema primeiro")

    with col3:
        st.subheader("📈 Etapa 3: Inventário Final")
        st.markdown("""
        **Processamento Completo**
        - Aplicação dos melhores modelos
        - Relatórios executivos
        - Análise por talhão
        - Downloads organizados
        """)

        disabled_inv = not configurado
        if st.button("🚀 Processar Inventário", use_container_width=True,
                     key="btn_inv_main", disabled=disabled_inv):
            st.switch_page("pages/3_📈_Inventário_Florestal.py")

        if disabled_inv:
            st.caption("Configure o sistema primeiro")


def mostrar_instrucoes():
    """Mostra instruções quando arquivos não estão carregados"""
    st.header("🌲 Sistema de Inventário Florestal")
    st.subheader("📋 Como Usar o Sistema")

    st.markdown("""
    ### 🎯 ** Fluxo de Trabalho Simplificado**
    1. **📁 Upload**: Carregue inventário e cubagem na barra lateral
    2. **⚙️ Etapa 0**: Configure uma vez todas as opções do sistema
    3. **🌳 Etapa 1**: Ajuste modelos hipsométricos (usa config automática)
    4. **📊 Etapa 2**: Ajuste modelos volumétricos (usa config automática)
    5. **📈 Etapa 3**: Processe o inventário completo (usa config automática)
    """)

    # NOVO: Destacar benefícios das configurações centralizadas
    st.info("""
    💡 **Vantagens das Configurações Centralizadas:**

    - **Consistência**: Mesmos filtros aplicados em todas as etapas
    - **Simplicidade**: Configure uma vez, use em todas as etapas
    - **Transparência**: Sempre saiba quais configurações estão sendo aplicadas
    - **Rastreabilidade**: Configurações salvas nos relatórios
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


def mostrar_alerta_configuracao():
    """NOVO: Mostra alerta sobre importância da configuração"""
    config_global = obter_configuracao_global()

    if not config_global.get('configurado', False):
        st.warning("""
        ⚠️ **Sistema Não Configurado**

        Configure o sistema na **Etapa 0** antes de executar as análises. 
        As configurações definidas lá serão aplicadas automaticamente em todas as etapas.

        **Configurações importantes:**
        - Filtros de dados (talhões a excluir, diâmetro mínimo)
        - Método de cálculo de áreas
        - Parâmetros florestais
        - Configurações de modelos
        """)

        if st.button("⚙️ Ir para Configurações Agora", type="primary", use_container_width=True):
            st.switch_page("pages/0_⚙️_Configurações.py")


def debug_arquivos_session_state():
    """NOVA: Função de debug para verificar arquivos no session_state"""
    with st.expander("🔍 Debug - Arquivos no Session State"):
        st.write("**Status dos Arquivos Opcionais:**")

        # Verificar shapefile
        if hasattr(st.session_state, 'arquivo_shapefile'):
            shapefile = st.session_state.arquivo_shapefile
            if shapefile is not None:
                st.success(f"✅ Shapefile: {shapefile.name}")
                st.write(f"   - Tipo: {type(shapefile)}")
                st.write(f"   - Tamanho: {shapefile.size} bytes")
            else:
                st.info("ℹ️ Shapefile: None")
        else:
            st.warning("⚠️ Atributo 'arquivo_shapefile' não existe")

        # Verificar coordenadas
        if hasattr(st.session_state, 'arquivo_coordenadas'):
            coordenadas = st.session_state.arquivo_coordenadas
            if coordenadas is not None:
                st.success(f"✅ Coordenadas: {coordenadas.name}")
                st.write(f"   - Tipo: {type(coordenadas)}")
                st.write(f"   - Tamanho: {coordenadas.size} bytes")
            else:
                st.info("ℹ️ Coordenadas: None")
        else:
            st.warning("⚠️ Atributo 'arquivo_coordenadas' não existe")

        # Mostrar todos os atributos relacionados a arquivo
        st.write("**Todos os atributos 'arquivo_*':**")
        attrs_arquivo = [k for k in st.session_state.keys() if 'arquivo' in k.lower()]
        for attr in attrs_arquivo:
            value = getattr(st.session_state, attr, None)
            if value is not None and hasattr(value, 'name'):
                st.write(f"✅ {attr}: {value.name}")
            else:
                st.write(f"❌ {attr}: {value}")


def main_corrigido():
    # Inicializar sistema
    inicializar_session_state()

    # Usar sidebar com verificação
    arquivos = criar_sidebar_melhorada()

    # Usar processamento com persistência
    arquivos_ok = processar_uploads_com_persistencia(arquivos)

    # Resto do código continua igual...
    if arquivos_ok:
        # Mostrar progresso
        mostrar_progresso_sistema()

        # Preview dos dados na sidebar
        if st.sidebar.checkbox("👀 Preview Dados", key="checkbox_preview_main"):
            with st.sidebar.expander("📊 Inventário"):
                st.dataframe(st.session_state.dados_inventario.head(3))
            with st.sidebar.expander("📏 Cubagem"):
                st.dataframe(st.session_state.dados_cubagem.head(3))

        # Alerta sobre configuração
        mostrar_alerta_configuracao()

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

        # Mostrar preview configuração atual
        mostrar_preview_configuracao_atual()

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
            - Configurações centralizadas
            - Análise de significância
            - Gráficos interativos
            - Relatórios executivos
            """)


def mostrar_preview_configuracao_atual():
    """NOVO: Mostra preview das configurações atuais"""
    config_global = obter_configuracao_global()

    if config_global.get('configurado', False) and hasattr(st.session_state, 'dados_inventario'):
        with st.expander("⚙️ Preview das Configurações Atuais"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🔍 Filtros Aplicados:**")
                st.write(f"• Diâmetro mínimo: {config_global.get('diametro_min', 4.0)} cm")

                talhoes_excluir = config_global.get('talhoes_excluir', [])
                if talhoes_excluir:
                    st.write(f"• Talhões excluídos: {talhoes_excluir}")
                else:
                    st.write("• Talhões excluídos: Nenhum")

                codigos_excluir = config_global.get('codigos_excluir', [])
                if codigos_excluir:
                    st.write(f"• Códigos excluídos: {codigos_excluir}")
                else:
                    st.write("• Códigos excluídos: Nenhum")

            with col2:
                st.write("**📏 Configurações de Área:**")
                st.write(f"• Método: {config_global.get('metodo_area', 'Simular automaticamente')}")
                st.write(f"• Área da parcela: {config_global.get('area_parcela', 400)} m²")

                st.write("**🧮 Modelos:**")
                st.write(
                    f"• Não-lineares: {'Incluídos' if config_global.get('incluir_nao_lineares', True) else 'Excluídos'}")

            # Calcular impacto dos filtros
            try:
                from config.configuracoes_globais import aplicar_filtros_configuracao_global
                df_original = st.session_state.dados_inventario
                df_filtrado = aplicar_filtros_configuracao_global(df_original)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Registros Originais", len(df_original))
                with col2:
                    st.metric("Após Filtros", len(df_filtrado))
                with col3:
                    percentual = (len(df_filtrado) / len(df_original)) * 100 if len(df_original) > 0 else 0
                    st.metric("% Mantido", f"{percentual:.1f}%")

            except Exception as e:
                st.info("Calcule o impacto executando as configurações")


if __name__ == "__main__":
    main_corrigido()