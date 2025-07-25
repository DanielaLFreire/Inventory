# pages/0_⚙️_Configurações.py - VERSÃO CORRIGIDA
"""
Página de Configurações Globais - CORRIGIDA
Centraliza todas as configurações do sistema em um local único
CORREÇÕES:
1. Removida aba de debug desnecessária
2. Removida aba duplicada de parâmetros não-lineares
3. Corrigido problema do shapefile não aparecer na lista
"""

import streamlit as st
from config.configuracoes_globais import (
    inicializar_configuracoes_globais,
    mostrar_configuracoes_globais,
    obter_configuracao_global,
    verificar_configuracao_atualizada
)

st.set_page_config(
    page_title="Configurações Globais",
    page_icon="⚙️",
    layout="wide"
)


def verificar_dados_carregados():
    """Verifica se os dados básicos estão carregados"""
    if not hasattr(st.session_state, 'dados_inventario') or st.session_state.dados_inventario is None:
        st.warning("⚠️ Carregue os dados de inventário primeiro na página principal")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Ir para Página Principal", use_container_width=True):
                st.switch_page("Principal.py")

        with col2:
            if st.button("🔄 Atualizar Página", use_container_width=True):
                st.rerun()

        return False

    return True


def mostrar_introducao():
    """Mostra introdução sobre configurações centralizadas"""
    st.title("⚙️ Configurações Globais do Sistema")

    st.markdown("""
    ### 🎯 **Central de Configurações**

    Esta página centraliza **todas** as configurações que afetam o sistema:

    - **🔍 Filtros de Dados**: Aplicados em todas as 3 etapas
    - **📏 Áreas dos Talhões**: Usadas no inventário final  
    - **🌱 Parâmetros Florestais**: Para cálculos de biomassa e carbono
    - **🧮 Configurações de Modelos**: Ajustes avançados dos algoritmos (incluindo parâmetros não-lineares)

    > **💡 Vantagem**: Configure uma vez, use em todas as etapas automaticamente!
    """)

    # Status atual
    config_atual = obter_configuracao_global()
    if config_atual.get('configurado', False):
        st.success("✅ Sistema já configurado - você pode revisar e ajustar as configurações abaixo")
    else:
        st.info("ℹ️ Sistema ainda não configurado - defina as configurações abaixo antes de executar as análises")


def mostrar_impacto_configuracao():
    """Mostra como as configurações impactam cada etapa"""
    with st.expander("🔗 Como as Configurações Afetam Cada Etapa"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🌳 Etapa 1 - Hipsométricos**

            🔍 **Filtros aplicados:**
            - ✅ Talhões excluídos
            - ✅ Diâmetro mínimo
            - ✅ Códigos excluídos

            🧮 **Configurações de modelo:**
            - ✅ Modelos não-lineares
            - ✅ Máximo iterações
            - ✅ Tolerância ajuste
            - ✅ Parâmetros iniciais (Chapman, Weibull, Mononuclear)
            """)

        with col2:
            st.markdown("""
            **📊 Etapa 2 - Volumétricos**

            🔍 **Filtros aplicados:**
            - ✅ Talhões excluídos (na cubagem)
            - ✅ Diâmetro mínimo
            - ✅ Dados inválidos removidos

            📏 **Processamento:**
            - ✅ Método Smalian
            - ✅ Validação consistência
            """)

        with col3:
            st.markdown("""
            **📈 Etapa 3 - Inventário**

            🔍 **Todos os filtros:**
            - ✅ Aplicação completa

            📏 **Áreas:**
            - ✅ Método selecionado
            - ✅ Parâmetros específicos

            🌱 **Parâmetros florestais:**
            - ✅ Densidade, sobrevivência
            - ✅ Fator forma, densidade madeira
            """)


def mostrar_aviso_impacto():
    """Aviso sobre impacto das mudanças"""
    if verificar_configuracao_atualizada():
        st.warning("""
        ⚠️ **Configurações Alteradas**

        As configurações foram modificadas. Os resultados das análises anteriores 
        podem não refletir as novas configurações.

        **Recomendação**: Execute novamente as etapas para aplicar as novas configurações.
        """)


def verificar_arquivos_opcionais():
    """
    CORREÇÃO 3: Função dedicada para verificar arquivos opcionais de forma mais robusta

    Returns:
        dict: Status dos arquivos opcionais
    """
    status_arquivos = {
        'shapefile_disponivel': False,
        'coordenadas_disponivel': False,
        'shapefile_nome': None,
        'coordenadas_nome': None
    }

    # Verificar shapefile de múltiplas formas
    shapefile_encontrado = False
    shapefile_nome = None

    # 1. Verificar upload atual na sessão
    if hasattr(st.session_state, 'arquivo_shapefile'):
        if st.session_state.arquivo_shapefile is not None:
            shapefile_encontrado = True
            shapefile_nome = st.session_state.arquivo_shapefile.name

    # 2. Verificar se foi carregado via file_uploader (pode estar em outros states)
    for key in st.session_state.keys():
        if 'shapefile' in key.lower() and st.session_state[key] is not None:
            if hasattr(st.session_state[key], 'name'):
                shapefile_encontrado = True
                shapefile_nome = st.session_state[key].name
                break

    # Verificar coordenadas de múltiplas formas
    coordenadas_encontradas = False
    coordenadas_nome = None

    # 1. Verificar upload atual na sessão
    if hasattr(st.session_state, 'arquivo_coordenadas'):
        if st.session_state.arquivo_coordenadas is not None:
            coordenadas_encontradas = True
            coordenadas_nome = st.session_state.arquivo_coordenadas.name

    # 2. Verificar se foi carregado via file_uploader (pode estar em outros states)
    for key in st.session_state.keys():
        if 'coordenadas' in key.lower() and st.session_state[key] is not None:
            if hasattr(st.session_state[key], 'name'):
                coordenadas_encontradas = True
                coordenadas_nome = st.session_state[key].name
                break

    status_arquivos.update({
        'shapefile_disponivel': shapefile_encontrado,
        'coordenadas_disponivel': coordenadas_encontradas,
        'shapefile_nome': shapefile_nome,
        'coordenadas_nome': coordenadas_nome
    })

    return status_arquivos


def mostrar_status_arquivos_opcionais():
    """Mostra status dos arquivos opcionais de forma clara"""
    status = verificar_arquivos_opcionais()

    with st.expander("📁 Status dos Arquivos Opcionais"):
        col1, col2 = st.columns(2)

        with col1:
            if status['shapefile_disponivel']:
                st.success(f"✅ **Shapefile detectado**")
                st.info(f"📄 {status['shapefile_nome']}")
                st.caption("Método 'Upload shapefile' disponível")
            else:
                st.warning("⚠️ **Shapefile não carregado**")
                st.caption("Upload na página principal para habilitar")

        with col2:
            if status['coordenadas_disponivel']:
                st.success(f"✅ **Coordenadas detectadas**")
                st.info(f"📄 {status['coordenadas_nome']}")
                st.caption("Método 'Coordenadas das parcelas' disponível")
            else:
                st.warning("⚠️ **Coordenadas não carregadas**")
                st.caption("Upload na página principal para habilitar")


def mostrar_navegacao_rapida():
    """Navegação rápida para as etapas"""
    st.markdown("---")
    st.subheader("🚀 Navegação Rápida")

    config_atual = obter_configuracao_global()
    configurado = config_atual.get('configurado', False)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🏠 Página Principal", use_container_width=True):
            st.switch_page("Principal.py")

    with col2:
        disabled_hip = not configurado
        if st.button("🌳 Etapa 1: Hipsométricos", use_container_width=True, disabled=disabled_hip):
            st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")
        if disabled_hip:
            st.caption("Configure primeiro")

    with col3:
        disabled_vol = not configurado
        if st.button("📊 Etapa 2: Volumétricos", use_container_width=True, disabled=disabled_vol):
            st.switch_page("pages/2_📊_Modelos_Volumétricos.py")
        if disabled_vol:
            st.caption("Configure primeiro")

    with col4:
        disabled_inv = not configurado
        if st.button("📈 Etapa 3: Inventário", use_container_width=True, disabled=disabled_inv):
            st.switch_page("pages/3_📈_Inventário_Florestal.py")
        if disabled_inv:
            st.caption("Configure primeiro")


def main():
    # Inicializar configurações globais se necessário
    inicializar_configuracoes_globais()

    # Mostrar introdução
    mostrar_introducao()

    # Verificar se dados estão carregados
    if not verificar_dados_carregados():
        return

    # Mostrar como configurações afetam etapas
    mostrar_impacto_configuracao()

    # CORREÇÃO 3: Mostrar status dos arquivos opcionais
    mostrar_status_arquivos_opcionais()

    # Aviso sobre impacto das mudanças
    mostrar_aviso_impacto()

    # Interface principal de configurações
    st.markdown("---")
    configuracoes = mostrar_configuracoes_globais()

    # Navegação rápida
    mostrar_navegacao_rapida()

    # Informações adicionais
    with st.expander("ℹ️ Informações Importantes"):
        st.markdown("""
        ### 📋 **Como Usar as Configurações Centralizadas**

        1. **🔧 Configure uma vez**: Defina todos os parâmetros nesta página
        2. **🚀 Execute as etapas**: As configurações serão aplicadas automaticamente
        3. **✏️ Ajuste conforme necessário**: Volte aqui para modificar configurações
        4. **🔄 Re-execute se necessário**: Mudanças importantes podem requerer nova execução

        ### 🎯 **Configurações Críticas**

        - **Talhões excluídos**: Afeta TODAS as análises
        - **Diâmetro mínimo**: Impacta número de árvores consideradas
        - **Método de área**: Define como calcular estoques por talhão
        - **Modelos não-lineares**: Aumenta tempo de processamento mas pode melhorar precisão
        - **Parâmetros iniciais**: Fundamentais para convergência dos modelos não-lineares

        ### 💾 **Persistência**

        - Configurações ficam salvas durante toda a sessão
        - Use "Exportar" para salvar permanentemente
        - "Resetar Padrão" volta às configurações iniciais

        ### ⚠️ **Avisos**

        - Mudanças nas configurações invalidam resultados anteriores
        - Filtros muito restritivos podem reduzir drasticamente os dados
        - Configurações de área impactam diretamente os estoques calculados
        - Parâmetros inadequados podem impedir convergência dos modelos não-lineares

        ### 📁 **Arquivos Opcionais**

        - **Shapefile**: Carregue na página principal para habilitar método "Upload shapefile"
        - **Coordenadas**: Carregue na página principal para habilitar método "Coordenadas das parcelas"
        - Estes arquivos ficam persistentes na sessão após o upload
        """)


if __name__ == "__main__":
    main()