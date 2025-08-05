# pages/0_⚙️_Configurações.py - VERSÃO LIMPA E OBJETIVA
"""
Página de Configurações Globais - VERSÃO LIMPA
Centraliza todas as configurações do sistema em um local único
"""

import streamlit as st
from config.configuracoes_globais import (
    inicializar_configuracoes_globais,
    mostrar_configuracoes_globais,
    obter_configuracao_global,
    verificar_configuracao_atualizada
)

# Importar componentes visuais padronizados
from ui.components import (
    configurar_pagina_greenvista,
    criar_cabecalho_greenvista,
    criar_navegacao_rapida_botoes
)
from ui.sidebar import criar_sidebar_melhorada

# Configurar página com identidade visual
configurar_pagina_greenvista("Configurações Globais", "⚙️")


def verificar_dados_carregados():
    """Verifica se os dados básicos estão carregados"""
    if not hasattr(st.session_state, 'dados_inventario') or st.session_state.dados_inventario is None:
        st.warning("⚠️ Carregue os dados de inventário primeiro na página principal")

        if st.button("🏠 Ir para Página Principal", use_container_width=True):
            st.switch_page("Principal.py")

        return False

    return True


def mostrar_introducao():
    """Mostra introdução sobre configurações centralizadas"""
    st.markdown("""
    ### 🎯 **Central de Configurações**

    Esta página centraliza **todas** as configurações que afetam o sistema:

    - **🔍 Filtros de Dados**: Aplicados em todas as 3 etapas
    - **📏 Áreas dos Talhões**: Usadas no inventário final  
    - **🌱 Parâmetros Florestais**: Para cálculos de biomassa e carbono
    - **🧮 Configurações de Modelos**: Ajustes avançados dos algoritmos

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
    Função dedicada para verificar arquivos opcionais de forma mais robusta

    Returns:
        dict: Status dos arquivos opcionais
    """
    status_arquivos = {
        'shapefile_disponivel': False,
        'coordenadas_disponivel': False,
        'shapefile_nome': None,
        'coordenadas_nome': None
    }

    # Verificar shapefile
    if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None:
        status_arquivos['shapefile_disponivel'] = True
        status_arquivos['shapefile_nome'] = st.session_state.arquivo_shapefile.name

    # Verificar coordenadas
    if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None:
        status_arquivos['coordenadas_disponivel'] = True
        status_arquivos['coordenadas_nome'] = st.session_state.arquivo_coordenadas.name

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


def main():
    # Inicializar configurações globais
    inicializar_configuracoes_globais()

    # Criar cabeçalho padronizado
    criar_cabecalho_greenvista("Configurações Globais")

    # Criar sidebar padronizada
    criar_sidebar_melhorada()

    # Mostrar introdução
    mostrar_introducao()

    # Verificar se dados estão carregados
    if not verificar_dados_carregados():
        return

    # Mostrar como configurações afetam etapas
    mostrar_impacto_configuracao()

    # Mostrar status dos arquivos opcionais
    mostrar_status_arquivos_opcionais()

    # Aviso sobre impacto das mudanças
    mostrar_aviso_impacto()

    # Interface principal de configurações
    st.markdown("---")
    configuracoes = mostrar_configuracoes_globais()

    # Informações importantes (simplificadas)
    with st.expander("ℹ️ Informações Importantes"):
        st.markdown("""
        ### 📋 **Como Usar**

        1. **🔧 Configure uma vez**: Defina todos os parâmetros nesta página
        2. **🚀 Execute as etapas**: As configurações serão aplicadas automaticamente
        3. **✏️ Ajuste conforme necessário**: Volte aqui para modificar configurações

        ### ⚠️ **Avisos**

        - Mudanças nas configurações invalidam resultados anteriores
        - Filtros muito restritivos podem reduzir drasticamente os dados
        - Configurações de área impactam diretamente os estoques calculados
        """)

    # Navegação rápida final (apenas uma vez)
    st.markdown("---")
    criar_navegacao_rapida_botoes()


if __name__ == "__main__":
    main()