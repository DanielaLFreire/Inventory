# ui/sidebar.py - VERSÃO AJUSTADA PARA CONFIGURAÇÕES GLOBAIS
'''
Interface da barra lateral para upload de arquivos - Versão integrada com configurações globais
NOVO: Melhor integração com sistema de configurações centralizadas
'''

import streamlit as st


def criar_sidebar():
    '''
    Cria a interface da barra lateral com uploads e status das etapas
    ATUALIZADO: Melhor integração com configurações globais

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

    # Upload opcional de shapefile para áreas COM PERSISTÊNCIA
    arquivo_shapefile = st.sidebar.file_uploader(
        "🗺️ Shapefile Áreas (Opcional)",
        type=['shp', 'zip'],
        help="Arquivo shapefile com áreas dos talhões",
        key="upload_shapefile_persistente"
    )

    # Armazenar shapefile no session_state de forma segura
    try:
        if arquivo_shapefile is not None:
            st.session_state.arquivo_shapefile = arquivo_shapefile
            st.sidebar.success(f"✅ Shapefile salvo: {arquivo_shapefile.name}")
        elif not hasattr(st.session_state, 'arquivo_shapefile'):
            st.session_state.arquivo_shapefile = None
    except Exception as e:
        st.sidebar.error(f"Erro ao salvar shapefile: {e}")

    # Upload opcional de coordenadas COM PERSISTÊNCIA
    arquivo_coordenadas = st.sidebar.file_uploader(
        "📍 Coordenadas Parcelas (Opcional)",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo com coordenadas X,Y das parcelas",
        key="upload_coordenadas_persistente"
    )

    # Armazenar coordenadas no session_state de forma segura
    try:
        if arquivo_coordenadas is not None:
            st.session_state.arquivo_coordenadas = arquivo_coordenadas
            st.sidebar.success(f"✅ Coordenadas salvas: {arquivo_coordenadas.name}")
        elif not hasattr(st.session_state, 'arquivo_coordenadas'):
            st.session_state.arquivo_coordenadas = None
    except Exception as e:
        st.sidebar.error(f"Erro ao salvar coordenadas: {e}")

    arquivos = {
        'inventario': arquivo_inventario,
        'cubagem': arquivo_cubagem,
        'shapefile': arquivo_shapefile,
        'coordenadas': arquivo_coordenadas
    }

    # Mostrar status dos arquivos
    mostrar_status_arquivos(arquivos)

    # NOVO: Mostrar status das configurações globais na sidebar
    mostrar_status_configuracao_sidebar()

    # Mostrar progresso das etapas na sidebar
    mostrar_progresso_etapas_sidebar()

    # Mostrar informações adicionais
    mostrar_informacoes_adicionais()

    return arquivos


def mostrar_status_configuracao_sidebar():
    '''NOVO: Mostra status da configuração global na sidebar'''
    try:
        # Importar aqui para evitar importação circular
        from config.configuracoes_globais import obter_configuracao_global

        config_global = obter_configuracao_global()
        configurado = config_global.get('configurado', False)

        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Status Configuração")

        if configurado:
            st.sidebar.success("✅ Sistema Configurado")

            # Mostrar timestamp da última configuração
            timestamp = config_global.get('timestamp_config')
            if timestamp:
                import pandas as pd
                if isinstance(timestamp, pd.Timestamp):
                    st.sidebar.caption(f"Atualizado: {timestamp.strftime('%H:%M')}")
                else:
                    st.sidebar.caption("Configurado nesta sessão")

            # NOVO: Mostrar resumo das configurações principais
            with st.sidebar.expander("📋 Resumo Config"):
                # Filtros básicos
                st.write(f"🔍 **Filtros:**")
                st.write(f"• Diâmetro min: {config_global.get('diametro_min', 4.0)} cm")

                talhoes_excluir = config_global.get('talhoes_excluir', [])
                if talhoes_excluir:
                    st.write(f"• Talhões excluídos: {len(talhoes_excluir)}")
                else:
                    st.write("• Talhões excluídos: Nenhum")

                # Método de área
                metodo_area = config_global.get('metodo_area', 'Simular automaticamente')
                st.write(f"📏 **Área:** {metodo_area[:15]}...")

                # NOVO: Status dos modelos não-lineares
                incluir_nao_lineares = config_global.get('incluir_nao_lineares', True)
                st.write(f"🧮 **Modelos:** {'Lineares+NL' if incluir_nao_lineares else 'Só Lineares'}")

                # NOVO: Mostrar se parâmetros foram customizados
                parametros_customizados = verificar_parametros_customizados(config_global)
                if parametros_customizados:
                    st.success("🔧 Parâmetros customizados")
                else:
                    st.info("⚙️ Parâmetros padrão")

        else:
            st.sidebar.warning("⚠️ Sistema Não Configurado")
            st.sidebar.caption("Configure na Etapa 0 primeiro")

            if st.sidebar.button("⚙️ Ir para Configurações", use_container_width=True, key="btn_config_sidebar"):
                st.switch_page("pages/0_⚙️_Configurações.py")

    except ImportError:
        # Se não conseguir importar configurações, mostrar aviso simples
        st.sidebar.warning("⚠️ Configurações não disponíveis")


def verificar_parametros_customizados(config):
    '''NOVO: Verifica se parâmetros não-lineares foram customizados'''
    parametros_padrao = {
        'parametros_chapman': {'b0': 42.12, 'b1': 0.01, 'b2': 1.00},
        'parametros_weibull': {'a': 42.12, 'b': 0.01, 'c': 1.00},
        'parametros_mononuclear': {'a': 42.12, 'b': 1.00, 'c': 0.10}
    }

    for modelo, params_padrao in parametros_padrao.items():
        params_config = config.get(modelo, {})
        for param, valor_padrao in params_padrao.items():
            if params_config.get(param, valor_padrao) != valor_padrao:
                return True
    return False


def mostrar_status_arquivos(arquivos):
    '''
    Mostra o status dos arquivos carregados - VERSÃO MELHORADA
    '''
    st.sidebar.subheader("📊 Status dos Arquivos")

    # Inventário
    if arquivos['inventario'] is not None:
        st.sidebar.success("✅ Inventário carregado")
        st.sidebar.caption(f"📄 {arquivos['inventario'].name}")
    else:
        st.sidebar.error("❌ Inventário necessário")

    # Cubagem
    if arquivos['cubagem'] is not None:
        st.sidebar.success("✅ Cubagem carregada")
        st.sidebar.caption(f"📄 {arquivos['cubagem'].name}")
    else:
        st.sidebar.error("❌ Cubagem necessária")

    # Seção para arquivos opcionais
    st.sidebar.markdown("**Arquivos Opcionais:**")

    # Shapefile - Verificar tanto upload atual quanto session_state
    shapefile_ativo = None
    if arquivos['shapefile'] is not None:
        shapefile_ativo = arquivos['shapefile']
    elif hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None:
        shapefile_ativo = st.session_state.arquivo_shapefile

    if shapefile_ativo is not None:
        st.sidebar.info("📁 Shapefile ativo")
        st.sidebar.caption(f"📄 {shapefile_ativo.name}")
    else:
        st.sidebar.warning("📁 Shapefile: Não carregado")

    # Coordenadas - Verificar tanto upload atual quanto session_state
    coordenadas_ativas = None
    if arquivos['coordenadas'] is not None:
        coordenadas_ativas = arquivos['coordenadas']
    elif hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None:
        coordenadas_ativas = st.session_state.arquivo_coordenadas

    if coordenadas_ativas is not None:
        st.sidebar.info("📍 Coordenadas ativas")
        st.sidebar.caption(f"📄 {coordenadas_ativas.name}")
    else:
        st.sidebar.warning("📍 Coordenadas: Não carregadas")


def mostrar_progresso_etapas_sidebar():
    '''Mostra o progresso das etapas na sidebar - VERSÃO ATUALIZADA'''
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Progresso das Etapas")

    # NOVO: Verificar configurações primeiro (importação segura)
    config_status = False
    try:
        from config.configuracoes_globais import obter_configuracao_global
        config_global = obter_configuracao_global()
        config_status = config_global.get('configurado', False)
    except:
        config_status = False

    if config_status:
        st.sidebar.success("✅ **Etapa 0** - Configurado")
    else:
        st.sidebar.warning("⚠️ **Etapa 0** - Configure primeiro")
        st.sidebar.caption("Necessário para Etapas 1-3")

    # Verificar session states de forma segura
    etapas_info = [
        ('resultados_hipsometricos', 'Etapa 1 - Hipsométricos'),
        ('resultados_volumetricos', 'Etapa 2 - Volumétricos'),
        ('inventario_processado', 'Etapa 3 - Inventário')
    ]

    etapas_concluidas = 0

    for state_key, nome_etapa in etapas_info:
        try:
            resultado = getattr(st.session_state, state_key, None)

            if resultado is not None:
                st.sidebar.success(f"✅ **{nome_etapa}**")

                # NOVO: Mostrar mais detalhes dos resultados
                if isinstance(resultado, dict):
                    melhor = resultado.get('melhor_modelo', 'N/A')
                    if melhor != 'N/A':
                        st.sidebar.caption(f"🏆 Melhor: {melhor}")

                    # NOVO: Mostrar se configuração foi aplicada
                    config_aplicada = resultado.get('config_aplicada')
                    if config_aplicada and config_aplicada.get('configurado', False):
                        st.sidebar.caption("⚙️ Config aplicada")

                etapas_concluidas += 1
            else:
                st.sidebar.info(f"⏳ **{nome_etapa}**")

                # NOVO: Mostrar dependências
                if state_key == 'resultados_hipsometricos' and not config_status:
                    st.sidebar.caption("Precisa: Configuração")
                elif state_key == 'resultados_volumetricos' and not config_status:
                    st.sidebar.caption("Precisa: Configuração")
                elif state_key == 'inventario_processado':
                    hip_ok = getattr(st.session_state, 'resultados_hipsometricos', None) is not None
                    vol_ok = getattr(st.session_state, 'resultados_volumetricos', None) is not None
                    if not hip_ok or not vol_ok:
                        st.sidebar.caption("Precisa: Etapas 1 e 2")

        except Exception:
            st.sidebar.info(f"⏳ **{nome_etapa}**")

    # Mostrar progresso geral
    total_etapas = 3
    if etapas_concluidas > 0:
        progresso = etapas_concluidas / total_etapas
        st.sidebar.progress(progresso, text=f"Progresso: {etapas_concluidas}/{total_etapas} etapas")

        if etapas_concluidas == total_etapas:
            st.sidebar.success("🎉 Análise Completa!")
        elif etapas_concluidas >= 2:
            st.sidebar.info("🚀 Quase lá! Falta 1 etapa")


def mostrar_informacoes_adicionais():
    '''Mostra informações adicionais na sidebar - VERSÃO ATUALIZADA'''
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

    # Informações sobre configurações centralizadas
    with st.sidebar.expander("⚙️ Sistema de Configurações"):
        st.markdown('''
        **Configurações Centralizadas:**
        - ✅ Configure uma vez (Etapa 0)
        - ✅ Aplica em todas as etapas
        - ✅ Parâmetros dos modelos não-lineares
        - ✅ Filtros de dados globais
        - ✅ Configurações de área

        **Vantagens:**
        - Consistência total
        - Fácil de usar
        - Transparente
        - Rastreável
        ''')

    # Informações sobre persistência de arquivos
    with st.sidebar.expander("💾 Persistência de Arquivos"):
        st.markdown('''
        **Arquivos Obrigatórios:**
        - Recarregados a cada navegação
        - Use sempre os mesmos arquivos

        **Arquivos Opcionais:**
        - ✅ Ficam salvos na sessão
        - ✅ Persistem entre páginas
        - ✅ Detectados nas configurações

        **Dica:** Carregue shapefile/coordenadas uma vez e navegue livremente!
        ''')

    # NOVO: Dicas sobre parâmetros não-lineares
    with st.sidebar.expander("🔧 Modelos Não-Lineares"):
        st.markdown('''
        **Parâmetros Configuráveis:**
        - Chapman: b₀, b₁, b₂
        - Weibull: a, b, c
        - Mononuclear: a, b, c

        **Dicas de Configuração:**
        - Altura assintótica: 20-50m típico
        - Começar com valores padrão
        - Ajustar baseado na convergência
        - Monitorar relatórios de qualidade
        ''')

    # Mostrar ações rápidas se necessário
    mostrar_acoes_rapidas_sidebar()


def mostrar_acoes_rapidas_sidebar():
    '''Seção de ações rápidas na sidebar - VERSÃO ATUALIZADA'''
    # Verificar se há resultados para mostrar ações
    tem_resultados = False
    try:
        resultados_disponiveis = [
            getattr(st.session_state, 'resultados_hipsometricos', None),
            getattr(st.session_state, 'resultados_volumetricos', None),
            getattr(st.session_state, 'inventario_processado', None)
        ]
        tem_resultados = any(resultado is not None for resultado in resultados_disponiveis)
    except:
        tem_resultados = False

    if tem_resultados:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Ações Rápidas")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("🔄 Limpar", use_container_width=True, key="limpar_resultados"):
                keys_para_limpar = [
                    'resultados_hipsometricos',
                    'resultados_volumetricos',
                    'inventario_processado'
                ]

                for key in keys_para_limpar:
                    if hasattr(st.session_state, key):
                        delattr(st.session_state, key)

                st.sidebar.success("✅ Resultados limpos!")
                st.rerun()

        with col2:
            if st.button("📊 Relatório", use_container_width=True, key="gerar_relatorio_rapido"):
                st.switch_page("pages/3_📈_Inventário_Florestal.py")

        # NOVO: Botão para reconfigurar sistema
        if st.button("⚙️ Reconfigurar Sistema", use_container_width=True, key="reconfigurar_sistema"):
            st.switch_page("pages/0_⚙️_Configurações.py")

        # Download rápido se inventário foi processado
        try:
            inventario_resultado = getattr(st.session_state, 'inventario_processado', None)

            if inventario_resultado is not None and isinstance(inventario_resultado, dict):
                if 'resumo_talhoes' in inventario_resultado:
                    resumo_df = inventario_resultado['resumo_talhoes']
                    csv_dados = resumo_df.to_csv(index=False)

                    st.sidebar.download_button(
                        "📥 Download Resumo",
                        data=csv_dados,
                        file_name="resumo_inventario_rapido.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Download rápido do resumo por talhões"
                    )
        except:
            pass

    # NOVO: Mostrar dicas contextuais baseadas no estado
    mostrar_dicas_contextuais()


def mostrar_dicas_contextuais():
    '''NOVO: Dicas contextuais baseadas no estado atual do sistema'''
    st.sidebar.markdown("---")

    # Determinar contexto atual
    dados_carregados = (
            hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None and
            hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None
    )

    configurado = False
    try:
        from config.configuracoes_globais import obter_configuracao_global
        config_global = obter_configuracao_global()
        configurado = config_global.get('configurado', False)
    except:
        pass

    # Dicas baseadas no contexto
    if not dados_carregados:
        st.sidebar.info('''
        **🚀 Próximo Passo:**
        1. Carregue Inventário e Cubagem
        2. Configure o sistema (Etapa 0)
        3. Execute as análises (Etapas 1-3)
        ''')
    elif not configurado:
        st.sidebar.warning('''
        **⚙️ Configure o Sistema:**
        Os dados estão carregados!

        **Na Etapa 0 você define:**
        - Filtros de dados
        - Parâmetros dos modelos
        - Configurações de área
        - Tolerâncias de ajuste
        ''')
    else:
        st.sidebar.success('''
        **✅ Sistema Pronto:**
        Execute as Etapas 1, 2 e 3 em qualquer ordem.

        **Configurações aplicam automaticamente:**
        - Filtros globais
        - Parâmetros não-lineares
        - Validações automáticas
        - Relatórios com configurações
        ''')


def criar_sidebar_melhorada():
    '''
    Versão melhorada da sidebar com integração total às configurações globais

    Returns:
        dict: Dicionário com os arquivos carregados
    '''
    try:
        # Criar sidebar principal
        arquivos = criar_sidebar()
        return arquivos

    except Exception as e:
        st.sidebar.error(f"Erro na sidebar: {e}")
        # Retornar estrutura mínima em caso de erro
        return {
            'inventario': None,
            'cubagem': None,
            'shapefile': None,
            'coordenadas': None
        }


# Função de compatibilidade (manter para não quebrar código existente)
def mostrar_status_configuracao_sidebar_compat():
    '''Função de compatibilidade para o código existente'''
    return mostrar_status_configuracao_sidebar()