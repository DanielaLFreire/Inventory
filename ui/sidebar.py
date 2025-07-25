# ui/sidebar.py - VERSÃO MELHORADA COM CORREÇÃO DE PERSISTÊNCIA
'''
Interface da barra lateral para upload de arquivos - Versão melhorada com status das etapas
NOVO: Correção para persistir arquivos opcionais no session_state
'''

import streamlit as st


def criar_sidebar():
    '''
    Cria a interface da barra lateral com uploads e status das etapas
    CORREÇÃO: Agora persiste arquivos opcionais no session_state de forma segura

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

    # CORREÇÃO: Upload opcional de shapefile para áreas COM PERSISTÊNCIA
    arquivo_shapefile = st.sidebar.file_uploader(
        "🗺️ Shapefile Áreas (Opcional)",
        type=['shp', 'zip'],
        help="Arquivo shapefile com áreas dos talhões",
        key="upload_shapefile_persistente"
    )

    # CORREÇÃO: Armazenar shapefile no session_state de forma segura
    try:
        if arquivo_shapefile is not None:
            st.session_state.arquivo_shapefile = arquivo_shapefile
            st.sidebar.success(f"✅ Shapefile salvo: {arquivo_shapefile.name}")
        elif not hasattr(st.session_state, 'arquivo_shapefile'):
            st.session_state.arquivo_shapefile = None
    except Exception as e:
        st.sidebar.error(f"Erro ao salvar shapefile: {e}")

    # CORREÇÃO: Upload opcional de coordenadas COM PERSISTÊNCIA
    arquivo_coordenadas = st.sidebar.file_uploader(
        "📍 Coordenadas Parcelas (Opcional)",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo com coordenadas X,Y das parcelas",
        key="upload_coordenadas_persistente"
    )

    # CORREÇÃO: Armazenar coordenadas no session_state de forma segura
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

    # Mostrar progresso das etapas na sidebar (versão corrigida)
    mostrar_progresso_etapas_sidebar()

    # Mostrar informações adicionais
    mostrar_informacoes_adicionais()

    return arquivos


def mostrar_status_arquivos(arquivos):
    '''
    Mostra o status dos arquivos carregados - VERSÃO MELHORADA
    NOVO: Mostra também arquivos persistidos no session_state

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

    # NOVO: Seção para arquivos opcionais (verifica session_state também)
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
        st.sidebar.info("📁 Coordenadas ativas")
        st.sidebar.caption(f"📄 {coordenadas_ativas.name}")
    else:
        st.sidebar.warning("📁 Coordenadas: Não carregadas")

    # NOVO: Debug opcional para desenvolvedores
    mostrar_debug_arquivos_opcional()


def mostrar_debug_arquivos_opcional():
    '''NOVO: Debug opcional para verificar session_state'''
    if st.sidebar.checkbox("🔍 Debug Session State", key="debug_session_arquivos"):
        st.sidebar.markdown("**Session State - Arquivos:**")

        attrs_arquivo = ['arquivo_shapefile', 'arquivo_coordenadas']
        for attr in attrs_arquivo:
            if hasattr(st.session_state, attr):
                value = getattr(st.session_state, attr)
                if value is not None and hasattr(value, 'name'):
                    st.sidebar.success(f"✅ {attr}: {value.name}")
                else:
                    st.sidebar.warning(f"⚠️ {attr}: None")
            else:
                st.sidebar.error(f"❌ {attr}: Não existe")


def mostrar_progresso_etapas_sidebar():
    '''Mostra o progresso das etapas na sidebar - VERSÃO CORRIGIDA'''
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Progresso das Etapas")

    # CORREÇÃO: Verificar configurações primeiro
    config_status = False
    if hasattr(st.session_state, 'config_global'):
        config_status = st.session_state.config_global.get('configurado', False)

    if config_status:
        st.sidebar.success("✅ **Etapa 0** - Configurado")
    else:
        st.sidebar.warning("⚠️ **Etapa 0** - Configure primeiro")

    # CORREÇÃO: Verificar session states de forma segura
    etapas_info = [
        ('resultados_hipsometricos', 'Etapa 1 - Hipsométricos'),
        ('resultados_volumetricos', 'Etapa 2 - Volumétricos'),
        ('inventario_processado', 'Etapa 3 - Inventário')
    ]

    etapas_concluidas = 0

    for state_key, nome_etapa in etapas_info:
        # CORREÇÃO: Usar getattr em vez de __dict__
        try:
            resultado = getattr(st.session_state, state_key, None)

            if resultado is not None:
                st.sidebar.success(f"✅ **{nome_etapa}**")

                # Mostrar melhor modelo se disponível
                if isinstance(resultado, dict):
                    melhor = resultado.get('melhor_modelo', 'N/A')
                    if melhor != 'N/A':
                        st.sidebar.caption(f"🏆 Melhor: {melhor}")

                etapas_concluidas += 1
            else:
                st.sidebar.info(f"⏳ **{nome_etapa}**")

        except Exception as e:
            # Se houver erro, apenas mostrar como pendente
            st.sidebar.info(f"⏳ **{nome_etapa}**")

    # Mostrar progresso geral
    if etapas_concluidas > 0:
        progresso = etapas_concluidas / 3
        st.sidebar.progress(progresso, text=f"Progresso: {etapas_concluidas}/3 etapas")

        if etapas_concluidas == 3:
            st.sidebar.success("🎉 Análise Completa!")


def mostrar_informacoes_adicionais():
    '''Mostra informações adicionais na sidebar - VERSÃO MELHORADA'''
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

    # NOVO: Informações sobre persistência de arquivos
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

    # NOVO: Mostrar informações sobre configurações
    if hasattr(st.session_state, 'config_global'):
        config = st.session_state.config_global
        configurado = config.get('configurado', False)

        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Status Configuração")

        if configurado:
            st.sidebar.success("✅ Sistema Configurado")

            # Mostrar resumo das configurações principais
            with st.sidebar.expander("📋 Resumo Config"):
                st.write(f"• Diâmetro min: {config.get('diametro_min', 4.0)} cm")

                talhoes_excluir = config.get('talhoes_excluir', [])
                if talhoes_excluir:
                    st.write(f"• Talhões excluídos: {len(talhoes_excluir)}")
                else:
                    st.write("• Talhões excluídos: Nenhum")

                metodo_area = config.get('metodo_area', 'Simular automaticamente')
                st.write(f"• Método área: {metodo_area[:20]}...")

                # NOVO: Mostrar se arquivos opcionais foram detectados
                if metodo_area == "Coordenadas das parcelas":
                    st.success("📍 Coordenadas detectadas")
                elif metodo_area == "Upload shapefile":
                    st.success("📁 Shapefile detectado")
        else:
            st.sidebar.warning("⚠️ Configure o Sistema")
            if st.sidebar.button("⚙️ Ir para Configurações", use_container_width=True):
                st.switch_page("pages/0_⚙️_Configurações.py")

    # NOVO: Botões de ação rápida melhorados
    mostrar_acoes_rapidas_sidebar()


def mostrar_acoes_rapidas_sidebar():
    '''Seção de ações rápidas na sidebar - VERSÃO CORRIGIDA'''
    # CORREÇÃO: Verificar resultados de forma segura
    tem_resultados = False

    try:
        # Verificar cada resultado individualmente
        resultados_disponiveis = [
            getattr(st.session_state, 'resultados_hipsometricos', None),
            getattr(st.session_state, 'resultados_volumetricos', None),
            getattr(st.session_state, 'inventario_processado', None)
        ]

        tem_resultados = any(resultado is not None for resultado in resultados_disponiveis)

    except Exception:
        tem_resultados = False

    if tem_resultados:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Ações Rápidas")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("🔄 Limpar", use_container_width=True, key="limpar_resultados"):
                # CORREÇÃO: Limpar de forma segura
                keys_para_limpar = ['resultados_hipsometricos', 'resultados_volumetricos', 'inventario_processado']

                for key in keys_para_limpar:
                    if hasattr(st.session_state, key):
                        delattr(st.session_state, key)

                st.sidebar.success("✅ Resultados limpos!")
                st.rerun()

        with col2:
            if st.button("📊 Relatório", use_container_width=True, key="gerar_relatorio_rapido"):
                st.switch_page("pages/3_📈_Inventário_Florestal.py")

        # Download rápido se inventário foi processado - VERSÃO SEGURA
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
        except Exception:
            # Silenciosamente ignorar erro de download se não houver dados
            pass

    # Mostrar dicas contextuais
    mostrar_dicas_contextuais()


def mostrar_dicas_contextuais():
    '''NOVO: Dicas contextuais baseadas no estado atual'''
    st.sidebar.markdown("---")

    # Determinar contexto atual
    dados_carregados = (
            hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None and
            hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None
    )

    configurado = (
            hasattr(st.session_state, 'config_global') and
            st.session_state.config_global.get('configurado', False)
    )

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
        Agora configure filtros e parâmetros na Etapa 0.
        ''')
    else:
        st.sidebar.success('''
        **✅ Sistema Pronto:**
        Execute as Etapas 1, 2 e 3 em qualquer ordem.
        As configurações se aplicam automaticamente!
        ''')


def mostrar_resumo_sidebar():
    '''Mostra resumo dos resultados na sidebar - VERSÃO CORRIGIDA'''
    # CORREÇÃO: Verificar de forma segura
    try:
        inventario_resultado = getattr(st.session_state, 'inventario_processado', None)

        if inventario_resultado is None:
            return

        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Resumo Rápido")

        # Verificar se há estatísticas gerais
        if isinstance(inventario_resultado, dict) and 'estatisticas_gerais' in inventario_resultado:
            stats = inventario_resultado['estatisticas_gerais']

            # Métricas principais na sidebar
            if 'volume_medio_ha' in stats:
                st.sidebar.metric(
                    "🌲 Produtividade",
                    f"{stats['volume_medio_ha']:.1f} m³/ha"
                )

            if 'area_total' in stats:
                st.sidebar.metric(
                    "📏 Área Total",
                    f"{stats['area_total']:.1f} ha"
                )

            if 'estoque_total' in stats:
                st.sidebar.metric(
                    "🌳 Estoque Total",
                    f"{stats['estoque_total']:,.0f} m³"
                )

            # Classificação rápida
            vol_medio = stats.get('volume_medio_ha', 0)
            if vol_medio >= 150:
                st.sidebar.success("🌟 Alta Produtividade")
            elif vol_medio >= 100:
                st.sidebar.info("📊 Média Produtividade")
            else:
                st.sidebar.warning("📈 Baixa Produtividade")

    except Exception as e:
        # Em caso de erro, apenas não mostrar o resumo
        pass


def debug_session_state_seguro():
    '''NOVA: Debug seguro do session_state'''
    if st.sidebar.checkbox("🔍 Debug Session State", key="debug_session_seguro"):
        st.sidebar.markdown("**Session State - Verificação:**")

        # Verificar atributos importantes
        atributos_importantes = [
            'dados_inventario',
            'dados_cubagem',
            'config_global',
            'resultados_hipsometricos',
            'resultados_volumetricos',
            'inventario_processado',
            'arquivo_shapefile',
            'arquivo_coordenadas'
        ]

        for attr in atributos_importantes:
            try:
                if hasattr(st.session_state, attr):
                    value = getattr(st.session_state, attr)
                    if value is not None:
                        if hasattr(value, 'name'):  # É um arquivo
                            st.sidebar.success(f"✅ {attr}: {value.name}")
                        elif isinstance(value, dict):
                            st.sidebar.success(f"✅ {attr}: Dict ({len(value)} items)")
                        elif hasattr(value, '__len__'):  # DataFrame ou lista
                            st.sidebar.success(f"✅ {attr}: {type(value).__name__} ({len(value)})")
                        else:
                            st.sidebar.success(f"✅ {attr}: {type(value).__name__}")
                    else:
                        st.sidebar.info(f"ℹ️ {attr}: None")
                else:
                    st.sidebar.warning(f"⚠️ {attr}: Não existe")
            except Exception as e:
                st.sidebar.error(f"❌ {attr}: Erro ({e})")


def criar_sidebar_melhorada():
    '''
    Versão melhorada da sidebar com todas as funcionalidades
    CORREÇÃO: Todas as funções agora são à prova de erros

    Returns:
        dict: Dicionário com os arquivos carregados
    '''
    try:
        # Criar sidebar principal com correções
        arquivos = criar_sidebar()

        # Adicionar resumo se análise foi executada (versão segura)
        mostrar_resumo_sidebar()

        # Debug opcional e seguro
        debug_session_state_seguro()

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


def verificar_e_restaurar_arquivos():
    """
    NOVA: Verifica e restaura status dos arquivos na sidebar
    """
    # Verificar se dados principais ainda existem
    if (hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None and
            hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None):
        st.sidebar.success("✅ Dados principais ativos")
        st.sidebar.caption(f"📊 Inventário: {len(st.session_state.dados_inventario)} registros")
        st.sidebar.caption(f"📏 Cubagem: {len(st.session_state.dados_cubagem)} medições")

        # Garantir que arquivos_carregados esteja True
        st.session_state.arquivos_carregados = True

    # Verificar arquivos opcionais
    if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None:
        st.sidebar.info(f"📁 Shapefile persistido: {st.session_state.arquivo_shapefile.name}")

    if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None:
        st.sidebar.info(f"📍 Coordenadas persistidas: {st.session_state.arquivo_coordenadas.name}")


# CORREÇÃO 4: Modificar a função criar_sidebar() para incluir verificação

def criar_sidebar_com_verificacao():
    """
    Cria sidebar com verificação de arquivos persistidos
    """
    st.sidebar.header("📁 Upload de Dados")

    # NOVO: Verificar e mostrar arquivos persistidos primeiro
    verificar_e_restaurar_arquivos()

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
    mostrar_status_arquivos_melhorado(arquivos)

    # Mostrar progresso das etapas na sidebar
    mostrar_progresso_etapas_sidebar()

    # Mostrar informações adicionais
    mostrar_informacoes_adicionais()

    return arquivos


def mostrar_status_arquivos_melhorado(arquivos):
    """
    Mostra status melhorado dos arquivos
    """
    st.sidebar.subheader("📊 Status dos Arquivos")

    # Verificar dados principais (considerar tanto upload quanto session_state)
    inventario_ativo = (
            arquivos['inventario'] is not None or
            (hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None)
    )

    cubagem_ativa = (
            arquivos['cubagem'] is not None or
            (hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None)
    )

    # Status inventário
    if inventario_ativo:
        if arquivos['inventario'] is not None:
            st.sidebar.success("✅ Inventário carregado (novo)")
            st.sidebar.caption(f"📄 {arquivos['inventario'].name}")
        else:
            st.sidebar.success("✅ Inventário ativo (sessão)")
    else:
        st.sidebar.error("❌ Inventário necessário")

    # Status cubagem
    if cubagem_ativa:
        if arquivos['cubagem'] is not None:
            st.sidebar.success("✅ Cubagem carregada (nova)")
            st.sidebar.caption(f"📄 {arquivos['cubagem'].name}")
        else:
            st.sidebar.success("✅ Cubagem ativa (sessão)")
    else:
        st.sidebar.error("❌ Cubagem necessária")

    # Arquivos opcionais
    st.sidebar.markdown("**Arquivos Opcionais:**")

    # Shapefile
    shapefile_ativo = (
            arquivos['shapefile'] is not None or
            (hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None)
    )

    if shapefile_ativo:
        if arquivos['shapefile'] is not None:
            st.sidebar.info("📁 Shapefile (novo)")
            st.sidebar.caption(f"📄 {arquivos['shapefile'].name}")
        else:
            st.sidebar.info("📁 Shapefile (persistido)")
            st.sidebar.caption(f"📄 {st.session_state.arquivo_shapefile.name}")
    else:
        st.sidebar.warning("📁 Shapefile: Não carregado")

    # Coordenadas
    coordenadas_ativas = (
            arquivos['coordenadas'] is not None or
            (hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None)
    )

    if coordenadas_ativas:
        if arquivos['coordenadas'] is not None:
            st.sidebar.info("📍 Coordenadas (novas)")
            st.sidebar.caption(f"📄 {arquivos['coordenadas'].name}")
        else:
            st.sidebar.info("📍 Coordenadas (persistidas)")
            st.sidebar.caption(f"📄 {st.session_state.arquivo_coordenadas.name}")
    else:
        st.sidebar.warning("📍 Coordenadas: Não carregadas")


# FUNÇÃO AUXILIAR PARA VERIFICAR ARQUIVOS PERSISTIDOS
def obter_arquivos_com_persistencia():
    '''
    NOVA: Obtém arquivos considerando tanto upload atual quanto session_state

    Returns:
        dict: Arquivos disponíveis (atuais + persistidos)
    '''
    arquivos_persistidos = {}

    # Shapefile
    if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None:
        arquivos_persistidos['shapefile'] = st.session_state.arquivo_shapefile

    # Coordenadas
    if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None:
        arquivos_persistidos['coordenadas'] = st.session_state.arquivo_coordenadas

    return arquivos_persistidos


# FUNÇÃO PARA LIMPAR ARQUIVOS PERSISTIDOS
def limpar_arquivos_persistidos():
    '''NOVA: Limpa arquivos opcionais do session_state'''
    if hasattr(st.session_state, 'arquivo_shapefile'):
        st.session_state.arquivo_shapefile = None

    if hasattr(st.session_state, 'arquivo_coordenadas'):
        st.session_state.arquivo_coordenadas = None

    st.sidebar.success("🗑️ Arquivos opcionais removidos!")