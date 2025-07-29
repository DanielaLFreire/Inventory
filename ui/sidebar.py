# ui/sidebar.py - VERSÃO MELHORADA INTEGRADA COM PRINCIPAL.PY
'''
Interface da barra lateral para upload de arquivos - Versão integrada completa
Inclui funcionalidades do Principal.py para processamento automático
'''

import streamlit as st
import pandas as pd
import traceback

# Importar processadores
from utils.arquivo_handler import carregar_arquivo, validar_estrutura_arquivo
from utils.formatacao import formatar_brasileiro, formatar_numero_inteligente


def processar_dados_inventario_sidebar(arquivo_inventario):
    """
    Processa dados do inventário na sidebar com feedback completo

    Args:
        arquivo_inventario: Arquivo de inventário carregado

    Returns:
        DataFrame processado ou None se erro
    """
    try:
        if arquivo_inventario is None:
            return None

        # Carregar arquivo
        df_inventario = carregar_arquivo(arquivo_inventario)

        if df_inventario is None:
            st.sidebar.error("❌ Falha no carregamento")
            return None

        # Validar estrutura básica
        colunas_obrigatorias = ['D_cm', 'H_m', 'talhao', 'parcela']
        validacao = validar_estrutura_arquivo(df_inventario, colunas_obrigatorias, "inventário")

        if not validacao['valido']:
            st.sidebar.error("❌ Estrutura inválida")
            for erro in validacao['erros'][:2]:  # Mostrar apenas 2 primeiros erros na sidebar
                st.sidebar.error(f"• {erro}")
            return None

        # Limpar e validar dados
        df_limpo = limpar_dados_inventario_sidebar(df_inventario)

        if len(df_limpo) == 0:
            st.sidebar.error("❌ Sem dados válidos")
            return None

        # Feedback de sucesso
        percentual_mantido = (len(df_limpo) / len(df_inventario)) * 100
        st.sidebar.success(f"✅ Inventário OK")
        st.sidebar.info(f"📊 {len(df_limpo):,} registros ({percentual_mantido:.1f}%)")

        return df_limpo

    except Exception as e:
        st.sidebar.error(f"❌ Erro: {str(e)[:50]}...")
        return None


def processar_dados_cubagem_sidebar(arquivo_cubagem):
    """
    Processa dados de cubagem na sidebar com feedback completo

    Args:
        arquivo_cubagem: Arquivo de cubagem carregado

    Returns:
        DataFrame processado ou None se erro
    """
    try:
        if arquivo_cubagem is None:
            return None

        # Carregar arquivo
        df_cubagem = carregar_arquivo(arquivo_cubagem)

        if df_cubagem is None:
            st.sidebar.error("❌ Falha no carregamento")
            return None

        # Validar estrutura básica
        colunas_obrigatorias = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']
        validacao = validar_estrutura_arquivo(df_cubagem, colunas_obrigatorias, "cubagem")

        if not validacao['valido']:
            st.sidebar.error("❌ Estrutura inválida")
            for erro in validacao['erros'][:2]:
                st.sidebar.error(f"• {erro}")
            return None

        # Limpar e validar dados
        df_limpo = limpar_dados_cubagem_sidebar(df_cubagem)

        if len(df_limpo) == 0:
            st.sidebar.error("❌ Sem dados válidos")
            return None

        # Feedback de sucesso
        arvores_cubadas = df_limpo['arv'].nunique()
        st.sidebar.success(f"✅ Cubagem OK")
        st.sidebar.info(f"📏 {arvores_cubadas} árvores cubadas")

        return df_limpo

    except Exception as e:
        st.sidebar.error(f"❌ Erro: {str(e)[:50]}...")
        return None


def limpar_dados_inventario_sidebar(df_inventario):
    """Versão otimizada da limpeza para sidebar"""
    df_limpo = df_inventario.copy()

    # Converter tipos básicos
    try:
        df_limpo['D_cm'] = pd.to_numeric(df_limpo['D_cm'], errors='coerce')
        df_limpo['H_m'] = pd.to_numeric(df_limpo['H_m'], errors='coerce')
        df_limpo['talhao'] = pd.to_numeric(df_limpo['talhao'], errors='coerce').astype('Int64')
        df_limpo['parcela'] = pd.to_numeric(df_limpo['parcela'], errors='coerce').astype('Int64')

        if 'idade_anos' in df_limpo.columns:
            df_limpo['idade_anos'] = pd.to_numeric(df_limpo['idade_anos'], errors='coerce')

        if 'cod' in df_limpo.columns:
            df_limpo['cod'] = df_limpo['cod'].astype(str)

    except Exception:
        pass  # Continuar mesmo com problemas de conversão

    # Filtros básicos de qualidade
    mask_valido = (
            df_limpo['D_cm'].notna() &
            df_limpo['H_m'].notna() &
            df_limpo['talhao'].notna() &
            df_limpo['parcela'].notna() &
            (df_limpo['D_cm'] > 0) &
            (df_limpo['H_m'] > 1.3)
    )

    df_limpo = df_limpo[mask_valido]

    # Remover outliers extremos (apenas os mais óbvios)
    try:
        # DAP entre 1 e 100 cm (limites muito amplos)
        df_limpo = df_limpo[(df_limpo['D_cm'] >= 1) & (df_limpo['D_cm'] <= 100)]

        # Altura entre 1.3 e 60 m (limites muito amplos)
        df_limpo = df_limpo[(df_limpo['H_m'] >= 1.3) & (df_limpo['H_m'] <= 60)]

    except Exception:
        pass

    return df_limpo


def limpar_dados_cubagem_sidebar(df_cubagem):
    """Versão otimizada da limpeza para sidebar"""
    df_limpo = df_cubagem.copy()

    # Converter tipos básicos
    try:
        df_limpo['arv'] = pd.to_numeric(df_limpo['arv'], errors='coerce').astype('Int64')
        df_limpo['talhao'] = pd.to_numeric(df_limpo['talhao'], errors='coerce').astype('Int64')
        df_limpo['d_cm'] = pd.to_numeric(df_limpo['d_cm'], errors='coerce')
        df_limpo['h_m'] = pd.to_numeric(df_limpo['h_m'], errors='coerce')
        df_limpo['D_cm'] = pd.to_numeric(df_limpo['D_cm'], errors='coerce')
        df_limpo['H_m'] = pd.to_numeric(df_limpo['H_m'], errors='coerce')

    except Exception:
        pass

    # Filtros básicos de qualidade
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

    # Validação de consistência básica
    try:
        mask_consistente = df_limpo['d_cm'] <= df_limpo['D_cm'] * 1.5  # Tolerância ampla
        df_limpo = df_limpo[mask_consistente]
    except Exception:
        pass

    return df_limpo


def criar_sidebar():
    '''
    Cria a interface da barra lateral com uploads e processamento automático
    VERSÃO INTEGRADA com funcionalidades do Principal.py

    Returns:
        dict: Dicionário com os arquivos carregados e processados
    '''
    st.sidebar.header("📁 Upload de Dados")

    # Upload do arquivo de inventário
    arquivo_inventario = st.sidebar.file_uploader(
        "📋 Arquivo de Inventário",
        type=['csv', 'xlsx', 'xls'],
        help="Dados de parcelas (D_cm, H_m, talhao, parcela, cod, idade_anos)",
        key="upload_inventario_principal"
    )

    # Upload do arquivo de cubagem
    arquivo_cubagem = st.sidebar.file_uploader(
        "📏 Arquivo de Cubagem",
        type=['csv', 'xlsx', 'xls'],
        help="Medições detalhadas (arv, talhao, d_cm, h_m, D_cm, H_m)",
        key="upload_cubagem_principal"
    )

    # Upload opcional de shapefile para áreas COM PERSISTÊNCIA
    arquivo_shapefile = st.sidebar.file_uploader(
        "🗺️ Shapefile Áreas (Opcional)",
        type=['shp', 'zip'],
        help="Arquivo shapefile com áreas dos talhões",
        key="upload_shapefile_persistente"
    )

    # Gerenciar persistência do shapefile
    if arquivo_shapefile is not None:
        st.session_state.arquivo_shapefile = arquivo_shapefile
        st.sidebar.success(f"✅ Shapefile salvo")
        st.sidebar.caption(f"📄 {arquivo_shapefile.name}")
    elif not hasattr(st.session_state, 'arquivo_shapefile'):
        st.session_state.arquivo_shapefile = None

    # Upload opcional de coordenadas COM PERSISTÊNCIA
    arquivo_coordenadas = st.sidebar.file_uploader(
        "📍 Coordenadas Parcelas (Opcional)",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo com coordenadas X,Y das parcelas",
        key="upload_coordenadas_persistente"
    )

    # Gerenciar persistência das coordenadas
    if arquivo_coordenadas is not None:
        st.session_state.arquivo_coordenadas = arquivo_coordenadas
        st.sidebar.success(f"✅ Coordenadas salvas")
        st.sidebar.caption(f"📄 {arquivo_coordenadas.name}")
    elif not hasattr(st.session_state, 'arquivo_coordenadas'):
        st.session_state.arquivo_coordenadas = None

    # PROCESSAMENTO AUTOMÁTICO DOS DADOS
    dados_processados = {
        'inventario': None,
        'cubagem': None,
        'shapefile': arquivo_shapefile if arquivo_shapefile else st.session_state.arquivo_shapefile,
        'coordenadas': arquivo_coordenadas if arquivo_coordenadas else st.session_state.arquivo_coordenadas
    }

    # Processar inventário se carregado
    if arquivo_inventario is not None:
        with st.sidebar.expander("🔄 Processando Inventário..."):
            dados_processados['inventario'] = processar_dados_inventario_sidebar(arquivo_inventario)

            # Salvar no session_state se processado com sucesso
            if dados_processados['inventario'] is not None:
                st.session_state.dados_inventario = dados_processados['inventario']

    # Processar cubagem se carregada
    if arquivo_cubagem is not None:
        with st.sidebar.expander("🔄 Processando Cubagem..."):
            dados_processados['cubagem'] = processar_dados_cubagem_sidebar(arquivo_cubagem)

            # Salvar no session_state se processado com sucesso
            if dados_processados['cubagem'] is not None:
                st.session_state.dados_cubagem = dados_processados['cubagem']

    # Mostrar status dos arquivos
    mostrar_status_arquivos_melhorado(dados_processados)

    # Mostrar status das configurações globais na sidebar
    mostrar_status_configuracao_sidebar()

    # Mostrar progresso das etapas na sidebar
    mostrar_progresso_etapas_sidebar()

    # Mostrar informações adicionais e ações rápidas
    mostrar_informacoes_e_acoes_sidebar()

    return dados_processados


def mostrar_status_arquivos_melhorado(arquivos):
    '''
    Mostra status detalhado dos arquivos carregados e processados
    '''
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status dos Dados")

    # Inventário
    if arquivos['inventario'] is not None:
        st.sidebar.success("✅ Inventário processado")

        df_inv = arquivos['inventario']
        st.sidebar.info(f"📊 {len(df_inv):,} registros")
        st.sidebar.info(f"🌳 {df_inv['talhao'].nunique()} talhões")

        # Estatísticas rápidas
        dap_medio = df_inv['D_cm'].mean()
        altura_media = df_inv['H_m'].mean()
        st.sidebar.caption(f"DAP: {formatar_brasileiro(dap_medio, 1)} cm")
        st.sidebar.caption(f"Altura: {formatar_brasileiro(altura_media, 1)} m")

    elif hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
        st.sidebar.info("✅ Inventário carregado")
        df_inv = st.session_state.dados_inventario
        st.sidebar.caption(f"📊 {len(df_inv):,} registros")
    else:
        st.sidebar.error("❌ Inventário necessário")

    # Cubagem
    if arquivos['cubagem'] is not None:
        st.sidebar.success("✅ Cubagem processada")

        df_cub = arquivos['cubagem']
        arvores = df_cub['arv'].nunique()
        secoes_media = df_cub.groupby(['talhao', 'arv']).size().mean()

        st.sidebar.info(f"📏 {arvores} árvores")
        st.sidebar.caption(f"Seções/árvore: {formatar_brasileiro(secoes_media, 1)}")

    elif hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None:
        st.sidebar.info("✅ Cubagem carregada")
        df_cub = st.session_state.dados_cubagem
        arvores = df_cub['arv'].nunique()
        st.sidebar.caption(f"📏 {arvores} árvores")
    else:
        st.sidebar.error("❌ Cubagem necessária")

    # Arquivos opcionais
    st.sidebar.markdown("**Arquivos Opcionais:**")

    # Shapefile
    shapefile_ativo = arquivos['shapefile']
    if shapefile_ativo is not None:
        st.sidebar.info("📁 Shapefile ativo")
        st.sidebar.caption(f"📄 {shapefile_ativo.name}")
    else:
        st.sidebar.warning("📁 Shapefile: Não carregado")

    # Coordenadas
    coordenadas_ativas = arquivos['coordenadas']
    if coordenadas_ativas is not None:
        st.sidebar.info("📍 Coordenadas ativas")
        st.sidebar.caption(f"📄 {coordenadas_ativas.name}")
    else:
        st.sidebar.warning("📍 Coordenadas: Não carregadas")


def mostrar_status_configuracao_sidebar():
    '''Mostra status da configuração global na sidebar'''
    try:
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
                if hasattr(timestamp, 'strftime'):
                    st.sidebar.caption(f"Atualizado: {timestamp.strftime('%H:%M')}")
                else:
                    st.sidebar.caption("Configurado nesta sessão")

            # Mostrar resumo das configurações principais
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

                # Status dos modelos não-lineares
                incluir_nao_lineares = config_global.get('incluir_nao_lineares', True)
                st.write(f"🧮 **Modelos:** {'Lineares+NL' if incluir_nao_lineares else 'Só Lineares'}")

                # Verificar se parâmetros foram customizados
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
        st.sidebar.warning("⚠️ Configurações não disponíveis")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro nas configurações: {str(e)[:30]}...")


def verificar_parametros_customizados(config):
    '''Verifica se parâmetros não-lineares foram customizados'''
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


def mostrar_progresso_etapas_sidebar():
    '''Mostra o progresso das etapas na sidebar com melhorias'''
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Progresso das Etapas")

    # Verificar configurações primeiro
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
        ('resultados_hipsometricos', 'Etapa 1 - Hipsométricos', '🌳'),
        ('resultados_volumetricos', 'Etapa 2 - Volumétricos', '📊'),
        ('inventario_processado', 'Etapa 3 - Inventário', '📈')
    ]

    etapas_concluidas = 0

    for state_key, nome_etapa, icone in etapas_info:
        try:
            resultado = getattr(st.session_state, state_key, None)

            if resultado is not None:
                st.sidebar.success(f"✅ **{nome_etapa}**")

                # Mostrar detalhes dos resultados
                if isinstance(resultado, dict):
                    melhor = resultado.get('melhor_modelo', 'N/A')
                    if melhor != 'N/A':
                        st.sidebar.caption(f"🏆 Melhor: {melhor}")

                    # Mostrar qualidade se disponível
                    if 'resultados' in resultado and melhor in resultado['resultados']:
                        r2 = resultado['resultados'][melhor].get('r2', resultado['resultados'][melhor].get('r2g', 0))
                        if r2 > 0:
                            st.sidebar.caption(f"📊 R²: {formatar_brasileiro(r2, 3)}")

                etapas_concluidas += 1
            else:
                st.sidebar.info(f"⏳ **{nome_etapa}**")

                # Mostrar dependências
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
            st.sidebar.balloons()
        elif etapas_concluidas >= 2:
            st.sidebar.info("🚀 Quase lá! Falta 1 etapa")


def mostrar_informacoes_e_acoes_sidebar():
    '''Mostra informações e ações rápidas na sidebar'''

    # Seção de informações
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Informações")

    st.sidebar.markdown('''
    **Formatos aceitos:**
    - CSV (separadores: ; , tab)
    - Excel (.xlsx, .xls, .xlsb)
    - Shapefile (.shp ou .zip)

    **Tamanho máximo:**
    - 200MB por arquivo

    **Processamento:**
    - ✅ Automático na sidebar
    - ✅ Validação em tempo real
    - ✅ Feedback imediato
    ''')

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
            if st.button("🔄 Limpar", use_container_width=True, key="limpar_resultados_sidebar"):
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
            if st.button("📊 Relatório", use_container_width=True, key="gerar_relatorio_rapido_sidebar"):
                st.switch_page("pages/3_📈_Inventário_Florestal.py")

        # Botão para reconfigurar sistema
        if st.sidebar.button("⚙️ Reconfigurar Sistema", use_container_width=True, key="reconfigurar_sistema_sidebar"):
            st.switch_page("pages/0_⚙️_Configurações.py")

        # Download rápido se inventário foi processado
        try:
            inventario_resultado = getattr(st.session_state, 'inventario_processado', None)

            if inventario_resultado is not None and isinstance(inventario_resultado, dict):
                if 'resumo_talhoes' in inventario_resultado:
                    resumo_df = inventario_resultado['resumo_talhoes']
                    csv_dados = resumo_df.to_csv(index=False, sep=';')

                    st.sidebar.download_button(
                        "📥 Download Resumo",
                        data=csv_dados,
                        file_name="resumo_inventario_rapido.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Download rápido do resumo por talhões"
                    )
        except Exception:
            pass

    # Mostrar dicas contextuais
    mostrar_dicas_contextuais_sidebar()


def mostrar_dicas_contextuais_sidebar():
    '''Dicas contextuais baseadas no estado atual do sistema'''
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

    # Verificar etapas executadas
    hip_executado = hasattr(st.session_state,
                            'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None
    vol_executado = hasattr(st.session_state,
                            'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None
    inv_executado = hasattr(st.session_state,
                            'inventario_processado') and st.session_state.inventario_processado is not None

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
    elif not hip_executado and not vol_executado:
        st.sidebar.success('''
        **✅ Sistema Pronto:**
        Execute as Etapas 1, 2 e 3.

        **Configurações aplicam automaticamente:**
        - Filtros globais
        - Parâmetros não-lineares
        - Validações automáticas
        ''')
    elif hip_executado and vol_executado and not inv_executado:
        st.sidebar.info('''
        **🎯 Finalize:**
        Execute a Etapa 3 para gerar o inventário final com relatórios completos.
        ''')
    elif inv_executado:
        st.sidebar.success('''
        **🎉 Análise Completa:**
        Todos os modelos foram executados!

        **Disponível:**
        - Relatórios completos
        - Downloads organizados
        - Gráficos detalhados
        ''')

    # Informações sobre arquivos opcionais
    with st.sidebar.expander("📁 Arquivos Opcionais"):
        st.markdown('''
        **Shapefile:**
        - Upload na sidebar
        - Fica persistente na sessão
        - Habilita método avançado de área

        **Coordenadas:**
        - Upload na sidebar  
        - Fica persistente na sessão
        - Cálculo preciso de áreas por GPS

        **Vantagem:** Carregue uma vez e navegue livremente entre as páginas!
        ''')


def mostrar_metricas_rapidas_sidebar():
    '''Mostra métricas rápidas dos dados carregados'''
    if hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
        df_inv = st.session_state.dados_inventario

        with st.sidebar.expander("📊 Métricas Rápidas"):
            col1, col2 = st.sidebar.columns(2)

            with col1:
                st.metric("Registros", f"{len(df_inv):,}")
                st.metric("Talhões", df_inv['talhao'].nunique())

            with col2:
                dap_medio = df_inv['D_cm'].mean()
                altura_media = df_inv['H_m'].mean()
                st.metric("DAP Médio", f"{formatar_brasileiro(dap_medio, 1)} cm")
                st.metric("Alt. Média", f"{formatar_brasileiro(altura_media, 1)} m")

            # Gráfico de distribuição simples
            if st.checkbox("📈 Distribuições", key="show_dist_sidebar"):
                st.write("**DAP (cm):**")
                st.bar_chart(df_inv['D_cm'].value_counts().head(10))


def criar_sidebar_melhorada():
    '''
    Versão melhorada da sidebar com processamento automático e feedback completo

    Returns:
        dict: Dicionário com os arquivos carregados e processados
    '''
    try:
        # Criar sidebar principal com processamento automático
        arquivos = criar_sidebar()

        # Mostrar métricas rápidas se dados estão carregados
        mostrar_metricas_rapidas_sidebar()

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


# Funções de compatibilidade para manter código existente funcionando
def mostrar_status_configuracao_sidebar_compat():
    '''Função de compatibilidade para o código existente'''
    return mostrar_status_configuracao_sidebar()


def criar_sidebar_compat():
    '''Função de compatibilidade que mantém interface original'''
    return criar_sidebar_melhorada()


# Função utilitária para verificar status geral do sistema
def obter_status_sistema_completo():
    '''
    Obtém status completo do sistema para uso em outras páginas

    Returns:
        dict: Status completo do sistema
    '''
    status = {
        'dados_inventario': hasattr(st.session_state,
                                    'dados_inventario') and st.session_state.dados_inventario is not None,
        'dados_cubagem': hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None,
        'configurado': False,
        'hip_executado': hasattr(st.session_state,
                                 'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None,
        'vol_executado': hasattr(st.session_state,
                                 'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None,
        'inv_executado': hasattr(st.session_state,
                                 'inventario_processado') and st.session_state.inventario_processado is not None,
        'shapefile_disponivel': hasattr(st.session_state,
                                        'arquivo_shapefile') and st.session_state.arquivo_shapefile is not None,
        'coordenadas_disponiveis': hasattr(st.session_state,
                                           'arquivo_coordenadas') and st.session_state.arquivo_coordenadas is not None
    }

    # Verificar configuração
    try:
        from config.configuracoes_globais import obter_configuracao_global
        config_global = obter_configuracao_global()
        status['configurado'] = config_global.get('configurado', False)
    except:
        pass

    # Calcular progresso geral
    etapas_base = [status['dados_inventario'] and status['dados_cubagem'], status['configurado']]
    etapas_analise = [status['hip_executado'], status['vol_executado'], status['inv_executado']]

    status['progresso_base'] = sum(etapas_base) / len(etapas_base)
    status['progresso_analise'] = sum(etapas_analise) / len(etapas_analise)
    status['progresso_total'] = (sum(etapas_base) + sum(etapas_analise)) / (len(etapas_base) + len(etapas_analise))

    return status