# ui/sidebar.py - VERSÃO COMPLETA COM LAS/LAZ
'''
Interface da barra lateral para upload de arquivos - Versão completa
INCLUI: Upload LAS/LAZ, persistência de sessão, processamento automático
'''

import streamlit as st
import pandas as pd
import traceback

# Importar processadores
from utils.arquivo_handler import carregar_arquivo_seguro, validar_estrutura_arquivo
from utils.formatacao import formatar_brasileiro, formatar_numero_inteligente

# Importar processador LAS se disponível
try:
    from processors.las_processor_integrado import (
        ProcessadorLASIntegrado,
        integrar_com_pagina_lidar
    )

    PROCESSAMENTO_LAS_DISPONIVEL = True
except ImportError:
    PROCESSAMENTO_LAS_DISPONIVEL = False


def verificar_disponibilidade_las():
    """Verifica se processamento LAS está disponível"""
    if not PROCESSAMENTO_LAS_DISPONIVEL:
        return False

    try:
        processador = ProcessadorLASIntegrado()
        disponivel, _ = processador.verificar_disponibilidade()
        return disponivel
    except:
        return False


def processar_dados_inventario_sidebar(arquivo_inventario):
    """
    Processa dados do inventário com persistência garantida - VERSÃO CORRIGIDA
    """
    try:
        if arquivo_inventario is None:
            return None

        # Usar função segura de carregamento
        df_inventario = carregar_arquivo_seguro(arquivo_inventario, "inventário")

        if df_inventario is None:
            st.sidebar.error("❌ Falha no carregamento")
            return None

        # Validar estrutura básica
        colunas_obrigatorias = ['D_cm', 'H_m', 'talhao', 'parcela']
        validacao = validar_estrutura_arquivo(df_inventario, colunas_obrigatorias, "inventário")

        if not validacao['valido']:
            st.sidebar.error("❌ Estrutura inválida")
            for erro in validacao['erros'][:2]:
                st.sidebar.error(f"• {erro}")
            return None

        # Limpar e validar dados
        df_limpo = limpar_dados_inventario_sidebar(df_inventario)

        if len(df_limpo) == 0:
            st.sidebar.error("❌ Sem dados válidos")
            return None

        # CORREÇÃO PRINCIPAL: Salvar de forma mais robusta
        try:
            # 1. Fazer cópia profunda
            df_para_salvar = df_limpo.copy(deep=True)

            # 2. Limpar dados anteriores
            if hasattr(st.session_state, 'dados_inventario'):
                del st.session_state.dados_inventario

            # 3. Salvar dados principais
            st.session_state.dados_inventario = df_para_salvar

            # 4. Salvar flags de controle
            st.session_state.arquivos_carregados = True
            st.session_state.timestamp_carregamento_inventario = pd.Timestamp.now()

            # 5. VERIFICAR SE SALVAMENTO FOI BEM-SUCEDIDO
            if (hasattr(st.session_state, 'dados_inventario') and
                    st.session_state.dados_inventario is not None and
                    len(st.session_state.dados_inventario) == len(df_limpo)):

                # Feedback de sucesso
                percentual_mantido = (len(df_limpo) / len(df_inventario)) * 100
                st.sidebar.success(f"✅ Inventário Persistido!")
                st.sidebar.info(f"📊 {len(df_limpo):,} registros ({percentual_mantido:.1f}%)")
                st.sidebar.info(f"🌳 {df_limpo['talhao'].nunique()} talhões")

                return df_limpo
            else:
                st.sidebar.error("❌ Erro: Dados não persistiram")
                return None

        except Exception as e:
            st.sidebar.error(f"❌ Erro ao persistir: {str(e)[:30]}...")
            return None

    except Exception as e:
        st.sidebar.error(f"❌ Erro: {str(e)[:50]}...")
        if st.sidebar.button("🔍 Debug", key="debug_inventario"):
            st.sidebar.code(str(e))
        return None


def processar_dados_cubagem_sidebar(arquivo_cubagem):
    """
    Processa dados de cubagem com persistência garantida - VERSÃO CORRIGIDA
    """
    try:
        if arquivo_cubagem is None:
            return None

        # Usar função segura de carregamento
        df_cubagem = carregar_arquivo_seguro(arquivo_cubagem, "cubagem")

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

        # CORREÇÃO: Salvar de forma mais robusta
        try:
            # 1. Fazer cópia profunda
            df_para_salvar = df_limpo.copy(deep=True)

            # 2. Limpar dados anteriores
            if hasattr(st.session_state, 'dados_cubagem'):
                del st.session_state.dados_cubagem

            # 3. Salvar dados principais
            st.session_state.dados_cubagem = df_para_salvar

            # 4. Salvar flags de controle
            st.session_state.cubagem_carregada = True
            st.session_state.timestamp_carregamento_cubagem = pd.Timestamp.now()

            # 5. VERIFICAR SALVAMENTO
            if (hasattr(st.session_state, 'dados_cubagem') and
                    st.session_state.dados_cubagem is not None and
                    len(st.session_state.dados_cubagem) == len(df_limpo)):

                # Feedback de sucesso
                arvores_cubadas = df_limpo['arv'].nunique()
                st.sidebar.success(f"✅ Cubagem Persistida!")
                st.sidebar.info(f"📏 {arvores_cubadas} árvores cubadas")
                st.sidebar.info(f"📊 {len(df_limpo):,} seções")

                return df_limpo
            else:
                st.sidebar.error("❌ Erro: Cubagem não persistiu")
                return None

        except Exception as e:
            st.sidebar.error(f"❌ Erro ao persistir cubagem: {str(e)[:30]}...")
            return None

    except Exception as e:
        st.sidebar.error(f"❌ Erro: {str(e)[:50]}...")
        if st.sidebar.button("🔍 Debug", key="debug_cubagem"):
            st.sidebar.code(str(e))
        return None


def processar_arquivo_las_sidebar(arquivo_las):
    """
    Processa arquivo LAS/LAZ na sidebar (preview apenas)

    Args:
        arquivo_las: Arquivo LAS/LAZ carregado

    Returns:
        bool: True se arquivo foi validado
    """
    try:
        if arquivo_las is None:
            return False

        # Validações básicas
        tamanho_mb = arquivo_las.size / (1024 * 1024)
        nome_arquivo = arquivo_las.name.lower()

        # Verificar extensão
        if not (nome_arquivo.endswith('.las') or nome_arquivo.endswith('.laz')):
            st.sidebar.error("❌ Formato inválido")
            st.sidebar.caption("Apenas arquivos .las ou .laz")
            return False

        # Verificar tamanho
        if tamanho_mb > 500:
            st.sidebar.error("❌ Arquivo muito grande")
            st.sidebar.caption(f"{tamanho_mb:.1f}MB (máx: 500MB)")
            return False

        # Feedback de sucesso
        st.sidebar.success("✅ Arquivo LAS válido")
        st.sidebar.info(f"📁 {arquivo_las.name}")
        st.sidebar.caption(f"💾 {tamanho_mb:.1f} MB")

        return True

    except Exception as e:
        st.sidebar.error(f"❌ Erro na validação LAS")
        return False


def limpar_dados_inventario_sidebar(df_inventario):
    """Versão otimizada da limpeza para sidebar"""
    if not isinstance(df_inventario, pd.DataFrame):
        st.sidebar.error("❌ Dados de inventário inválidos")
        return pd.DataFrame()

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

    except Exception as e:
        st.sidebar.warning(f"⚠️ Problema na conversão: {str(e)[:30]}...")

    # Filtros básicos de qualidade
    try:
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
        # DAP entre 1 e 100 cm (limites muito amplos)
        df_limpo = df_limpo[(df_limpo['D_cm'] >= 1) & (df_limpo['D_cm'] <= 100)]

        # Altura entre 1.3 e 60 m (limites muito amplos)
        df_limpo = df_limpo[(df_limpo['H_m'] >= 1.3) & (df_limpo['H_m'] <= 60)]

    except Exception as e:
        st.sidebar.warning(f"⚠️ Problema na filtragem: {str(e)[:30]}...")

    return df_limpo


def limpar_dados_cubagem_sidebar(df_cubagem):
    """Versão otimizada da limpeza para sidebar"""
    if not isinstance(df_cubagem, pd.DataFrame):
        st.sidebar.error("❌ Dados de cubagem inválidos")
        return pd.DataFrame()

    df_limpo = df_cubagem.copy()

    # Converter tipos básicos
    try:
        df_limpo['arv'] = pd.to_numeric(df_limpo['arv'], errors='coerce').astype('Int64')
        df_limpo['talhao'] = pd.to_numeric(df_limpo['talhao'], errors='coerce').astype('Int64')
        df_limpo['d_cm'] = pd.to_numeric(df_limpo['d_cm'], errors='coerce')
        df_limpo['h_m'] = pd.to_numeric(df_limpo['h_m'], errors='coerce')
        df_limpo['D_cm'] = pd.to_numeric(df_limpo['D_cm'], errors='coerce')
        df_limpo['H_m'] = pd.to_numeric(df_limpo['H_m'], errors='coerce')

    except Exception as e:
        st.sidebar.warning(f"⚠️ Problema na conversão: {str(e)[:30]}...")

    # Filtros básicos de qualidade
    try:
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
        mask_consistente = df_limpo['d_cm'] <= df_limpo['D_cm'] * 1.5  # Tolerância ampla
        df_limpo = df_limpo[mask_consistente]

    except Exception as e:
        st.sidebar.warning(f"⚠️ Problema na filtragem: {str(e)[:30]}...")

    return df_limpo


def criar_sidebar():
    '''
    Cria a interface da barra lateral com uploads e processamento automático
    VERSÃO COMPLETA: Inclui upload LAS/LAZ com persistência

    Returns:
        dict: Dicionário com os arquivos carregados e processados
    '''
    st.sidebar.header("📁 Upload de Dados")

    mostrar_debug_persistencia_sidebar()

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

    # === SEÇÃO LAS/LAZ ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛩️ Dados LiDAR")

    # Verificar disponibilidade do processamento LAS
    las_disponivel = verificar_disponibilidade_las()

    if las_disponivel:
        # Upload de arquivo LAS/LAZ COM PERSISTÊNCIA
        arquivo_las = st.sidebar.file_uploader(
            "🛩️ Arquivo LAS/LAZ",
            type=['las', 'laz'],
            help="Dados LiDAR brutos (máximo 500MB)",
            key="upload_las_persistente"
        )

        # Gerenciar persistência do arquivo LAS
        if arquivo_las is not None:
            # Validar arquivo
            if processar_arquivo_las_sidebar(arquivo_las):
                st.session_state.arquivo_las = arquivo_las
                st.sidebar.success("✅ LAS/LAZ salvo na sessão")
            else:
                # Limpar arquivo inválido
                if hasattr(st.session_state, 'arquivo_las'):
                    delattr(st.session_state, 'arquivo_las')
        elif not hasattr(st.session_state, 'arquivo_las'):
            st.session_state.arquivo_las = None

        # Mostrar status do arquivo LAS persistente
        if hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None:
            arquivo_las_ativo = st.session_state.arquivo_las
            st.sidebar.info("📁 LAS/LAZ ativo")
            st.sidebar.caption(f"📄 {arquivo_las_ativo.name}")

            # Botão para limpar arquivo LAS
            if st.sidebar.button("🗑️ Remover LAS", key="remove_las"):
                delattr(st.session_state, 'arquivo_las')
                st.sidebar.success("🗑️ Arquivo LAS removido!")
                st.rerun()
        else:
            st.sidebar.warning("🛩️ Sem arquivo LAS")

    else:
        st.sidebar.warning("⚠️ Processamento LAS indisponível")
        st.sidebar.caption("Instale: pip install laspy geopandas")
        if st.sidebar.button("📋 Ver Instruções", key="instrucoes_las"):
            with st.sidebar.expander("📦 Instalação LAS", expanded=True):
                st.code("""
pip install laspy[lazrs,laszip]
pip install geopandas
pip install shapely
pip install scipy
                """)

    # Upload de métricas LiDAR processadas COM PERSISTÊNCIA
    arquivo_metricas_lidar = st.sidebar.file_uploader(
        "📊 Métricas LiDAR (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Métricas já processadas do LiDAR",
        key="upload_metricas_lidar_persistente"
    )

    # Gerenciar persistência das métricas LiDAR
    if arquivo_metricas_lidar is not None:
        st.session_state.arquivo_metricas_lidar = arquivo_metricas_lidar
        st.sidebar.success("✅ Métricas LiDAR salvas")
        st.sidebar.caption(f"📄 {arquivo_metricas_lidar.name}")
    elif not hasattr(st.session_state, 'arquivo_metricas_lidar'):
        st.session_state.arquivo_metricas_lidar = None

    # === SEÇÃO ARQUIVOS OPCIONAIS ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Arquivos Opcionais")

    # Upload opcional de shapefile para áreas COM PERSISTÊNCIA
    arquivo_shapefile = st.sidebar.file_uploader(
        "🗺️ Shapefile Áreas",
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
        "📍 Coordenadas Parcelas",
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

    # === PROCESSAMENTO AUTOMÁTICO DOS DADOS ===
    dados_processados = {
        'inventario': None,
        'cubagem': None,
        'las': arquivo_las if las_disponivel and 'arquivo_las' else st.session_state.get('arquivo_las'),
        'metricas_lidar': arquivo_metricas_lidar if arquivo_metricas_lidar else st.session_state.get(
            'arquivo_metricas_lidar'),
        'shapefile': arquivo_shapefile if arquivo_shapefile else st.session_state.get('arquivo_shapefile'),
        'coordenadas': arquivo_coordenadas if arquivo_coordenadas else st.session_state.get('arquivo_coordenadas')
    }

    # Processar inventário se carregado
    if arquivo_inventario is not None:
        with st.sidebar.expander("🔄 Processando Inventário..."):
            dados_processados['inventario'] = processar_dados_inventario_sidebar(arquivo_inventario)
            # REMOVER esta linha - o salvamento já é feito dentro da função
            # if dados_processados['inventario'] is not None:
            #     st.session_state.dados_inventario = dados_processados['inventario']


    # Processar cubagem se carregada
    if arquivo_cubagem is not None:
        with st.sidebar.expander("🔄 Processando Cubagem..."):
            dados_processados['cubagem'] = processar_dados_cubagem_sidebar(arquivo_cubagem)
            # REMOVER esta linha - o salvamento já é feito dentro da função
            # if dados_processados['cubagem'] is not None:
            #     st.session_state.dados_cubagem = dados_processados['cubagem']

    # Mostrar status dos arquivos
    mostrar_status_arquivos_completo(dados_processados)

    # Mostrar status das configurações globais na sidebar
    mostrar_status_configuracao_sidebar()

    # Mostrar progresso das etapas na sidebar
    mostrar_progresso_etapas_sidebar()

    # Mostrar informações adicionais e ações rápidas
    mostrar_informacoes_e_acoes_sidebar()

    # VERIFICAÇÃO FINAL DE PERSISTÊNCIA
    if dados_processados['inventario'] is not None or dados_processados['cubagem'] is not None:
        # Verificar se realmente foram salvos no session_state
        inventario_ok = (hasattr(st.session_state, 'dados_inventario') and
                         st.session_state.dados_inventario is not None and
                         len(st.session_state.dados_inventario) > 0)

        cubagem_ok = (hasattr(st.session_state, 'dados_cubagem') and
                      st.session_state.dados_cubagem is not None and
                      len(st.session_state.dados_cubagem) > 0)

        if inventario_ok and cubagem_ok:
            st.sidebar.success("🎉 Dados Totalmente Persistidos!")
            st.sidebar.info("✅ Pode navegar livremente")
        elif inventario_ok:
            st.sidebar.info("✅ Inventário persistiu - falta cubagem")
        elif cubagem_ok:
            st.sidebar.info("✅ Cubagem persistiu - falta inventário")
        else:
            st.sidebar.warning("⚠️ Problemas na persistência detectados")
            st.sidebar.caption("Use debug para investigar")

    return dados_processados


def mostrar_status_arquivos_completo(arquivos):
    '''
    Mostra status detalhado dos arquivos carregados e processados
    VERSÃO COMPLETA: Inclui status LAS/LAZ
    '''
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status dos Dados")

    # === DADOS PRINCIPAIS ===
    # Inventário
    if arquivos['inventario'] is not None:
        st.sidebar.success("✅ Inventário processado")

        df_inv = arquivos['inventario']
        st.sidebar.info(f"📊 {len(df_inv):,} registros")
        st.sidebar.info(f"🌳 {df_inv['talhao'].nunique()} talhões")

        # Estatísticas rápidas
        try:
            dap_medio = df_inv['D_cm'].mean()
            altura_media = df_inv['H_m'].mean()
            st.sidebar.caption(f"DAP: {formatar_brasileiro(dap_medio, 1)} cm")
            st.sidebar.caption(f"Altura: {formatar_brasileiro(altura_media, 1)} m")
        except Exception:
            st.sidebar.caption("Estatísticas indisponíveis")


    elif hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:

        st.sidebar.success("✅ Inventário persistido")

        try:

            df_inv = st.session_state.dados_inventario

            if isinstance(df_inv, pd.DataFrame) and len(df_inv) > 0:

                st.sidebar.caption(f"📊 {len(df_inv):,} registros")

                st.sidebar.caption(f"🌳 {df_inv['talhao'].nunique()} talhões")

                # Mostrar timestamp se disponível

                if hasattr(st.session_state, 'timestamp_carregamento_inventario'):

                    timestamp = st.session_state.timestamp_carregamento_inventario

                    tempo_decorrido = pd.Timestamp.now() - timestamp

                    if tempo_decorrido.total_seconds() < 3600:  # Menos de 1 hora

                        minutos = int(tempo_decorrido.total_seconds() / 60)

                        st.sidebar.caption(f"⏰ Há {minutos} min")

            else:

                st.sidebar.warning("⚠️ Inventário existe mas inválido")

        except Exception as e:

            st.sidebar.error(f"❌ Erro no inventário: {str(e)[:20]}...")

    else:
        st.sidebar.error("❌ Inventário necessário")

    # Cubagem
    if arquivos['cubagem'] is not None:
        st.sidebar.success("✅ Cubagem processada")

        df_cub = arquivos['cubagem']
        try:
            arvores = df_cub['arv'].nunique()
            secoes_media = df_cub.groupby(['talhao', 'arv']).size().mean()

            st.sidebar.info(f"📏 {arvores} árvores")
            st.sidebar.caption(f"Seções/árvore: {formatar_brasileiro(secoes_media, 1)}")
        except Exception:
            st.sidebar.info(f"📏 Dados processados")


    elif hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None:

        st.sidebar.success("✅ Cubagem persistida")

        try:

            df_cub = st.session_state.dados_cubagem

            if isinstance(df_cub, pd.DataFrame) and len(df_cub) > 0:

                arvores = df_cub['arv'].nunique()

                st.sidebar.caption(f"📏 {arvores} árvores")

                st.sidebar.caption(f"📊 {len(df_cub):,} seções")

                # Mostrar timestamp se disponível

                if hasattr(st.session_state, 'timestamp_carregamento_cubagem'):

                    timestamp = st.session_state.timestamp_carregamento_cubagem

                    tempo_decorrido = pd.Timestamp.now() - timestamp

                    if tempo_decorrido.total_seconds() < 3600:
                        minutos = int(tempo_decorrido.total_seconds() / 60)

                        st.sidebar.caption(f"⏰ Há {minutos} min")

            else:

                st.sidebar.warning("⚠️ Cubagem existe mas inválida")

        except Exception as e:

            st.sidebar.error(f"❌ Erro na cubagem: {str(e)[:20]}...")
    else:
        st.sidebar.error("❌ Cubagem necessária")

    # === DADOS LIDAR ===
    st.sidebar.markdown("**🛩️ Dados LiDAR:**")

    # Arquivo LAS/LAZ
    arquivo_las_ativo = arquivos['las']
    if arquivo_las_ativo is not None:
        st.sidebar.success("✅ Arquivo LAS/LAZ ativo")
        try:
            nome_arquivo = getattr(arquivo_las_ativo, 'name', 'arquivo.las')
            tamanho_mb = getattr(arquivo_las_ativo, 'size', 0) / (1024 * 1024)
            st.sidebar.caption(f"📄 {nome_arquivo}")
            st.sidebar.caption(f"💾 {tamanho_mb:.1f} MB")
        except Exception:
            st.sidebar.caption("📄 Arquivo LAS disponível")
    else:
        st.sidebar.warning("🛩️ Sem arquivo LAS/LAZ")

    # Métricas LiDAR
    arquivo_metricas_ativo = arquivos['metricas_lidar']
    if arquivo_metricas_ativo is not None:
        st.sidebar.success("✅ Métricas LiDAR ativas")
        try:
            nome_arquivo = getattr(arquivo_metricas_ativo, 'name', 'metricas.csv')
            st.sidebar.caption(f"📄 {nome_arquivo}")
        except Exception:
            st.sidebar.caption("📄 Métricas disponíveis")
    else:
        st.sidebar.info("📊 Sem métricas LiDAR")

    # === ARQUIVOS OPCIONAIS ===
    st.sidebar.markdown("**📁 Arquivos Opcionais:**")

    # Shapefile
    shapefile_ativo = arquivos['shapefile']
    if shapefile_ativo is not None:
        st.sidebar.info("🗺️ Shapefile ativo")
        try:
            nome_arquivo = getattr(shapefile_ativo, 'name', 'shapefile.zip')
            st.sidebar.caption(f"📄 {nome_arquivo}")
        except Exception:
            st.sidebar.caption("📄 Shapefile carregado")
    else:
        st.sidebar.warning("🗺️ Shapefile: Não carregado")

    # Coordenadas
    coordenadas_ativas = arquivos['coordenadas']
    if coordenadas_ativas is not None:
        st.sidebar.info("📍 Coordenadas ativas")
        try:
            nome_arquivo = getattr(coordenadas_ativas, 'name', 'coordenadas.csv')
            st.sidebar.caption(f"📄 {nome_arquivo}")
        except Exception:
            st.sidebar.caption("📄 Coordenadas carregadas")
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
                try:
                    if hasattr(timestamp, 'strftime'):
                        st.sidebar.caption(f"Atualizado: {timestamp.strftime('%H:%M')}")
                    else:
                        st.sidebar.caption("Configurado nesta sessão")
                except Exception:
                    st.sidebar.caption("Configurado nesta sessão")

            # Mostrar resumo das configurações principais
            with st.sidebar.expander("📋 Resumo Config"):
                try:
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

                except Exception as e:
                    st.write("❌ Erro ao exibir configurações")

        else:
            st.sidebar.warning("⚠️ Sistema Não Configurado")
            st.sidebar.caption("Configure na Etapa 0 primeiro")

            if st.sidebar.button("⚙️ Ir para Configurações", use_container_width=True, key="btn_config_sidebar"):
                st.switch_page("pages/0_⚙️_Configurações.py")

    except ImportError:
        st.sidebar.warning("⚠️ Configurações não disponíveis")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Erro nas configurações")


def verificar_parametros_customizados(config):
    '''Verifica se parâmetros não-lineares foram customizados'''
    try:
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
    except Exception:
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
        ('inventario_processado', 'Etapa 3 - Inventário', '📈'),
        (None, 'Etapa 4 - LiDAR', '🛩️')  # Etapa especial para LiDAR
    ]

    etapas_concluidas = 0

    for state_key, nome_etapa, icone in etapas_info:
        try:
            if state_key is None:  # Etapa LiDAR
                # Verificar se há dados LiDAR processados
                lidar_las = hasattr(st.session_state,
                                    'dados_lidar_las') and st.session_state.dados_lidar_las is not None
                lidar_metrics = hasattr(st.session_state, 'dados_lidar') and st.session_state.dados_lidar is not None

                if lidar_las or lidar_metrics:
                    st.sidebar.success(f"✅ **{nome_etapa}**")
                    if lidar_las:
                        st.sidebar.caption("🛩️ Processamento LAS")
                    if lidar_metrics:
                        st.sidebar.caption("📊 Métricas processadas")
                    etapas_concluidas += 0.5  # Conta como meia etapa (opcional)
                else:
                    st.sidebar.info(f"⏳ **{nome_etapa}** (Opcional)")
                continue

            resultado = getattr(st.session_state, state_key, None)

            if resultado is not None:
                st.sidebar.success(f"✅ **{nome_etapa}**")

                # Mostrar detalhes dos resultados
                try:
                    if isinstance(resultado, dict):
                        melhor = resultado.get('melhor_modelo', 'N/A')
                        if melhor != 'N/A':
                            st.sidebar.caption(f"🏆 Melhor: {melhor}")

                        # Mostrar qualidade se disponível
                        if 'resultados' in resultado and melhor in resultado['resultados']:
                            r2 = resultado['resultados'][melhor].get('r2',
                                                                     resultado['resultados'][melhor].get('r2g', 0))
                            if r2 > 0:
                                st.sidebar.caption(f"📊 R²: {formatar_brasileiro(r2, 3)}")
                except Exception:
                    pass  # Não quebrar se não conseguir mostrar detalhes

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
    total_etapas = 3  # Não contar LiDAR como obrigatória
    if etapas_concluidas > 0:
        progresso = min(etapas_concluidas / total_etapas, 1.0)  # Máximo 100%
        st.sidebar.progress(progresso, text=f"Progresso: {int(etapas_concluidas)}/{total_etapas} etapas")

        if etapas_concluidas >= total_etapas:
            st.sidebar.success("🎉 Análise Completa!")
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
    - **LAS/LAZ (máx 500MB)**

    **Processamento:**
    - ✅ Automático na sidebar
    - ✅ Validação em tempo real
    - ✅ Persistência na sessão
    - ✅ Processamento LAS integrado
    ''')

    # Verificar se há resultados para mostrar ações
    tem_resultados = False
    try:
        resultados_disponiveis = [
            getattr(st.session_state, 'resultados_hipsometricos', None),
            getattr(st.session_state, 'resultados_volumetricos', None),
            getattr(st.session_state, 'inventario_processado', None),
            getattr(st.session_state, 'dados_lidar_las', None),
            getattr(st.session_state, 'dados_lidar', None)
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
                    'inventario_processado',
                    'dados_lidar_las',
                    'dados_lidar',
                    'calibracao_lidar'
                ]

                for key in keys_para_limpar:
                    if hasattr(st.session_state, key):
                        delattr(st.session_state, key)

                st.sidebar.success("✅ Resultados limpos!")
                st.rerun()

        with col2:
            if st.button("📊 Relatório", use_container_width=True, key="gerar_relatorio_rapido_sidebar"):
                st.switch_page("pages/3_📈_Inventário_Florestal.py")

        # Botão para LiDAR se há dados LAS
        if hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None:
            if st.sidebar.button("🛩️ Processar LAS", use_container_width=True, key="processar_las_sidebar"):
                st.switch_page("pages/4_🛩️_Dados_LiDAR.py")

        # Botão para reconfigurar sistema
        if st.sidebar.button("⚙️ Reconfigurar Sistema", use_container_width=True, key="reconfigurar_sistema_sidebar"):
            st.switch_page("pages/0_⚙️_Configurações.py")

        # Download rápido se inventário foi processado
        try:
            inventario_resultado = getattr(st.session_state, 'inventario_processado', None)

            if inventario_resultado is not None and isinstance(inventario_resultado, dict):
                if 'resumo_talhoes' in inventario_resultado:
                    resumo_df = inventario_resultado['resumo_talhoes']
                    if isinstance(resumo_df, pd.DataFrame):
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

    # Verificar dados LiDAR
    dados_lidar = (
            (hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None) or
            (hasattr(st.session_state,
                     'arquivo_metricas_lidar') and st.session_state.arquivo_metricas_lidar is not None)
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
        2. **OPCIONAL:** Carregue dados LiDAR
        3. Configure o sistema (Etapa 0)
        4. Execute as análises (Etapas 1-3)
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
    elif dados_lidar and not hip_executado:
        st.sidebar.success('''
        **🛩️ LiDAR Detectado:**
        Execute primeiro as Etapas 1-3, depois use os dados LiDAR na Etapa 4 para:
        - Validar modelos
        - Calibrar equações
        - Mapear estrutura florestal
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
    elif inv_executado and dados_lidar:
        st.sidebar.success('''
        **🎉 Análise Completa + LiDAR:**
        Tudo pronto! Agora você pode:
        - Usar Etapa 4 para dados LiDAR
        - Validar com sensoriamento remoto
        - Gerar relatórios integrados
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

    # Informações sobre arquivos LiDAR
    if dados_lidar:
        with st.sidebar.expander("🛩️ Dados LiDAR Disponíveis"):
            if hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None:
                st.markdown('''
                **📁 Arquivo LAS/LAZ:**
                - Processamento direto no sistema
                - Extração automática de métricas
                - Integração com inventário
                - Análise estrutural avançada
                ''')

            if hasattr(st.session_state,
                       'arquivo_metricas_lidar') and st.session_state.arquivo_metricas_lidar is not None:
                st.markdown('''
                **📊 Métricas LiDAR:**
                - Dados pré-processados
                - Integração direta
                - Comparação campo vs remoto
                - Calibração de modelos
                ''')

    # Informações sobre arquivos opcionais
    with st.sidebar.expander("📁 Arquivos Opcionais"):
        st.markdown('''
        **Shapefile/Coordenadas:**
        - Upload na sidebar
        - Fica persistente na sessão
        - Habilita métodos avançados de área
        - Navegue livremente entre páginas

        **Dados LiDAR:**
        - LAS/LAZ: Processamento completo
        - Métricas CSV: Integração rápida
        - Ambos persistem na sessão
        - Análise na Etapa 4
        ''')


def mostrar_metricas_rapidas_sidebar():
    '''Mostra métricas rápidas dos dados carregados incluindo LiDAR'''
    if hasattr(st.session_state, 'dados_inventario') and st.session_state.dados_inventario is not None:
        try:
            df_inv = st.session_state.dados_inventario

            if not isinstance(df_inv, pd.DataFrame):
                return

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

                # Informações LiDAR se disponível
                if hasattr(st.session_state, 'arquivo_las') and st.session_state.arquivo_las is not None:
                    st.info("🛩️ LAS/LAZ disponível para processamento")

                if hasattr(st.session_state, 'dados_lidar_las') and st.session_state.dados_lidar_las is not None:
                    dados_las = st.session_state.dados_lidar_las
                    if 'df_metricas' in dados_las:
                        parcelas_lidar = len(dados_las['df_metricas'])
                        st.success(f"✅ {parcelas_lidar} parcelas LiDAR processadas")

                # Gráfico de distribuição simples
                if st.checkbox("📈 Distribuições", key="show_dist_sidebar"):
                    st.write("**DAP (cm):**")
                    st.bar_chart(df_inv['D_cm'].value_counts().head(10))

        except Exception as e:
            st.sidebar.caption("⚠️ Erro nas métricas")


def limpar_dados_lidar_sidebar():
    '''Limpa dados LiDAR da sessão'''
    keys_lidar = [
        'arquivo_las',
        'arquivo_metricas_lidar',
        'dados_lidar_las',
        'dados_lidar',
        'calibracao_lidar'
    ]

    for key in keys_lidar:
        if hasattr(st.session_state, key):
            delattr(st.session_state, key)

    st.sidebar.success("🗑️ Dados LiDAR limpos!")


def criar_sidebar_melhorada():
    '''
    Versão melhorada da sidebar com processamento automático e feedback completo
    VERSÃO COMPLETA: Inclui upload LAS/LAZ, persistência total, tratamento robusto

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
        st.sidebar.error(f"❌ Erro na sidebar")
        # Debug apenas se solicitado
        if st.sidebar.button("🔍 Ver Erro", key="debug_sidebar"):
            st.sidebar.code(str(e))

        # Retornar estrutura mínima em caso de erro
        return {
            'inventario': None,
            'cubagem': None,
            'las': None,
            'metricas_lidar': None,
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
    VERSÃO COMPLETA: Inclui status LiDAR

    Returns:
        dict: Status completo do sistema
    '''
    try:
        status = {
            # Dados principais
            'dados_inventario': hasattr(st.session_state,
                                        'dados_inventario') and st.session_state.dados_inventario is not None,
            'dados_cubagem': hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None,

            # Configuração
            'configurado': False,

            # Etapas principais
            'hip_executado': hasattr(st.session_state,
                                     'resultados_hipsometricos') and st.session_state.resultados_hipsometricos is not None,
            'vol_executado': hasattr(st.session_state,
                                     'resultados_volumetricos') and st.session_state.resultados_volumetricos is not None,
            'inv_executado': hasattr(st.session_state,
                                     'inventario_processado') and st.session_state.inventario_processado is not None,

            # Dados LiDAR
            'arquivo_las_disponivel': hasattr(st.session_state,
                                              'arquivo_las') and st.session_state.arquivo_las is not None,
            'metricas_lidar_disponivel': hasattr(st.session_state,
                                                 'arquivo_metricas_lidar') and st.session_state.arquivo_metricas_lidar is not None,
            'dados_lidar_processados': hasattr(st.session_state,
                                               'dados_lidar_las') and st.session_state.dados_lidar_las is not None,
            'dados_lidar_integrados': hasattr(st.session_state,
                                              'dados_lidar') and st.session_state.dados_lidar is not None,

            # Arquivos opcionais
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

        # LiDAR como etapa opcional
        etapa_lidar = status['dados_lidar_processados'] or status['dados_lidar_integrados']

        status['progresso_base'] = sum(etapas_base) / len(etapas_base)
        status['progresso_analise'] = sum(etapas_analise) / len(etapas_analise)
        status['progresso_total'] = (sum(etapas_base) + sum(etapas_analise)) / (len(etapas_base) + len(etapas_analise))

        # Progresso com LiDAR (bônus)
        if etapa_lidar:
            status['progresso_completo'] = min(status['progresso_total'] + 0.1, 1.0)  # Bônus de 10%
        else:
            status['progresso_completo'] = status['progresso_total']

        return status

    except Exception:
        # Retornar status básico em caso de erro
        return {
            'dados_inventario': False,
            'dados_cubagem': False,
            'configurado': False,
            'hip_executado': False,
            'vol_executado': False,
            'inv_executado': False,
            'arquivo_las_disponivel': False,
            'metricas_lidar_disponivel': False,
            'dados_lidar_processados': False,
            'dados_lidar_integrados': False,
            'shapefile_disponivel': False,
            'coordenadas_disponiveis': False,
            'progresso_base': 0,
            'progresso_analise': 0,
            'progresso_total': 0,
            'progresso_completo': 0
        }


def mostrar_debug_persistencia_sidebar():
    """
    NOVA FUNÇÃO: Debug da persistência dos dados na sidebar
    """
    if st.sidebar.checkbox("🔍 Debug Persistência", key="debug_persistencia"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Status Persistência")

        # Verificar dados de inventário
        if hasattr(st.session_state, 'dados_inventario'):
            dados = st.session_state.dados_inventario
            if dados is not None and len(dados) > 0:
                st.sidebar.success(f"✅ Inventário: {len(dados)} reg")
                st.sidebar.caption(f"Talhões: {dados['talhao'].nunique()}")
            else:
                st.sidebar.error("❌ Inventário vazio")
        else:
            st.sidebar.error("❌ Inventário: não existe")

        # Verificar dados de cubagem
        if hasattr(st.session_state, 'dados_cubagem'):
            dados = st.session_state.dados_cubagem
            if dados is not None and len(dados) > 0:
                st.sidebar.success(f"✅ Cubagem: {dados['arv'].nunique()} árv")
                st.sidebar.caption(f"Seções: {len(dados)}")
            else:
                st.sidebar.error("❌ Cubagem vazia")
        else:
            st.sidebar.error("❌ Cubagem: não existe")

        # Verificar flags
        if hasattr(st.session_state, 'arquivos_carregados'):
            if st.session_state.arquivos_carregados:
                st.sidebar.success("✅ Flag ativa")
            else:
                st.sidebar.warning("⚠️ Flag False")
        else:
            st.sidebar.error("❌ Flag não existe")

        # Timestamps
        if hasattr(st.session_state, 'timestamp_carregamento_inventario'):
            timestamp = st.session_state.timestamp_carregamento_inventario
            tempo_decorrido = pd.Timestamp.now() - timestamp
            minutos = int(tempo_decorrido.total_seconds() / 60)
            st.sidebar.caption(f"⏰ Inventário: há {minutos}min")

        # Botão para limpar e recarregar
        if st.sidebar.button("🔄 Forçar Recarregamento"):
            # Limpar session_state
            keys_para_limpar = ['dados_inventario', 'dados_cubagem', 'arquivos_carregados']
            for key in keys_para_limpar:
                if hasattr(st.session_state, key):
                    delattr(st.session_state, key)
            st.sidebar.success("🗑️ Session state limpo - recarregue arquivos")
            st.rerun()


def teste_persistencia_sidebar():
    """
    FUNÇÃO TEMPORÁRIA - Adicione no final do sidebar.py para testar
    """
    if st.sidebar.button("🧪 Teste Rápido Persistência"):
        st.sidebar.write("**Teste de Persistência:**")

        # Testar inventário
        if hasattr(st.session_state, 'dados_inventario'):
            dados = st.session_state.dados_inventario
            if dados is not None and len(dados) > 0:
                st.sidebar.success(f"✅ Inventário: {len(dados)} registros")

                # Testar acesso às colunas
                try:
                    talhoes = dados['talhao'].nunique()
                    dap_medio = dados['D_cm'].mean()
                    st.sidebar.success(f"✅ Acesso OK: {talhoes} talhões, DAP {dap_medio:.1f}")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro acesso: {e}")
            else:
                st.sidebar.error("❌ Inventário existe mas está vazio/inválido")
        else:
            st.sidebar.error("❌ dados_inventario não existe")

        # Testar cubagem
        if hasattr(st.session_state, 'dados_cubagem'):
            dados = st.session_state.dados_cubagem
            if dados is not None and len(dados) > 0:
                st.sidebar.success(f"✅ Cubagem: {len(dados)} seções")

                try:
                    arvores = dados['arv'].nunique()
                    st.sidebar.success(f"✅ Acesso OK: {arvores} árvores")
                except Exception as e:
                    st.sidebar.error(f"❌ Erro acesso: {e}")
            else:
                st.sidebar.error("❌ Cubagem existe mas está vazia/inválida")
        else:
            st.sidebar.error("❌ dados_cubagem não existe")

        # Mostrar todas as keys relevantes
        st.sidebar.write("**Keys relevantes:**")
        keys_relevantes = [k for k in st.session_state.keys()
                           if any(termo in k.lower() for termo in ['dados', 'inventario', 'cubagem', 'arquivo'])]

        for key in keys_relevantes:
            valor = st.session_state[key]
            if hasattr(valor, '__len__'):
                st.sidebar.caption(f"• {key}: {type(valor).__name__} ({len(valor)})")
            else:
                st.sidebar.caption(f"• {key}: {type(valor).__name__}")

teste_persistencia_sidebar()