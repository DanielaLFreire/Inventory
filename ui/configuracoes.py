# ui/configuracoes.py
'''
Interface para configurações do sistema
'''

import streamlit as st
import numpy as np
from config.config import DEFAULTS


def criar_configuracoes(df_inventario):
    '''
    Cria interface de configurações do sistema

    Args:
        df_inventario: DataFrame do inventário

    Returns:
        dict: Configurações selecionadas pelo usuário
    '''
    st.header("⚙️ Configurações")

    # Configurações principais
    config_principais = criar_configuracoes_principais(df_inventario)

    # Configurações de área
    config_areas = criar_configuracoes_areas(df_inventario)

    # Combinar todas as configurações
    config_completa = {**config_principais, **config_areas}

    # Mostrar resumo das configurações
    mostrar_resumo_configuracoes(config_completa, df_inventario)

    return config_completa


def criar_configuracoes_principais(df_inventario):
    '''
    Cria configurações principais de filtros

    Args:
        df_inventario: DataFrame do inventário

    Returns:
        dict: Configurações principais
    '''
    col1, col2, col3 = st.columns(3)

    with col1:
        talhoes_excluir = st.multiselect(
            "🚫 Talhões a excluir",
            options=sorted(df_inventario['talhao'].unique()),
            help="Ex: Pinus, áreas experimentais"
        )

    with col2:
        diametro_min = st.number_input(
            "📏 Diâmetro mínimo (cm)",
            min_value=0.0,
            max_value=20.0,
            value=DEFAULTS['diametro_min'],
            step=0.5,
            help="Diâmetro mínimo para inclusão na análise"
        )

    with col3:
        codigos_disponiveis = sorted(df_inventario['cod'].unique()) if 'cod' in df_inventario.columns else []
        codigos_excluir = st.multiselect(
            "🏷️ Códigos a excluir",
            options=codigos_disponiveis,
            default=[c for c in DEFAULTS['codigos_excluir'] if c in codigos_disponiveis],
            help="Códigos de árvores a excluir (C=Cortada, I=Invasora)"
        )

    return {
        'talhoes_excluir': talhoes_excluir,
        'diametro_min': diametro_min,
        'codigos_excluir': codigos_excluir
    }


def criar_configuracoes_areas(df_inventario):
    '''
    Cria configurações específicas para áreas

    Args:
        df_inventario: DataFrame do inventário

    Returns:
        dict: Configurações de área
    '''
    st.subheader("📏 Configurações de Área")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Determinar métodos disponíveis
        metodos_disponiveis = ["Simular automaticamente", "Valores informados manualmente"]

        # Verificar se existem arquivos opcionais carregados
        if 'arquivo_shapefile' in st.session_state and st.session_state.arquivo_shapefile is not None:
            metodos_disponiveis.append("Upload shapefile")

        if 'arquivo_coordenadas' in st.session_state and st.session_state.arquivo_coordenadas is not None:
            metodos_disponiveis.append("Coordenadas das parcelas")

        metodo_area = st.selectbox(
            "🗺️ Método para Área dos Talhões",
            options=metodos_disponiveis,
            help="Como calcular as áreas dos talhões"
        )

    with col2:
        area_parcela = st.number_input(
            "📐 Área da Parcela (m²)",
            min_value=100,
            max_value=2000,
            value=DEFAULTS['area_parcela'],
            step=100,
            help="Área padrão: 400m² (20x20m) ou 1000m² (círculo r=17.84m)"
        )

    with col3:
        if metodo_area == "Coordenadas das parcelas":
            raio_parcela = st.number_input(
                "📐 Raio da Parcela (m)",
                min_value=5.0,
                max_value=30.0,
                value=DEFAULTS['raio_parcela'],
                step=0.1,
                help="Raio para calcular área circular (11.28m = 400m²)"
            )
            area_calculada = np.pi * (raio_parcela ** 2)
            st.write(f"**Área calculada**: {area_calculada:.0f} m²")
        else:
            raio_parcela = DEFAULTS['raio_parcela']

    # Configurações específicas por método
    config_areas_especifica = processar_configuracao_area_especifica(metodo_area, df_inventario)

    return {
        'metodo_area': metodo_area,
        'area_parcela': area_parcela,
        'raio_parcela': raio_parcela,
        **config_areas_especifica
    }


def processar_configuracao_area_especifica(metodo_area, df_inventario):
    '''
    Processa configurações específicas para cada método de área

    Args:
        metodo_area: Método selecionado
        df_inventario: DataFrame do inventário

    Returns:
        dict: Configurações específicas
    '''
    config = {}

    if metodo_area == "Valores informados manualmente":
        config.update(criar_interface_areas_manuais(df_inventario))

    elif metodo_area == "Upload shapefile":
        st.success("📁 Shapefile carregado!")
        st.write("✅ Áreas serão extraídas automaticamente")
        config['usar_shapefile'] = True

    elif metodo_area == "Coordenadas das parcelas":
        st.success("📍 Coordenadas carregadas!")
        st.write("✅ Áreas serão calculadas automaticamente")
        config['usar_coordenadas'] = True

    else:  # Simular automaticamente
        config.update(criar_interface_simulacao_areas())

    return config


def criar_interface_areas_manuais(df_inventario):
    '''
    Cria interface para entrada manual de áreas

    Args:
        df_inventario: DataFrame do inventário

    Returns:
        dict: Áreas manuais configuradas
    '''
    st.write("**📝 Áreas por Talhão (hectares):**")

    talhoes_disponiveis = sorted(df_inventario['talhao'].unique())
    areas_manuais = {}

    # Criar interface em colunas para melhor organização
    n_colunas = min(4, len(talhoes_disponiveis))
    colunas = st.columns(n_colunas)

    for i, talhao in enumerate(talhoes_disponiveis):
        col_idx = i % n_colunas
        with colunas[col_idx]:
            areas_manuais[talhao] = st.number_input(
                f"Talhão {talhao}",
                min_value=0.1,
                max_value=1000.0,
                value=25.0,
                step=0.1,
                key=f"area_manual_talhao_{talhao}"
            )

    # Mostrar resumo das áreas manuais
    if areas_manuais:
        area_total = sum(areas_manuais.values())
        area_media = np.mean(list(areas_manuais.values()))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{area_total:.1f} ha")
        with col2:
            st.metric("Média", f"{area_media:.1f} ha")
        with col3:
            st.metric("Talhões", len(areas_manuais))

    return {'areas_manuais': areas_manuais}


def criar_interface_simulacao_areas():
    '''
    Cria interface para configuração da simulação de áreas

    Returns:
        dict: Configurações da simulação
    '''
    st.info("💡 **Simulação Automática de Áreas**")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Método de simulação:**")
        st.write("- Baseado no número de parcelas por talhão")
        st.write("- Cada parcela representa 2-5 hectares")
        st.write("- Variação aleatória realística aplicada")

    with col2:
        st.write("**Parâmetros:**")

        fator_expansao_min = st.slider(
            "Fator mínimo (ha/parcela)",
            min_value=1.0,
            max_value=10.0,
            value=2.5,
            step=0.5,
            help="Área mínima representada por parcela"
        )

        fator_expansao_max = st.slider(
            "Fator máximo (ha/parcela)",
            min_value=fator_expansao_min,
            max_value=15.0,
            value=4.0,
            step=0.5,
            help="Área máxima representada por parcela"
        )

        variacao_percentual = st.slider(
            "Variação aleatória (%)",
            min_value=0,
            max_value=50,
            value=20,
            step=5,
            help="Percentual de variação aleatória aplicada"
        )

    return {
        'simulacao_fator_min': fator_expansao_min,
        'simulacao_fator_max': fator_expansao_max,
        'simulacao_variacao': variacao_percentual / 100
    }


def mostrar_resumo_configuracoes(config, df_inventario):
    '''
    Mostra resumo das configurações selecionadas

    Args:
        config: Configurações completas
        df_inventario: DataFrame do inventário
    '''
    st.subheader("📋 Resumo das Configurações")

    # Estatísticas dos dados originais
    total_registros = len(df_inventario)
    total_talhoes = df_inventario['talhao'].nunique()
    total_parcelas = df_inventario['parcela'].nunique() if 'parcela' in df_inventario.columns else 0

    # Calcular impacto dos filtros de forma segura
    try:
        dados_filtrados = aplicar_preview_filtros(df_inventario, config)
        if dados_filtrados is not None and len(dados_filtrados) > 0:
            registros_apos_filtro = len(dados_filtrados)
            talhoes_apos_filtro = dados_filtrados['talhao'].nunique()
            parcelas_apos_filtro = dados_filtrados['parcela'].nunique() if 'parcela' in dados_filtrados.columns else 0
        else:
            # Fallback se filtros resultarem em dataset vazio
            registros_apos_filtro = 0
            talhoes_apos_filtro = 0
            parcelas_apos_filtro = 0
    except Exception as e:
        st.warning(f"⚠️ Erro ao aplicar preview dos filtros: {e}")
        # Usar dados originais como fallback
        registros_apos_filtro = total_registros
        talhoes_apos_filtro = total_talhoes
        parcelas_apos_filtro = total_parcelas

    # Mostrar estatísticas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_registros = registros_apos_filtro - total_registros
        st.metric(
            "Registros",
            f"{registros_apos_filtro:,}",
            delta=f"{delta_registros:+,}" if delta_registros != 0 else None,
            delta_color="normal"
        )

    with col2:
        delta_talhoes = talhoes_apos_filtro - total_talhoes
        st.metric(
            "Talhões",
            talhoes_apos_filtro,
            delta=f"{delta_talhoes:+}" if delta_talhoes != 0 else None,
            delta_color="normal"
        )

    with col3:
        if total_parcelas > 0:
            delta_parcelas = parcelas_apos_filtro - total_parcelas
            st.metric(
                "Parcelas",
                parcelas_apos_filtro,
                delta=f"{delta_parcelas:+}" if delta_parcelas != 0 else None,
                delta_color="normal"
            )
        else:
            st.metric("Parcelas", "N/A")

    with col4:
        if total_registros > 0:
            percentual_mantido = (registros_apos_filtro / total_registros) * 100
        else:
            percentual_mantido = 0
        st.metric("Dados Mantidos", f"{percentual_mantido:.1f}%")

    # Mostrar configurações aplicadas
    with st.expander("🔧 Detalhes das Configurações"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Filtros:**")
            st.write(f"- Diâmetro mínimo: {config.get('diametro_min', 'N/A')} cm")
            if config.get('talhoes_excluir'):
                st.write(f"- Talhões excluídos: {config['talhoes_excluir']}")
            if config.get('codigos_excluir'):
                st.write(f"- Códigos excluídos: {config['codigos_excluir']}")

        with col2:
            st.write("**Áreas:**")
            st.write(f"- Método: {config.get('metodo_area', 'N/A')}")
            st.write(f"- Área da parcela: {config.get('area_parcela', 'N/A')} m²")
            if config.get('metodo_area') == "Coordenadas das parcelas":
                st.write(f"- Raio da parcela: {config.get('raio_parcela', 'N/A')} m")


def aplicar_preview_filtros(df_inventario, config):
    '''
    Aplica filtros para preview sem modificar dados originais

    Args:
        df_inventario: DataFrame original
        config: Configurações

    Returns:
        DataFrame filtrado para preview
    '''
    try:
        df_preview = df_inventario.copy()

        # Aplicar filtros básicos de forma segura
        if config.get('talhoes_excluir'):
            df_preview = df_preview[~df_preview['talhao'].isin(config['talhoes_excluir'])]

        diametro_min = config.get('diametro_min', 0)
        if 'D_cm' in df_preview.columns:
            df_preview = df_preview[df_preview['D_cm'] >= diametro_min]

        if config.get('codigos_excluir') and 'cod' in df_preview.columns:
            df_preview = df_preview[~df_preview['cod'].isin(config['codigos_excluir'])]

        # Remover dados inválidos de forma segura
        if 'D_cm' in df_preview.columns:
            df_preview = df_preview[
                (df_preview['D_cm'].notna()) &
                (df_preview['D_cm'] > 0)
                ]

        return df_preview

    except Exception as e:
        st.warning(f"Erro ao aplicar filtros: {e}")
        # Retornar DataFrame original em caso de erro
        return df_inventario


def validar_configuracoes(config, df_inventario):
    '''
    Valida as configurações antes de executar análise

    Args:
        config: Configurações selecionadas
        df_inventario: DataFrame do inventário

    Returns:
        dict: Resultado da validação
    '''
    validacao = {
        'valido': True,
        'alertas': [],
        'erros': []
    }

    # Verificar se dados suficientes após filtros
    dados_filtrados = aplicar_preview_filtros(df_inventario, config)

    if len(dados_filtrados) < 20:
        validacao['erros'].append("Poucos dados após aplicar filtros (mínimo: 20 registros)")
        validacao['valido'] = False

    if dados_filtrados['talhao'].nunique() < 2:
        validacao['alertas'].append("Apenas um talhão após filtros - considere revisar exclusões")

    # Verificar diâmetro mínimo
    if config['diametro_min'] > 15:
        validacao['alertas'].append("Diâmetro mínimo muito alto - pode excluir muitos dados")

    # Verificar configurações de área
    if config['metodo_area'] == "Valores informados manualmente":
        areas_manuais = config.get('areas_manuais', {})
        if not areas_manuais:
            validacao['erros'].append("Nenhuma área manual foi informada")
            validacao['valido'] = False
        else:
            # Verificar áreas muito pequenas ou grandes
            areas_pequenas = [t for t, a in areas_manuais.items() if a < 1]
            areas_grandes = [t for t, a in areas_manuais.items() if a > 200]

            if areas_pequenas:
                validacao['alertas'].append(f"Áreas muito pequenas: talhões {areas_pequenas}")
            if areas_grandes:
                validacao['alertas'].append(f"Áreas muito grandes: talhões {areas_grandes}")

    return validacao


def criar_configuracoes_avancadas():
    '''
    Cria interface para configurações avançadas (opcional)

    Returns:
        dict: Configurações avançadas
    '''
    with st.expander("🔧 Configurações Avançadas"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Modelos Hipsométricos:**")

            incluir_prodan = st.checkbox(
                "Incluir modelo Prodan",
                value=True,
                help="Modelo complexo que usa idade (se disponível)"
            )

            incluir_nao_lineares = st.checkbox(
                "Incluir modelos não-lineares",
                value=True,
                help="Chapman, Weibull, Mononuclear (mais demorados)"
            )

            tolerancia_ajuste = st.slider(
                "Tolerância para ajuste",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Tolerância para convergência dos modelos"
            )

        with col2:
            st.write("**Processamento:**")

            max_iteracoes = st.number_input(
                "Máximo de iterações",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=500,
                help="Máximo de iterações para modelos não-lineares"
            )

            usar_bootstrap = st.checkbox(
                "Usar bootstrap para validação",
                value=False,
                help="Validação adicional (mais demorado)"
            )

            mostrar_detalhes = st.checkbox(
                "Mostrar detalhes dos ajustes",
                value=False,
                help="Exibir informações detalhadas durante ajuste"
            )

    return {
        'incluir_prodan': incluir_prodan,
        'incluir_nao_lineares': incluir_nao_lineares,
        'tolerancia_ajuste': tolerancia_ajuste,
        'max_iteracoes': max_iteracoes,
        'usar_bootstrap': usar_bootstrap,
        'mostrar_detalhes': mostrar_detalhes
    }


def salvar_configuracoes(config):
    '''
    Salva configurações no session_state para reutilização

    Args:
        config: Configurações a serem salvas
    '''
    st.session_state['config_inventario'] = config
    st.success("✅ Configurações salvas!")


def carregar_configuracoes_salvas():
    '''
    Carrega configurações salvas do session_state

    Returns:
        dict: Configurações salvas ou None
    '''
    return st.session_state.get('config_inventario', None)


def mostrar_interface_configuracoes_salvas():
    '''
    Mostra interface para gerenciar configurações salvas
    '''
    st.sidebar.subheader("💾 Configurações")

    config_salva = carregar_configuracoes_salvas()

    if config_salva:
        st.sidebar.success("✅ Configuração salva disponível")

        if st.sidebar.button("🔄 Usar configuração salva"):
            st.success("Configuração carregada!")
            return config_salva
    else:
        st.sidebar.info("💡 Nenhuma configuração salva")

    return None


def exportar_configuracoes(config):
    '''
    Prepara configurações para exportação

    Args:
        config: Configurações atuais

    Returns:
        str: Configurações em formato JSON
    '''
    import json
    from datetime import datetime

    config_export = {
        'data_criacao': datetime.now().isoformat(),
        'versao_sistema': '1.0',
        'configuracoes': config
    }

    return json.dumps(config_export, indent=2, ensure_ascii=False)


def mostrar_botoes_configuracao(config):
    '''
    Mostra botões para ações com configurações

    Args:
        config: Configurações atuais
    '''
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Salvar Configurações"):
            salvar_configuracoes(config)

    with col2:
        config_json = exportar_configuracoes(config)
        st.download_button(
            label="📁 Exportar Configurações",
            data=config_json,
            file_name="config_inventario.json",
            mime="application/json"
        )

    with col3:
        if st.button("🔄 Resetar para Padrão"):
            st.session_state.clear()
            st.experimental_rerun()