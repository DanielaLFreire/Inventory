# config/configuracoes_globais.py
"""
Configurações globais centralizadas para todo o sistema
Gerencia filtros e configurações compartilhadas entre todas as etapas
"""

import streamlit as st
import pandas as pd


def inicializar_configuracoes_globais():
    """Inicializa configurações globais no session_state - VERSÃO COMPLETA"""
    if 'config_global' not in st.session_state:
        st.session_state.config_global = {
            # Filtros de dados
            'talhoes_excluir': [],
            'diametro_min': 4.0,
            'codigos_excluir': ['C', 'I'],

            # Configurações de áreas
            'metodo_area': 'Simular automaticamente',
            'area_parcela': 400,
            'raio_parcela': 11.28,
            'areas_manuais': {},

            # Parâmetros florestais
            'densidade_plantio': 1667,
            'sobrevivencia': 0.85,
            'fator_forma': 0.5,
            'densidade_madeira': 500,
            'idade_padrao': 5.0,

            # Configurações de modelos
            'incluir_nao_lineares': True,
            'max_iteracoes': 5000,
            'tolerancia_ajuste': 0.01,

            # NOVO: Parâmetros iniciais para modelos não-lineares hipsométricos
            'parametros_chapman': {
                'b0': 42.12,  # Altura assintótica
                'b1': 0.01,   # Taxa de crescimento
                'b2': 1.00    # Parâmetro de forma
            },
            'parametros_weibull': {
                'a': 42.12,   # Altura assintótica
                'b': 0.01,    # Parâmetro de escala
                'c': 1.00     # Parâmetro de forma
            },
            'parametros_mononuclear': {
                'a': 42.12,   # Altura assintótica
                'b': 1.00,    # Parâmetro de intercepto
                'c': 0.10     # Taxa de decaimento
            },

            # NOVO: Parâmetros iniciais para modelos volumétricos não-lineares (se houver)
            'parametros_vol_nao_lineares': {
                'enabled': False,  # Por padrão volumétricos são lineares
                'modelo_customizado': {
                    'param1': 1.0,
                    'param2': 0.1,
                    'param3': 1.0
                }
            },

            # Estado de configuração
            'configurado': False,
            'timestamp_config': None
        }


def limpar_tipos_nao_serializaveis(config):
    """
    NOVA FUNÇÃO: Limpa tipos não-serializáveis das configurações

    Args:
        config: Configurações a serem limpas

    Returns:
        dict: Configurações com tipos Python nativos
    """
    import numpy as np
    import pandas as pd

    def converter_recursivo(obj):
        if isinstance(obj, dict):
            return {k: converter_recursivo(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [converter_recursivo(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(converter_recursivo(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # Para scalars do NumPy
            return obj.item()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        else:
            return obj

    return converter_recursivo(config)


def mostrar_configuracoes_globais():
    """
    Interface unificada para todas as configurações do sistema - VERSÃO COM PARÂMETROS NÃO-LINEARES
    """
    st.header("⚙️ Configurações Globais do Sistema")
    st.info("💡 Estas configurações se aplicam a todas as etapas da análise")

    # Verificar se dados estão carregados
    if not hasattr(st.session_state, 'dados_inventario') or st.session_state.dados_inventario is None:
        st.warning("⚠️ Carregue os dados primeiro para configurar filtros específicos")
        return st.session_state.config_global

    df_inventario = st.session_state.dados_inventario

    # Criar abas para organizar configurações - ADICIONANDO ABA DE PARÂMETROS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Filtros de Dados",
        "📏 Áreas dos Talhões",
        "🌱 Parâmetros Florestais",
        "🧮 Configurações de Modelos",
        "🔧 Parâmetros Não-Lineares"  # NOVA ABA
    ])

    with tab1:
        config_filtros = configurar_filtros_dados(df_inventario)

    with tab2:
        config_areas = configurar_areas_talhoes(df_inventario)

    with tab3:
        config_parametros = configurar_parametros_florestais()

    with tab4:
        config_modelos = configurar_modelos_avancados()

    with tab5:  # NOVA ABA
        st.subheader("🔧 Parâmetros de Modelos Não-Lineares")
        st.info("⚙️ Estes parâmetros são usados como valores iniciais para os modelos Chapman, Weibull e Mononuclear")

        # Mostrar validação
        mostrar_validacao_parametros()

        # Botão para resetar
        resetar_parametros_padrao()

        # Download específico dos parâmetros
        st.download_button(
            "📥 Exportar Parâmetros Não-Lineares",
            data=exportar_parametros_nao_lineares(),
            file_name="parametros_nao_lineares.json",
            mime="application/json",
            key="download_parametros_nao_lineares"
        )

    # Combinar todas as configurações (config_modelos já inclui os parâmetros)
    config_completa = {
        **st.session_state.config_global,
        **config_filtros,
        **config_areas,
        **config_parametros,
        **config_modelos
    }

    # Limpar tipos não-serializáveis
    config_completa = limpar_tipos_nao_serializaveis(config_completa)

    # Atualizar session_state
    st.session_state.config_global = config_completa
    st.session_state.config_global['configurado'] = True
    st.session_state.config_global['timestamp_config'] = pd.Timestamp.now()

    # Mostrar resumo das configurações
    mostrar_resumo_configuracoes_completas(config_completa, df_inventario)

    # Botões de ação
    mostrar_botoes_configuracao_globais(config_completa)

    return config_completa


def configurar_filtros_dados(df_inventario):
    """Configurações de filtros de dados - VERSÃO CORRIGIDA"""
    st.subheader("🔍 Filtros de Dados")
    st.markdown("*Estes filtros serão aplicados em todas as etapas*")

    col1, col2, col3 = st.columns(3)

    with col1:
        talhoes_disponiveis = sorted(df_inventario['talhao'].unique())
        # CORREÇÃO: Converter para tipos Python nativos
        talhoes_disponiveis_nativos = [int(t) for t in talhoes_disponiveis]

        talhoes_excluir = st.multiselect(
            "🚫 Talhões a Excluir",
            options=talhoes_disponiveis_nativos,
            default=[int(t) for t in st.session_state.config_global.get('talhoes_excluir', [])],
            help="Talhões que serão excluídos de TODAS as análises",
            key="global_talhoes_excluir"
        )

    with col2:
        diametro_min = st.number_input(
            "📏 Diâmetro Mínimo (cm)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state.config_global.get('diametro_min', 4.0)),
            step=0.5,
            help="Diâmetro mínimo para TODAS as análises",
            key="global_diametro_min"
        )

    with col3:
        codigos_disponiveis = sorted(df_inventario['cod'].unique()) if 'cod' in df_inventario.columns else []
        # CORREÇÃO: Garantir que códigos sejam strings
        codigos_disponiveis_nativos = [str(c) for c in codigos_disponiveis]

        codigos_excluir = st.multiselect(
            "🏷️ Códigos a Excluir",
            options=codigos_disponiveis_nativos,
            default=[str(c) for c in st.session_state.config_global.get('codigos_excluir', ['C', 'I'])],
            help="Códigos de árvores excluídos (C=Cortada, I=Invasora)",
            key="global_codigos_excluir"
        )

    # Preview do impacto dos filtros
    if st.checkbox("👀 Preview do Impacto dos Filtros", key="preview_filtros_global"):
        mostrar_preview_filtros(df_inventario, talhoes_excluir, diametro_min, codigos_excluir)

    return {
        'talhoes_excluir': [int(t) for t in talhoes_excluir],  # CORREÇÃO: Garantir int nativo
        'diametro_min': float(diametro_min),  # CORREÇÃO: Garantir float nativo
        'codigos_excluir': [str(c) for c in codigos_excluir]  # CORREÇÃO: Garantir string nativa
    }


def configurar_areas_talhoes(df_inventario):
    """Configurações de áreas dos talhões - VERSÃO CORRIGIDA"""
    st.subheader("📏 Configuração de Áreas")

    # CORREÇÃO: Verificar arquivos de forma mais robusta
    metodos_disponiveis = ["Simular automaticamente", "Valores específicos por talhão"]

    # Debug: Mostrar status dos arquivos
    with st.expander("🔍 Debug - Status dos Arquivos"):
        st.write("**Verificando arquivos disponíveis:**")

        # Verificar shapefile
        shapefile_disponivel = False
        if hasattr(st.session_state, 'arquivo_shapefile'):
            if st.session_state.arquivo_shapefile is not None:
                shapefile_disponivel = True
                st.success(f"✅ Shapefile detectado: {st.session_state.arquivo_shapefile.name}")
            else:
                st.info("ℹ️ Shapefile não carregado")
        else:
            st.info("ℹ️ Atributo 'arquivo_shapefile' não existe no session_state")

        # Verificar coordenadas
        coordenadas_disponivel = False
        if hasattr(st.session_state, 'arquivo_coordenadas'):
            if st.session_state.arquivo_coordenadas is not None:
                coordenadas_disponivel = True
                st.success(f"✅ Coordenadas detectadas: {st.session_state.arquivo_coordenadas.name}")
            else:
                st.info("ℹ️ Arquivo de coordenadas não carregado")
        else:
            st.info("ℹ️ Atributo 'arquivo_coordenadas' não existe no session_state")

        # Mostrar todos os atributos do session_state relacionados a arquivos
        st.write("**Todos os atributos do session_state:**")
        arquivo_attrs = {k: v for k, v in st.session_state.items() if 'arquivo' in k.lower()}
        if arquivo_attrs:
            for k, v in arquivo_attrs.items():
                st.write(f"- {k}: {type(v)} = {v is not None}")
        else:
            st.write("Nenhum atributo relacionado a 'arquivo' encontrado")

    # Adicionar métodos baseado nos arquivos disponíveis
    if shapefile_disponivel:
        metodos_disponiveis.append("Upload shapefile")
        st.info("📁 Método 'Upload shapefile' adicionado")

    if coordenadas_disponivel:
        metodos_disponiveis.append("Coordenadas das parcelas")
        st.info("📍 Método 'Coordenadas das parcelas' adicionado")

    # Se nenhum arquivo adicional foi detectado, mostrar aviso
    if not shapefile_disponivel and not coordenadas_disponivel:
        st.warning("""
        ⚠️ **Arquivos adicionais não detectados**

        Para usar métodos avançados de área:
        - Upload um shapefile para usar "Upload shapefile"
        - Upload coordenadas das parcelas para usar "Coordenadas das parcelas"

        **Verifique se os arquivos foram carregados na página principal**
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        # CORREÇÃO: Garantir que o método atual está na lista
        metodo_atual = st.session_state.config_global.get('metodo_area', 'Simular automaticamente')
        if metodo_atual not in metodos_disponiveis:
            st.warning(f"⚠️ Método '{metodo_atual}' não disponível. Revertendo para 'Simular automaticamente'")
            metodo_atual = 'Simular automaticamente'

        metodo_area = st.selectbox(
            "🗺️ Método para Áreas",
            options=metodos_disponiveis,
            index=metodos_disponiveis.index(metodo_atual),
            key="global_metodo_area",
            help=f"Métodos disponíveis baseados nos arquivos carregados. Total: {len(metodos_disponiveis)}"
        )

    with col2:
        area_parcela = st.number_input(
            "📐 Área da Parcela (m²)",
            min_value=100,
            max_value=2000,
            value=st.session_state.config_global.get('area_parcela', 400),
            step=50,
            key="global_area_parcela"
        )

    with col3:
        raio_parcela = st.number_input(
            "📐 Raio da Parcela (m)",
            min_value=5.0,
            max_value=30.0,
            value=st.session_state.config_global.get('raio_parcela', 11.28),
            step=0.1,
            key="global_raio_parcela"
        )

    # Configurações específicas por método
    areas_manuais = {}
    if metodo_area == "Valores específicos por talhão":
        areas_manuais = configurar_areas_manuais_global(df_inventario)
    elif metodo_area == "Coordenadas das parcelas":
        mostrar_info_coordenadas()
    elif metodo_area == "Upload shapefile":
        mostrar_info_shapefile()

    return {
        'metodo_area': metodo_area,
        'area_parcela': area_parcela,
        'raio_parcela': raio_parcela,
        'areas_manuais': areas_manuais
    }

def configurar_areas_manuais_global(df_inventario):
    """Interface para áreas manuais - VERSÃO CORRIGIDA"""
    st.write("**📝 Áreas por Talhão (hectares):**")

    talhoes_disponiveis = sorted(df_inventario['talhao'].unique())
    areas_manuais = {}

    # Criar interface em colunas
    n_colunas = min(4, len(talhoes_disponiveis))
    colunas = st.columns(n_colunas)

    for i, talhao in enumerate(talhoes_disponiveis):
        col_idx = i % n_colunas
        with colunas[col_idx]:
            valor_anterior = st.session_state.config_global.get('areas_manuais', {}).get(talhao, 25.0)

            # CORREÇÃO: Garantir que seja float nativo do Python
            if hasattr(valor_anterior, 'item'):  # Se for numpy scalar
                valor_anterior = float(valor_anterior.item())
            else:
                valor_anterior = float(valor_anterior)

            area_valor = st.number_input(
                f"Talhão {talhao}",
                min_value=0.1,
                max_value=1000.0,
                value=valor_anterior,
                step=0.1,
                key=f"global_area_talhao_{talhao}"
            )

            # CORREÇÃO: Converter talhao e area para tipos Python nativos
            areas_manuais[int(talhao)] = float(area_valor)

    # Mostrar resumo
    if areas_manuais:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Área Total", f"{sum(areas_manuais.values()):.1f} ha")
        with col2:
            st.metric("Área Média", f"{sum(areas_manuais.values()) / len(areas_manuais):.1f} ha")
        with col3:
            st.metric("Talhões", len(areas_manuais))

    return areas_manuais


def configurar_parametros_florestais():
    """Configurações de parâmetros florestais"""
    st.subheader("🌱 Parâmetros Florestais")

    col1, col2, col3 = st.columns(3)

    with col1:
        densidade_plantio = st.number_input(
            "🌱 Densidade de Plantio (árv/ha)",
            min_value=500,
            max_value=5000,
            value=st.session_state.config_global.get('densidade_plantio', 1667),
            step=50,
            key="global_densidade_plantio"
        )

        sobrevivencia = st.slider(
            "🌲 Taxa de Sobrevivência (%)",
            min_value=50,
            max_value=100,
            value=int(st.session_state.config_global.get('sobrevivencia', 0.85) * 100),
            step=5,
            key="global_sobrevivencia"
        ) / 100

    with col2:
        fator_forma = st.number_input(
            "📊 Fator de Forma",
            min_value=0.3,
            max_value=0.8,
            value=st.session_state.config_global.get('fator_forma', 0.5),
            step=0.05,
            key="global_fator_forma"
        )

        densidade_madeira = st.number_input(
            "🌱 Densidade da Madeira (kg/m³)",
            min_value=300,
            max_value=800,
            value=st.session_state.config_global.get('densidade_madeira', 500),
            step=25,
            key="global_densidade_madeira"
        )

    with col3:
        idade_padrao = st.number_input(
            "📅 Idade Padrão (anos)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.config_global.get('idade_padrao', 5.0),
            step=0.5,
            key="global_idade_padrao"
        )

    return {
        'densidade_plantio': densidade_plantio,
        'sobrevivencia': sobrevivencia,
        'fator_forma': fator_forma,
        'densidade_madeira': densidade_madeira,
        'idade_padrao': idade_padrao
    }


def configurar_modelos_avancados():
    """Configurações avançadas de modelos - VERSÃO COM PARÂMETROS NÃO-LINEARES"""
    st.subheader("🧮 Configurações de Modelos")

    # Configurações básicas
    col1, col2 = st.columns(2)

    with col1:
        incluir_nao_lineares = st.checkbox(
            "Incluir modelos não-lineares",
            value=st.session_state.config_global.get('incluir_nao_lineares', True),
            help="Chapman, Weibull, Mononuclear (mais demorados)",
            key="global_incluir_nao_lineares"
        )

        max_iteracoes = st.number_input(
            "Máximo de iterações",
            min_value=1000,
            max_value=10000,
            value=st.session_state.config_global.get('max_iteracoes', 5000),
            step=500,
            key="global_max_iteracoes"
        )

    with col2:
        tolerancia_ajuste = st.number_input(
            "Tolerância para ajuste",
            min_value=0.001,
            max_value=0.1,
            value=st.session_state.config_global.get('tolerancia_ajuste', 0.01),
            step=0.001,
            format="%.3f",
            key="global_tolerancia_ajuste"
        )

    # NOVO: Configuração de parâmetros iniciais para modelos não-lineares
    if incluir_nao_lineares:
        st.markdown("---")
        st.subheader("🔧 Configuração de Parâmetros Iniciais")
        st.info("⚙️ Configure os valores iniciais para os modelos não-lineares (importante para convergência)")

        # Abas para cada modelo não-linear
        tab_chapman, tab_weibull, tab_mononuclear = st.tabs([
            "🔧 Chapman", "📈 Weibull", "📊 Mononuclear"
        ])

        # Chapman
        with tab_chapman:
            st.write("**Modelo de Chapman:** H = b₀ × (1 - exp(-b₁ × D))^b₂")

            col1, col2, col3 = st.columns(3)

            with col1:
                chapman_b0 = st.number_input(
                    "b₀ - Altura assintótica",
                    min_value=10.0,
                    max_value=100.0,
                    value=float(st.session_state.config_global.get('parametros_chapman', {}).get('b0', 42.12)),
                    step=0.01,
                    help="Altura máxima teórica que a árvore pode atingir",
                    key="chapman_b0"
                )

            with col2:
                chapman_b1 = st.number_input(
                    "b₁ - Taxa de crescimento",
                    min_value=0.001,
                    max_value=1.0,
                    value=float(st.session_state.config_global.get('parametros_chapman', {}).get('b1', 0.01)),
                    step=0.001,
                    format="%.3f",
                    help="Velocidade de crescimento em altura",
                    key="chapman_b1"
                )

            with col3:
                chapman_b2 = st.number_input(
                    "b₂ - Parâmetro de forma",
                    min_value=0.1,
                    max_value=5.0,
                    value=float(st.session_state.config_global.get('parametros_chapman', {}).get('b2', 1.00)),
                    step=0.01,
                    help="Forma da curva de crescimento",
                    key="chapman_b2"
                )

        # Weibull
        with tab_weibull:
            st.write("**Modelo de Weibull:** H = a × (1 - exp(-b × D^c))")

            col1, col2, col3 = st.columns(3)

            with col1:
                weibull_a = st.number_input(
                    "a - Altura assintótica",
                    min_value=10.0,
                    max_value=100.0,
                    value=float(st.session_state.config_global.get('parametros_weibull', {}).get('a', 42.12)),
                    step=0.01,
                    help="Altura máxima teórica",
                    key="weibull_a"
                )

            with col2:
                weibull_b = st.number_input(
                    "b - Parâmetro de escala",
                    min_value=0.001,
                    max_value=1.0,
                    value=float(st.session_state.config_global.get('parametros_weibull', {}).get('b', 0.01)),
                    step=0.001,
                    format="%.3f",
                    help="Parâmetro de escala da distribuição",
                    key="weibull_b"
                )

            with col3:
                weibull_c = st.number_input(
                    "c - Parâmetro de forma",
                    min_value=0.1,
                    max_value=5.0,
                    value=float(st.session_state.config_global.get('parametros_weibull', {}).get('c', 1.00)),
                    step=0.01,
                    help="Forma da distribuição Weibull",
                    key="weibull_c"
                )

        # Mononuclear
        with tab_mononuclear:
            st.write("**Modelo Mononuclear:** H = a × (1 - b × exp(-c × D))")

            col1, col2, col3 = st.columns(3)

            with col1:
                mono_a = st.number_input(
                    "a - Altura assintótica",
                    min_value=10.0,
                    max_value=100.0,
                    value=float(st.session_state.config_global.get('parametros_mononuclear', {}).get('a', 42.12)),
                    step=0.01,
                    help="Altura máxima teórica",
                    key="mono_a"
                )

            with col2:
                mono_b = st.number_input(
                    "b - Parâmetro de intercepto",
                    min_value=0.1,
                    max_value=2.0,
                    value=float(st.session_state.config_global.get('parametros_mononuclear', {}).get('b', 1.00)),
                    step=0.01,
                    help="Intercepto do modelo mononuclear",
                    key="mono_b"
                )

            with col3:
                mono_c = st.number_input(
                    "c - Taxa de decaimento",
                    min_value=0.01,
                    max_value=1.0,
                    value=float(st.session_state.config_global.get('parametros_mononuclear', {}).get('c', 0.10)),
                    step=0.01,
                    help="Taxa de decaimento exponencial",
                    key="mono_c"
                )

        # Salvar parâmetros configurados
        parametros_modelos = {
            'parametros_chapman': {
                'b0': chapman_b0,
                'b1': chapman_b1,
                'b2': chapman_b2
            },
            'parametros_weibull': {
                'a': weibull_a,
                'b': weibull_b,
                'c': weibull_c
            },
            'parametros_mononuclear': {
                'a': mono_a,
                'b': mono_b,
                'c': mono_c
            }
        }

        # Preview dos parâmetros
        with st.expander("👀 Preview dos Parâmetros Configurados"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Chapman:**")
                st.write(f"• b₀ = {chapman_b0:.2f}")
                st.write(f"• b₁ = {chapman_b1:.3f}")
                st.write(f"• b₂ = {chapman_b2:.2f}")

            with col2:
                st.write("**Weibull:**")
                st.write(f"• a = {weibull_a:.2f}")
                st.write(f"• b = {weibull_b:.3f}")
                st.write(f"• c = {weibull_c:.2f}")

            with col3:
                st.write("**Mononuclear:**")
                st.write(f"• a = {mono_a:.2f}")
                st.write(f"• b = {mono_b:.2f}")
                st.write(f"• c = {mono_c:.2f}")

    else:
        # Se não incluir não-lineares, usar parâmetros padrão
        parametros_modelos = {
            'parametros_chapman': {'b0': 42.12, 'b1': 0.01, 'b2': 1.00},
            'parametros_weibull': {'a': 42.12, 'b': 0.01, 'c': 1.00},
            'parametros_mononuclear': {'a': 42.12, 'b': 1.00, 'c': 0.10}
        }

    # Retornar todas as configurações
    return {
        'incluir_nao_lineares': incluir_nao_lineares,
        'max_iteracoes': max_iteracoes,
        'tolerancia_ajuste': tolerancia_ajuste,
        **parametros_modelos
    }


def obter_parametros_modelo_nao_linear(nome_modelo):
    """
    NOVA: Obtém parâmetros iniciais para um modelo não-linear específico

    Args:
        nome_modelo: 'Chapman', 'Weibull', ou 'Mononuclear'

    Returns:
        dict: Parâmetros iniciais para o modelo
    """
    config = obter_configuracao_global()

    parametros_map = {
        'Chapman': config.get('parametros_chapman', {'b0': 42.12, 'b1': 0.01, 'b2': 1.00}),
        'Weibull': config.get('parametros_weibull', {'a': 42.12, 'b': 0.01, 'c': 1.00}),
        'Mononuclear': config.get('parametros_mononuclear', {'a': 42.12, 'b': 1.00, 'c': 0.10})
    }

    return parametros_map.get(nome_modelo, {})


def validar_parametros_nao_lineares():
    """
    NOVA: Valida se os parâmetros não-lineares são adequados

    Returns:
        dict: {'valido': bool, 'avisos': list, 'erros': list}
    """
    config = obter_configuracao_global()

    avisos = []
    erros = []

    if config.get('incluir_nao_lineares', True):
        # Validar Chapman
        chapman = config.get('parametros_chapman', {})
        if chapman.get('b0', 0) < 10:
            avisos.append("Chapman: Altura assintótica muito baixa (< 10m)")
        if chapman.get('b1', 0) > 0.5:
            avisos.append("Chapman: Taxa de crescimento muito alta (> 0.5)")

        # Validar Weibull
        weibull = config.get('parametros_weibull', {})
        if weibull.get('a', 0) < 10:
            avisos.append("Weibull: Altura assintótica muito baixa (< 10m)")
        if weibull.get('c', 0) > 3:
            avisos.append("Weibull: Parâmetro de forma muito alto (> 3)")

        # Validar Mononuclear
        mono = config.get('parametros_mononuclear', {})
        if mono.get('a', 0) < 10:
            avisos.append("Mononuclear: Altura assintótica muito baixa (< 10m)")
        if mono.get('b', 0) < 0.5:
            avisos.append("Mononuclear: Parâmetro de intercepto muito baixo (< 0.5)")

    return {
        'valido': len(erros) == 0,
        'avisos': avisos,
        'erros': erros
    }


def mostrar_validacao_parametros():
    """NOVA: Mostra validação dos parâmetros configurados"""
    validacao = validar_parametros_nao_lineares()

    if validacao['valido']:
        st.success("✅ Parâmetros válidos!")

    if validacao['avisos']:
        st.warning("⚠️ **Avisos sobre os parâmetros:**")
        for aviso in validacao['avisos']:
            st.warning(f"• {aviso}")

    if validacao['erros']:
        st.error("❌ **Erros nos parâmetros:**")
        for erro in validacao['erros']:
            st.error(f"• {erro}")


def resetar_parametros_padrao():
    """NOVA: Reseta parâmetros para valores padrão recomendados"""
    if st.button("🔄 Resetar para Valores Padrão", key="reset_parametros"):
        st.session_state.config_global.update({
            'parametros_chapman': {'b0': 42.12, 'b1': 0.01, 'b2': 1.00},
            'parametros_weibull': {'a': 42.12, 'b': 0.01, 'c': 1.00},
            'parametros_mononuclear': {'a': 42.12, 'b': 1.00, 'c': 0.10}
        })
        st.success("✅ Parâmetros resetados para valores padrão!")
        st.rerun()


def exportar_parametros_nao_lineares():
    """NOVA: Exporta apenas os parâmetros dos modelos não-lineares"""
    config = obter_configuracao_global()

    parametros_export = {
        'chapman': config.get('parametros_chapman', {}),
        'weibull': config.get('parametros_weibull', {}),
        'mononuclear': config.get('parametros_mononuclear', {}),
        'configuracoes_modelo': {
            'incluir_nao_lineares': config.get('incluir_nao_lineares', True),
            'max_iteracoes': config.get('max_iteracoes', 5000),
            'tolerancia_ajuste': config.get('tolerancia_ajuste', 0.01)
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }

    import json
    return json.dumps(parametros_export, indent=2, ensure_ascii=False)


# NOVA FUNÇÃO: Para ser usada na Etapa 1 (Hipsométricos)
def aplicar_parametros_nao_lineares_etapa1():
    """
    Aplica parâmetros não-lineares na Etapa 1
    Para ser importada e usada em pages/1_🌳_Modelos_Hipsométricos.py
    """
    config = obter_configuracao_global()

    if not config.get('incluir_nao_lineares', True):
        return None

    # Retornar configurações completas para modelos não-lineares
    return {
        'chapman_params': config.get('parametros_chapman', {'b0': 42.12, 'b1': 0.01, 'b2': 1.00}),
        'weibull_params': config.get('parametros_weibull', {'a': 42.12, 'b': 0.01, 'c': 1.00}),
        'mononuclear_params': config.get('parametros_mononuclear', {'a': 42.12, 'b': 1.00, 'c': 0.10}),
        'max_iteracoes': config.get('max_iteracoes', 5000),
        'tolerancia': config.get('tolerancia_ajuste', 0.01)
    }


def mostrar_preview_filtros(df_inventario, talhoes_excluir, diametro_min, codigos_excluir):
    """Mostra preview do impacto dos filtros"""
    st.write("**📊 Impacto dos Filtros:**")

    # Dados originais
    total_original = len(df_inventario)
    talhoes_original = df_inventario['talhao'].nunique()

    # Aplicar filtros
    df_filtrado = df_inventario.copy()

    if talhoes_excluir:
        df_filtrado = df_filtrado[~df_filtrado['talhao'].isin(talhoes_excluir)]

    df_filtrado = df_filtrado[df_filtrado['D_cm'] >= diametro_min]

    if codigos_excluir and 'cod' in df_filtrado.columns:
        df_filtrado = df_filtrado[~df_filtrado['cod'].isin(codigos_excluir)]

    # Estatísticas após filtros
    total_filtrado = len(df_filtrado)
    talhoes_filtrado = df_filtrado['talhao'].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_registros = total_filtrado - total_original
        st.metric("Registros", f"{total_filtrado:,}", delta=f"{delta_registros:+,}")

    with col2:
        delta_talhoes = talhoes_filtrado - talhoes_original
        st.metric("Talhões", talhoes_filtrado, delta=f"{delta_talhoes:+}")

    with col3:
        percentual = (total_filtrado / total_original) * 100 if total_original > 0 else 0
        st.metric("Dados Mantidos", f"{percentual:.1f}%")

    with col4:
        if total_filtrado < 20:
            st.error("⚠️ Poucos dados!")
        elif percentual > 80:
            st.success("✅ Filtros OK")
        else:
            st.warning("⚠️ Muitos dados excluídos")


def mostrar_resumo_configuracoes_completas(config, df_inventario):
    """Mostra resumo completo das configurações"""
    with st.expander("📋 Resumo Completo das Configurações"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**🔍 Filtros:**")
            st.write(f"- Diâmetro mínimo: {config['diametro_min']} cm")
            if config['talhoes_excluir']:
                st.write(f"- Talhões excluídos: {config['talhoes_excluir']}")
            if config['codigos_excluir']:
                st.write(f"- Códigos excluídos: {config['codigos_excluir']}")

            st.write("**📏 Áreas:**")
            st.write(f"- Método: {config['metodo_area']}")
            st.write(f"- Área da parcela: {config['area_parcela']} m²")

        with col2:
            st.write("**🌱 Parâmetros:**")
            st.write(f"- Densidade plantio: {config['densidade_plantio']} árv/ha")
            st.write(f"- Sobrevivência: {config['sobrevivencia'] * 100:.0f}%")
            st.write(f"- Fator de forma: {config['fator_forma']}")

            st.write("**🧮 Modelos:**")
            st.write(f"- Não-lineares: {'Sim' if config['incluir_nao_lineares'] else 'Não'}")
            st.write(f"- Max iterações: {config['max_iteracoes']}")


def mostrar_botoes_configuracao_globais(config):
    """Botões de ação para configurações - VERSÃO DEBUGADA"""
    import json  # CORREÇÃO: Adicionar import
    import pandas as pd  # CORREÇÃO: Adicionar import pandas

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Salvar Configurações", use_container_width=True):
            # Limpar antes de salvar
            config_limpa = limpar_tipos_nao_serializaveis(config)
            st.session_state.config_global = config_limpa
            st.success("✅ Configurações salvas!")

    with col2:
        if st.button("🔄 Resetar Padrão", use_container_width=True):
            inicializar_configuracoes_globais()
            st.rerun()

    with col3:
        # DEBUG: Mostrar tipos problemáticos antes da exportação
        try:
            with st.expander("🔍 Debug Config"):
                st.write("Tipos detectados:")
                for k, v in config.items():
                    st.write(f"- {k}: {type(v)} = {v}")

            config_json = exportar_configuracoes_globais(config)
            st.download_button(
                "📁 Exportar",
                data=config_json,
                file_name="configuracoes_globais.json",
                mime="application/json",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ Erro na exportação: {e}")
            st.write("🔍 Tentando exportação simplificada...")

            # Exportação de emergência
            config_emergencia = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'erro': str(e),
                'configuracoes_basicas': 'Erro na serialização'
            }

            try:
                config_json_emergencia = json.dumps(config_emergencia, indent=2, ensure_ascii=False)
                st.download_button(
                    "📁 Exportar (Emergência)",
                    data=config_json_emergencia,
                    file_name="configuracoes_emergencia.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_emergencia"
                )
            except Exception as e2:
                st.error(f"❌ Erro crítico: {e2}")

    with col4:
        if st.button("🎯 Aplicar a Todas Etapas", use_container_width=True):
            aplicar_configuracao_todas_etapas(config)
            st.success("✅ Configurações aplicadas!")


def exportar_configuracoes_globais(config):
    """Exporta configurações em JSON - VERSÃO ULTRA ROBUSTA"""
    import json
    from datetime import datetime
    import numpy as np
    import pandas as pd

    class ConfigJSONEncoder(json.JSONEncoder):
        """Encoder personalizado para configurações"""

        def default(self, obj):
            # Tipos NumPy
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)

            # Tipos Pandas
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif hasattr(obj, 'item'):  # Scalars NumPy
                try:
                    return obj.item()
                except:
                    return str(obj)

            # Outros tipos problemáticos
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return super().default(obj)

    def limpar_config_completa(obj):
        """Limpeza mais agressiva das configurações"""
        if isinstance(obj, dict):
            resultado = {}
            for k, v in obj.items():
                try:
                    # Tentar converter a chave
                    if hasattr(k, 'item'):
                        k = k.item()
                    elif isinstance(k, (np.integer, np.int64)):
                        k = int(k)

                    # Converter o valor recursivamente
                    resultado[str(k)] = limpar_config_completa(v)
                except Exception as e:
                    # Se falhar, usar string
                    resultado[str(k)] = str(v)
            return resultado

        elif isinstance(obj, (list, tuple)):
            resultado = []
            for item in obj:
                try:
                    resultado.append(limpar_config_completa(item))
                except Exception:
                    resultado.append(str(item))
            return resultado

        # Conversões específicas
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'item'):
            try:
                return obj.item()
            except:
                return str(obj)
        else:
            return obj

    try:
        # Primeira tentativa: limpar configurações
        config_limpa = limpar_config_completa(config)

        config_export = {
            'timestamp': datetime.now().isoformat(),
            'versao': '1.0',
            'configuracoes': config_limpa
        }

        # Tentar serializar com encoder personalizado
        return json.dumps(config_export, cls=ConfigJSONEncoder, indent=2, ensure_ascii=False)

    except Exception as e1:
        st.warning(f"⚠️ Erro na serialização completa: {e1}")

        try:
            # Segunda tentativa: versão mais simples
            config_basica = {
                'timestamp': datetime.now().isoformat(),
                'versao': '1.0',
                'configuracoes_basicas': {
                    'diametro_min': float(config.get('diametro_min', 4.0)) if config.get(
                        'diametro_min') is not None else 4.0,
                    'metodo_area': str(config.get('metodo_area', 'Simular automaticamente')),
                    'area_parcela': float(config.get('area_parcela', 400)) if config.get(
                        'area_parcela') is not None else 400,
                    'configurado': bool(config.get('configurado', False)),
                    'talhoes_excluir': [int(t) for t in config.get('talhoes_excluir', []) if t is not None],
                    'codigos_excluir': [str(c) for c in config.get('codigos_excluir', []) if c is not None],
                },
                'observacao': 'Configurações básicas devido a erro de serialização'
            }

            return json.dumps(config_basica, indent=2, ensure_ascii=False)

        except Exception as e2:
            st.error(f"❌ Erro crítico na serialização: {e2}")

            # Terceira tentativa: mínima
            config_minima = {
                'timestamp': datetime.now().isoformat(),
                'versao': '1.0',
                'erro': f'Serialização falhou: {str(e1)} | {str(e2)}',
                'status': 'Erro na exportação'
            }

            return json.dumps(config_minima, indent=2, ensure_ascii=False)


def aplicar_configuracao_todas_etapas(config):
    """Aplica configurações a todas as etapas"""
    # Limpar resultados existentes para forçar reaplicação das configurações
    for key in ['resultados_hipsometricos', 'resultados_volumetricos', 'inventario_processado']:
        if key in st.session_state:
            del st.session_state[key]

    # Marcar que configurações mudaram
    st.session_state.config_alterada = True


def obter_configuracao_global():
    """
    Função para outras páginas obterem a configuração global

    Returns:
        dict: Configuração global atual
    """
    if 'config_global' not in st.session_state:
        inicializar_configuracoes_globais()

    return st.session_state.config_global


def aplicar_filtros_configuracao_global(df_inventario):
    """
    Aplica filtros baseados na configuração global

    Args:
        df_inventario: DataFrame do inventário

    Returns:
        DataFrame filtrado
    """
    config = obter_configuracao_global()

    df_filtrado = df_inventario.copy()

    # Aplicar filtros da configuração global
    if config.get('talhoes_excluir'):
        df_filtrado = df_filtrado[~df_filtrado['talhao'].isin(config['talhoes_excluir'])]

    df_filtrado = df_filtrado[df_filtrado['D_cm'] >= config.get('diametro_min', 4.0)]

    if config.get('codigos_excluir') and 'cod' in df_filtrado.columns:
        df_filtrado = df_filtrado[~df_filtrado['cod'].isin(config['codigos_excluir'])]

    # Remover dados inválidos
    df_filtrado = df_filtrado[
        (df_filtrado['D_cm'].notna()) &
        (df_filtrado['H_m'].notna()) &
        (df_filtrado['D_cm'] > 0) &
        (df_filtrado['H_m'] > 1.3)
        ]

    return df_filtrado


def verificar_configuracao_atualizada():
    """
    Verifica se a configuração foi atualizada desde a última execução

    Returns:
        bool: True se configuração foi atualizada
    """
    if 'config_alterada' in st.session_state:
        alterada = st.session_state.config_alterada
        st.session_state.config_alterada = False  # Reset flag
        return alterada

    return False


def mostrar_status_configuracao_sidebar():
    """Mostra status da configuração na sidebar"""
    if 'config_global' in st.session_state and st.session_state.config_global.get('configurado', False):
        st.sidebar.success("✅ Configurado")

        config = st.session_state.config_global
        timestamp = config.get('timestamp_config')
        if timestamp:
            st.sidebar.caption(f"Última atualização: {timestamp.strftime('%H:%M')}")

        # Mostrar resumo rápido
        with st.sidebar.expander("⚙️ Resumo Config"):
            st.write(f"Diâmetro min: {config.get('diametro_min', 4.0)} cm")
            if config.get('talhoes_excluir'):
                st.write(f"Talhões excluídos: {len(config['talhoes_excluir'])}")
            st.write(f"Método área: {config.get('metodo_area', 'Automático')}")
    else:
        st.sidebar.warning("⚠️ Não configurado")
        if st.sidebar.button("⚙️ Configurar", use_container_width=True):
            st.switch_page("pages/0_⚙️_Configurações.py")


def mostrar_info_coordenadas():
    """Mostra informações sobre arquivo de coordenadas"""
    st.info("""
    📍 **Método: Coordenadas das Parcelas**

    - Arquivo de coordenadas detectado
    - Áreas serão calculadas baseadas nas coordenadas das parcelas
    - Método mais preciso quando coordenadas GPS estão disponíveis
    """)

    # Tentar mostrar preview se possível
    if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas:
        try:
            import pandas as pd
            df_coords = pd.read_csv(st.session_state.arquivo_coordenadas)
            st.write("**Preview do arquivo de coordenadas:**")
            st.dataframe(df_coords.head(3))
        except Exception as e:
            st.warning(f"Não foi possível visualizar o arquivo: {e}")


def mostrar_info_shapefile():
    """Mostra informações sobre shapefile"""
    st.info("""
    📁 **Método: Upload Shapefile**

    - Shapefile detectado
    - Áreas serão extraídas dos polígonos
    - Método mais preciso para talhões irregulares
    """)

    if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile:
        st.write(f"**Arquivo:** {st.session_state.arquivo_shapefile.name}")


# ADICIONAR FUNÇÃO PARA VERIFICAR UPLOAD DE ARQUIVOS NA SIDEBAR
def verificar_uploads_arquivos():
    """
    Função para verificar se uploads adicionais foram realizados
    Deve ser chamada na sidebar principal
    """
    st.sidebar.subheader("📎 Arquivos Adicionais")

    # Upload de coordenadas
    arquivo_coords = st.sidebar.file_uploader(
        "📍 Coordenadas das Parcelas (CSV)",
        type=['csv'],
        help="Arquivo com coordenadas GPS das parcelas",
        key="upload_coordenadas_sidebar"
    )

    if arquivo_coords is not None:
        st.session_state.arquivo_coordenadas = arquivo_coords
        st.sidebar.success(f"✅ Coordenadas: {arquivo_coords.name}")

    # Upload de shapefile
    arquivo_shape = st.sidebar.file_uploader(
        "📁 Shapefile (ZIP)",
        type=['zip', 'shp'],
        help="Arquivo shapefile com polígonos dos talhões",
        key="upload_shapefile_sidebar"
    )

    if arquivo_shape is not None:
        st.session_state.arquivo_shapefile = arquivo_shape
        st.sidebar.success(f"✅ Shapefile: {arquivo_shape.name}")

    # Mostrar status
    if hasattr(st.session_state, 'arquivo_coordenadas') and st.session_state.arquivo_coordenadas:
        st.sidebar.info(f"📍 Coordenadas ativas: {st.session_state.arquivo_coordenadas.name}")

    if hasattr(st.session_state, 'arquivo_shapefile') and st.session_state.arquivo_shapefile:
        st.sidebar.info(f"📁 Shapefile ativo: {st.session_state.arquivo_shapefile.name}")


# FUNÇÃO PARA CORRIGIR O PROBLEMA NO PRINCIPAL.PY
def processar_uploads_corrigido(arquivos):
    """
    Versão corrigida do processamento de uploads no Principal.py
    """
    arquivos_processados = False

    # Processar inventário
    if arquivos['inventario'] is not None:
        df_inventario = carregar_arquivo(arquivos['inventario'])
        if df_inventario is not None:
            st.session_state.dados_inventario = df_inventario
            st.sidebar.success(f"✅ Inventário: {len(df_inventario)} registros")
            arquivos_processados = True

    # Processar cubagem
    if arquivos['cubagem'] is not None:
        df_cubagem = carregar_arquivo(arquivos['cubagem'])
        if df_cubagem is not None:
            st.session_state.dados_cubagem = df_cubagem
            st.sidebar.success(f"✅ Cubagem: {len(df_cubagem)} medições")
            if arquivos_processados:
                st.session_state.arquivos_carregados = True

    # CORREÇÃO: Armazenar arquivos opcionais corretamente
    if arquivos.get('shapefile') is not None:
        st.session_state.arquivo_shapefile = arquivos['shapefile']
        st.sidebar.info(f"📁 Shapefile: {arquivos['shapefile'].name}")
    else:
        # IMPORTANTE: Manter None se não foi carregado
        if 'arquivo_shapefile' not in st.session_state:
            st.session_state.arquivo_shapefile = None

    if arquivos.get('coordenadas') is not None:
        st.session_state.arquivo_coordenadas = arquivos['coordenadas']
        st.sidebar.info(f"📍 Coordenadas: {arquivos['coordenadas'].name}")
    else:
        # IMPORTANTE: Manter None se não foi carregado
        if 'arquivo_coordenadas' not in st.session_state:
            st.session_state.arquivo_coordenadas = None

    return st.session_state.arquivos_carregados


# FUNÇÃO PARA DEBUG COMPLETO DO SESSION_STATE
def debug_session_state():
    """Função de debug para verificar todo o session_state"""
    with st.expander("🔍 Debug Completo - Session State"):
        st.write("**Todos os atributos do session_state:**")

        for key, value in st.session_state.items():
            if 'arquivo' in key.lower() or 'config' in key.lower():
                st.write(f"**{key}:**")
                if value is None:
                    st.write("  → None")
                elif hasattr(value, 'name'):
                    st.write(f"  → Arquivo: {value.name}")
                elif isinstance(value, dict):
                    st.write(f"  → Dict com {len(value)} itens")
                    for k, v in value.items():
                        st.write(f"    - {k}: {v}")
                else:
                    st.write(f"  → {type(value)}: {value}")
