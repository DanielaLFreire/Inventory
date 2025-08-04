# config/configuracoes_globais.py - VERSÃO CORRIGIDA
"""
Configurações globais centralizadas para todo o sistema - CORRIGIDA
Gerencia filtros e configurações compartilhadas entre todas as etapas

CORREÇÕES:
1. Removida aba duplicada de parâmetros não-lineares
2. Corrigido problema do shapefile não aparecer na lista
3. Removidas funções de debug desnecessárias
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
            'codigos_excluir': [],

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

            # Configurações de modelos (incluindo parâmetros não-lineares)
            'incluir_nao_lineares': True,
            'max_iteracoes': 5000,
            'tolerancia_ajuste': 0.01,

            # Parâmetros iniciais para modelos não-lineares hipsométricos
            'parametros_chapman': {
                'b0': 42.12,  # Altura assintótica
                'b1': 0.01,  # Taxa de crescimento
                'b2': 1.00  # Parâmetro de forma
            },
            'parametros_weibull': {
                'a': 42.12,  # Altura assintótica
                'b': 0.01,  # Parâmetro de escala
                'c': 1.00  # Parâmetro de forma
            },
            'parametros_mononuclear': {
                'a': 42.12,  # Altura assintótica
                'b': 1.00,  # Parâmetro de intercepto
                'c': 0.10  # Taxa de decaimento
            },

            # Estado de configuração
            'configurado': False,
            'timestamp_config': None
        }


def limpar_tipos_nao_serializaveis(config):
    """
    Limpa tipos não-serializáveis das configurações

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


def verificar_arquivos_opcionais_disponiveis():
    """
    CORREÇÃO 2: Função robusta para verificar disponibilidade de arquivos opcionais

    Returns:
        dict: Status dos arquivos opcionais
    """
    arquivos_status = {
        'shapefile_disponivel': False,
        'coordenadas_disponivel': False
    }

    # Verificar shapefile - múltiplas estratégias
    try:
        # Estratégia 1: Verificar atributo direto
        if hasattr(st.session_state, 'arquivo_shapefile'):
            if st.session_state.arquivo_shapefile is not None:
                arquivos_status['shapefile_disponivel'] = True

        # Estratégia 2: Buscar em todas as keys que contenham 'shapefile'
        if not arquivos_status['shapefile_disponivel']:
            for key in st.session_state.keys():
                if 'shapefile' in key.lower():
                    valor = st.session_state[key]
                    if valor is not None and hasattr(valor, 'name'):
                        arquivos_status['shapefile_disponivel'] = True
                        # Atualizar referência principal se necessário
                        if not hasattr(st.session_state, 'arquivo_shapefile'):
                            st.session_state.arquivo_shapefile = valor
                        break

    except Exception as e:
        st.sidebar.warning(f"Erro ao verificar shapefile: {e}")

    # Verificar coordenadas - múltiplas estratégias
    try:
        # Estratégia 1: Verificar atributo direto
        if hasattr(st.session_state, 'arquivo_coordenadas'):
            if st.session_state.arquivo_coordenadas is not None:
                arquivos_status['coordenadas_disponivel'] = True

        # Estratégia 2: Buscar em todas as keys que contenham 'coordenadas'
        if not arquivos_status['coordenadas_disponivel']:
            for key in st.session_state.keys():
                if 'coordenadas' in key.lower():
                    valor = st.session_state[key]
                    if valor is not None and hasattr(valor, 'name'):
                        arquivos_status['coordenadas_disponivel'] = True
                        # Atualizar referência principal se necessário
                        if not hasattr(st.session_state, 'arquivo_coordenadas'):
                            st.session_state.arquivo_coordenadas = valor
                        break

    except Exception as e:
        st.sidebar.warning(f"Erro ao verificar coordenadas: {e}")

    return arquivos_status


def mostrar_configuracoes_globais():
    """
    Interface unificada para todas as configurações do sistema - VERSÃO CORRIGIDA
    CORREÇÃO 1: Removida aba duplicada de parâmetros não-lineares
    """
    st.header("⚙️ Configurações Globais do Sistema")
    st.info("💡 Estas configurações se aplicam a todas as etapas da análise")

    # Verificar se dados estão carregados
    if not hasattr(st.session_state, 'dados_inventario') or st.session_state.dados_inventario is None:
        st.warning("⚠️ Carregue os dados primeiro para configurar filtros específicos")
        return st.session_state.config_global

    df_inventario = st.session_state.dados_inventario

    # CORREÇÃO 1: Criar apenas 4 abas (removida aba duplicada)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Filtros de Dados",
        "📏 Áreas dos Talhões",
        "🌱 Parâmetros Florestais",
        "🧮 Configurações de Modelos"  # Esta aba já inclui parâmetros não-lineares
    ])

    with tab1:
        config_filtros = configurar_filtros_dados(df_inventario)

    with tab2:
        config_areas = configurar_areas_talhoes(df_inventario)

    with tab3:
        config_parametros = configurar_parametros_florestais()

    with tab4:
        config_modelos = configurar_modelos_avancados()

    # Combinar todas as configurações
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
            #default=[str(c) for c in st.session_state.config_global.get('codigos_excluir', ['C', 'I'])],
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

    # CORREÇÃO 2: Verificar arquivos usando função robusta
    arquivos_status = verificar_arquivos_opcionais_disponiveis()

    metodos_disponiveis = ["Simular automaticamente", "Valores específicos por talhão"]

    # CORREÇÃO 2: Adicionar métodos baseado na verificação robusta
    if arquivos_status['shapefile_disponivel']:
        metodos_disponiveis.append("Upload shapefile")
        st.success("📁 Shapefile detectado - Método 'Upload shapefile' disponível")

    if arquivos_status['coordenadas_disponivel']:
        metodos_disponiveis.append("Coordenadas das parcelas")
        st.success("📍 Coordenadas detectadas - Método 'Coordenadas das parcelas' disponível")

    # Se nenhum arquivo adicional foi detectado, mostrar informação
    if not arquivos_status['shapefile_disponivel'] and not arquivos_status['coordenadas_disponivel']:
        st.info("""
        💡 **Para habilitar métodos avançados de área:**
        - Upload um shapefile na página principal para usar "Upload shapefile"
        - Upload coordenadas das parcelas na página principal para usar "Coordenadas das parcelas"
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
    """
    CORREÇÃO 1: Configurações avançadas de modelos incluindo parâmetros não-lineares
    Esta função agora inclui TUDO sobre configuração de modelos em uma única aba
    """
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

    # CORREÇÃO 1: Incluir configuração de parâmetros iniciais diretamente nesta aba
    parametros_modelos = {}

    if incluir_nao_lineares:
        st.markdown("---")
        st.markdown("### 🔧 Parâmetros Iniciais dos Modelos Não-Lineares")
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

        # Botão para resetar parâmetros
        if st.button("🔄 Resetar Parâmetros para Valores Padrão", key="reset_parametros_modelos"):
            st.session_state.config_global.update({
                'parametros_chapman': {'b0': 42.12, 'b1': 0.01, 'b2': 1.00},
                'parametros_weibull': {'a': 42.12, 'b': 0.01, 'c': 1.00},
                'parametros_mononuclear': {'a': 42.12, 'b': 1.00, 'c': 0.10}
            })
            st.success("✅ Parâmetros resetados para valores padrão!")
            st.rerun()

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

        # Calcular impacto dos filtros
        try:
            df_original = df_inventario
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


def mostrar_botoes_configuracao_globais(config):
    """Botões de ação para configurações"""
    import json
    import pandas as pd

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Salvar Configurações", use_container_width=True):
            config_limpa = limpar_tipos_nao_serializaveis(config)
            st.session_state.config_global = config_limpa
            st.success("✅ Configurações salvas!")

    with col2:
        if st.button("🔄 Resetar Padrão", use_container_width=True):
            inicializar_configuracoes_globais()
            st.rerun()

    with col3:
        try:
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

    with col4:
        if st.button("🎯 Aplicar a Todas Etapas", use_container_width=True):
            aplicar_configuracao_todas_etapas(config)
            st.success("✅ Configurações aplicadas!")


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


def exportar_configuracoes_globais(config):
    """Exporta configurações em JSON - VERSÃO CORRIGIDA"""
    import json
    from datetime import datetime
    import numpy as np
    import pandas as pd

    class ConfigJSONEncoder(json.JSONEncoder):
        """Encoder personalizado para lidar com tipos não serializáveis"""

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

    try:
        # Limpar config antes da exportação
        config_limpo = limpar_tipos_nao_serializaveis(config)

        config_export = {
            'timestamp': datetime.now().isoformat(),
            'versao': '1.0',
            'configuracoes': config_limpo
        }

        return json.dumps(config_export, cls=ConfigJSONEncoder, indent=2, ensure_ascii=False)

    except Exception as e:
        st.error(f"❌ Erro na serialização: {e}")

        # Fallback: exportar apenas configurações básicas
        config_basico = {
            'timestamp': datetime.now().isoformat(),
            'versao': '1.0',
            'erro': str(e),
            'configuracoes_basicas': {
                'diametro_min': float(config.get('diametro_min', 4.0)),
                'metodo_area': str(config.get('metodo_area', 'Simular automaticamente')),
                'area_parcela': float(config.get('area_parcela', 400)),
                'configurado': bool(config.get('configurado', False))
            }
        }

        return json.dumps(config_basico, indent=2, ensure_ascii=False)


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